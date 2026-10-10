"""Veda attention functions plugged into H3DiT.forward.

* TeacherCollector (stage 1): returns the dense output and, per layer and
  head group, trains the predictor against the teacher heat (the residual
  stream always follows the dense teacher).
* SparseStudent (stage 2): block-sparse attention with predictor masks.

Both share ClipTiling, which builds each tile permutation once per clip.
"""

from __future__ import annotations

import dataclasses

import torch

from miowtion.h3 import attention as h3_attention
from miowtion.h3 import layout as h3_layout
from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.kernels import tile_gather_triton
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

# Default VedaConfig.collect_bytes. Note the TeacherCollector holds *two*
# copies at once (q and k, until the heat is built), so its peak is twice
# this. 256 MiB, not 512: on a 24 GB card at 104k tokens the 1 GiB that
# cost left 20 MB free.
DEFAULT_COLLECT_BYTES = 256 * 2**20


@dataclasses.dataclass(frozen=True)
class VedaConfig:
    """Sparsity settings shared by training and evaluation.

    Attributes:
        target_budget: Budget of the target column block.
        ref_budget: Budget of the condition column block (tiled conditions).
        tile_conditions: Tile keyframes / reference visuals as their own
            spans (least-padding shape) instead of keeping them global.
        teacher_q_tiles: Supervised video query tiles per layer and head
            group: a fraction of the clip's video tiles when <= 1, else an
            absolute count. Resolved per clip, since tile counts differ ~3x
            between geometries.
        recall_every: Compute the mask diagnostics on every n-th layer.
        dense_layers: Layers that stay dense in the sparse student.
        transport_weight: Weight on the retained-transport-mass loss
            (heatmap.transport_loss). It is the only one of the three
            that is invariant to the per-row affine maps a top-k read-out
            is invariant to, so it is the one whose gradient cannot be
            spent on directions the kernel cannot see, and the only one
            whose optimum is the metric the run is judged on.
        kl_weight: Weight on the seer KL. 0.0 leaves the oracle top-k BCE
            as the only objective, which is the one that matches what the
            kernel reads; it used to be impossible to express because the
            trainer's scale multiplied both terms at once.
        kl_direction: 'forward' for KL(teacher || student), which is
            mass-covering and flattens, or 'reverse' for
            KL(student || teacher), which is mode-seeking and is what a
            top-k read-out wants. See heatmap.seer_kl.
        heat_reduce: Which teacher heat the predictor is distilled
            against: 'max' for the block's peak probability (Veda1) or
            'sum' for its total attention mass (Veda2). Mass is what
            determines the output error; see docs/features/veda2.md.
        zero_order_compensation: Replace each dropped tile by one
            virtual key at its centroid instead of discarding it, which
            is Sol's zero-order term. Only for the end-to-end ablation
            (plan E4): it is the one comparison where nothing but the
            compensation differs -- same tiles, same router, same kernel.
            Off by default; it costs a [rows, n_tiles] score matrix on
            top of the kernel, which is the cost the ablation measures.
        contiguous_tiles: Cut the sequence into blocks in its own order
            instead of permuting it into 3D tiles. Only for ablations
            against methods that block the sequence as it lies (Sol-Attn
            and most block-sparse attention): `span_tiles` moves both the
            block size and the block shape at once, so attributing a
            difference to either needs each moved separately. No plan
            applies, every head gets the same contiguous cut.
        collect_bytes: Bound on one tile-ordered q / k / v / out copy; a
            head group is processed in chunks of heads under it. Heads are
            independent, so it changes only the launch count, never the
            result. The default fits a 24 GB card at latent_t 102; on a
            card with room, a larger bound runs a whole group per launch,
            which is much faster (docs/benchmark/performance.md §11.3).
    """

    heat_reduce: str = 'max'
    transport_weight: float = 0.0
    gauge_kl_weight: float = 0.0
    kl_weight: float = 1.0
    kl_direction: str = 'forward'
    target_budget: veda_mask.Budget = veda_mask.Budget(ratio=0.1)
    ref_budget: veda_mask.Budget | None = None
    tile_conditions: bool = False
    contiguous_tiles: bool = False
    zero_order_compensation: bool = False
    teacher_q_tiles: float = 1.0
    recall_every: int = 1
    dense_layers: frozenset[int] = frozenset()
    collect_bytes: int = DEFAULT_COLLECT_BYTES


class ClipTiling:
    """Tile layouts of one clip, one per target shape used by the plan."""

    def __init__(self, layout: h3_layout.PackedLayout,
                 config: VedaConfig, device: torch.device,
                 condition_spans: tuple[tiling.TiledSpan, ...] = ()):
        """Initializes the cache.

        Args:
            layout: Packed layout of the clip.
            config: Veda settings.
            device: Device of the tile layouts.
            condition_spans: Extra tiled condition spans that are not
                layout.spans (e.g. vision-language rows inside the text
                segment whose grid is known).
        """
        self.layout = layout
        self.config = config
        self.device = device
        spans = list(condition_spans)
        if config.tile_conditions:
            spans += [tiling.TiledSpan(s.start, s.grid,
                                       tiling.least_padding_shape(s.grid))
                      for s in layout.spans[:-1]]
        self._condition_spans = sorted(spans, key=lambda s: s.start)
        self._cache: dict[tiling.TileShape, tiling.TileLayout] = {}

    def get(self, shape: tiling.TileShape) -> tiling.TileLayout:
        if shape not in self._cache:
            target = self.layout.target
            spans = self._condition_spans + [
                tiling.TiledSpan(target.start, target.grid, shape)]
            tiler = (tiling.contiguous_span_tiles
                     if self.config.contiguous_tiles else tiling.span_tiles)
            self._cache[shape] = tiling.build_tile_layout(
                spans, self.layout.used, self.layout.seq_len, self.device,
                tiler=tiler)
        return self._cache[shape]

    def blocks(self, tile_layout: tiling.TileLayout
               ) -> list[veda_mask.ColumnBlock]:
        return veda_mask.column_blocks(tile_layout, self.config.target_budget,
                                       self.config.ref_budget)

    def num_supervised(self, tile_layout: tiling.TileLayout) -> int:
        n_video = tile_layout.n_video_tiles
        value = self.config.teacher_q_tiles
        count = round(value * n_video) if value <= 1 else int(value)
        return max(1, min(n_video, count))


# Bytes of one bf16 head_dim-128 row of a tile-ordered copy.
_HEAD_DIM_BYTES = 128 * 2


def _chunk_heads(tile_layout: tiling.TileLayout, collect_bytes: int) -> int:
    """Heads per chunk so one tile-ordered copy stays under collect_bytes."""
    return max(1, collect_bytes // (tile_layout.num_slots * _HEAD_DIM_BYTES))


def _fused(x: torch.Tensor) -> bool:
    return tile_gather_triton.supports(x)


def _gather(x: torch.Tensor, tile_layout: tiling.TileLayout,
            heads: torch.Tensor) -> torch.Tensor:
    if _fused(x):
        return tile_gather_triton.gather_tiles(x, tile_layout, heads)
    return tiling.gather_tiles(x, tile_layout, heads)


def _gather_and_pool(x: torch.Tensor, tile_layout: tiling.TileLayout,
                     heads: torch.Tensor, second: str | None = None
                     ) -> tuple:
    """Tile-ordered rows and predictor features (one fused pass on CUDA).

    Args:
        x: [S, H, D] packed rows.
        tile_layout: Tile layout.
        heads: Heads to gather.
        second: Also return the second moment the predictor's extra head
            needs on this side: `predictor.SECOND_RAW` for the query side,
            `SECOND_CENTRAL` for the key side, None for neither.

    Returns:
        (rows, features), plus the second moment when asked.
    """
    if second is None:
        if _fused(x):
            return tile_gather_triton.gather_and_pool(x, tile_layout, heads)
        tiles = tiling.gather_tiles(x, tile_layout, heads)
        return tiles, veda_predictor.pool_tiles(tiles, tile_layout)
    if _fused(x):
        tiles, feats, sq, var = tile_gather_triton.gather_and_pool(
            x, tile_layout, heads, second=True)
        return tiles, feats, (sq if second == veda_predictor.SECOND_RAW
                              else var)
    tiles = tiling.gather_tiles(x, tile_layout, heads)
    return (tiles, veda_predictor.pool_tiles(tiles, tile_layout),
            veda_predictor.pool_tiles(tiles, tile_layout, second))


def _extra_features(predictor: veda_predictor.TileScorePredictor,
                    tile_layout: tiling.TileLayout,
                    sq_q: torch.Tensor | None,
                    var_k: torch.Tensor | None,
                    rows: torch.Tensor | None = None) -> dict:
    """Inputs the predictor's optional score terms need, as kwargs.

    The second moments come from `_gather_and_pool`, which computes them
    inside the fused gather kernel. The log B_j term wants nothing but the
    layout. Returns {} for a plain predictor, so the call site reads the
    same either way.

    Args:
        predictor: The predictor about to be called.
        tile_layout: Layout of the gathered tiles.
        sq_q: [H', n_tiles, D] pooled E[q^2], or None.
        var_k: [H', n_tiles, D] pooled Var(k), or None.
        rows: Query tiles the logits are restricted to, if any; the query
            side is sliced to match.

    Returns:
        Keyword arguments for `LayerPredictor.forward`.

    Raises:
        ValueError: If the predictor wants a second moment and none came.
    """
    extra: dict = {}
    if predictor.count_term:
        extra['log_count'] = torch.log(
            tile_layout.valid_count.clamp(min=1).to(torch.float32))
    if predictor.second_order_rank:
        if sq_q is None or var_k is None:
            raise ValueError('the second-order head needs the pooled '
                             'second moments; gather with second=...')
        extra['sq_q'] = sq_q if rows is None else sq_q[:, rows]
        extra['var_k'] = var_k
    return extra



def _scatter_(out: torch.Tensor, tiled: torch.Tensor,
              tile_layout: tiling.TileLayout, heads: torch.Tensor) -> None:
    if _fused(out):
        tile_gather_triton.scatter_tiles_(out, tiled, tile_layout, heads)
    else:
        tiling.scatter_tiles_(out, tiled, tile_layout, heads)


def _gather_lse(lse: torch.Tensor, tile_layout: tiling.TileLayout,
                heads: torch.Tensor) -> torch.Tensor:
    """[S, H] LSE -> [N, H'] tile order, 0 on padding slots."""
    out = lse[tile_layout.gather_index[:, None], heads[None, :]]
    if tile_layout.pad_slots.numel():
        out.index_fill_(0, tile_layout.pad_slots, 0.0)
    return out


# Diagnostic fields of LayerStats, in the order resolve() transfers them.
_STAT_FIELDS = ('kl', 'topk_bce', 'transport', 'logit_std', 'recall',
                'gauge_kl', 'heat_kept', 'heat_ceiling', 'retained')


@dataclasses.dataclass
class LayerStats:
    """Training diagnostics, kept on the device until `resolve()`.

    Every entry is a 0-d device tensor. Reading them one by one would
    synchronize once per layer and head group (50 layers x 2 groups per
    micro step), so they are stacked and transferred once at the end.

    `kl` has exactly one entry per layer; the others have one per layer and
    head group, and only for the layers where they were computed.
    """

    kl: list[torch.Tensor] = dataclasses.field(default_factory=list)
    topk_bce: list[torch.Tensor] = dataclasses.field(default_factory=list)
    transport: list[torch.Tensor] = dataclasses.field(default_factory=list)
    gauge_kl: list[torch.Tensor] = dataclasses.field(default_factory=list)
    retained: list[torch.Tensor] = dataclasses.field(default_factory=list)
    logit_std: list[torch.Tensor] = dataclasses.field(default_factory=list)
    recall: list[torch.Tensor] = dataclasses.field(default_factory=list)
    heat_kept: list[torch.Tensor] = dataclasses.field(default_factory=list)
    heat_ceiling: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def resolve(self) -> dict[str, list[float]]:
        """Host values of every field, in one device transfer."""
        fields = {name: getattr(self, name) for name in _STAT_FIELDS}
        flat = [t for values in fields.values() for t in values]
        if not flat:
            return {name: [] for name in fields}
        host = torch.stack([t.detach().float().reshape(())
                            for t in flat]).cpu().tolist()
        out, start = {}, 0
        for name, values in fields.items():
            out[name] = host[start:start + len(values)]
            start += len(values)
        return out


class TeacherCollector:
    """Stage-1 attention: dense output + immediate predictor KL backward.

    The KL of every (layer, head group) is backpropagated right away (scaled
    so the sum equals the head-weighted mean over layers, divided by the
    gradient-accumulation steps), so no graph is held across layers.
    """

    def __init__(self, clip: ClipTiling, plan: veda_plan.TilePlan,
                 predictor: veda_predictor.TileScorePredictor,
                 generator: torch.Generator, grad_scale: float,
                 dense_backend: str = 'auto', topk_weight: float = 0.0):
        """Initializes the collector.

        Args:
            clip: Tile layouts and sparsity settings of this clip.
            plan: Tile shapes per layer and head.
            predictor: The trained scorer.
            generator: Draws the supervised query tiles.
            grad_scale: Scale of every KL backward.
            dense_backend: Kernel for the teacher's dense attention.
            topk_weight: Weight of heatmap.oracle_bce() added to the KL.
                0 disables the term and skips its oracle top-k.
        """
        self.clip = clip
        self.plan = plan
        self.predictor = predictor
        self.generator = generator
        self.grad_scale = grad_scale
        self.dense_backend = dense_backend
        self.topk_weight = topk_weight
        self.stats = LayerStats()
        self._num_heads = sum(
            g.heads.numel() for g in plan.head_groups(0, clip.device))

    def _sample_rows(self, tile_layout: tiling.TileLayout) -> torch.Tensor:
        count = self.clip.num_supervised(tile_layout)
        rows = torch.randperm(tile_layout.n_video_tiles,
                              generator=self.generator)[:count]
        return rows.sort().values.to(self.clip.device)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        out, lse = h3_attention.dense_attention(
            q, k, v, self.clip.layout.used, return_lse=True,
            backend=self.dense_backend)
        layer_kl = None
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            rows = self._sample_rows(tile_layout)
            # Heads are independent, so a group is processed in chunks of
            # heads: the tile-ordered q / k copies of a whole group (1.5 GB
            # each at 103k tokens) would not fit next to the trunk.
            for heads in group.heads.split(
                    _chunk_heads(tile_layout, self.clip.config.collect_bytes)):
                want = self.predictor.second_order_rank
                q_tiles, feats_q, sq_q = (
                    _gather_and_pool(q, tile_layout, heads,
                                     veda_predictor.SECOND_RAW) if want
                    else (*_gather_and_pool(q, tile_layout, heads), None))
                k_tiles, feats_k, var_k = (
                    _gather_and_pool(k, tile_layout, heads,
                                     veda_predictor.SECOND_CENTRAL) if want
                    else (*_gather_and_pool(k, tile_layout, heads), None))
                heat = heatmap.teacher_heat(
                    q_tiles, k_tiles, _gather_lse(lse, tile_layout, heads),
                    tile_layout, rows, self.clip.config.heat_reduce)
                extra = _extra_features(self.predictor, tile_layout, sq_q,
                                        var_k, rows)
                del q_tiles, k_tiles
                weight = heads.numel() / self._num_heads
                with torch.enable_grad():
                    logits = self.predictor.layers[layer_index](
                        feats_q[:, rows], feats_k, heads, **extra)
                    kl = heatmap.seer_kl(logits, heat, tile_layout,
                                         self.clip.config.kl_direction)
                    loss = self.clip.config.kl_weight * kl
                    if self.clip.config.transport_weight:
                        transport = heatmap.transport_loss(
                            logits, heat, tile_layout,
                            self.clip.blocks(tile_layout), rows)
                        loss = loss + (self.clip.config.transport_weight
                                       * transport)
                        self.stats.transport.append(transport.detach())
                    if self.clip.config.gauge_kl_weight:
                        gkl = heatmap.gauge_fixed_kl(
                            logits, heat, tile_layout, rows)
                        loss = loss + (self.clip.config.gauge_kl_weight
                                       * gkl)
                        self.stats.gauge_kl.append(gkl.detach())
                    if self.topk_weight:
                        bce = heatmap.oracle_bce(
                            logits, heat, tile_layout,
                            self.clip.blocks(tile_layout), rows)
                        loss = loss + self.topk_weight * bce
                        self.stats.topk_bce.append(bce.detach())
                    (loss * (weight * self.grad_scale)).backward()
                # Accumulated on the device: .item() here would synchronize
                # once per layer and head group.
                term = kl.detach() * weight
                layer_kl = term if layer_kl is None else layer_kl + term
                # A collapsing predictor scores every key tile alike; the
                # spread of its logits is the cheapest way to see it.
                # Video columns only. Global key tiles are always kept
                # and never ranked, and their second-order term sits at a
                # wildly different level, so including them made this read
                # 192 where the spread that actually decides anything is
                # about 4.
                self.stats.logit_std.append(
                    logits.detach()[:, :, :tile_layout.n_video_tiles]
                    .std(dim=-1).mean())
                # Diagnostics are cheap: they work on the /128 tile grid,
                # where a top-k costs ~1e-6 of the heat that produced it.
                if layer_index % self.clip.config.recall_every == 0:
                    values = heatmap.mask_diagnostics(
                        logits.detach(), heat, tile_layout,
                        self.clip.blocks(tile_layout), rows)
                    for name, value in values.items():
                        getattr(self.stats, name).append(value)
        self.stats.kl.append(layer_kl)
        return out




ORACLE_TARGETS = ('mass', 'max', 'mean')


class OracleStudent:
    """Block sparsity masked by the *true* block statistic, not a predictor.

    An upper bound, and the experiment that says which upper bound. Veda
    showed that approaching an oracle is what matters; this answers which
    oracle a router should approach, at the level of the generated video
    rather than of the attention output error.

    Three targets, which differ by exactly the terms of the cumulant
    expansion (see miowtion/veda/predictor.py):

    * `mass`: the block's summed attention probability. Derived as
      optimal -- restricting a one-sided entropic transport plan to a
      block set S attains log sum_{v in S} exp(s_uv), so the best set of
      size k is the top-k by mass, exactly.
    * `max`: the block's peak probability, which is the target Veda's
      released scorer was distilled against. An extreme-value statistic.
    * `mean`: mass divided by the key tile's real row count, i.e. mass
      with the log B_j term removed. Separates "mass is right" from
      "counting rows is right".

    Cost is dense plus sparse per layer: the heat needs the dense LSE, so
    this is an analysis mode and never an inference path.
    """

    def __init__(self, clip: ClipTiling, plan: veda_plan.TilePlan,
                 target: str = 'mass',
                 allow_reference_kernel: bool = False):
        """Initializes the oracle.

        Args:
            clip: Clip tiling cache.
            plan: Tile plan of the clip geometry.
            target: One of ORACLE_TARGETS.
            allow_reference_kernel: Permit the token-level reference
                kernel (CPU tests only).

        Raises:
            ValueError: On an unknown target.
            RuntimeError: If FA4 is unavailable and the reference kernel
                is not explicitly allowed.
        """
        if target not in ORACLE_TARGETS:
            raise ValueError(f'target must be one of {ORACLE_TARGETS}: '
                             f'{target!r}')
        if not fa4.available(clip.device) and not allow_reference_kernel:
            raise RuntimeError('OracleStudent needs the FA4 block-sparse '
                               'kernel; refusing to fall back silently')
        self.clip = clip
        self.plan = plan
        self.target = target
        self.use_fa4 = fa4.available(clip.device)
        self.calls = {'sparse': 0, 'dense': 0}

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        used = self.clip.layout.used
        if layer_index in self.clip.config.dense_layers:
            self.calls['dense'] += 1
            return h3_attention.dense_attention(q, k, v, used)[0]
        self.calls['sparse'] += 1
        # The oracle statistic is defined through the teacher's own
        # normalisation, so the dense pass is not optional here.
        with torch.no_grad():
            _, lse = h3_attention.dense_attention(q, k, v, used,
                                                  return_lse=True)
        seq_len = q.shape[0]
        out = q.new_zeros(seq_len + 1, *q.shape[1:])
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            for heads in group.heads.split(
                    _chunk_heads(tile_layout,
                                 self.clip.config.collect_bytes)):
                self._attend(q, k, v, lse, tile_layout, heads, out)
        out = out[:seq_len]
        if used < seq_len:
            out[used:] = 0
        return out

    def _attend(self, q, k, v, lse, tile_layout: tiling.TileLayout,
                heads: torch.Tensor, out: torch.Tensor) -> None:
        """Oracle-masked attention of some heads of one group."""
        q_tiles = _gather(q, tile_layout, heads)
        k_tiles = _gather(k, tile_layout, heads)
        v_tiles = _gather(v, tile_layout, heads)
        rows = torch.arange(tile_layout.n_video_tiles, device=q.device)
        with torch.no_grad():
            reduce = 'max' if self.target == 'max' else 'sum'
            heat = heatmap.teacher_heat(
                q_tiles, k_tiles, _gather_lse(lse, tile_layout, heads),
                tile_layout, rows, reduce)
            if self.target == 'mean':
                # Mass per real key row: the same ranking as mass with the
                # log B_j term taken back out.
                counts = tile_layout.valid_count.clamp(min=1).to(heat.dtype)
                heat = heat / counts[None, None, :]
            selection = veda_mask.select_video_blocks(
                heat[:, :, :tile_layout.n_video_tiles], tile_layout,
                self.clip.blocks(tile_layout), rows)
            block_mask = veda_mask.dense_block_mask(selection, tile_layout)
        if self.use_fa4:
            o_tiles = fa4.block_sparse_attention(q_tiles, k_tiles, v_tiles,
                                                 block_mask, tile_layout)
        else:
            o_tiles = reference.block_sparse_attention(
                q_tiles, k_tiles, v_tiles, block_mask,
                tile_layout.valid_count)
        _scatter_(out, o_tiles, tile_layout, heads)


def zero_order_compensation(
        o_kept: torch.Tensor, lse_kept: torch.Tensor, q_tiles: torch.Tensor,
        k_tiles: torch.Tensor, v_tiles: torch.Tensor,
        block_mask: torch.Tensor, tile_layout: tiling.TileLayout,
        chunk_rows: int = 4096) -> torch.Tensor:
    """Adds Sol's zero-order term for the tiles the mask dropped.

    A dropped tile is not discarded but replaced by one virtual key at
    its centroid: key `mean_j k` carrying value `sum_j v`, weighted as if
    it stood for all `B_j` of the tile's real rows. This is the `c = 1`
    arm of `solattn.relative_errors`, the one whose ceiling we measured,
    reproduced here so it can run end to end rather than only offline.

    Two facts make that possible inside a sparse path:

    * **The exponential reference cancels.** Numerator and denominator
      both carry `exp(-ref_u)`, so the output does not depend on `ref`
      and in particular does not need the dense LSE, which is what made
      this look impossible. Verified to 4e-16 in float64 against the
      dense-LSE form.
    * **Taking `ref = lse_kept` makes the kept denominator exactly 1**,
      because `lse_kept` is by definition `log sum_kept exp(s)`. The
      kernel's output is already `num/den`, so with that reference it
      *is* the numerator and the whole combination collapses to
      `(o_kept + extra) / (1 + den_extra)`.

    Args:
        o_kept: [R*128, H', D] block-sparse kernel output, i.e. num/den
            over the kept keys.
        lse_kept: [H', R*128] fp32 log-sum-exp over the kept keys.
        q_tiles: [R*128, H', D] tile-ordered queries.
        k_tiles: [N*128, H', D] tile-ordered keys, padding slots zeroed.
        v_tiles: [N*128, H', D] tile-ordered values, padding slots zeroed.
        block_mask: [H', R, n_tiles] bool, True where the tile was kept.
        tile_layout: Tile layout, for the real row count of each key tile.
        chunk_rows: Query rows per chunk. The term needs a
            [rows, n_tiles] score matrix, which is per *row* and not per
            query tile, so it cannot be materialised whole.

    Returns:
        [R*128, H', D] in o_kept.dtype.

    Raises:
        ValueError: On a shape mismatch.
        RuntimeError: If a query row kept no key at all, which would make
            the reference -inf. Every real row keeps its own diagonal
            tile, so this means the mask is malformed rather than that a
            fallback is needed.
    """
    n_rows, n_heads, head_dim = o_kept.shape
    n_tiles = tile_layout.n_tiles
    if lse_kept.shape != (n_heads, n_rows):
        raise ValueError(f'lse_kept is {tuple(lse_kept.shape)}, expected '
                         f'{(n_heads, n_rows)}')
    if block_mask.shape != (n_heads, n_rows // tiling.TILE_SIZE, n_tiles):
        raise ValueError(
            f'block_mask is {tuple(block_mask.shape)}, expected '
            f'{(n_heads, n_rows // tiling.TILE_SIZE, n_tiles)}')
    scale = head_dim ** -0.5
    counts = tile_layout.valid_count.to(torch.float32).clamp(min=1.0)
    # Padding slots are zeroed by the gather, so a sum over all 128 slots
    # of a tile is a sum over its real rows.
    k_mean = (k_tiles.float().view(n_tiles, tiling.TILE_SIZE, n_heads,
                                   head_dim).sum(1) / counts[:, None, None])
    v_sum = v_tiles.float().view(n_tiles, tiling.TILE_SIZE, n_heads,
                                 head_dim).sum(1)
    out = torch.empty_like(o_kept)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        ref = lse_kept[:, start:stop]                        # [H', C]
        owner = torch.arange(start, stop, device=o_kept.device
                             ) // tiling.TILE_SIZE
        keep = block_mask[:, owner, :]                       # [H', C, N]
        if not bool(torch.isfinite(ref[keep.any(-1)]).all()):
            raise RuntimeError('a query row kept at least one tile but '
                               'its lse is not finite')
        drop = (~keep).to(torch.float32)
        s_hat = torch.einsum(
            'chd,nhd->hcn', q_tiles[start:stop].float(), k_mean) * scale
        # A row that kept nothing has ref -inf; it carries no mass either
        # way, and the gather already zeroed it, so neutralise the term
        # rather than producing inf - inf.
        safe = torch.where(torch.isfinite(ref), ref,
                           torch.zeros_like(ref))
        p_hat = torch.exp(s_hat - safe[..., None]) * drop     # [H', C, N]
        num = torch.einsum('hcn,nhd->chd', p_hat, v_sum)
        den = torch.einsum('hcn,n->hc', p_hat, counts)
        out[start:stop] = ((o_kept[start:stop].float() + num)
                           / (1.0 + den).transpose(0, 1)[..., None]
                           ).to(o_kept.dtype)
    return out


class SparseStudent:
    """Stage-2 / evaluation attention: predictor-masked block sparsity."""

    def __init__(self, clip: ClipTiling, plan: veda_plan.TilePlan,
                 predictor: veda_predictor.TileScorePredictor,
                 allow_reference_kernel: bool = False):
        """Initializes the student.

        Args:
            clip: Clip tiling cache.
            plan: Tile plan of the clip geometry.
            predictor: Trained predictor (used without gradient).
            allow_reference_kernel: Permit the token-level reference kernel
                (CPU tests only; far too slow for real sequences).

        Raises:
            RuntimeError: If FA4 is unavailable and the reference kernel is
                not explicitly allowed.
        """
        # Ask about the device the clip is actually on: a CUDA machine
        # answers True for a CPU clip otherwise, and the kernel then dies.
        if not fa4.available(clip.device) and not allow_reference_kernel:
            raise RuntimeError('SparseStudent needs the FA4 block-sparse '
                               'kernel; refusing to fall back silently')
        self.clip = clip
        self.plan = plan
        self.predictor = predictor
        self.use_fa4 = fa4.available(clip.device)
        self.calls = {'sparse': 0, 'dense': 0}

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        used = self.clip.layout.used
        if layer_index in self.clip.config.dense_layers:
            self.calls['dense'] += 1
            return h3_attention.dense_attention(q, k, v, used)[0]
        self.calls['sparse'] += 1
        seq_len = q.shape[0]
        out = q.new_zeros(seq_len + 1, *q.shape[1:])
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            # Chunks of heads bound the tile-ordered q / k / v / out copies
            # (heads are independent throughout, so this is exact).
            for heads in group.heads.split(
                    _chunk_heads(tile_layout, self.clip.config.collect_bytes)):
                self._attend(q, k, v, layer_index, tile_layout, heads, out)
        out = out[:seq_len]
        if used < seq_len:
            out[used:] = 0
        return out

    def _attend(self, q, k, v, layer_index: int,
                tile_layout: tiling.TileLayout, heads: torch.Tensor,
                out: torch.Tensor) -> None:
        """Sparse attention of some heads of one group, scattered to out."""
        want = self.predictor.second_order_rank
        q_tiles, feats_q, sq_q = (
            _gather_and_pool(q, tile_layout, heads,
                             veda_predictor.SECOND_RAW) if want
            else (*_gather_and_pool(q, tile_layout, heads), None))
        k_tiles, feats_k, var_k = (
            _gather_and_pool(k, tile_layout, heads,
                             veda_predictor.SECOND_CENTRAL) if want
            else (*_gather_and_pool(k, tile_layout, heads), None))
        v_tiles = _gather(v, tile_layout, heads)
        with torch.no_grad():
            logits = self.predictor.layers[layer_index](
                feats_q, feats_k, heads,
                **_extra_features(self.predictor, tile_layout, sq_q,
                                  var_k))
            # Only video query tiles are selected; global rows are dense.
            selection = veda_mask.select_video_blocks(
                logits[:, :tile_layout.n_video_tiles], tile_layout,
                self.clip.blocks(tile_layout))
            block_mask = veda_mask.dense_block_mask(selection, tile_layout)
        compensate = self.clip.config.zero_order_compensation
        if self.use_fa4:
            result = fa4.block_sparse_attention(
                q_tiles, k_tiles, v_tiles, block_mask, tile_layout,
                return_lse=compensate)
        else:
            result = reference.block_sparse_attention(
                q_tiles, k_tiles, v_tiles, block_mask,
                tile_layout.valid_count, return_lse=compensate)
        if compensate:
            o_tiles = zero_order_compensation(
                *result, q_tiles, k_tiles, v_tiles, block_mask, tile_layout)
        else:
            o_tiles = result
        _scatter_(out, o_tiles, tile_layout, heads)
