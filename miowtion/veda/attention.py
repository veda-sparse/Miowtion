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
        recall_every: Compute mask recall on every n-th layer (diagnostic).
        dense_layers: Layers that stay dense in the sparse student.
    """

    target_budget: veda_mask.Budget = veda_mask.Budget(ratio=0.1)
    ref_budget: veda_mask.Budget | None = None
    tile_conditions: bool = False
    teacher_q_tiles: float = 1.0
    recall_every: int = 4
    dense_layers: frozenset[int] = frozenset()


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
            self._cache[shape] = tiling.build_tile_layout(
                spans, self.layout.used, self.layout.seq_len, self.device)
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


# Bound on one tile-ordered q / k / v / out copy in TeacherCollector and
# SparseStudent (heads are processed in chunks under it), and the bytes of
# one bf16 head_dim-128 row.
_COLLECT_BYTES = 512 * 2**20
_HEAD_DIM_BYTES = 128 * 2


def _chunk_heads(tile_layout: tiling.TileLayout) -> int:
    """Heads per chunk so one tile-ordered copy stays under _COLLECT_BYTES."""
    return max(1, _COLLECT_BYTES // (tile_layout.num_slots * _HEAD_DIM_BYTES))


def _fused(x: torch.Tensor) -> bool:
    return x.is_cuda and tile_gather_triton.available()


def _gather(x: torch.Tensor, tile_layout: tiling.TileLayout,
            heads: torch.Tensor) -> torch.Tensor:
    if _fused(x):
        return tile_gather_triton.gather_tiles(x, tile_layout, heads)
    return tiling.gather_tiles(x, tile_layout, heads)


def _gather_and_pool(x: torch.Tensor, tile_layout: tiling.TileLayout,
                     heads: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile-ordered rows and predictor features (one fused pass on CUDA)."""
    if _fused(x):
        return tile_gather_triton.gather_and_pool(x, tile_layout, heads)
    tiles = tiling.gather_tiles(x, tile_layout, heads)
    return tiles, veda_predictor.pool_tiles(tiles, tile_layout)


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


@dataclasses.dataclass
class LayerStats:
    kl: list[float] = dataclasses.field(default_factory=list)
    recall: list[float] = dataclasses.field(default_factory=list)


class TeacherCollector:
    """Stage-1 attention: dense output + immediate predictor KL backward.

    The KL of every (layer, head group) is backpropagated right away (scaled
    so the sum equals the head-weighted mean over layers, divided by the
    gradient-accumulation steps), so no graph is held across layers.
    """

    def __init__(self, clip: ClipTiling, plan: veda_plan.TilePlan,
                 predictor: veda_predictor.TileScorePredictor,
                 generator: torch.Generator, grad_scale: float,
                 dense_backend: str = 'auto'):
        self.clip = clip
        self.plan = plan
        self.predictor = predictor
        self.generator = generator
        self.grad_scale = grad_scale
        self.dense_backend = dense_backend
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
        layer_kl = 0.0
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            rows = self._sample_rows(tile_layout)
            # Heads are independent, so a group is processed in chunks of
            # heads: the tile-ordered q / k copies of a whole group (1.5 GB
            # each at 103k tokens) would not fit next to the trunk.
            for heads in group.heads.split(_chunk_heads(tile_layout)):
                q_tiles, feats_q = _gather_and_pool(q, tile_layout, heads)
                k_tiles, feats_k = _gather_and_pool(k, tile_layout, heads)
                heat = heatmap.teacher_heat(
                    q_tiles, k_tiles, _gather_lse(lse, tile_layout, heads),
                    tile_layout, rows)
                del q_tiles, k_tiles
                weight = heads.numel() / self._num_heads
                with torch.enable_grad():
                    logits = self.predictor.layers[layer_index](
                        feats_q[:, rows], feats_k, heads)
                    kl = heatmap.seer_kl(logits, heat, tile_layout)
                    (kl * (weight * self.grad_scale)).backward()
                layer_kl += weight * kl.item()
                if layer_index % self.clip.config.recall_every == 0:
                    self.stats.recall.append(heatmap.mask_recall(
                        logits.detach(), heat, tile_layout,
                        self.clip.blocks(tile_layout), rows).item())
        self.stats.kl.append(layer_kl)
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
        if not fa4.available() and not allow_reference_kernel:
            raise RuntimeError('SparseStudent needs the FA4 block-sparse '
                               'kernel; refusing to fall back silently')
        self.clip = clip
        self.plan = plan
        self.predictor = predictor
        self.use_fa4 = fa4.available()
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
            for heads in group.heads.split(_chunk_heads(tile_layout)):
                self._attend(q, k, v, layer_index, tile_layout, heads, out)
        out = out[:seq_len]
        if used < seq_len:
            out[used:] = 0
        return out

    def _attend(self, q, k, v, layer_index: int,
                tile_layout: tiling.TileLayout, heads: torch.Tensor,
                out: torch.Tensor) -> None:
        """Sparse attention of some heads of one group, scattered to out."""
        q_tiles, feats_q = _gather_and_pool(q, tile_layout, heads)
        k_tiles, feats_k = _gather_and_pool(k, tile_layout, heads)
        v_tiles = _gather(v, tile_layout, heads)
        with torch.no_grad():
            logits = self.predictor.layers[layer_index](feats_q, feats_k,
                                                        heads)
            # Only video query tiles are selected; global rows are dense.
            selection = veda_mask.select_video_blocks(
                logits[:, :tile_layout.n_video_tiles], tile_layout,
                self.clip.blocks(tile_layout))
            block_mask = veda_mask.dense_block_mask(selection, tile_layout)
        if self.use_fa4:
            o_tiles = fa4.block_sparse_attention(q_tiles, k_tiles, v_tiles,
                                                 block_mask, tile_layout)
        else:
            o_tiles = reference.block_sparse_attention(
                q_tiles, k_tiles, v_tiles, block_mask,
                tile_layout.valid_count)
        _scatter_(out, o_tiles, tile_layout, heads)
