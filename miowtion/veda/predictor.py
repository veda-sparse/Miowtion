"""Tile-score predictor: which key tiles each query tile should attend.

Per tile, q and k are pooled into [mean | max | min] (3D features). Per
layer and head a residual projection P maps them to D dims:
    q_hat = pool_q @ P_q[h] + mean_q,   k_hat likewise,
and block logits are q_hat . k_hat / sqrt(D) in fp32.

P is initialized N(0, 1e-4), so an untrained predictor already equals
mean-pooled QK, a usable block score.

Mean-pooled QK is not an arbitrary starting point: it is the zero-order
term of the block's log attention mass,

    log E_ij = log B_j + Qbar_i . Kbar_j / sqrt(D)
               + Qbar2_i^T Cov_j Qbar2_i / (2 D) + ...

The second cumulant is real signal that a bilinear form on [mean|max|min]
cannot express, because its diagonal part needs E[q^2] on one side and
Var(k) on the other. Supplying those as a fourth pooled feature is not
enough either: the term is itself an inner product of two D-vectors, so
folding it into the existing rank-D bilinear would make the two compete,
and giving it its own rank-D bilinear would double the n^2 cost that
dominates the predictor.

`second_order_rank` therefore gives it a *low-rank* head of its own:

    logits = q_hat . k_hat / sqrt(D) + (sq_q @ So_q) . (var_k @ So_k)

at a cost of r / D extra on the n^2 term. Set it to 0 (the default) and
the predictor is bit-for-bit what it was. Set it to head_dim and
`init_exact_second_order_()` makes an *untrained* predictor equal the full
diagonal estimate of log mass rather than the zero-order one.

At a rank below head_dim there is no identity to copy, but there is
still a best answer. The term the full-rank head computes is
`c * sq_q . var_k` with `c = 1 / (2 D)`, so a rank-r head wants
projections A, B with `A B^T` as close to `c I` as the data makes
necessary. Weighting by how the two features actually spread,

    error(A, B) = E_ij (sq_q_i (A B^T - c I) var_k_j^T)^2
                = tr(C_u M C_v M^T),   M = A B^T - c I,

with `C_u = E[sq_q^T sq_q]` and `C_v = E[var_k^T var_k]`, the minimizer is
the rank-r truncated SVD of `c * C_u^(1/2) C_v^(1/2)`, undone on each
side. `init_low_rank_second_order_` does that from moments a calibration
pass collects, and at r = head_dim it reproduces the exact term, which is
what pins the two initializations against each other.

`count_term` supplies `log B_j`, which is free: a per-key-tile constant
read straight off the layout, added with one learned gain per head that
starts at the exact coefficient of 1. Without it the score ranks a key
tile holding 32 real keys level with one holding 128, even though the
former carries a quarter of the mass. That blindness is self-consistent
under a block-*maximum* target, which barely moves with the row count,
and becomes a systematic bias the moment the target is mass.

The predictor is a side branch: pooling runs on detached activations under
no_grad, and gradients never reach the trunk.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from miowtion.veda import tiling

INIT_STD = 1e-4

# Pooled statistics a side can ask for beyond [mean | max | min].
SECOND_RAW = 'raw'          # E[x^2], what the query side of the term needs
SECOND_CENTRAL = 'central'  # Var(x), what the key side needs

# Row-chunk bound of the fp32 slab the second moments accumulate over.
# They cannot reuse the mean's trick of summing bf16 straight into an fp32
# accumulator, because the values have to be squared (and centred) first.
_POOL_CHUNK_BYTES = 64 << 20


@torch.no_grad()
def pool_tiles(x: torch.Tensor, layout: tiling.TileLayout,
               second: str | None = None) -> torch.Tensor:
    """Masked mean/max/min over the real rows of every tile.

    Args:
        x: [N, H', D] bf16 tile-ordered rows (padding slots are zero), as
            returned by tiling.gather_tiles.
        layout: Its tile layout.
        second: Also return a second-moment feature: 'raw' for E[x^2] (the
            query side of the second cumulant) or 'central' for Var(x)
            (the key side). None returns the three base features only.

    Returns:
        [H', n_tiles, 3D] fp32 features, or [H', n_tiles, D] of the second
        moment alone when `second` is set; exactly 0 for empty tiles.

    Raises:
        ValueError: On an unknown `second`.
    """
    if second not in (None, SECOND_RAW, SECOND_CENTRAL):
        raise ValueError(f'unknown second moment {second!r}')
    n = layout.n_tiles
    tiles = x.detach().view(n, tiling.TILE_SIZE, *x.shape[1:])  # [n,128,H,D]
    count = layout.valid_count.clamp(min=1).to(torch.float32)
    # Sum bf16 with fp32 accumulation instead of upcasting the whole tensor.
    mean = tiles.sum(dim=1, dtype=torch.float32) / count[:, None, None]
    # Padding rows are 0, which is a legal value, so max/min are recomputed
    # with masking only for the partial tiles.
    tmax = tiles.amax(dim=1)
    tmin = tiles.amin(dim=1)
    partial = layout.partial_tiles
    if partial.numel():
        sub = tiles.index_select(0, partial)
        valid = (torch.arange(tiling.TILE_SIZE, device=x.device)[None, :]
                 < layout.valid_count.index_select(0, partial)[:, None])
        valid = valid[:, :, None, None]
        tmax.index_copy_(0, partial, sub.masked_fill(
            ~valid, float('-inf')).amax(dim=1))
        tmin.index_copy_(0, partial, sub.masked_fill(
            ~valid, float('inf')).amin(dim=1))
    if second is not None:
        feats = _second_moment(tiles, mean, count, layout,
                               second)
    else:
        feats = torch.cat([mean, tmax.float(), tmin.float()], dim=-1)
    # where (not multiply): -inf * 0 would be NaN for empty tiles.
    feats = torch.where(layout.kv_ok[:, None, None], feats, 0.0)
    return feats.permute(1, 0, 2).contiguous()  # [H', n, 3D] or [H', n, D]


def _second_moment(tiles: torch.Tensor, mean: torch.Tensor,
                   count: torch.Tensor, layout: tiling.TileLayout,
                   second: str) -> torch.Tensor:
    """[n, H', D] fp32 E[x^2] or Var(x) over the real rows of every tile.

    Centred in two passes on purpose. The one-pass identity
    `E[x^2] - E[x]^2` subtracts two nearly equal large numbers whenever a
    tile's rows sit close together, which is the common case for RMSNorm'd
    keys inside one spatial block: measured relative error reached 31% on a
    random layout, far too much for a feature the predictor then ranks by.
    Squaring in fp32 rather than bf16 comes along for free.

    Args:
        tiles: [n, 128, H', D] tile-ordered rows, padding slots zero.
        mean: [n, H', D] fp32 masked mean of every tile.
        count: [n] fp32 real rows per tile, clamped to >= 1.
        layout: Tile layout.
        second: SECOND_RAW or SECOND_CENTRAL.

    Returns:
        [n, H', D] fp32.
    """
    n_tiles = tiles.shape[0]
    per_row = max(1, n_tiles * mean.shape[-2] * mean.shape[-1] * 4)
    chunk = max(1, _POOL_CHUNK_BYTES // per_row)
    valid = (torch.arange(tiling.TILE_SIZE, device=tiles.device)[None, :]
             < layout.valid_count[:, None])
    acc = torch.zeros_like(mean)
    for start in range(0, tiling.TILE_SIZE, chunk):
        sl = slice(start, min(start + chunk, tiling.TILE_SIZE))
        rows = tiles[:, sl].float()
        if second == SECOND_CENTRAL:
            # Padding rows hold 0, and 0 - mean is not 0, so mask after.
            rows = (rows - mean[:, None]) * valid[:, sl, None, None]
        acc += rows.square().sum(1)
    return acc / count[:, None, None]


def _psd_roots(matrix: torch.Tensor,
               floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric square root of a PSD matrix and its pseudo-inverse.

    Args:
        matrix: [D, D] fp32 symmetric PSD.
        floor: Eigenvalues below this fraction of the largest are dropped
            from the inverse, which is therefore a pseudo-inverse.

    Returns:
        (root, inverse_root) with `root @ root == matrix` on the kept
        subspace.
    """
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp(min=0.0)
    cut = float(values.max()) * floor
    kept = values > cut
    root_values = torch.where(kept, values.sqrt(),
                              torch.zeros_like(values))
    inv_values = torch.where(kept, values.clamp(min=cut).rsqrt(),
                             torch.zeros_like(values))
    root = (vectors * root_values[None, :]) @ vectors.transpose(0, 1)
    inverse = (vectors * inv_values[None, :]) @ vectors.transpose(0, 1)
    return root, inverse


class LayerPredictor(nn.Module):
    """Projections of one layer: proj_q, proj_k [num_heads, 3D, D].

    With `second_order_rank > 0` it also holds so_q, so_k
    [num_heads, D, rank] for the low-rank second-cumulant head, and with
    `count_term` a gain [num_heads] on log B_j.
    """

    def __init__(self, num_heads: int, head_dim: int,
                 second_order_rank: int = 0, count_term: bool = False):
        super().__init__()
        if not 0 <= second_order_rank <= head_dim:
            raise ValueError(f'second_order_rank {second_order_rank} must be '
                             f'in [0, {head_dim}]')
        self.head_dim = head_dim
        self.second_order_rank = second_order_rank
        self.proj_q = nn.Parameter(torch.empty(num_heads, 3 * head_dim,
                                               head_dim))
        self.proj_k = nn.Parameter(torch.empty(num_heads, 3 * head_dim,
                                               head_dim))
        nn.init.normal_(self.proj_q, std=INIT_STD)
        nn.init.normal_(self.proj_k, std=INIT_STD)
        if second_order_rank:
            self.so_q = nn.Parameter(torch.empty(num_heads, head_dim,
                                                 second_order_rank))
            self.so_k = nn.Parameter(torch.empty(num_heads, head_dim,
                                                 second_order_rank))
            nn.init.normal_(self.so_q, std=INIT_STD)
            nn.init.normal_(self.so_k, std=INIT_STD)
        if count_term:
            # Starts at the exact coefficient, so an untrained predictor
            # already scores log B_j correctly.
            self.count_gain = nn.Parameter(torch.ones(num_heads))

    @torch.no_grad()
    def init_exact_second_order_(self) -> None:
        """Warm-start the head to the exact diagonal cumulant term.

        The term is `E[q^2] . Var(k) / (2 D)`, so splitting the factor
        evenly over the two sides and using the identity reproduces it
        exactly. Only possible at full rank.

        Raises:
            ValueError: If the rank is not head_dim.
        """
        if self.second_order_rank != self.head_dim:
            raise ValueError('the exact term needs second_order_rank == '
                             f'head_dim, not {self.second_order_rank}')
        eye = torch.eye(self.head_dim, dtype=self.so_q.dtype,
                        device=self.so_q.device)
        half = math.sqrt(0.5 / self.head_dim)
        self.so_q.copy_(eye * half)
        self.so_k.copy_(eye * half)

    def embed(self, feats: torch.Tensor, heads: torch.Tensor,
              proj: torch.Tensor) -> torch.Tensor:
        """[H', n, 3D] fp32 features -> [H', n, D] tile embeddings."""
        mean = feats[..., :self.head_dim]
        return torch.bmm(feats, proj.index_select(0, heads).float()) + mean

    @torch.no_grad()
    def init_low_rank_second_order_(self, c_u: torch.Tensor,
                                    c_v: torch.Tensor,
                                    floor: float = 1e-6) -> None:
        """Best rank-r head for the diagonal cumulant, given its moments.

        See the module docstring for the derivation. At full rank this is
        `init_exact_second_order_` to within the inverse square roots'
        conditioning, which the test pins.

        Args:
            c_u: [num_heads, D, D] fp32 `E[sq_q^T sq_q]` per head.
            c_v: [num_heads, D, D] fp32 `E[var_k^T var_k]` per head.
            floor: Eigenvalues below this fraction of a head's largest are
                clamped before the inverse square root. Both moments are
                outer products of non-negative vectors, so they are PSD
                but routinely near-singular.

        Raises:
            ValueError: If the head has no rank, or the moments are the
                wrong shape.
        """
        if not self.second_order_rank:
            raise ValueError('no second-order head to initialize')
        heads, dim = self.so_q.shape[0], self.head_dim
        want = (heads, dim, dim)
        for name, moment in (('c_u', c_u), ('c_v', c_v)):
            if tuple(moment.shape) != want:
                raise ValueError(f'{name} has shape {tuple(moment.shape)}, '
                                 f'expected {want}')
        rank = self.second_order_rank
        coefficient = 0.5 / dim
        for head in range(heads):
            root_u, inv_u = _psd_roots(c_u[head].float(), floor)
            root_v, inv_v = _psd_roots(c_v[head].float(), floor)
            gram = coefficient * (root_u @ root_v)
            left, values, right = torch.linalg.svd(gram)
            keep = values[:rank].clamp(min=0.0).sqrt()
            self.so_q[head] = (inv_u @ left[:, :rank]) * keep[None, :]
            self.so_k[head] = (inv_v @ right[:rank].transpose(0, 1)
                               ) * keep[None, :]

    def forward(self, feats_q: torch.Tensor, feats_k: torch.Tensor,
                heads: torch.Tensor, sq_q: torch.Tensor | None = None,
                var_k: torch.Tensor | None = None,
                log_count: torch.Tensor | None = None) -> torch.Tensor:
        """Block logits [H', n_q, n_k] fp32.

        Args:
            feats_q: [H', n_q, 3D] pooled query features.
            feats_k: [H', n_k, 3D] pooled key features.
            heads: [H'] int64 global head indices of the group.
            sq_q: [H', n_q, D] pooled E[q^2]; required iff the second-order
                head is enabled.
            var_k: [H', n_k, D] pooled Var(k); likewise.
            log_count: [n_k] fp32 log of every key tile's real row count;
                required iff `count_term` is on.

        Raises:
            ValueError: If a feature is missing, or given when its term is
                disabled.
        """
        has_count = log_count is not None
        if hasattr(self, 'count_gain') != has_count:
            raise ValueError('log_count and count_term must be set '
                             'together')
        q_hat = self.embed(feats_q, heads, self.proj_q)
        k_hat = self.embed(feats_k, heads, self.proj_k)
        logits = torch.bmm(q_hat, k_hat.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        has_second = sq_q is not None and var_k is not None
        if bool(self.second_order_rank) != has_second:
            raise ValueError('second-order features and second_order_rank '
                             'must be set together')
        if has_count:
            gain = self.count_gain.index_select(0, heads).float()
            logits = logits + gain[:, None, None] * log_count[None, None, :]
        if not self.second_order_rank:
            return logits
        u = torch.bmm(sq_q, self.so_q.index_select(0, heads).float())
        w = torch.bmm(var_k, self.so_k.index_select(0, heads).float())
        return logits + torch.bmm(u, w.transpose(1, 2))


class TileScorePredictor(nn.Module):
    """All layers: `layers.{i}.proj_q` / `layers.{i}.proj_k`.

    With 50 layers x 56 heads x 384 x 128 x 2 this is 275M parameters. It is
    kept fully replicated (never FSDP-sharded): dim 0 is the head axis and
    per-head indexing of a sharded DTensor is not supported; replication
    also costs one gradient all-reduce instead of an all-gather per layer.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 second_order_rank: int = 0, count_term: bool = False):
        super().__init__()
        self.second_order_rank = second_order_rank
        self.count_term = count_term
        self.layers = nn.ModuleList(
            LayerPredictor(num_heads, head_dim, second_order_rank,
                           count_term)
            for _ in range(num_layers))

    @torch.no_grad()
    def init_exact_second_order_(self) -> None:
        """Warm-start every layer's second-order head (see LayerPredictor)."""
        for layer in self.layers:
            layer.init_exact_second_order_()

    @torch.no_grad()
    def init_low_rank_second_order_(self, c_u: torch.Tensor,
                                    c_v: torch.Tensor,
                                    floor: float = 1e-6) -> None:
        """Per-layer low-rank warm start from calibration moments.

        Args:
            c_u: [num_layers, num_heads, D, D] fp32.
            c_v: Same shape.
            floor: See LayerPredictor.init_low_rank_second_order_.

        Raises:
            ValueError: If the leading dimension is not num_layers.
        """
        if c_u.shape[0] != len(self.layers) or c_v.shape[0] != len(
                self.layers):
            raise ValueError(f'moments cover {c_u.shape[0]} layers, '
                             f'expected {len(self.layers)}')
        for index, layer in enumerate(self.layers):
            layer.init_low_rank_second_order_(c_u[index], c_v[index], floor)

    def scores(self, layer: int, q_tiles: torch.Tensor, k_tiles: torch.Tensor,
               layout: tiling.TileLayout, heads: torch.Tensor) -> torch.Tensor:
        """Block logits of one head group.

        Args:
            layer: Layer index.
            q_tiles: [N, H', D] tile-ordered queries.
            k_tiles: [N, H', D] tile-ordered keys.
            layout: Tile layout of the group.
            heads: [H'] global head indices.

        Returns:
            [H', n_tiles, n_tiles] fp32 logits.
        """
        feats_q = pool_tiles(q_tiles, layout)
        feats_k = pool_tiles(k_tiles, layout)
        extra = {}
        if self.count_term:
            extra['log_count'] = torch.log(
                layout.valid_count.clamp(min=1).to(torch.float32))
        if self.second_order_rank:
            extra['sq_q'] = pool_tiles(q_tiles, layout, SECOND_RAW)
            extra['var_k'] = pool_tiles(k_tiles, layout, SECOND_CENTRAL)
        return self.layers[layer](feats_q, feats_k, heads, **extra)
