"""Veda's tile scorer on MLX: pooled q/k features and block logits.

The torch reference is `miowtion.veda.predictor`: every tile is pooled into
[mean | max | min] over its real rows, a per-head residual projection maps
the 3D features to D dims (`q_hat = pool_q @ P_q[h] + mean_q`), and the
block logits are `q_hat . k_hat / sqrt(D)` in fp32.

This module runs that on the layer's own q and k inside the MLX denoise
loop, which is what makes the selection content-dependent: the plan of a
layer cannot be built before the step that produces its activations. Only
the scoring lives here -- the top-k, the Bresenham budget split and the
plan construction stay on the torch side (see `miowtion.mlx.veda_plan`),
because they work on the tile grid, where a layer's logits are ~2 MB.

`proj_q` / `proj_k` are optional. Without them `embed` returns the pooled
mean, which is exactly what the reference predictor computes at
initialization (P ~ N(0, 1e-4)): mean-pooled QK, the block score Veda
bootstraps from. That is the honest scorer to use while no trained
predictor checkpoint exists -- it is content-based, unlike
`veda_plan.random_scores`, but it is not a trained selection either.

Alignment with the torch reference: max / min pooling is bitwise equal,
while the mean, the projection and the logit matmul are reductions whose
accumulation order differs between the two backends (~1e-6 relative, see
docs/features/mlx_inference.md).
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx


@dataclasses.dataclass(frozen=True)
class TileGeometry:
    """The parts of a `veda.tiling.TileLayout` the scorer needs, in MLX.

    Attributes:
        gather: [N] int32 packed row of every tile slot, padding slots
            redirected to row 0 (`TileLayout.gather_index`).
        slot_valid: [N] bool, False on padding slots.
        valid_count: [n_tiles] fp32 real rows per tile, clamped to >= 1 so
            it can divide the mean of an empty tile (whose sum is 0).
        kv_ok: [n_tiles] bool, tile has at least one real row.
        n_tiles: Tiles in the permutation.
        n_video_tiles: Leading tiles that are video (the ones Veda selects
            for); the rest are global rows and stay dense.
        tile_size: Rows per tile.
    """

    gather: mx.array
    slot_valid: mx.array
    valid_count: mx.array
    kv_ok: mx.array
    n_tiles: int
    n_video_tiles: int
    tile_size: int


def gather_tiles(x: mx.array, geom: TileGeometry) -> mx.array:
    """Permutes packed rows into tile order, zeroing padding slots.

    Args:
        x: [S, H', D] packed rows of one head group.
        geom: Tile geometry of the group.

    Returns:
        [n_tiles, tile_size, H', D] in x's dtype.
    """
    rows = mx.take(x, geom.gather, axis=0)
    rows = mx.where(geom.slot_valid[:, None, None], rows, mx.zeros_like(rows))
    return rows.reshape(geom.n_tiles, geom.tile_size, *x.shape[1:])


def pool_tiles(tiles: mx.array, geom: TileGeometry) -> mx.array:
    """Masked mean / max / min over the real rows of every tile.

    Args:
        tiles: [n_tiles, tile_size, H', D] tile-ordered rows, padding slots
            zero (i.e. the output of `gather_tiles`).
        geom: Tile geometry.

    Returns:
        [H', n_tiles, 3D] fp32 features, exactly 0 for empty tiles.
    """
    valid = geom.slot_valid.reshape(geom.n_tiles, geom.tile_size, 1, 1)
    mean = tiles.astype(mx.float32).sum(axis=1) / geom.valid_count[:, None,
                                                                   None]
    # Padding rows hold 0, which is a legal value, so max / min are taken
    # over the real rows only. An empty tile then yields -inf / +inf, which
    # the kv_ok select below replaces (a multiply would give NaN).
    big = mx.array(float('inf'), dtype=tiles.dtype)
    tmax = mx.where(valid, tiles, -big).max(axis=1)
    tmin = mx.where(valid, tiles, big).min(axis=1)
    feats = mx.concatenate([mean, tmax.astype(mx.float32),
                            tmin.astype(mx.float32)], axis=-1)
    feats = mx.where(geom.kv_ok[:, None, None], feats, 0.0)
    return feats.transpose(1, 0, 2)


def embed(feats: mx.array, head_dim: int,
          proj: mx.array | None = None) -> mx.array:
    """[H', n, 3D] fp32 features -> [H', n, D] tile embeddings.

    Args:
        feats: Pooled features.
        head_dim: D.
        proj: [H', 3D, D] fp32 projection of these heads, or None for the
            residual-only (mean-pooled) embedding.

    Returns:
        The embeddings, fp32.
    """
    mean = feats[..., :head_dim]
    if proj is None:
        return mean
    return feats @ proj + mean


def tile_logits(feats_q: mx.array, feats_k: mx.array, head_dim: int,
                proj_q: mx.array | None = None,
                proj_k: mx.array | None = None) -> mx.array:
    """Block logits [H', n_q, n_k] fp32, as `predictor.LayerPredictor`.

    Args:
        feats_q: [H', n_q, 3D] pooled query features.
        feats_k: [H', n_k, 3D] pooled key features.
        head_dim: D.
        proj_q: [H', 3D, D] query projection of these heads, or None.
        proj_k: [H', 3D, D] key projection of these heads, or None.

    Returns:
        The logits.
    """
    q_hat = embed(feats_q, head_dim, proj_q)
    k_hat = embed(feats_k, head_dim, proj_k)
    return (q_hat @ k_hat.transpose(0, 2, 1)) / math.sqrt(head_dim)


def score_tiles(q: mx.array, k: mx.array, geom: TileGeometry,
                proj_q: mx.array | None = None,
                proj_k: mx.array | None = None) -> mx.array:
    """Scores a head group's video query tiles from its own activations.

    Args:
        q: [S, H', D] packed queries of the group (post qk-norm and RoPE,
            as the attention sees them).
        k: [S, H', D] packed keys.
        geom: Tile geometry of the group.
        proj_q: [H', 3D, D] query projection, or None (mean-pooled QK).
        proj_k: [H', 3D, D] key projection, or None.

    Returns:
        [H', n_video_tiles, n_tiles] fp32 logits: one row per query tile
        Veda selects for, one column per key tile.

    Raises:
        ValueError: If q and k disagree in shape.
    """
    if q.shape != k.shape:
        raise ValueError(f'q {q.shape} and k {k.shape} must match')
    head_dim = q.shape[-1]
    feats_q = pool_tiles(gather_tiles(q, geom), geom)
    feats_k = pool_tiles(gather_tiles(k, geom), geom)
    return tile_logits(feats_q[:, :geom.n_video_tiles], feats_k, head_dim,
                       proj_q, proj_k)
