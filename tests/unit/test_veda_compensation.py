"""Sol's zero-order compensation, run inside the sparse path.

The offline `solattn.relative_errors` c=1 arm is the formula whose
ceiling we measured, so that is what this has to reproduce: the point of
the end-to-end arm (plan E4) is that nothing but the compensation
differs from a plain Veda run.
"""

import math

import pytest
import torch

from miowtion.kernels import reference
from miowtion.veda import attention as veda_attention
from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE


def _layout(n_tiles: int, last_valid: int) -> tiling.TileLayout:
    """A layout of `n_tiles` tiles, the last one only partly filled."""
    used = (n_tiles - 1) * _TILE + last_valid
    span = tiling.TiledSpan(start=0, grid=(used, 1, 1),
                            shape=tiling.TileShape.parse('128x1x1'))
    return tiling.build_tile_layout([span], used=used,
                                    seq_len=n_tiles * _TILE,
                                    tiler=tiling.contiguous_span_tiles)


def _dense_c1(q, k, v, keep_rows, counts, key_valid) -> torch.Tensor:
    """c = 1 written out directly, the way relative_errors writes it.

    `key_valid` is not cosmetic: a partial tile's padding slots have k = 0,
    so their score is 0 and exp(0 - ref) is not small. Leaving them in the
    kept sum inflates the denominator for every query row that keeps that
    tile, which is why the offline code carries `slot_valid` too.
    """
    n_tiles, head_dim = counts.numel(), q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    k_mean = k.view(n_tiles, _TILE, head_dim).sum(1) / counts[:, None]
    v_sum = v.view(n_tiles, _TILE, head_dim).sum(1)
    scores = (q @ k.transpose(0, 1)) * scale
    scores = scores.masked_fill(~key_valid[None, :], float('-inf'))
    ref = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - ref[:, None])
    kept = keep_rows.to(q.dtype).repeat_interleave(_TILE, dim=1)
    num = (probs * kept) @ v
    den = (probs * kept).sum(-1)
    s_hat = (q @ k_mean.transpose(0, 1)) * scale
    p_hat = torch.exp(s_hat - ref[:, None]) * (1.0 - keep_rows.to(q.dtype))
    num = num + p_hat @ v_sum
    den = den + (p_hat * counts[None, :]).sum(-1)
    return num / den[:, None]


@pytest.mark.parametrize('last_valid', [_TILE, 37])
def test_matches_the_offline_c1_formula(last_valid):
    """Same number as the arm whose ceiling we published."""
    torch.manual_seed(0)
    n_tiles, head_dim = 4, 16
    layout = _layout(n_tiles, last_valid)
    rows = n_tiles * _TILE
    q = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    k = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    v = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    # The gather zeroes padding slots; the compensation relies on it.
    valid = (torch.arange(_TILE)[None, :]
             < layout.valid_count[:, None]).view(-1)
    k[~valid] = 0.0
    v[~valid] = 0.0

    keep = torch.zeros(1, n_tiles, n_tiles, dtype=torch.bool)
    keep[0] = torch.eye(n_tiles, dtype=torch.bool)       # forced diagonal
    keep[0, 0, 2] = True
    keep[0, 3, 1] = True
    keep_eff = keep & layout.kv_ok[None, None, :]

    o_kept, lse_kept = reference.block_sparse_attention(
        q, k, v, keep_eff, layout.valid_count, return_lse=True)
    got = veda_attention.zero_order_compensation(
        o_kept, lse_kept, q, k, v, keep_eff, layout, chunk_rows=97)

    keep_rows = keep_eff[0].repeat_interleave(_TILE, dim=0)
    want = _dense_c1(q[:, 0], k[:, 0], v[:, 0], keep_rows,
                     layout.valid_count.to(q.dtype).clamp(min=1), valid)
    # Only real query rows are meaningful; padding rows are zeroed later.
    real = valid
    # float64 inputs do not buy float64 accuracy: the reference kernel
    # casts to float32 internally, so the lse it hands back is good to
    # about 1e-7 and that is the floor for anything built on it.
    assert torch.allclose(got[real, 0], want[real], atol=1e-6, rtol=1e-6)


def test_the_exponential_reference_cancels():
    """`_dense_c1` uses the dense lse, the path uses the kept-only lse.

    They differ by more than one nat here, and the outputs still agree:
    numerator and denominator both carry exp(-ref), so the compensated
    output does not depend on the reference. That is the fact that lets
    this run inside a sparse path at all, since the dense lse is exactly
    what a sparse path does not have.
    """
    torch.manual_seed(0)
    n_tiles, head_dim = 4, 16
    layout = _layout(n_tiles, _TILE)
    rows = n_tiles * _TILE
    q = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    k = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    v = torch.randn(rows, 1, head_dim, dtype=torch.float64)
    keep = torch.eye(n_tiles, dtype=torch.bool)[None]

    _, lse_kept = reference.block_sparse_attention(
        q, k, v, keep, layout.valid_count, return_lse=True)
    dense_ref = torch.logsumexp(
        (q[:, 0] @ k[:, 0].transpose(0, 1)) * head_dim ** -0.5, dim=-1)
    assert (dense_ref - lse_kept[0]).abs().max() > 1.0


def test_keeping_everything_leaves_the_output_alone():
    """With nothing dropped the term is identically zero.

    This is the invariant that makes the arm a clean ablation: at full
    density the compensated path must be the uncompensated path.
    """
    torch.manual_seed(1)
    n_tiles, head_dim = 3, 16
    layout = _layout(n_tiles, _TILE)
    rows = n_tiles * _TILE
    q = torch.randn(rows, 2, head_dim, dtype=torch.float64)
    k = torch.randn(rows, 2, head_dim, dtype=torch.float64)
    v = torch.randn(rows, 2, head_dim, dtype=torch.float64)
    keep = torch.ones(2, n_tiles, n_tiles, dtype=torch.bool)

    o_kept, lse_kept = reference.block_sparse_attention(
        q, k, v, keep, layout.valid_count, return_lse=True)
    got = veda_attention.zero_order_compensation(
        o_kept, lse_kept, q, k, v, keep, layout)
    assert torch.allclose(got, o_kept, atol=1e-12, rtol=1e-12)


def test_shape_mismatches_raise():
    layout = _layout(2, _TILE)
    rows = 2 * _TILE
    q = torch.zeros(rows, 1, 16)
    keep = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='lse_kept'):
        veda_attention.zero_order_compensation(
            q, torch.zeros(1, rows + 1), q, q, q, keep, layout)
    with pytest.raises(ValueError, match='block_mask'):
        veda_attention.zero_order_compensation(
            q, torch.zeros(1, rows), q, q, q,
            torch.ones(1, 2, 3, dtype=torch.bool), layout)


def test_the_two_compensation_implementations_are_exclusive():
    """They compute the same term; running both would double it."""
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import mask as veda_mask

    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.05),
        zero_order_compensation=True, fused_compensation=True)
    with pytest.raises(ValueError, match='pick one'):
        veda_attention.ClipTiling(
            h3_layout.PackedLayout.__new__(h3_layout.PackedLayout),
            config, torch.device('cpu'))
