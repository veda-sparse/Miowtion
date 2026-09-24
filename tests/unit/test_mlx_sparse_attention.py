"""Tests for miowtion.mlx.sparse_attention (Veda block sparsity on MLX)."""

import pytest

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import sparse_attention as sa  # noqa: E402

_HEADS, _DIM, _SEQ = 4, 128, 512
_QB, _KB = 64, 32


def _qkv(seed=0, seq=_SEQ, heads=_HEADS, dtype=mx.float32):
    mx.random.seed(seed)
    arrays = [mx.random.normal((heads, seq, _DIM)).astype(dtype)
              for _ in range(3)]
    mx.eval(*arrays)
    return arrays


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()


def test_matches_dense_under_the_equivalent_mask():
    # The gathered problem contains exactly the kept tiles, so the two
    # kernels see the same keys in the same order: the results must agree
    # bit for bit, not just numerically.
    q, k, v = _qkv()
    index = sa.random_index(_SEQ // _QB, _SEQ // _KB, budget=4)
    mx.eval(index)
    got = sa.block_sparse_attention(q, k, v, index, q_block=_QB, k_block=_KB)
    mask = sa.block_mask_from_index(index, _QB, _KB, _SEQ)
    want = sa.dense_reference(q, k, v, mask=mask)
    mx.eval(got, want)
    assert got.shape == q.shape
    assert mx.array_equal(got, want).item()


def test_full_budget_is_dense_attention():
    q, k, v = _qkv(seed=1)
    n_k = _SEQ // _KB
    index = mx.broadcast_to(mx.arange(n_k, dtype=mx.int32)[None],
                            (_SEQ // _QB, n_k))
    mx.eval(index)
    got = sa.block_sparse_attention(q, k, v, index, q_block=_QB, k_block=_KB)
    want = sa.dense_reference(q, k, v)
    mx.eval(got, want)
    assert _rel(got, want) < 1e-6


def test_head_chunking_is_exact():
    q, k, v = _qkv(seed=2, dtype=mx.bfloat16)
    index = sa.random_index(_SEQ // _QB, _SEQ // _KB, budget=3)
    mx.eval(index)
    whole = sa.block_sparse_attention(q, k, v, index, q_block=_QB,
                                      k_block=_KB)
    for chunk in (1, 2, _HEADS):
        part = sa.block_sparse_attention(q, k, v, index, q_block=_QB,
                                         k_block=_KB, head_chunk=chunk)
        mx.eval(whole, part)
        assert mx.array_equal(part, whole).item(), chunk


def test_selection_actually_restricts_attention():
    # A key tile that is not selected must not influence the output.
    q, k, v = _qkv(seed=3)
    index = mx.array([[0]] * (_SEQ // _QB), dtype=mx.int32)
    mx.eval(index)
    before = sa.block_sparse_attention(q, k, v, index, q_block=_QB,
                                       k_block=_KB)
    v2 = mx.array(v)
    v2[:, _KB:] += 100.0  # everything outside key tile 0
    mx.eval(v2)
    after = sa.block_sparse_attention(q, k, v2, index, q_block=_QB,
                                      k_block=_KB)
    mx.eval(before, after)
    assert mx.array_equal(before, after).item()


def test_block_mask_from_index_marks_exactly_the_tiles():
    index = mx.array([[0, 2], [1, 3]], dtype=mx.int32)
    mask = sa.block_mask_from_index(index, q_block=4, k_block=2, seq_len=8)
    mx.eval(mask)
    assert mask.shape == (8, 8)
    # 2 query tiles x 2 key tiles, each 4x2 rows.
    assert mx.sum(mask.astype(mx.int32)).item() == 2 * 2 * 4 * 2
    assert bool(mask[0, 0].item()) and bool(mask[0, 5].item())
    assert not bool(mask[0, 2].item())
    assert bool(mask[4, 2].item()) and not bool(mask[4, 0].item())
    with pytest.raises(ValueError):
        sa.block_mask_from_index(index, q_block=2, k_block=2, seq_len=8)


def test_plan_reports_density_and_budget():
    index = sa.random_index(_SEQ // _QB, _SEQ // _KB, budget=4)
    plan = sa.SparsePlan(index, _QB, _KB)
    assert plan.budget == 4
    assert plan.density(_SEQ) == pytest.approx(4 / (_SEQ / _KB))


def test_rejects_bad_arguments():
    q, k, v = _qkv(seed=4)
    index = sa.random_index(_SEQ // _QB, _SEQ // _KB, budget=2)
    mx.eval(index)
    good = dict(q_block=_QB, k_block=_KB)
    with pytest.raises(ValueError):  # seq_len not a multiple of the tile
        sa.block_sparse_attention(q, k, v, index, q_block=48, k_block=_KB)
    with pytest.raises(ValueError):  # wrong number of query tiles
        sa.block_sparse_attention(q, k, v, index[:-1], **good)
    with pytest.raises(ValueError):  # mismatched k
        sa.block_sparse_attention(q, k[:, :_SEQ // 2], v, index, **good)
    with pytest.raises(ValueError):
        sa.block_sparse_attention(q, k, v, index, head_chunk=0, **good)
    with pytest.raises(ValueError):
        sa.random_index(4, 4, budget=5)


def test_flop_and_byte_models_scale_as_expected():
    flops = sa.attention_flops(1024, 8, 128, density=0.1)
    assert flops == pytest.approx(0.1 * sa.attention_flops(1024, 8, 128, 1.0))
    small = sa.gathered_bytes(4096, 8, 128, q_block=512, density=0.1)
    large = sa.gathered_bytes(4096, 8, 128, q_block=1024, density=0.1)
    # A larger query tile gathers proportionally fewer rows.
    assert large == pytest.approx(small / 2)
