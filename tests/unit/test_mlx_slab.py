"""Tests for miowtion.mlx.slab (the NVMe offloading format)."""

import os

import pytest

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import block as mlx_block  # noqa: E402
from miowtion.mlx import slab as mlx_slab  # noqa: E402

_HIDDEN = 64
_HEADS = 2
_HEAD_DIM = 16
_FFN = 32


def _weights(seed: int) -> mlx_block.BlockWeights:
    mx.random.seed(seed)
    inner = _HEADS * _HEAD_DIM
    shapes = {
        'norm1.weight': (_HIDDEN,), 'norm2.weight': (_HIDDEN,),
        'attn.q_norm.weight': (_HEAD_DIM,), 'attn.k_norm.weight': (_HEAD_DIM,),
        'attn.qkv_proj.weight': (3 * inner, _HIDDEN),
        'attn.out_proj.weight': (_HIDDEN, inner),
        'mlp.fc1.weight': (2 * _FFN, _HIDDEN),
        'mlp.fc2.weight': (_HIDDEN, _FFN),
    }
    tensors = {n: mx.random.normal(s).astype(mx.bfloat16)
               for n, s in shapes.items()}
    mx.eval(*tensors.values())
    return mlx_block.BlockWeights.from_tensors(tensors)


def _roundtrip(tmp_path, weights, bits=0, group_size=0, **reader_kwargs):
    arrays = weights.to_tensors()
    mx.eval(*arrays.values())
    path = mlx_slab.slab_path(str(tmp_path), 0)
    mlx_slab.write_slab(path, 0, arrays, bits, group_size)
    reader = mlx_slab.SlabReader([path], slots=1, **reader_kwargs)
    try:
        reader.read(0, 0)
        got = reader.weights(0).to_tensors()
        mx.eval(*got.values())
        assert set(got) == set(arrays)
        for name, value in arrays.items():
            # A slab is a pure data transform: it must come back bit for bit.
            assert mx.array_equal(got[name], value), name
    finally:
        reader.close()


def test_slab_roundtrip_is_bitwise(tmp_path):
    for nocache in (True, False):
        for threads, piece in ((1, mlx_slab.SLAB_ALIGNMENT), (4, 1 << 20)):
            _roundtrip(tmp_path, _weights(0), nocache=nocache,
                       threads=threads, piece_bytes=piece)


def test_quantized_slab_roundtrip_is_bitwise(tmp_path):
    for bits in (8, 4):
        _roundtrip(tmp_path, _weights(1).quantize(bits, 32), bits, 32)


def test_layout_is_aligned_and_reread(tmp_path):
    arrays = _weights(2).to_tensors()
    mx.eval(*arrays.values())
    path = mlx_slab.slab_path(str(tmp_path), 7)
    written = mlx_slab.write_slab(path, 7, arrays, 0, 0)
    layout = mlx_slab.read_layout(path)
    assert layout == written
    assert layout.block == 7
    assert layout.data_bytes == sum(a.nbytes for a in arrays.values())
    assert os.path.getsize(path) >= layout.file_bytes
    for entry in layout.tensors:
        assert entry.offset % mlx_slab.SLAB_ALIGNMENT == 0
        assert entry.nbytes == arrays[entry.name].nbytes


def test_reader_rejects_mixed_layouts(tmp_path):
    for block, weights in ((0, _weights(3)), (1, _weights(4).quantize(8, 32))):
        arrays = weights.to_tensors()
        mx.eval(*arrays.values())
        mlx_slab.write_slab(mlx_slab.slab_path(str(tmp_path), block), block,
                            arrays, 8 if block else 0, 32 if block else 0)
    paths = [mlx_slab.slab_path(str(tmp_path), i) for i in (0, 1)]
    with pytest.raises(ValueError):
        mlx_slab.SlabReader(paths)


def test_reader_rejects_bad_arguments(tmp_path):
    arrays = _weights(5).to_tensors()
    mx.eval(*arrays.values())
    path = mlx_slab.slab_path(str(tmp_path), 0)
    mlx_slab.write_slab(path, 0, arrays)
    for kwargs in ({'slots': 0}, {'threads': 0}, {'piece_bytes': 4096}):
        with pytest.raises(ValueError):
            mlx_slab.SlabReader([path], **kwargs)
    bad = str(tmp_path / 'not_a_slab.slab')
    with open(bad, 'wb') as f:
        f.write(b'XXXXXXXX' + b'\0' * 64)
    with pytest.raises(ValueError):
        mlx_slab.read_layout(bad)


def test_prefetcher_streams_every_block_in_order(tmp_path):
    order = [0, 1, 2, 3]
    for block in order:
        arrays = _weights(block).to_tensors()
        mx.eval(*arrays.values())
        mlx_slab.write_slab(mlx_slab.slab_path(str(tmp_path), block), block,
                            arrays)
    paths = [mlx_slab.slab_path(str(tmp_path), i) for i in order]
    expected = [mlx_slab.SlabReader([p], slots=1) for p in paths]
    reader = mlx_slab.SlabReader(paths, slots=2)
    try:
        seen = []
        prefetcher = mlx_slab.BlockPrefetcher(reader, order, depth=1)
        for index, weights in prefetcher:
            reference = expected[index]
            reference.read(0, 0)
            got, want = weights.to_tensors(), reference.weights(0).to_tensors()
            mx.eval(*got.values(), *want.values())
            for name in want:
                assert mx.array_equal(got[name], want[name]), (index, name)
            seen.append(index)
        assert seen == order
        assert len(prefetcher.stats.read) == len(order)
        assert len(prefetcher.stats.wait) == len(order)
    finally:
        reader.close()
        for r in expected:
            r.close()

    with pytest.raises(ValueError):
        # depth + 1 slots are required.
        mlx_slab.BlockPrefetcher(mlx_slab.SlabReader(paths, slots=1), order,
                                 depth=1)
