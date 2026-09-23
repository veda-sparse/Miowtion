"""Tests for the CPU-side parts of miowtion.kernels.fa4."""

import torch

from miowtion.kernels import fa4
from miowtion.veda import mask as veda_mask
from miowtion.veda import tiling


def _selection_mask(heads=3, seed=0):
    target = tiling.TiledSpan(130, (7, 6, 10), tiling.TileShape.parse('8x4x4'))
    used = target.start + target.num_rows
    lay = tiling.build_tile_layout([target], used, used)
    assert lay.partial_tiles.numel() > 0  # padded grid -> partial key tiles
    scores = torch.randn(heads, lay.n_tiles, lay.n_tiles,
                         generator=torch.Generator().manual_seed(seed))
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.5))
    sel = veda_mask.select_video_blocks(scores[:, :lay.n_video_tiles], lay,
                                        blocks)
    return veda_mask.dense_block_mask(sel, lay), lay


def _rebuild(cnt, idx, shape):
    out = torch.zeros(shape, dtype=torch.bool)
    for h in range(shape[0]):
        for i in range(shape[1]):
            out[h, i, idx[0, h, i, :cnt[0, h, i]].long()] = True
    return out


def test_index_lists_rebuild_the_mask():
    mask, lay = _selection_mask()
    part_cnt, part_idx, full_cnt, full_idx = fa4.index_lists(mask, lay)
    full = _rebuild(full_cnt, full_idx, mask.shape)
    part = _rebuild(part_cnt, part_idx, mask.shape)
    assert not (full & part).any()
    assert torch.equal(full, mask & lay.full_tile)
    assert torch.equal(full | part, mask & lay.kv_ok)
    # Lists are sorted by key tile.
    for h in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            row = full_idx[0, h, i, :full_cnt[0, h, i]]
            assert torch.equal(row, row.sort().values)


def test_transposed_index_lists_are_per_key_tile():
    mask, lay = _selection_mask()
    part_cnt, part_idx, full_cnt, full_idx = fa4.index_lists(mask, lay,
                                                             transpose=True)
    mask_t = (mask & lay.kv_ok).transpose(1, 2)
    full = _rebuild(full_cnt, full_idx, mask_t.shape)
    part = _rebuild(part_cnt, part_idx, mask_t.shape)
    # A block is full iff its key tile (the row here) is full.
    assert torch.equal(full, mask_t & lay.full_tile[:, None])
    assert torch.equal(part, mask_t & ~lay.full_tile[:, None])
