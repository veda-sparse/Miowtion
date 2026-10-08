"""Tests for the CPU-side parts of miowtion.kernels.fa4."""

import threading
import time

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


def test_module_import_is_serialized_across_threads(monkeypatch):
    """Concurrent first calls must not run the import side by side.

    The SM8x patch puts half-executed modules in sys.modules; a second
    thread importing flash_attn.cute at that moment sees a module without
    its attributes. functools.cache alone does not prevent that, because it
    holds no lock while the wrapped call runs.
    """
    state = {'inside': 0, 'peak': 0}

    def slow_import():
        state['inside'] += 1
        state['peak'] = max(state['peak'], state['inside'])
        time.sleep(0.01)
        state['inside'] -= 1
        return None

    monkeypatch.setattr(fa4, '_import_modules', slow_import)
    monkeypatch.setattr(fa4, '_modules', fa4._modules.__wrapped__)
    threads = [threading.Thread(target=fa4._modules) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert state['peak'] == 1


def test_available_is_false_for_a_non_cuda_device():
    """A CUDA machine must not claim the kernel for a CPU tensor.

    `SparseStudent` asks this before choosing between FA4 and the
    reference kernel. When it ignored the device, a CPU clip on a CUDA box
    picked FA4 and died inside the kernel on 'Expected a cuda device'.
    """
    from miowtion.kernels import fa4
    assert fa4.available(torch.device('cpu')) is False
    assert fa4.dense_available(torch.device('cpu')) is False


def test_tile_kernel_support_accounts_for_the_tensor():
    """`available()` says nothing about head_dim or device; `supports` does.

    On a CUDA machine the tiny test configs used to route a head_dim of 16
    into the Triton tile kernels, which raise. The predicate the callers
    use has to answer for the tensor they would pass.
    """
    from miowtion.kernels import block_heat_triton, tile_gather_triton
    cpu = torch.zeros(4, 2, 128)
    assert tile_gather_triton.supports(cpu) is False
    assert block_heat_triton.supports(cpu) is False
    small = torch.zeros(4, 2, 16)
    assert tile_gather_triton.supports(small) is False
    assert block_heat_triton.supports(small) is False
