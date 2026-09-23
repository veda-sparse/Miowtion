"""FlashAttention-4 (CuTe DSL): the single entry point for FA4 in Miowtion.

Every FA4 import goes through `_modules()`, which on SM8x first installs the
vendored block-sparse patch (miowtion.kernels.fa4_sm8x; it must precede any
`flash_attn.cute` import).

Block-sparse integration rules (each one silently costs speed or
correctness if broken):
  * The mask_mod is a module-level singleton. FA4 hashes the callable to key
    its compile cache; a fresh closure per call re-hashes (and may recompile)
    every time.
  * Full key tiles (valid_count == 128) and partial key tiles go into
    separate lists. Full blocks skip the per-token mask_mod; putting all
    blocks in the masked list is ~25-30% slower.
  * BlockSparseTensorsTorch's 5th positional field is cu_total_m_blocks
    (varlen), not the block size; block_size must be passed by keyword.
  * Kernels process q_stage * tile_m query rows per CTA and require the
    sparse Q block to be a multiple of it. SM90 uses tile_m = 128 with
    q_stage = 1; SM8x (patched) splits 128x128 blocks into sub-tiles. SM100
    picks q_stage = 2 whenever seqlen_q > 128, so the forward config is
    overridden to q_stage = 1 there (_force_q_stage_one; pending B200
    validation).
  * Backward needs Q-direction (transposed) block lists. Without them FA4's
    SM90/SM100 backward silently computes dense gradients, so they are
    always built when gradients are required.

q, k, v are tile-ordered [N, H', D] (seq-major), which is FA4's native
[B, S, H, D] layout after adding a batch axis, so no transposes are needed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import threading

import torch

from miowtion.kernels import fa4_sm8x
from miowtion.veda import mask as veda_mask
from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE
# Architectures (compute capability major) whose FA4 kernels implement block
# sparsity; 8 only with the vendored SM8x patch installed.
_SPARSE_MAJOR_ARCHS = (9, 10, 11)
_sm8x_error: str | None = None


def _any_sm8x() -> bool:
    return any(torch.cuda.get_device_capability(i)[0] == 8
               for i in range(torch.cuda.device_count()))


@functools.cache
def _modules():
    """Imports FA4 (patched on SM8x); returns None when unavailable."""
    global _sm8x_error
    if torch.cuda.is_available() and _any_sm8x():
        try:
            fa4_sm8x.install()
        except (RuntimeError, ImportError) as e:
            _sm8x_error = str(e)
    try:
        import cutlass  # pylint: disable=import-outside-toplevel
        import cutlass.cute as cute  # pylint: disable=import-outside-toplevel
        from flash_attn.cute import block_sparsity  # pylint: disable=import-outside-toplevel
        from flash_attn.cute import interface  # pylint: disable=import-outside-toplevel
        from flash_attn.cute import utils  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return cutlass, cute, block_sparsity, interface, utils


def interface():
    """The (possibly patched) flash_attn.cute.interface module, or None."""
    modules = _modules()
    return None if modules is None else modules[3]


def dense_available(device: torch.device) -> bool:
    return device.type == 'cuda' and _modules() is not None


def available(device: torch.device | None = None) -> bool:
    """FA4 is importable and implements block sparsity on `device`."""
    if not torch.cuda.is_available() or _modules() is None:
        return False
    major = torch.cuda.get_device_capability(device)[0]
    if major == 8:
        return fa4_sm8x.installed()
    return major in _SPARSE_MAJOR_ARCHS


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float, return_lse: bool
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dense attention. q, k, v: [S, H, D] -> out [S, H, D], lse [S, H]."""
    out = interface().flash_attn_func(q[None], k[None], v[None],
                                      softmax_scale=scale,
                                      return_lse=return_lse)
    if not return_lse:
        return (out[0] if isinstance(out, tuple) else out)[0], None
    out, lse = out
    return out[0], lse[0].transpose(0, 1)


@functools.cache
def _valid_key_mask_mod():
    """The singleton mask_mod: key slot n is real iff aux_tensors[0][n] != 0.

    aux_tensors[0] is the int32 [N] slot-validity vector of the tile layout,
    which encodes every key tile's valid prefix.
    """
    cutlass, cute, _, _, utils = _modules()

    @cute.jit
    def valid_key_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        del batch, head, m_idx, seqlen_info
        slot_valid = aux_tensors[0]
        valid = utils.scalar_to_ssa(slot_valid[n_idx[0]], cutlass.Int32)
        zero = utils.scalar_to_ssa(0, cutlass.Int32)
        return valid != zero

    return valid_key_mask


_Q_STAGE_ONE = threading.local()


@functools.cache
def _install_q_stage_hook() -> None:
    """Wraps FA4's forward config selection so q_stage can be forced to 1."""
    iface = interface()
    original = getattr(iface, '_get_fwd_config', None)
    if original is None or 'q_stage' not in {
            f.name for f in dataclasses.fields(iface.FwdConfig)}:
        raise RuntimeError('FA4 internals changed: _get_fwd_config/FwdConfig '
                           'q_stage not found; re-validate the SM100 path')

    @functools.wraps(original)
    def patched(*args, **kwargs):
        config = original(*args, **kwargs)
        if getattr(_Q_STAGE_ONE, 'active', False) and config.q_stage != 1:
            config = dataclasses.replace(config, q_stage=1)
        return config

    iface._get_fwd_config = patched  # pylint: disable=protected-access


@contextlib.contextmanager
def _force_q_stage_one(device: torch.device):
    if torch.cuda.get_device_capability(device)[0] < 10:
        yield
        return
    _install_q_stage_hook()
    _Q_STAGE_ONE.active = True
    try:
        yield
    finally:
        _Q_STAGE_ONE.active = False


def _transpose_indices(block_mask: torch.Tensor, layout: tiling.TileLayout
                       ) -> tuple[torch.Tensor, ...]:
    """Q-direction lists for backward: per key tile, the query tiles."""
    mask_t = block_mask.transpose(1, 2)  # [H', nk, nq]
    heads, nk, nq = mask_t.shape
    rows_full = layout.full_tile[None, :, None]
    cols = torch.arange(nq, device=mask_t.device).expand(heads, nk, nq)

    def pack(member):
        order = torch.argsort((~member).to(torch.int8), dim=-1, stable=True)
        return (member.sum(-1)[None].to(torch.int32),
                torch.gather(cols, -1, order)[None].to(torch.int32)
                .contiguous())

    full_cnt, full_idx = pack(mask_t & rows_full)
    part_cnt, part_idx = pack(mask_t & ~rows_full)
    return part_cnt, part_idx, full_cnt, full_idx


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           indices: veda_mask.KernelIndices,
                           layout: tiling.TileLayout,
                           block_mask: torch.Tensor | None = None
                           ) -> torch.Tensor:
    """FA4 block-sparse attention on tile-ordered tensors.

    Args:
        q: [R * 128, H', D] bf16 query tiles (all tiles, or a subset whose
            lists `indices` describes).
        k: [N, H', D] bf16 (all key tiles).
        v: [N, H', D] bf16.
        indices: Full/partial lists from mask.kernel_indices.
        layout: Tile layout (slot validity for the mask_mod).
        block_mask: [H', n, n] bool; required when gradients are needed
            (backward lists are its transpose).

    Returns:
        [R * 128, H', D] bf16.

    Raises:
        NotImplementedError: On architectures without block sparsity.
    """
    if not available(q.device):
        reason = f' ({_sm8x_error})' if _sm8x_error else ''
        raise NotImplementedError(
            'FA4 block sparsity unavailable on sm'
            f'{"".join(map(str, torch.cuda.get_device_capability(q.device)))}'
            f'{reason}; refusing to fall back to dense attention')
    _, _, block_sparsity, iface, _ = _modules()
    tensors = block_sparsity.BlockSparseTensorsTorch(
        mask_block_cnt=indices.partial_cnt,
        mask_block_idx=indices.partial_idx,
        full_block_cnt=indices.full_cnt,
        full_block_idx=indices.full_idx,
        block_size=(_TILE, _TILE))
    tensors_bwd = None
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v))
    if needs_grad:
        if block_mask is None:
            raise ValueError('block_mask is required for backward')
        part_cnt, part_idx, full_cnt, full_idx = _transpose_indices(
            block_mask, layout)
        tensors_bwd = block_sparsity.BlockSparseTensorsTorch(
            mask_block_cnt=part_cnt, mask_block_idx=part_idx,
            full_block_cnt=full_cnt, full_block_idx=full_idx,
            block_size=(_TILE, _TILE))
    with _force_q_stage_one(q.device):
        out = iface.flash_attn_func(
            q[None], k[None], v[None],
            softmax_scale=q.shape[-1]**-0.5,
            mask_mod=_valid_key_mask_mod(),
            aux_tensors=[layout.slot_valid],
            block_sparse_tensors=tensors,
            block_sparse_tensors_bwd=tensors_bwd)
    if isinstance(out, tuple):
        out = out[0]
    return out[0]
