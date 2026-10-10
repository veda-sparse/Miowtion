"""Sol + Veda: the fused zero-order compensation from the ComfyUI node.

Our own `veda.attention.zero_order_compensation` computes the same term
in torch, which is enough to measure what the compensation does to the
video (plan E4) but not what it costs: it materialises a
[rows, n_tiles] score matrix outside the kernel, where a fused
implementation merges the approximate branch into the same online
softmax as the exact blocks and never writes those scores to memory.

`third_party/Veda-on-ComfyUI` @ 3f9be82 (`dev/veda-sol`, MIT) does that
fusion in one Triton kernel, and its entry point already takes the three
tensors we hand to FA4, so it is a drop-in replacement for
`fa4.block_sparse_attention`. Verified against that source line by line:
`_summaries` is `mean(K)` and `sum(V)` over each tile's valid rows and
the kernel accumulates `acc += p @ vc`, `denom += p * lengths` -- the
same c = 1 arm as `solattn.relative_errors` and as Sol-Attn's own
`preprocess.py` / `fwd.py`.

One difference that must travel with every number from here: it
quantises the *selected* blocks to INT8 (Sage) while the pooled
summaries stay BF16, where our FA4 path is BF16 throughout. A quality
difference against our `veda` arm therefore mixes precision with
compensation; E4 is the arm that separates them.
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys

import torch

from miowtion.veda import tiling

_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    'third_party', 'Veda-on-ComfyUI')

# The pinned commit. The kernel is vendored as a submodule rather than
# copied, so this is the only place that needs to know the revision.
PINNED_COMMIT = '3f9be82'


# sha256 of the upstream kernel this copy was patched from. Checked at
# import so an upstream edit surfaces as an error rather than as a
# silently stale vendored file (AGENTS.md 3).
UPSTREAM_SHA256 = ('47affe24a193e8b5ba614152fdf96d317d784624'
                   '813040e099441c3ec3e026b4')
_UPSTREAM = os.path.join(_ROOT, 'veda_comfy', 'kernels', 'sol_veda',
                         'attention.py')


def _check_upstream() -> None:
    """Fails if the vendored patch no longer matches its source.

    Raises:
        RuntimeError: If the upstream file is missing or its hash moved.
    """
    if not os.path.exists(_UPSTREAM):
        raise RuntimeError(
            f'{_UPSTREAM} is missing; run git submodule update --init')
    with open(_UPSTREAM, 'rb') as handle:
        got = hashlib.sha256(handle.read()).hexdigest()
    if got != UPSTREAM_SHA256:
        raise RuntimeError(
            f'{_UPSTREAM} is sha256 {got}, the patch was generated against '
            f'{UPSTREAM_SHA256}; regenerate it before trusting this kernel')


@functools.cache
def _modules() -> tuple:
    """(attend, sage) for the patched kernel, or raise with the reason.

    The kernel comes from our patched copy because the published one does
    not compile; the quantiser comes from the submodule unchanged. See
    `miowtion/kernels/sol_veda_patched/attention.py` for the two patches.
    """
    _check_upstream()
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from veda_comfy.kernels.sage import sparse_int8      # noqa: PLC0415

    from miowtion.kernels.sol_veda_patched import attend  # noqa: PLC0415
    return attend, sparse_int8


def available(device: torch.device | str) -> bool:
    """True when the submodule imports and the device is CUDA SM80+."""
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        return False
    try:
        _modules()
    except Exception:                                    # noqa: BLE001
        return False
    return torch.cuda.get_device_capability(device)[0] >= 8


def unavailable_reason(device: torch.device | str = 'cuda') -> str | None:
    """Why `available` is False, or None when it is True."""
    if not torch.cuda.is_available():
        return 'no CUDA device'
    if torch.device(device).type != 'cuda':
        return f'device {device} is not CUDA'
    try:
        _modules()
    except Exception as error:                           # noqa: BLE001
        return str(error)
    if torch.cuda.get_device_capability(device)[0] < 8:
        return 'needs SM80 or newer'
    return None


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor, block_mask: torch.Tensor,
                           layout: tiling.TileLayout) -> torch.Tensor:
    """Same contract as `fa4.block_sparse_attention`, plus compensation.

    Args:
        q: [R * 128, H', D] bf16 tile-ordered queries, R == layout.n_tiles.
        k: [N * 128, H', D] bf16 tile-ordered keys.
        v: [N * 128, H', D] bf16 tile-ordered values.
        block_mask: [H', R, n_tiles] bool, True where the tile is kept
            exactly; every other non-empty tile is compensated rather
            than discarded, which is the whole difference from FA4.
        layout: Tile layout, for the real row count of each tile.

    Returns:
        [R * 128, H', D] bf16.

    Raises:
        RuntimeError: If the submodule or the device cannot run it.
        ValueError: If the query tiles are a subset; the kernel indexes
            queries and keys with the same tile id, so a partial mask
            would silently mis-pair them.
    """
    reason = unavailable_reason(q.device)
    if reason:
        raise RuntimeError(f'Sol + Veda is unavailable: {reason}')
    if block_mask.shape[1] != layout.n_tiles:
        raise ValueError(
            f'block_mask covers {block_mask.shape[1]} query tiles but the '
            f'layout has {layout.n_tiles}; this kernel needs all of them')
    attend, sage = _modules()
    out, _exact = attend(
        q.contiguous(), k.contiguous(), v.contiguous(),
        layout.valid_count.to(torch.int32),
        (block_mask & layout.kv_ok[None, None, :]).contiguous(), sage)
    return out
