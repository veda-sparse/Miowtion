"""FA4 CuTe with block sparsity on the SM80 family (sm80 / sm86 / sm89) and
on SM120 (sm120: RTX PRO 6000 Blackwell, Blackwell GeForce, DGX Spark).

Upstream FlashAttention-4 accepts block-sparse tensors on SM8x but its SM80
kernels ignore them and compute dense attention; on SM120 it rejects them
outright. SM120 needs no kernel of its own: upstream's
FlashAttentionForward/BackwardSm120 subclass the SM80 classes and override
only can_implement (99 KB of SMEM instead of 163 KB), the forward also forcing
self.arch back to sm_80, so patching the SM80 kernels covers SM120 too and
only the interface's arch-12 gates have to be lifted. SM120's SMEM capacity
equals sm86/sm89's, so the sm8x sparse tile tuning carries over.

This package vendors the five FA4 modules changed by the patch series in
`patches/`, applied on top of
flash-attention d15f1531a460ba456f41b01a774f33ab2db8febf (modified copies;
BSD-3-Clause, see LICENSE and AUTHORS):
  * block-sparse forward and backward main loops for the SM80 kernels that
    visit blocks in the dense kernel's order (descending KV forward,
    ascending Q backward), so results are bit-identical to the dense kernel
    with an equivalent mask_mod (dQ excepted: its fp32 atomics are
    nondeterministic in the dense kernel too);
  * `DenseBlockMaskTorch`: a [B, H, M, N] block mask (0 skip, 1 full, 2
    partial, plus per-KV-column partial flags) read directly by the kernels,
    for forward and backward alike, so no index lists or transposed lists
    are built;
  * an SM8x launch cache and a no-grad fast path (~30 us host time per call),
    8-warp SM8x backward, and KV sub-tiling for 128x128 sparse blocks.

`install()` swaps them in for the installed FA4 package. It must run before
anything imports `flash_attn.cute` (whose __init__ imports the interface), so
all FA4 access in Miowtion goes through miowtion.kernels.fa4. The installed
package must be exactly the pinned base: its unpatched copies of the five
modules are hash-checked, as are the two SM120 modules that are left in place
but relied on for inheritance, and anything else raises.

Regenerate the vendored files from the patches:
    git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
    git checkout d15f1531a460ba456f41b01a774f33ab2db8febf
    git am <this dir>/patches/*.patch
    cp flash_attn/cute/{block_sparsity,block_sparse_utils,flash_fwd,flash_bwd,interface}.py \
        AUTHORS LICENSE <this dir>
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import sys

_BASE_VERSION = '4.0.0b32'
# sha256 of the unpatched modules at the base commit, in load order: each
# module is registered before the modules that import it (block_sparse_utils
# imports block_sparsity, the kernels import block_sparse_utils, and the
# interface imports all of them).
_BASE_SHA256 = {
    'block_sparsity':
        '9331cb2abd4be4da87a74c0839b665d5f914ac6f51cb97491e97b26199668ac9',
    'block_sparse_utils':
        '608f6c5a1c68daadfca402ec1259917019a195dc748ce9f24b92f0551d2ff6ba',
    'flash_fwd':
        'be6e88c7d9f122aa5ab29ff5946f6270af0b580e4403351b869f1330b71ca53c',
    'flash_bwd':
        '80cb6bfb436160d73b7214529f94d28b7333c98dff75a53d20552eb33645fea2',
    'interface':
        '144a3dd6f72f955e43834500808c7d47b3b4a76fdcd0b7188f9b459d85007cab',
}
# Modules we do NOT replace but whose contents we depend on: SM120 gets block
# sparsity purely by inheritance, because both classes below subclass their
# SM80 counterpart and override only can_implement (the SMEM bound), with the
# forward additionally forcing self.arch back to sm_80. If upstream ever gives
# SM120 a real kernel of its own, the patched SM80 main loops would quietly
# stop being used and SM120 would compute dense attention behind a sparse
# mask, so pin them by hash too.
_INHERITED_SHA256 = {
    'flash_fwd_sm120':
        'abd017add69914e46f0fcfc017ad51c3d07d79fd87d15f969036ddd82bcc7da2',
    'flash_bwd_sm120':
        'ad2d802c97dbf654fd03724bcfa91702b18acec8518e5dd4a33f137ba238bf7d',
}
_PACKAGE = 'flash_attn.cute'
_HERE = os.path.dirname(os.path.abspath(__file__))
_installed = False


def installed() -> bool:
    return _installed


def _sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def install() -> None:
    """Replaces the five FA4 modules with the block-sparse versions.

    Raises:
        RuntimeError: If flash_attn.cute was imported already, or the
            installed FA4 is not the pinned base version.
    """
    global _installed
    if _installed:
        return
    if _PACKAGE in sys.modules:
        raise RuntimeError('flash_attn.cute was imported before the SM8x '
                           'patch; import FA4 only via miowtion.kernels.fa4')
    version = importlib.metadata.version('flash-attn-4')
    if version != _BASE_VERSION:
        raise RuntimeError(f'flash-attn-4 {version} installed; the SM8x '
                           f'patch is built on {_BASE_VERSION}')
    spec = importlib.util.find_spec(_PACKAGE)
    base_dir = spec.submodule_search_locations[0]
    for name, digest in _BASE_SHA256.items():
        if _sha256(os.path.join(base_dir, f'{name}.py')) != digest:
            raise RuntimeError(f'installed {_PACKAGE}.{name} differs from '
                               'the pinned base; refusing to patch')
    for name, digest in _INHERITED_SHA256.items():
        if _sha256(os.path.join(base_dir, f'{name}.py')) != digest:
            raise RuntimeError(
                f'installed {_PACKAGE}.{name} differs from the pinned base; '
                'SM120 block sparsity relies on it subclassing the patched '
                'SM80 kernel, so refusing to patch')
    # Create the package without running its __init__ (which would import
    # the upstream interface), register the patched modules, then run it.
    package = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = package
    try:
        for name in _BASE_SHA256:
            full = f'{_PACKAGE}.{name}'
            module_spec = importlib.util.spec_from_file_location(
                full, os.path.join(_HERE, f'{name}.py'))
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[full] = module
            module_spec.loader.exec_module(module)
            setattr(package, name, module)
        spec.loader.exec_module(package)
    except BaseException:
        for key in [k for k in sys.modules if k == _PACKAGE
                    or k.startswith(_PACKAGE + '.')]:
            del sys.modules[key]
        raise
    _installed = True
