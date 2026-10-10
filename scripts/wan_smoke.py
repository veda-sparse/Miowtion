"""Smoke test: Veda's block-sparse attention on real Wan2.1 weights.

Step three of P0-3 is the plan search, which needs the dense teacher's
activations and hours of GPU. Before paying for that, this proves the
chain runs end to end on the real checkpoint: Wan's weights, our
PackedLayout, build_tile_layout, the FA4 block-sparse kernel, and the
processor writing the result back.

It needs a GPU and the 27 GB checkpoint, so it is a script rather than a
unit test; the parts that can be checked on CPU are in
tests/unit/test_wan_layout.py and tests/unit/test_wan_attention.py.

Example:
    python scripts/wan_smoke.py --size 256 --frames 9
"""
import argparse
import os
import sys

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from miowtion.wan import layout as wan_layout, attention as wan_attention
from miowtion.veda import tiling, mask as veda_mask
from miowtion.kernels import fa4

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--root', default='weights/wan/Wan2.1-T2V-1.3B')
_parser.add_argument('--size', type=int, default=256)
_parser.add_argument('--frames', type=int, default=9)
_parser.add_argument('--density', type=float, default=0.2)
_args = _parser.parse_args()
ROOT = _args.root
dev = torch.device('cuda')
from diffusers import WanTransformer3DModel
model = WanTransformer3DModel.from_pretrained(
    ROOT, subfolder='transformer', torch_dtype=torch.bfloat16).to(dev).eval()
cfg = model.config
print(f'loaded: {cfg.num_layers} layers, {cfg.num_attention_heads} heads, '
      f'dim {cfg.attention_head_dim}')

# A small clip so one forward fits comfortably.
W = H = _args.size
F = _args.frames
lay = wan_layout.packed_layout(W, H, F, text_len=0)
grid = lay.target.grid
print(f'{W}x{H}x{F} -> grid {grid}, {lay.used} tokens')

shape = tiling.least_padding_shape(grid)
tl = tiling.build_tile_layout(
    [tiling.TiledSpan(0, grid, shape)], used=lay.used,
    seq_len=lay.used, device=dev)
print(f'tile shape {shape}: {tl.n_tiles} tiles, {tl.n_video_tiles} video')

calls = {'n': 0}
def veda_attention(q, k, v, layer_index):
    """Block-sparse at ~20% with the forced diagonal, through FA4."""
    calls['n'] += 1
    heads = torch.arange(q.shape[1], device=q.device)
    qt = tiling.gather_tiles(q, tl, heads)
    kt = tiling.gather_tiles(k, tl, heads)
    vt = tiling.gather_tiles(v, tl, heads)
    n = tl.n_tiles
    keep = (torch.rand(q.shape[1], n, n, device=q.device) < _args.density)
    keep |= torch.eye(n, device=q.device, dtype=torch.bool)[None]
    keep &= tl.kv_ok[None, None, :]
    out = fa4.block_sparse_attention(qt, kt, vt, keep, tl)
    full = q.new_zeros(tl.num_slots + 1, *q.shape[1:])
    tiling.scatter_tiles_(full, out, tl, heads)
    return full[:q.shape[0]]

for i, block in enumerate(model.blocks):
    block.attn1.set_processor(wan_attention.make_processor(veda_attention, i))

x = torch.randn(1, lay.used, cfg.num_attention_heads * cfg.attention_head_dim,
                device=dev, dtype=torch.bfloat16)
enc = torch.randn(1, 512, cfg.text_dim, device=dev, dtype=torch.bfloat16)
t = torch.tensor([500], device=dev)
print('processors swapped on all', len(model.blocks), 'blocks')

# Forward one block directly: that is what the port has to make work.
blk = model.blocks[0]
# rope wants the latent grid, before the [1,2,2] patchify; the token
# grid is already patchified, so passing it halves each spatial axis again.
latent = (grid[0], grid[1] * 2, grid[2] * 2)
rot = model.rope(torch.randn(1, cfg.in_channels, *latent, device=dev,
                             dtype=torch.bfloat16))
with torch.no_grad():
    y = blk.attn1(x, None, None, rot)
print('block-0 self-attention out', tuple(y.shape), y.dtype,
      '| veda calls', calls['n'], '| finite', bool(torch.isfinite(y).all()))
