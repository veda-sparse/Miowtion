"""Second-moment calibration on Wan, step five of P0-3.

Veda2's dispersion head approximates the within-block score variance.
For q and k drawn from a block, Var(q.k) = tr(C_q C_k)/D - mu^2, and the
head keeps only the diagonal, sum_d E[q_d^2] Var(k_d) / D, because a
full covariance product is a rank-D object per head. On H3 that
truncation captures 77% of the exact term
(docs/features/veda2.md 2.x); whether it holds on a second model is what
decides if the head transfers or needs refitting.

solattn.run_second_moments is bound to H3's teacher and trajectory, and
what the calibration needs is the statistic, not that plumbing: this
captures real q and k from Wan's own blocks and computes the same ratio.

Example:
    python scripts/wan_second_moments.py --size 256 --frames 9
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402
from miowtion.wan import attention as wan_attention        # noqa: E402
from miowtion.wan import layout as wan_layout              # noqa: E402


def _ratios(q: torch.Tensor, k: torch.Tensor) -> list[float]:
    """Diagonal share of tr(C_q C_k)/D, per head.

    Args:
        q: [S, H, D] queries of one layer.
        k: [S, H, D] keys of the same layer.

    Returns:
        One ratio per head; 1.0 would mean the diagonal is exact.
    """
    out = []
    for head in range(q.shape[1]):
        qh = q[:, head].float()
        kh = k[:, head].float()
        dim = qh.shape[-1]
        # Second moment of q (the head uses E[q^2], not the centred one)
        # against the centred covariance of k, exactly as the term is
        # defined in the predictor.
        c_q = qh.T @ qh / qh.shape[0]
        centred = kh - kh.mean(0, keepdim=True)
        c_k = centred.T @ centred / kh.shape[0]
        exact = torch.einsum('ij,ji->', c_q, c_k).item() / dim
        diagonal = (torch.diagonal(c_q) * torch.diagonal(c_k)).sum().item() / dim
        out.append(diagonal / exact if exact > 0 else float('nan'))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='weights/wan/Wan2.1-T2V-1.3B')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--frames', type=int, default=9)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 10, 20, 29])
    args = parser.parse_args()

    from diffusers import WanTransformer3DModel                # noqa: PLC0415

    device = torch.device('cuda')
    with progress.Timer(f'load {args.root}'):
        model = WanTransformer3DModel.from_pretrained(
            args.root, subfolder='transformer',
            torch_dtype=torch.bfloat16).to(device).eval()
    cfg = model.config
    layout = wan_layout.packed_layout(args.size, args.size, args.frames,
                                      text_len=0)
    grid = layout.target.grid
    progress.log(f'{args.size}x{args.size}x{args.frames} -> grid {grid}, '
                 f'{layout.used} tokens, {cfg.num_attention_heads} heads')

    captured: dict[int, tuple] = {}

    def capture(q, k, v, layer_index):
        captured[layer_index] = (q.detach(), k.detach())
        return v                                   # the value is unused

    for index in args.layers:
        model.blocks[index].attn1.set_processor(
            wan_attention.make_processor(capture, index))

    width = cfg.num_attention_heads * cfg.attention_head_dim
    hidden = torch.randn(1, layout.used, width, device=device,
                         dtype=torch.bfloat16)
    latent = (grid[0], grid[1] * 2, grid[2] * 2)
    rope = model.rope(torch.randn(1, cfg.in_channels, *latent, device=device,
                                  dtype=torch.bfloat16))
    with torch.no_grad():
        for index in args.layers:
            model.blocks[index].attn1(hidden, None, None, rope)

    print(f'\n{"layer":>6s} {"heads":>6s} {"diagonal share of tr(C_q C_k)/D":>34s}')
    everything = []
    for index in sorted(captured):
        q, k = captured[index]
        ratios = _ratios(q, k)
        everything += ratios
        print(f'{index:6d} {len(ratios):6d} {statistics.mean(ratios):20.3f}'
              f'  [{min(ratios):.3f}, {max(ratios):.3f}]')
    print(f'\nWan, all heads: {statistics.mean(everything):.3f}')
    print('H3 for comparison: 0.77 (docs/features/veda2.md)')


if __name__ == '__main__':
    main()
