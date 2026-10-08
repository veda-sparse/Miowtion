"""Bolts Veda2's two free score terms onto an existing predictor bundle.

Veda's predictor is a bilinear form on [mean|max|min] pooled tiles. Two
terms of a block's log attention mass are not expressible that way, so no
amount of training reaches them (see docs/features/veda2.md):

    log E_ij = log B_j + Qbar . Kbar / sqrt(D) + E[q^2] . Var(k) / (2 D)
                ^^^^^^^                          ^^^^^^^^^^^^^^^^^^^^^^^
                row count                        second cumulant

Both have closed-form coefficients, so they can be added to an already
trained predictor without training anything: the trained projections are
copied across untouched and the new terms are initialized to exactly those
coefficients. The result keeps every update the input bundle paid for and
gains the terms it structurally could not learn.

Example:
    python scripts/bolt_veda2.py \\
        --in weights/veda/<release>/<name>_fp8.safetensors \\
        --out weights/veda/veda2_bolted.safetensors
"""

import argparse

import torch

from miowtion.utils import progress
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import predictor as veda_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in', dest='source', required=True,
                        help='bundle to bolt the terms onto')
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-count-term', action='store_true',
                        help='leave out log B_j')
    parser.add_argument('--no-second-order', action='store_true',
                        help='leave out the second cumulant')
    parser.add_argument('--base', default='keep', choices=('keep', 'reset'),
                        help="'keep' carries the input's trained "
                        "projections across; 'reset' draws fresh ones at "
                        'N(0, 1e-4), which makes the base term mean-pooled '
                        'QK again. Measured: the closed-form coefficients '
                        'only compose with a reset base, because training '
                        "against the block maximum shrinks the base's "
                        'spread and the exact terms then drown it out')
    parser.add_argument('--dtype', default='bfloat16',
                        choices=sorted(veda_bundle.DTYPES))
    parser.add_argument('--second-order-scale', type=float, default=1.0,
                        help='multiplier on the closed-form second-order '
                        'coefficient. The exact value balances against an '
                        '*untrained* base term; a trained base has a '
                        'different spread, so this has to be re-fit')
    parser.add_argument('--count-scale', type=float, default=1.0,
                        help='multiplier on the log B_j gain, same reason')
    args = parser.parse_args()

    source = veda_bundle.load(args.source)
    meta = source.metadata
    layers = int(meta['num_layers'])
    heads = int(meta['num_heads'])
    head_dim = int(meta['head_dim'])
    if source.predictor.second_order_rank or source.predictor.count_term:
        raise ValueError(f'{args.source} already carries the Veda2 terms')
    rank = 0 if args.no_second_order else head_dim
    count = not args.no_count_term
    progress.log(f'bolting onto {layers} layers x {heads} heads x '
                 f'{head_dim}: second_order_rank {rank}, count_term '
                 f'{count}, base {args.base}')

    target = veda_predictor.TileScorePredictor(
        layers, heads, head_dim, second_order_rank=rank, count_term=count)
    missing, unexpected = target.load_state_dict(
        source.predictor.state_dict(), strict=False)
    if unexpected:
        raise ValueError(f'{args.source} has unexpected keys: {unexpected}')
    expected = set()
    for i in range(layers):
        if rank:
            expected |= {f'layers.{i}.so_q', f'layers.{i}.so_k'}
        if count:
            expected.add(f'layers.{i}.count_gain')
    if set(missing) != expected:
        raise ValueError(f'unexpected missing keys: {sorted(missing)}')
    if rank:
        target.init_exact_second_order_()
    if args.base == 'reset':
        with torch.no_grad():
            for layer in target.layers:
                torch.nn.init.normal_(layer.proj_q,
                                      std=veda_predictor.INIT_STD)
                torch.nn.init.normal_(layer.proj_k,
                                      std=veda_predictor.INIT_STD)
    with torch.no_grad():
        for layer in target.layers:
            if rank and args.second_order_scale != 1.0:
                # The term is a product of two factors, so each carries
                # the square root of the requested scale.
                root = abs(args.second_order_scale) ** 0.5
                sign = 1.0 if args.second_order_scale > 0 else -1.0
                layer.so_q.mul_(root * sign)
                layer.so_k.mul_(root)
            if count and args.count_scale != 1.0:
                layer.count_gain.mul_(args.count_scale)

    veda_bundle.save(
        args.out, target.state_dict(), source.plans,
        num_layers=layers, num_heads=heads, head_dim=head_dim,
        keep_ratio=source.keep_ratio,
        source=(f"bolted from {meta.get('source', args.source)} "
                f"(base {args.base}, so_scale {args.second_order_scale}, "
                f"count_scale {args.count_scale})"),
        source_weights=meta.get('source_weights', 'unknown'),
        step=int(meta.get('step', 0)),
        dtype=veda_bundle.DTYPES[args.dtype],
        second_order_rank=rank, count_term=count)
    progress.log(f'wrote {args.out}')
    check = veda_bundle.load(args.out)
    assert check.predictor.second_order_rank == rank
    assert check.predictor.count_term == count
    progress.log('reloaded and verified')


if __name__ == '__main__':
    main()
