"""Builds a tile plan from completed search score files.

Example:
    python scripts/build_plan.py --scores runs/search/scores/16x9_t37 \
        --out plans/16x9_t37.json --mirror-out plans/9x16_t37.json
"""

import argparse
import glob
import os

from miowtion.veda import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True,
                        help='directory of <clip>.json score files')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mirror-out', default=None,
                        help='also write the H<->W mirrored plan (e.g. 9:16 '
                        'from a 16:9 search) to this path')
    parser.add_argument('--max-padding', type=float, default=0.2)
    parser.add_argument('--max-shapes-per-layer', type=int, default=2)
    args = parser.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(args.scores, '*.json')))
    candidates, entries, meta = search.load_entries(paths)
    num_layers = max(e.layer for e in entries) + 1
    plan = search.build_plan(
        meta['geometry'], tuple(meta['grid']), candidates, entries,
        num_layers, args.max_padding, args.max_shapes_per_layer,
        meta={'search_keep_ratio': meta['keep_ratio'],
              'schedule': meta['schedule'],
              'num_steps': meta['num_steps'],
              'teacher_adapter': meta['teacher_adapter'],
              'variant': meta['variant']})
    plan.save(args.out)
    print(f'{args.out}: shapes {[str(s) for s in plan.shapes]}, '
          f'plan mse {plan.provenance["plan_mse"]:.4f}, best single '
          f'{plan.provenance["best_single_shape_mse"]:.4f}')
    if args.mirror_out:
        mirror = plan.mirrored()
        mirror.save(args.mirror_out)
        print(f'{args.mirror_out}: {mirror.geometry} mirrored from '
              f'{plan.geometry}')


if __name__ == '__main__':
    main()
