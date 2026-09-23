"""Scores tile shapes with oracle masks on one geometry (dense teacher).

Example:
    torchrun --nproc_per_node 4 scripts/search_tiles.py \
        --config configs/search_turbo8_16x9_t37_4090.yaml

Independent processes can split one config's geometries (e.g. one per GPU
group) with --geometries; they write to the same run directory.
"""

import argparse
import dataclasses

import yaml

from miowtion.veda import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--geometries', nargs='+', default=None,
                        help='score these geometry specs instead of the '
                        "config's, e.g. 16:9@37 4:3@102")
    args = parser.parse_args()
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    unknown = set(raw) - {f.name for f in dataclasses.fields(
        search.SearchConfig)}
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    if args.geometries:
        raw['geometries'] = args.geometries
    search.run_search(search.SearchConfig(**raw))


if __name__ == '__main__':
    main()
