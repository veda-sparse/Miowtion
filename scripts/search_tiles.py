"""Scores tile shapes with oracle masks on one geometry (dense teacher).

Example:
    torchrun --nproc_per_node 4 scripts/search_tiles.py \
        --config configs/search_16x9_t37.yaml
"""

import argparse
import dataclasses

import yaml

from miowtion.veda import search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    unknown = set(raw) - {f.name for f in dataclasses.fields(
        search.SearchConfig)}
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    search.run_search(search.SearchConfig(**raw))


if __name__ == '__main__':
    main()
