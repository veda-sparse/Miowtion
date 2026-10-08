"""Merges sharded sample caches, preserving manifest order."""

import argparse
import json

from miowtion.train import data
from miowtion.utils import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()
    with open(args.manifest) as f:
        sample_order = [json.loads(line)['id'] for line in f if line.strip()]
    with progress.Timer(
            f'merge {len(args.inputs)} caches ({len(sample_order)} samples)'):
        data.merge_sample_caches(
            args.inputs, args.out, sample_order=sample_order)


if __name__ == '__main__':
    main()
