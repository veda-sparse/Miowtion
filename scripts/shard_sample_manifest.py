"""Splits a sample JSONL manifest into round-robin shards."""

import argparse

from miowtion.train import data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-shards', required=True, type=int)
    args = parser.parse_args()
    paths = data.shard_jsonl_manifest(
        args.manifest, args.out, args.num_shards)
    for path in paths:
        print(path, flush=True)


if __name__ == '__main__':
    main()
