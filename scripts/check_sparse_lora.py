"""Checks one real H3 state on CUDA without any optimizer update."""

import argparse
import json

from miowtion.train import sparse_lora
from miowtion.train import sparse_lora_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sample-cache', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--geometry', default='16:9@37')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    report = sparse_lora_validation.check_native_state(
        sparse_lora.SparseLoraConfig.from_yaml(args.config), args.sample_cache,
        args.sample_id, args.geometry, args.out, args.seed)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
