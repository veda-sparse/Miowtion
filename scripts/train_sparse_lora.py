"""Prepares or trains the frozen-predictor T2VA Sparse LoRA experiment."""

import argparse
import json
from pathlib import Path

from miowtion.train import sparse_lora
from miowtion.train import sparse_lora_data
from miowtion.train import sparse_lora_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-data', action='store_true')
    modes.add_argument('--encode', action='store_true')
    modes.add_argument('--cpu-preflight', action='store_true')
    modes.add_argument('--train', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--manifest')
    parser.add_argument('--out')
    parser.add_argument('--root')
    parser.add_argument('--report')
    parser.add_argument('--resume')
    parser.add_argument('--max-memory', help='JSON map, e.g. {"0":"40GiB"}')
    args = parser.parse_args()
    if args.prepare_data or args.encode:
        if not args.manifest or not args.out:
            parser.error('--manifest and --out required')
        if args.prepare_data:
            report = sparse_lora_data.prepare_splits(args.manifest, args.out)
            print(json.dumps(report, indent=2), flush=True)
        else:
            if not args.root:
                parser.error('--root required for encoding')
            memory = json.loads(args.max_memory) if args.max_memory else None
            memory = {int(k) if k.isdigit() else k: v
                      for k, v in memory.items()} if memory else None
            sparse_lora_data.encode_manifest(args.root, args.manifest,
                                             args.out, memory)
        return
    if not args.config:
        parser.error('--config required')
    config = sparse_lora.SparseLoraConfig.from_yaml(args.config)
    if args.cpu_preflight:
        report = sparse_lora_trainer.cpu_preflight(config)
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    else:
        sparse_lora_trainer.SparseLoraTrainer(config, args.resume).train()


if __name__ == '__main__':
    main()
