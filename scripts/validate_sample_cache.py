"""Validates an encoded cache against its manifest and train config."""

import argparse
import json

from miowtion.train import data
from miowtion.train import trainer
from miowtion.utils import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = trainer.TrainConfig.from_yaml(args.config)
    config.validate()
    with open(args.manifest) as f:
        ids = [json.loads(line)['id'] for line in f if line.strip()]
    with progress.Timer(f'validate training cache ({len(ids)} samples)'):
        summary = data.validate_training_cache(
            args.cache, ids, 'train', config.tasks, config.geometries)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
