"""Trains the Veda predictor (stage 1) or runs LoRA recovery (stage 2).

Example (4 GPUs):
    torchrun --nproc_per_node 4 scripts/train.py --config configs/x.yaml
"""

import argparse

from miowtion.train import trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    trainer.Trainer(trainer.TrainConfig.from_yaml(args.config)).train()


if __name__ == '__main__':
    main()
