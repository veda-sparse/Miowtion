"""Prepare the official MiniMax-H3 Ref2VA example for Miowtion (CPU only)."""

import argparse

from miowtion.train.reference_example import prepare_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/examples/minimax_h3_ref2va')
    args = parser.parse_args()
    prepare_example(args.out)


if __name__ == '__main__':
    main()
