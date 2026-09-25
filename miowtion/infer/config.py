"""Versioned run configuration for the generation script.

Generation has as many knobs as training does -- offload depth, chunk rows,
plans, keep ratio -- and getting one of them wrong does not fail fast: a
24 GB card runs the short geometries happily and then dies of OOM an hour
in, on the first 14 s clip. So the launch parameters belong in a file under
`configs/` next to the training ones, not in a shell line that is retyped
every time.

The file is merged into argparse defaults, so a config is a complete,
runnable launch and any flag can still be overridden on the command line
(handy for `--sample-id` or `--out-dir`). Unknown keys raise: a silently
ignored `offload_blocks` is exactly the failure this is meant to prevent.
"""

from __future__ import annotations

import argparse

import yaml


def merge_into(parser: argparse.ArgumentParser, path: str) -> None:
    """Applies a YAML config as the parser's defaults.

    Keys are the argument dest names (underscores, no leading dashes).

    Args:
        parser: The parser to update, before `parse_args()`.
        path: YAML file.

    Raises:
        ValueError: If the file is not a mapping, or holds a key that is not
            an argument of `parser`.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected a mapping of option names')
    known = {action.dest for action in parser._actions}  # pylint: disable=protected-access
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f'{path}: unknown config keys {unknown}')
    parser.set_defaults(**raw)


def require(args: argparse.Namespace, names: list[str]) -> None:
    """Raises if an option was set neither in the config nor on the CLI.

    The options a config is expected to carry cannot be argparse-`required`
    (that would force them onto the command line even with a config), so
    they are checked after the merge instead.

    Raises:
        ValueError: If any of `names` is None.
    """
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise ValueError(f'missing required options: {sorted(missing)} '
                         '(pass them in --config or on the command line)')
