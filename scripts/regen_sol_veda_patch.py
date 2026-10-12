"""Rebuild the vendored Sol+Veda kernel from upstream plus `patches/`.

The vendored file names this script, and until now the script did not
exist and `patches/` was empty, so the contract in AGENTS.md 3 (original
licence, patch series, regeneration method, run-time hash check) was
only three quarters kept: nothing said what the vendored file *is*
relative to its source.

    python scripts/regen_sol_veda_patch.py            # verify
    python scripts/regen_sol_veda_patch.py --write    # regenerate

Verify is the mode for CI and for review: it applies the patches to the
submodule's file in a temporary directory and fails if the result is not
byte for byte the vendored copy. `--write` is for after editing a patch.
"""

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.kernels import sol_veda                      # noqa: E402

_VENDORED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'miowtion', 'kernels', 'sol_veda_patched', 'attention.py')
_PATCHES = os.path.join(os.path.dirname(_VENDORED), 'patches')


def rebuild() -> str:
    """Applies every patch in order to the upstream file.

    Returns:
        The regenerated source.

    Raises:
        RuntimeError: If the upstream hash moved or a patch does not
            apply, which is the point: either means the vendored copy
            can no longer be derived from its stated source.
    """
    sol_veda._check_upstream()                             # noqa: SLF001
    with tempfile.TemporaryDirectory() as work:
        target = os.path.join(work, 'attention.py')
        shutil.copy(sol_veda._UPSTREAM, target)            # noqa: SLF001
        for name in sorted(os.listdir(_PATCHES)):
            if not name.endswith('.patch'):
                continue
            done = subprocess.run(
                ['patch', '-p1', '-i', os.path.join(_PATCHES, name)],
                cwd=work, capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(
                    f'{name} does not apply: {done.stdout}{done.stderr}')
        with open(target, encoding='utf-8') as handle:
            return handle.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='overwrite the vendored file with the result')
    args = parser.parse_args()
    built = rebuild()
    with open(_VENDORED, encoding='utf-8') as handle:
        have = handle.read()
    if built == have:
        print('the vendored kernel matches upstream + patches/')
        return 0
    if args.write:
        with open(_VENDORED, 'w', encoding='utf-8') as handle:
            handle.write(built)
        print(f'rewrote {_VENDORED} from upstream + patches/')
        return 0
    print('the vendored kernel is NOT upstream + patches/:')
    print(''.join(difflib.unified_diff(
        built.splitlines(True), have.splitlines(True),
        'upstream+patches', 'vendored'))[:4000])
    return 1


if __name__ == '__main__':
    sys.exit(main())
