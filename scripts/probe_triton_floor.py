"""Which compute capabilities can Triton actually compile a kernel for.

`kernels/arch.py` floors `tl.dot` at sm_75 with the reason "Triton
dropped Volta's mma.m8n8k4", and floors bf16 at sm_80. Both are
plausible and neither was measured; the whole sm_7x table rests on
them, so they are worth a measurement rather than a recollection.

This probe compiles ahead of time for a target capability, which needs
`ptxas` but not the device, so it runs on any machine with Triton
installed and -- the reason it is written this way -- does not contend
with a GPU that somebody else is using.

Two kernel shapes are probed per dtype, because `arch.py` makes two
different claims: `tl.dot` (the MMA path) and a load/convert/reduce
kernel with no dot in it at all (`_TRITON_MOVE_DTYPES`, which is what
the summary pass and the tile gather are).

    python scripts/probe_triton_floor.py
"""

import argparse
import sys

CAPABILITIES = (70, 75, 80, 86, 89, 90, 120)
# (pointer type for the inputs, pointer type for the dot's output).
# int8 inputs accumulate to int32; the 16-bit floats accumulate to fp32.
DTYPES = (('fp16', 'fp32'), ('bf16', 'fp32'), ('fp32', 'fp32'),
          ('i8', 'i32'))


def _kernels():
    """The two probe kernels, built lazily so import needs no Triton."""
    import triton                                         # noqa: PLC0415
    import triton.language as tl                          # noqa: PLC0415

    @triton.jit
    def dot_kernel(A, B, Out, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr):
        m, n, k = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
        a = tl.load(A + m[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] * N + n[None, :])
        tl.store(Out + m[:, None] * N + n[None, :], tl.dot(a, b))

    @triton.jit
    def move_kernel(A, Out, M: tl.constexpr, K: tl.constexpr):
        """Load, widen to fp32, reduce, store. No MMA anywhere."""
        m, k = tl.arange(0, M), tl.arange(0, K)
        a = tl.load(A + m[:, None] * K + k[None, :]).to(tl.float32)
        tl.store(Out + m, tl.sum(a, 1))

    return dot_kernel, move_kernel


def _compile(kernel, signature, constexprs, capability: int):
    """Compiles `kernel` ahead of time for `capability`.

    Returns:
        (error, instruction): the first line of the failure and None, or
        None and what the dot lowered to -- the MMA shape if it reached
        a tensor core, `fma` if Triton fell back to CUDA cores, which is
        the difference between "runs" and "runs well".
    """
    import triton                                         # noqa: PLC0415
    from triton.backends.compiler import GPUTarget        # noqa: PLC0415
    from triton.compiler.compiler import ASTSource        # noqa: PLC0415
    try:
        source = ASTSource(fn=kernel, signature=signature,
                           constexprs=constexprs)
        compiled = triton.compile(source,
                                  target=GPUTarget('cuda', capability, 32))
    except Exception as error:                            # noqa: BLE001
        text = str(error).strip().split('\n')
        detail = next((line for line in text if line.strip()), '')
        return f'{type(error).__name__}: {detail[:70]}', None
    return None, _lowering(compiled.asm.get('ptx', ''))


def _ops(ptx: str) -> str:
    """The arithmetic instruction mix, commonest first."""
    import collections                                    # noqa: PLC0415
    counts = collections.Counter(
        line.strip().split()[0] for line in ptx.split('\n')
        if line.strip().startswith(('fma', 'mma', 'mul.f', 'add.f', 'cvt.rn',
                                    'wgmma')))
    return ', '.join(f'{name} x{n}' for name, n in counts.most_common(4))


def _compile_ptx(kernel, signature, constexprs, capability: int) -> str:
    """The PTX for `kernel` at `capability`, or '' if it does not build."""
    import triton                                         # noqa: PLC0415
    from triton.backends.compiler import GPUTarget        # noqa: PLC0415
    from triton.compiler.compiler import ASTSource        # noqa: PLC0415
    try:
        source = ASTSource(fn=kernel, signature=signature,
                           constexprs=constexprs)
        return triton.compile(
            source, target=GPUTarget('cuda', capability, 32)).asm.get('ptx',
                                                                      '')
    except Exception:                                     # noqa: BLE001
        return ''


def _lowering(ptx: str) -> str:
    """The tensor-core shape in `ptx`, or `fma` when there is none."""
    for line in ptx.split('\n'):
        word = line.strip().split(' ')[0]
        if word.startswith(('mma.sync', 'wgmma')):
            parts = [p for p in word.split('.') if p.startswith('m')
                     and 'n' in p and 'k' in p]
            return parts[0] if parts else word[:22]
    return 'fma' if 'fma.rn' in ptx else '-'



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, default=(64, 64, 64),
                        metavar=('M', 'N', 'K'))
    parser.add_argument('--ops', action='store_true',
                        help='the arithmetic instruction mix of the dot, '
                             'which says whether the fma fallback is fp32 '
                             'or packed fp16 -- that factor of two decides '
                             'whether a sparse kernel beats dense at all')
    args = parser.parse_args()
    try:
        import triton                                     # noqa: PLC0415
    except ImportError:
        print('Triton is not installed; nothing to probe')
        return 1
    dot_kernel, move_kernel = _kernels()
    rows, cols, depth = args.shape
    print(f'triton {triton.__version__}, dot shape {rows}x{cols}x{depth}\n')
    width = 14
    print('cap  ' + ''.join(f'{f"{dt} dot":<{width}}' for dt, _ in DTYPES)
          + ''.join(f'{f"{dt} move":<{width}}' for dt, _ in DTYPES))
    notes = {}
    for capability in CAPABILITIES:
        cells = []
        for in_dtype, out_dtype in DTYPES:
            why, how = _compile(
                dot_kernel,
                {'A': f'*{in_dtype}', 'B': f'*{in_dtype}',
                 'Out': f'*{out_dtype}', 'M': 'constexpr', 'N': 'constexpr',
                 'K': 'constexpr'},
                {'M': rows, 'N': cols, 'K': depth}, capability)
            cells.append(how if why is None else 'FAIL')
            if why:
                notes.setdefault((in_dtype, 'dot', why), []).append(capability)
        for in_dtype, _ in DTYPES:
            why, _how = _compile(
                move_kernel,
                {'A': f'*{in_dtype}', 'Out': '*fp32', 'M': 'constexpr',
                 'K': 'constexpr'},
                {'M': rows, 'K': depth}, capability)
            cells.append('ok' if why is None else 'FAIL')
            if why:
                notes.setdefault((in_dtype, 'move', why), []).append(
                    capability)
        print(f'{capability:<5}' + ''.join(f'{c:<{width}}' for c in cells))
    if args.ops:
        print('\ninstruction mix of the dot (is the fallback fp32 or '
              'packed fp16?):')
        for in_dtype, out_dtype in DTYPES[:2]:
            for capability in (70, 75, 80):
                ptx = _compile_ptx(
                    dot_kernel,
                    {'A': f'*{in_dtype}', 'B': f'*{in_dtype}',
                     'Out': f'*{out_dtype}', 'M': 'constexpr',
                     'N': 'constexpr', 'K': 'constexpr'},
                    {'M': rows, 'N': cols, 'K': depth}, capability)
                print(f'  {in_dtype} sm_{capability}: '
                      f'{_ops(ptx) if ptx else "FAIL"}')
    print('\nint8 dot by shape (is the failure the arch or the shape?):')
    for shape in ((16, 16, 32), (32, 32, 64), (64, 64, 64), (128, 64, 128)):
        line = f'  {shape[0]}x{shape[1]}x{shape[2]:<4}'
        for capability in (75, 80, 120):
            why, how = _compile(
                dot_kernel,
                {'A': '*i8', 'B': '*i8', 'Out': '*i32', 'M': 'constexpr',
                 'N': 'constexpr', 'K': 'constexpr'},
                {'M': shape[0], 'N': shape[1], 'K': shape[2]}, capability)
            line += f'  sm_{capability}: {how if why is None else "FAIL":<12}'
        print(line)
    if notes:
        print('\nfailures:')
        for (dtype, kind, why), caps in sorted(notes.items()):
            print(f'  {dtype} {kind} on {caps}: {why}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
