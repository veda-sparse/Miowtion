"""Asks ptxas which tensor-core and pipelining instructions a target has.

Vendor tables and kernel comments disagree about what the Blackwell
consumer / workstation parts (SM120) can do, and guessing wrong sends a
kernel port down a dead end: SM100's forward is built on tcgen05 with
accumulators in tensor memory, which SM120 does not have, while SM120 has a
block-scaled warp MMA that SM100 does not expose at warp level. The
assembler is the authority, so this asks it.

Each probe is one instruction in a minimal kernel. A probe is reported
`yes` when ptxas accepts it, `no` when ptxas says the instruction or a
feature of it is unsupported on the target, and `operands?` when ptxas
recognized the instruction but rejected the operand list -- that means the
instruction exists on the target and this probe's register counts are
wrong, so it is a bug in the probe, not a capability answer.

Example:
    python scripts/probe_ptx_isa.py --targets sm_89 sm_90a sm_100a sm_120a
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile

_KERNEL = """.version {version}
.target {target}
.address_size 64
.visible .entry probe(.param .u64 p)
{{
  .reg .b64 %rd<8>;
  .reg .b32 %r<64>;
  .reg .f32 %f<64>;
  ld.param.u64 %rd1, [p];
{body}
  ret;
}}
"""

# Instruction -> one line of PTX. Operand lists follow the PTX ISA for the
# shape named; where a probe reports 'operands?' the shape is right and the
# register counts are not.
PROBES = {
    # Warp-level MMA: the atom every SM80-family kernel is built on.
    'mma.sync bf16 m16n8k16':
        'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 '
        '{%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f5,%f6,%f7,%f8};',
    'mma.sync fp8 e4m3 m16n8k32':
        'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 '
        '{%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f5,%f6,%f7,%f8};',
    'mma.sync int8 m16n8k32':
        'mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 '
        '{%r20,%r21,%r22,%r23}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, '
        '{%r24,%r25,%r26,%r27};',
    # Block-scaled warp MMA (one scale per 16/32 values), the fp4 path.
    'mma.sync block_scale mxf8f6f4 1X':
        'mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale'
        '.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%f1,%f2,%f3,%f4}, '
        '{%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f5,%f6,%f7,%f8}, {%r9}, {0, 0}, '
        '{%r10}, {0, 0};',
    'mma.sync block_scale nvf4 4X':
        'mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale'
        '.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%f1,%f2,%f3,%f4}, '
        '{%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f5,%f6,%f7,%f8}, {%r9}, {0, 0}, '
        '{%r10}, {0, 0};',
    # Hopper's warpgroup MMA and Blackwell-datacenter's tensor-memory MMA.
    'wgmma.mma_async bf16':
        'wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 '
        '{%f1,%f2,%f3,%f4}, %rd1, %rd2, 1, 1, 1, 0, 0;',
    'tcgen05.alloc':
        'tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd1], 32;',
    'tcgen05.ld 16x64b':
        'tcgen05.ld.sync.aligned.16x64b.x1.b32 {%r1}, [%r2];',
    # Pipelining and scheduling, independent of the MMA.
    'cp.async.bulk.tensor (TMA)':
        'cp.async.bulk.tensor.2d.shared::cluster.global.tile'
        '.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3}], [%r4];',
    'stmatrix m8n8':
        'stmatrix.sync.aligned.m8n8.x1.shared.b16 [%rd1], {%r1};',
    'setmaxnreg':
        'setmaxnreg.inc.sync.aligned.u32 240;',
    'clusterlaunchcontrol.try_cancel':
        'clusterlaunchcontrol.try_cancel.async.shared::cta'
        '.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 '
        '[%r1], [%r2];',
}

_UNSUPPORTED = re.compile(r"(Instruction|Feature) '[^']*' not supported")
_OPERANDS = re.compile(r'(Argument vector size mismatch|Arguments mismatch)')


def probe(ptxas: str, target: str, body: str, version: str) -> str:
    """'yes', 'no' or 'operands?' for one instruction on one target."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'probe.ptx')
        with open(path, 'w') as f:
            f.write(_KERNEL.format(version=version, target=target, body=body))
        done = subprocess.run([ptxas, f'-arch={target}', '-o', os.devnull,
                               path], capture_output=True, text=True,
                              check=False)
    if done.returncode == 0:
        return 'yes'
    if _UNSUPPORTED.search(done.stderr):
        return 'no'
    if _OPERANDS.search(done.stderr):
        return 'operands?'
    return 'error'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+',
                        default=['sm_89', 'sm_90a', 'sm_100a', 'sm_120a'])
    parser.add_argument('--ptxas', default=shutil.which('ptxas'),
                        help='defaults to ptxas on PATH')
    parser.add_argument('--ptx-version', default='8.7',
                        help='.version of the probe kernels')
    args = parser.parse_args()
    if not args.ptxas:
        raise SystemExit('no ptxas found; pass --ptxas')
    width = max(len(name) for name in PROBES)
    print(f'{"instruction":<{width}} | ' + ' | '.join(
        f'{t:>9}' for t in args.targets))
    print('-' * (width + 12 * len(args.targets)))
    for name, body in PROBES.items():
        results = [probe(args.ptxas, target, body, args.ptx_version)
                   for target in args.targets]
        print(f'{name:<{width}} | ' + ' | '.join(
            f'{r:>9}' for r in results))


if __name__ == '__main__':
    main()
