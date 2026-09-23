# FA4 CuTe: block sparsity on the SM80 family (sm80/86/89)

Patch series on top of flash-attention `d15f1531a460ba456f41b01a774f33ab2db8febf`
(the version pinned in `pyproject.toml`). Upstream FA4 accepts block-sparse
tensors on SM8x but silently computes dense attention; these patches add the
block-sparse forward and backward main loops to the SM80 kernels.

```bash
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
git checkout d15f1531a460ba456f41b01a774f33ab2db8febf
git am /path/to/Miowtion/third_party/patches/fa4_sm89/*.patch
uv pip install -e flash_attn/cute --no-deps
```

Status and measurements: `docs/features/veda_kernel.md` (RTX 4090: forward
efficiency 0.97-1.00, backward 0.75-0.84 at 90% sparsity). Not yet wired into
`miowtion.veda.kernels.fa4.available()`, which only enables SM9x/10x/11x.
