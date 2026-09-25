"""Low-precision block scoring: how fp8 / fp4 move the top-k selection.

The selection pass (score every 128x128 block, keep the top-k per query
tile) reads q and k once and produces one number per block. It is the same
O(S^2 D) shape as attention itself, so running it in fp8 or fp4 is the
obvious saving -- but only if the *set of blocks it picks* survives the
rounding. Absolute score error is not the question: top-k is invariant to a
per-row constant and to any monotone rescaling of a row, so what matters is
whether the ordering near the budget boundary flips.

This module fake-quantizes (quantize then dequantize, arithmetic stays in
bf16/fp32) so any scheme can be measured on this hardware, including ones
with no tensor-core support here, and so the only difference against the
reference is the rounding itself.

Schemes, all with the scale factored per something that a kernel can
actually keep in registers (a dot product of two per-row-scaled vectors is
the scaled dot product, so per-row scaling is free at the GEMM level):

  fp8_e4m3_head   e4m3, one amax scale per head          (coarsest)
  fp8_e4m3_row    e4m3, one amax scale per row and head
  fp8_e5m2_row    e5m2, same granularity (more range, 2 mantissa bits)
  nvfp4           e2m1 values, e4m3 scale per 16 elements, fp32 per head
  mxfp8_e4m3      e4m3 values, e8m0 (power-of-two) scale per 32 elements
  mxfp4           e2m1 values, e8m0 scale per 32 elements   (finest 4-bit)

`bf16` is the identity, kept as a control: it must come back with recall
1.0, otherwise the measurement, not the scheme, is broken.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from miowtion.h3 import attention as h3_attention
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import tiling

# Largest finite magnitude of each element format.
_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_E2M1_MAX = 6.0

# The eight magnitudes of e2m1 (1 sign, 2 exponent, 1 mantissa bits), and
# the midpoints between them (round to nearest, ties away from zero -- the
# hardware rounds ties to even, which differs on 1 of 16 codes at most and
# is below the effect being measured here).
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

# Block sizes of the micro-scaled formats, along the head dimension.
_NVFP4_BLOCK = 16
_MX_BLOCK = 32


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Rounds |x| <= 6 to the nearest e2m1 magnitude, keeping the sign."""
    edges = torch.tensor(_E2M1_EDGES, device=x.device, dtype=torch.float32)
    levels = torch.tensor(_E2M1_LEVELS, device=x.device, dtype=torch.float32)
    index = torch.bucketize(x.abs().float(), edges)
    return torch.sign(x) * levels[index]


def _round_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Rounds a positive scale up to a power of two (the e8m0 of MX)."""
    return torch.exp2(torch.ceil(torch.log2(scale.clamp(min=1e-30))))


def _cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).to(torch.float32)


def _safe(scale: torch.Tensor) -> torch.Tensor:
    """An all-zero slice would divide by zero; its values are zero anyway."""
    return scale.clamp(min=torch.finfo(torch.float32).tiny)


def _scaled(x: torch.Tensor, dims: tuple[int, ...], limit: float,
            dtype: torch.dtype) -> torch.Tensor:
    """amax scaling over `dims`, cast to `dtype`, scaled back."""
    scale = _safe(x.abs().amax(dim=dims, keepdim=True).float() / limit)
    return (_cast((x.float() / scale).clamp(-limit, limit), dtype)
            * scale).to(x.dtype)


def _blocked(x: torch.Tensor, block: int) -> tuple[torch.Tensor, list[int]]:
    """[..., D] -> [..., D/block, block]; D must be a multiple of block."""
    if x.shape[-1] % block:
        raise ValueError(f'head_dim {x.shape[-1]} is not a multiple of '
                         f'the scaling block {block}')
    shape = list(x.shape)
    return x.float().view(*shape[:-1], shape[-1] // block, block), shape


def _micro_scaled(x: torch.Tensor, block: int, limit: float,
                  scale_dtype: torch.dtype | None) -> torch.Tensor:
    """Per-block amax scaling; `scale_dtype` None means e8m0 (power of two).

    With a finite-precision scale (NVFP4's e4m3) the block scale itself has
    to be representable, which needs a second, per-head fp32 scale: that is
    the two-level scheme NVFP4 specifies, not an extra approximation.
    """
    blocks, shape = _blocked(x, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    if scale_dtype is None:
        scale = _round_e8m0(amax / limit)
    else:
        # dims: [N, H, D/block, block] -> per head (dim 1) global scale.
        head_amax = amax.amax(dim=0, keepdim=True).amax(dim=2, keepdim=True)
        outer = _safe(head_amax / (limit * _E4M3_MAX))
        scale = outer * _cast((amax / limit / outer).clamp(
            max=_E4M3_MAX), scale_dtype)
    scale = _safe(scale)
    values = (blocks / scale).clamp(-limit, limit)
    if limit == _E2M1_MAX:
        values = _round_e2m1(values)
    else:
        values = _cast(values, torch.float8_e4m3fn)
    return (values * scale).view(*shape).to(x.dtype)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


# name -> fake-quantizer on a [N, H, D] tile-ordered tensor.
SCHEMES = {
    'bf16': _identity,
    'fp8_e4m3_head': lambda x: _scaled(x, (0, 2), _E4M3_MAX,
                                       torch.float8_e4m3fn),
    'fp8_e4m3_row': lambda x: _scaled(x, (2,), _E4M3_MAX,
                                      torch.float8_e4m3fn),
    'fp8_e5m2_row': lambda x: _scaled(x, (2,), _E5M2_MAX,
                                      torch.float8_e5m2),
    'nvfp4': lambda x: _micro_scaled(x, _NVFP4_BLOCK, _E2M1_MAX,
                                     torch.float8_e4m3fn),
    'mxfp8_e4m3': lambda x: _micro_scaled(x, _MX_BLOCK, _E4M3_MAX, None),
    'mxfp4': lambda x: _micro_scaled(x, _MX_BLOCK, _E2M1_MAX, None),
}


def fake_quantize(x: torch.Tensor, scheme: str) -> torch.Tensor:
    """Quantizes and dequantizes one tile-ordered tensor.

    Args:
        x: [N, H, D] tile-ordered q or k (bf16 in training and inference).
        scheme: A key of SCHEMES.

    Returns:
        A tensor of x's shape and dtype, holding only values the scheme can
        represent (times its scale).

    Raises:
        ValueError: On an unknown scheme.
    """
    if scheme not in SCHEMES:
        raise ValueError(f'scheme must be one of {sorted(SCHEMES)}, got '
                         f'{scheme!r}')
    if x.ndim != 3:
        raise ValueError(f'expected [N, H, D] tile-ordered rows, got '
                         f'{tuple(x.shape)}')
    return SCHEMES[scheme](x)


@dataclasses.dataclass
class QuantRecord:
    """One (step, layer, head group, scheme) measurement."""

    step: int
    layer: int
    shape: str
    heads: int
    scheme: str
    recall: float        # top-k overlap with the bf16 oracle, diagonal out
    heat_kept: float     # true heat mass the quantized top-k keeps
    heat_ceiling: float  # true heat mass the bf16 top-k keeps
    rel_l2: float        # ||heat_q - heat|| / ||heat||
    max_abs: float       # max |heat_q - heat|, heat is in [0, 1]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


class QuantHeatProbe:
    """Dense attention that also measures quantized block scoring.

    Plugged in where TeacherCollector goes: the residual stream is the
    dense teacher's, untouched, and every layer additionally builds the
    exact teacher heat plus one heat per scheme from quantized q / k, then
    compares the top-k sets they select.

    The exact LSE is used for every scheme. Top-k is invariant to a per-row
    constant, so the LSE cannot change the selection; using the same one
    keeps the reported heat errors on a single scale.
    """

    def __init__(self, clip, plan: veda_plan.TilePlan, schemes: list[str],
                 q_tile_fraction: float = 1.0, layer_every: int = 1,
                 dense_backend: str = 'auto',
                 generator: torch.Generator | None = None):
        """Initializes the probe.

        Args:
            clip: ClipTiling of the current clip.
            plan: Tile plan of the geometry.
            schemes: Scheme names to measure, in report order.
            q_tile_fraction: Fraction of video query tiles measured per
                layer and head group (the metrics are means over query
                tiles, so a subsample is unbiased).
            layer_every: Measure every n-th layer; the others only run the
                dense teacher.
            dense_backend: Kernel for the dense teacher.
            generator: Draws the measured query tiles (CPU generator).

        Raises:
            ValueError: On an unknown scheme or a non-positive setting.
        """
        for scheme in schemes:
            if scheme not in SCHEMES:
                raise ValueError(f'unknown scheme {scheme!r}')
        if not 0.0 < q_tile_fraction <= 1.0:
            raise ValueError('q_tile_fraction must be in (0, 1]')
        if layer_every < 1:
            raise ValueError('layer_every must be >= 1')
        self.clip = clip
        self.plan = plan
        self.schemes = schemes
        self.q_tile_fraction = q_tile_fraction
        self.layer_every = layer_every
        self.dense_backend = dense_backend
        self.generator = generator or torch.Generator().manual_seed(0)
        self.step = 0
        self.records: list[QuantRecord] = []

    def _rows(self, tile_layout: tiling.TileLayout) -> torch.Tensor:
        n_video = tile_layout.n_video_tiles
        count = max(1, min(n_video, round(self.q_tile_fraction * n_video)))
        rows = torch.randperm(n_video, generator=self.generator)[:count]
        return rows.sort().values.to(self.clip.device)

    @torch.no_grad()
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        # Imported here: veda.attention imports this module's siblings and
        # only the chunking helpers are needed, not a package cycle.
        from miowtion.veda import attention as veda_attention  # pylint: disable=import-outside-toplevel
        out, lse = h3_attention.dense_attention(
            q, k, v, self.clip.layout.used, return_lse=True,
            backend=self.dense_backend)
        if layer_index % self.layer_every:
            return out
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            blocks = self.clip.blocks(tile_layout)
            rows = self._rows(tile_layout)
            chunk = veda_attention._chunk_heads(tile_layout)  # pylint: disable=protected-access
            for heads in group.heads.split(chunk):
                q_tiles = veda_attention._gather(q, tile_layout, heads)  # pylint: disable=protected-access
                k_tiles = veda_attention._gather(k, tile_layout, heads)  # pylint: disable=protected-access
                lse_tiles = veda_attention._gather_lse(  # pylint: disable=protected-access
                    lse, tile_layout, heads)
                exact = heatmap.teacher_heat(q_tiles, k_tiles, lse_tiles,
                                             tile_layout, rows)
                norm = exact.float().pow(2).sum().sqrt()
                for scheme in self.schemes:
                    heat = heatmap.teacher_heat(
                        fake_quantize(q_tiles, scheme),
                        fake_quantize(k_tiles, scheme), lse_tiles,
                        tile_layout, rows)
                    diag = heatmap.mask_diagnostics(
                        heat, exact, tile_layout, blocks, rows)
                    delta = (heat.float() - exact.float())
                    self.records.append(QuantRecord(
                        step=self.step, layer=layer_index,
                        shape=str(group.shape), heads=int(heads.numel()),
                        scheme=scheme,
                        recall=float(diag['recall']),
                        heat_kept=float(diag['heat_kept']),
                        heat_ceiling=float(diag['heat_ceiling']),
                        rel_l2=float(delta.pow(2).sum().sqrt()
                                     / norm.clamp(min=1e-30)),
                        max_abs=float(delta.abs().max())))
                    del heat, delta
                del q_tiles, k_tiles
        return out


def summarize(records: list[QuantRecord]) -> dict[str, dict[str, float]]:
    """Head-weighted means per scheme over every step, layer and group.

    Head groups differ in size, so a plain mean over records would weigh a
    two-head group like a thirty-head one.

    Returns:
        scheme -> {recall, heat_kept, heat_ceiling, rel_l2, max_abs,
        kept_vs_ceiling}, the last being the share of the attainable heat
        mass the scheme's selection actually keeps.
    """
    out = {}
    for record in records:
        acc = out.setdefault(record.scheme, {'weight': 0.0})
        weight = float(record.heads)
        acc['weight'] += weight
        for field in ('recall', 'heat_kept', 'heat_ceiling', 'rel_l2'):
            value = getattr(record, field)
            if not math.isnan(value):
                acc[field] = acc.get(field, 0.0) + weight * value
        acc['max_abs'] = max(acc.get('max_abs', 0.0), record.max_abs)
    for acc in out.values():
        weight = acc.pop('weight')
        for field in ('recall', 'heat_kept', 'heat_ceiling', 'rel_l2'):
            acc[field] = acc.get(field, 0.0) / weight
        acc['kept_vs_ceiling'] = (acc['heat_kept']
                                  / max(acc['heat_ceiling'], 1e-30))
    return out
