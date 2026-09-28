"""Analytic FLOP counts of one H3 DiT forward, for MFU reporting.

A measured step time only becomes a number one can compare against other
hardware or other geometries after it is divided by the arithmetic the step
actually had to do, so the count here has to follow the code in
miowtion.h3.model rather than a generic "6 * N * D" transformer estimate:

- the packed sequence is padded to a 64-aligned `seq_len`, and every linear
  runs over all `seq_len` rows, while attention only sees the real rows
  (`used`, see miowtion.h3.attention.dense_attention), so the two use
  different lengths;
- the token refiner runs once per clip, not once per denoising step;
- the per-block AdaLN projection is replaced by precomputed tables in
  inference (miowtion.train.adaln), so it is counted separately and is zero
  on that path;
- Veda skips whole tiles of the score matrix, so its attention term is the
  dense one scaled by the kept fraction.

Counted: every matmul, at 2 FLOP per multiply-accumulate, plus the softmax
value matmul. Not counted: RMSNorm, SiLU, RoPE, the AdaLN modulation and
gated residuals, the softmax exponentials, and the index_select gathers.
Those are memory-bound elementwise work of O(seq_len * hidden), below 1% of
the matmul count at every geometry we run, and counting them would make the
MFU depend on how one prices a transcendental.

Use `mfu` to turn a count and a measured time into a fraction of the device
peak; the peaks are in `PEAK_BF16_FLOPS`.
"""

from __future__ import annotations

import dataclasses

from miowtion.h3 import config as h3_config
from miowtion.h3 import layout as h3_layout

# Dense bf16 tensor-core peak, FLOP/s, of the devices we measure on. These
# are the *dense* numbers: the 2:4-structured-sparsity figures the vendors
# also publish (2x these) are unreachable for our GEMMs and would halve
# every MFU we report.
PEAK_BF16_FLOPS = {
    'NVIDIA GeForce RTX 4090': 165.2e12,
    # Same figure as miowtion.kernels.bench (bf16, fp32 accumulate).
    'NVIDIA RTX PRO 6000 Blackwell Server Edition': 503.8e12,
    'NVIDIA RTX PRO 6000 Blackwell Workstation Edition': 503.8e12,
    'NVIDIA H100 80GB HBM3': 989.4e12,
    'NVIDIA H100 PCIe': 756.0e12,
    'NVIDIA A100-SXM4-80GB': 312.0e12,
}


@dataclasses.dataclass(frozen=True)
class Flops:
    """A FLOP count split by the part of the model it comes from.

    Attributes:
        attention: The two score matmuls (q k^T and p v).
        linear: Every weight matmul except the AdaLN projections.
        adaln: The AdaLN projections, zero when precomputed tables are used.
    """

    attention: int = 0
    linear: int = 0
    adaln: int = 0

    @property
    def total(self) -> int:
        return self.attention + self.linear + self.adaln

    def __add__(self, other: Flops) -> Flops:
        return Flops(self.attention + other.attention,
                     self.linear + other.linear,
                     self.adaln + other.adaln)

    def __mul__(self, factor: int) -> Flops:
        return Flops(self.attention * factor, self.linear * factor,
                     self.adaln * factor)

    def as_dict(self) -> dict[str, int]:
        return {'attention': self.attention, 'linear': self.linear,
                'adaln': self.adaln, 'total': self.total}


def _gemm(rows: int, in_features: int, out_features: int) -> int:
    """2 FLOP per multiply-accumulate of a [rows, in] x [in, out] matmul."""
    return 2 * rows * in_features * out_features


def attention_flops(used: int, inner_dim: int,
                    keep_ratio: float = 1.0) -> int:
    """Both score matmuls of one full-attention layer over `used` rows.

    q k^T and p v are each `used * used * inner_dim` multiply-accumulates.

    Args:
        used: Real (non-padding) rows; padding rows are never attended.
        inner_dim: num_heads * head_dim.
        keep_ratio: Fraction of the score matrix a sparse kernel evaluates
            (1.0 for dense).

    Raises:
        ValueError: When `keep_ratio` is outside (0, 1].
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f'keep_ratio must be in (0, 1], got {keep_ratio}')
    return int(4 * used * used * inner_dim * keep_ratio)


def block_flops(config: h3_config.H3Config, seq_len: int, used: int,
                keep_ratio: float = 1.0,
                adaln_tables: bool = True,
                num_slots: int = 2) -> Flops:
    """One trunk block on a packed sequence.

    Args:
        config: The architecture.
        seq_len: Padded rows; every linear runs over these.
        used: Real rows; attention runs over these.
        keep_ratio: See `attention_flops`.
        adaln_tables: True when the modulation vectors are precomputed once
            per step (inference); False when the block projects them itself.
        num_slots: Distinct timesteps in the step (the `M` of
            miowtion.h3.schedule.TimestepState); only used when
            `adaln_tables` is False.
    """
    hidden, inner = config.hidden_size, config.inner_dim
    linear = (_gemm(seq_len, hidden, 3 * inner)          # qkv_proj
              + _gemm(seq_len, inner, hidden)            # out_proj
              + _gemm(seq_len, hidden, 2 * config.ffn_dim)   # mlp.fc1
              + _gemm(seq_len, config.ffn_dim, hidden))      # mlp.fc2
    adaln = 0 if adaln_tables else _gemm(
        num_slots, config.time_embed_dim,
        6 * h3_config.MODALITY_NUM * hidden)
    return Flops(attention=attention_flops(used, inner, keep_ratio),
                 linear=linear, adaln=adaln)


def refiner_flops(config: h3_config.H3Config, text_len: int) -> Flops:
    """The text tower: condition_proj plus every RefinerBlock.

    Runs once per clip (H3DiT.refine_text), not once per denoising step.
    """
    hidden, inner = config.hidden_size, config.inner_dim
    linear = _gemm(text_len, config.text_dim, hidden)
    attention = 0
    for _ in range(config.num_refiner_layers):
        linear += (_gemm(text_len, hidden, 3 * inner)
                   + _gemm(text_len, inner, hidden)
                   + _gemm(text_len, hidden, 2 * config.ffn_dim)
                   + _gemm(text_len, config.ffn_dim, hidden))
        attention += attention_flops(text_len, inner)
    return Flops(attention=attention, linear=linear)


def step_flops(config: h3_config.H3Config, layout: h3_layout.PackedLayout,
               keep_ratio: float = 1.0, adaln_tables: bool = True,
               num_slots: int = 2) -> Flops:
    """One velocity evaluation: embedding, the trunk, and the output heads.

    This is the denominator of a step-time MFU: it is what H3DiT.forward
    does, and it excludes the text tower, which the trajectory pays once.

    Args:
        config: The architecture.
        layout: The packed request (miowtion.h3.layout.pack).
        keep_ratio: Kept fraction of the score matrix, 1.0 for dense.
        adaln_tables: See `block_flops`.
        num_slots: See `block_flops`.
    """
    hidden = config.hidden_size
    video_rows = int(layout.img_pos.numel())
    audio_rows = int(layout.audio_pos.numel())
    target_video = int(layout.target_img_pos.numel())
    target_audio = int(layout.target_audio_pos.numel())
    embed = (_gemm(video_rows, config.video_patch_dim, hidden)
             + _gemm(audio_rows, config.audio_channels, hidden))
    heads = (_gemm(target_video, hidden, config.video_patch_dim)
             + _gemm(target_audio, hidden, config.audio_channels))
    time = (_gemm(num_slots, config.freq_dim, config.time_embed_hidden)
            + _gemm(num_slots, config.time_embed_hidden,
                    config.time_embed_dim))
    total = Flops(linear=embed + heads + time,
                  adaln=_gemm(num_slots, config.time_embed_dim, 2 * hidden))
    block = block_flops(config, layout.seq_len, layout.used, keep_ratio,
                        adaln_tables, num_slots)
    return total + block * config.num_layers


def trajectory_flops(config: h3_config.H3Config,
                     layout: h3_layout.PackedLayout, num_steps: int,
                     keep_ratio: float = 1.0, dense_steps: int = 0,
                     adaln_tables: bool = True,
                     num_slots: int = 2) -> Flops:
    """A whole denoising trajectory, text tower included.

    Args:
        config: The architecture.
        layout: The packed request.
        num_steps: Velocity evaluations (the NFE).
        keep_ratio: Kept fraction on the sparse steps.
        dense_steps: Leading steps that run dense anyway
            (scripts/generate.py --dense-steps).
        adaln_tables: See `block_flops`.
        num_slots: See `block_flops`.

    Raises:
        ValueError: When `dense_steps` exceeds `num_steps`.
    """
    if dense_steps > num_steps:
        raise ValueError(f'dense_steps {dense_steps} > num_steps {num_steps}')
    sparse = step_flops(config, layout, keep_ratio, adaln_tables, num_slots)
    dense = step_flops(config, layout, 1.0, adaln_tables, num_slots)
    return (refiner_flops(config, layout.text_len)
            + dense * dense_steps
            + sparse * (num_steps - dense_steps))


def mfu(flops: int, seconds: float, peak_flops: float) -> float:
    """Achieved fraction of the device's dense bf16 peak.

    Args:
        flops: Arithmetic of the timed region.
        seconds: Wall time of that region.
        peak_flops: Device peak, e.g. PEAK_BF16_FLOPS[torch.cuda
            .get_device_name()].

    Raises:
        ValueError: On a non-positive time or peak.
    """
    if seconds <= 0.0:
        raise ValueError(f'seconds must be positive, got {seconds}')
    if peak_flops <= 0.0:
        raise ValueError(f'peak_flops must be positive, got {peak_flops}')
    return flops / seconds / peak_flops


def device_peak(name: str) -> float:
    """Dense bf16 peak FLOP/s of `torch.cuda.get_device_name()`.

    Raises:
        KeyError: For a device we have not recorded a peak for; add it to
            PEAK_BF16_FLOPS from the vendor's datasheet rather than
            guessing, since every MFU we publish is relative to it.
    """
    if name not in PEAK_BF16_FLOPS:
        raise KeyError(f'no bf16 peak recorded for {name!r}; add it to '
                       'miowtion.h3.flops.PEAK_BF16_FLOPS')
    return PEAK_BF16_FLOPS[name]
