"""MiniMax-H3 DiT (FL2VA and Ref2VA share this architecture).

Module and parameter names equal the keys of the released checkpoints
(`<root>/<FL2VA|Ref2VA>/transformer`), so loading is a 1:1 mapping except
for the fused QKV row order (see weights.py).

Numerics follow a fixed eager op chain: every elementwise op runs in the
activation dtype (bf16), RMSNorm accumulates in fp32, and the mixed-precision
islands of the checkpoint (patch projections, time embedder, output heads in
fp32) are preserved. Changing an op order here changes the teacher.

Attention is pluggable: `forward` receives an `AttentionFn` that is called
once per trunk layer with post-norm, post-RoPE q/k/v. The dense teacher, the
Veda student and the tile-search scorer are all AttentionFns.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from miowtion.h3 import attention as h3_attention
from miowtion.h3 import config as h3_config
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import schedule as h3_schedule

_BF16 = torch.bfloat16
_FP32 = torch.float32


class AttentionFn(Protocol):
    """Trunk attention: q, k, v [S, H, D] bf16 -> out [S, H, D] bf16."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        ...


class DenseAttention:
    """The dense teacher attention."""

    def __init__(self, used: int, backend: str = 'auto'):
        self.used = used
        self.backend = backend

    def __call__(self, q, k, v, layer_index):
        del layer_index
        return h3_attention.dense_attention(q, k, v, self.used,
                                            backend=self.backend)[0]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """Rotates the leading rope_dim channels of every head.

    Args:
        x: [S, H, D] bf16.
        cos: [S, 1, rope_dim] bf16.
        sin: [S, 1, rope_dim] bf16.
    """
    rope_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), -1)


class Rope(nn.Module):
    """3-axis RoPE over fp64 (t, h, w) coordinates; rotates 96 of 128 dims."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        n = config.rope_freqs_per_axis
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, 2 * n, 2, dtype=_FP32) / (2 * n)))
        self.register_buffer('inv_freq', inv_freq, persistent=True)

    def cos_sin(self, position_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """[S, 3] fp64 positions -> (cos, sin), each [S, 1, rope_dim] bf16.

        The fp64 grid is cast to fp32 before the frequency product.
        """
        pos = position_ids.to(self.inv_freq.device).to(_FP32)
        per_axis = pos[:, :, None] * self.inv_freq[None, None, :]
        half = torch.cat(per_axis.unbind(dim=1), dim=-1)  # [S, 3n]
        cos_half = torch.cos(half).to(_BF16)
        sin_half = torch.sin(half).to(_BF16)
        cos = torch.cat((cos_half, cos_half), dim=-1)[:, None, :]
        sin = torch.cat((sin_half, sin_half), dim=-1)[:, None, :]
        return cos, sin


class TimeEmbedder(nn.Module):
    """Sinusoidal (cos first, then sin) embedding + fp32 MLP."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.freq_dim = config.freq_dim
        self.proj_in = nn.Linear(config.freq_dim, config.time_embed_hidden,
                                 dtype=_FP32)
        self.proj_out = nn.Linear(config.time_embed_hidden,
                                  config.time_embed_dim, dtype=_FP32)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] in [0, 1] -> [M, time_embed_dim] fp32."""
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(
            half, dtype=_FP32, device=t.device) / half)
        args = t.to(_FP32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(t_freq)))


class Attention(nn.Module):
    """Fused QKV -> per-head RMSNorm on q, k -> RoPE -> attention -> out."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        inner = config.inner_dim
        # Rows are [q_all; k_all; v_all] (the checkpoint interleaves per head
        # and is reordered at load time).
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * inner, bias=False,
                                  dtype=_BF16)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.qk_norm_eps,
                                 dtype=_BF16)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.qk_norm_eps,
                                 dtype=_BF16)
        self.out_proj = nn.Linear(inner, config.hidden_size, bias=False,
                                  dtype=_BF16)

    def qkv(self, x: torch.Tensor,
            rope: tuple[torch.Tensor, torch.Tensor] | None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.num_heads * self.head_dim, -1)
        q = self.q_norm(q.view(seq, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(seq, self.num_heads, self.head_dim))
        v = v.view(seq, self.num_heads, self.head_dim)
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        return q, k, v

    def project_out(self, out: torch.Tensor) -> torch.Tensor:
        return self.out_proj(out.reshape(out.shape[0], -1))


class Mlp(nn.Module):
    """SwiGLU; fc1 outputs [gate; up] in one GEMM.

    `chunk_rows` bounds the [rows, 2 * ffn_dim] intermediate (5.7 GB for a
    100k-row sequence) by processing rows in chunks. GEMM results may depend
    on the row count, so one chunk size must be used consistently by search,
    training and evaluation.
    """

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_dim,
                             bias=False, dtype=_BF16)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size, bias=False,
                             dtype=_BF16)
        self.chunk_rows: int | None = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_rows is None or x.shape[0] <= self.chunk_rows:
            return self._forward(x)
        return torch.cat([self._forward(chunk)
                          for chunk in x.split(self.chunk_rows)])


class AdalnProj(nn.Module):
    """SiLU(t_emb) -> per-(timestep, modality) modulation vectors."""

    def __init__(self, config: h3_config.H3Config, expand: int,
                 modalities: int):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = config.hidden_size
        self.linear = nn.Linear(config.time_embed_dim,
                                expand * modalities * config.hidden_size,
                                dtype=_BF16)

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """[M, t_dim] bf16 -> `expand` tables of [M * modalities, hidden]."""
        m = adaln_input.shape[0]
        x = self.linear(adaln_input).view(m * self.modalities,
                                          self.expand * self.hidden)
        return tuple(x.chunk(self.expand, dim=-1))


class RefinerBlock(nn.Module):
    """Pre-norm text block: no AdaLN, no RoPE, full attention over text."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps,
                                dtype=_BF16)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps,
                                dtype=_BF16)
        self.attn = Attention(config)
        self.mlp = Mlp(config)

    def forward(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        q, k, v = self.attn.qkv(self.norm1(x), rope=None)
        out = h3_attention.dense_attention(q, k, v, x.shape[0],
                                           backend=backend)[0]
        x = x + self.attn.project_out(out)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.blocks = nn.ModuleList(
            RefinerBlock(config) for _ in range(config.num_refiner_layers))
        self.final_norm = nn.RMSNorm(config.hidden_size,
                                     eps=config.final_norm_eps, dtype=_BF16)

    def forward(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, backend)
        return self.final_norm(x)


def _modulate(x: torch.Tensor, one_plus_scale: torch.Tensor,
              shift: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x * one_plus_scale.index_select(0, index) + shift.index_select(
        0, index)


class Block(nn.Module):
    """AdaLN-modulated pre-norm block over the whole packed sequence."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps,
                                dtype=_BF16)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps,
                                dtype=_BF16)
        self.attn = Attention(config)
        self.mlp = Mlp(config)
        self.adaln_proj = AdalnProj(config, expand=6,
                                    modalities=h3_config.MODALITY_NUM)

    def forward(self, x: torch.Tensor, adaln: Sequence[torch.Tensor] | None,
                adaln_input: torch.Tensor, adaln_index: torch.Tensor,
                rope: tuple[torch.Tensor, torch.Tensor],
                attention_fn: AttentionFn, layer_index: int) -> torch.Tensor:
        if adaln is None:
            if self.adaln_proj is None:
                raise ValueError('adaln_proj was dropped; pass adaln tables')
            adaln = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln
        h = _modulate(self.norm1(x), 1.0 + scale_msa, shift_msa, adaln_index)
        q, k, v = self.attn.qkv(h, rope)
        h = self.attn.project_out(attention_fn(q, k, v, layer_index))
        x = x + gate_msa.index_select(0, adaln_index) * h
        h = _modulate(self.norm2(x), 1.0 + scale_mlp, shift_mlp, adaln_index)
        return x + gate_mlp.index_select(0, adaln_index) * self.mlp(h)


class FinalLayer(nn.Module):

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps,
                               dtype=_BF16)
        self.adaln_proj = AdalnProj(config, expand=2, modalities=1)
        self.video_out = nn.Linear(config.hidden_size, config.video_patch_dim,
                                   dtype=_FP32)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_channels,
                                   dtype=_FP32)

    def forward(self, x: torch.Tensor, adaln_input: torch.Tensor,
                slot: torch.Tensor, video_rows: torch.Tensor,
                audio_rows: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = _modulate(self.norm(x), 1.0 + scale, shift, slot)
        # The output heads are fp32 islands; only the rows that are denoised
        # go through them.
        video = self.video_out(h.index_select(0, video_rows).to(_FP32))
        audio = self.audio_out(h.index_select(0, audio_rows).to(_FP32))
        return video, audio


@dataclasses.dataclass
class ClipInputs:
    """Device tensors that stay fixed over one trajectory.

    Attributes:
        text: [text_len, hidden] bf16 refined text (see H3DiT.refine_text).
        img_pos: [Nv] int64 packed rows of video-latent rows.
        audio_pos: [Na] int64 packed rows of audio rows.
        target_img_pos: [Nt] int64 packed rows whose velocity is predicted.
        target_audio_pos: [Nta] int64.
        rope: (cos, sin) from Rope.cos_sin.
        seq_len: Packed length.
        used: Real rows.
    """

    text: torch.Tensor
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    target_img_pos: torch.Tensor
    target_audio_pos: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    seq_len: int
    used: int


class H3DiT(nn.Module):
    """The 50-layer trunk plus its embedding and output layers."""

    def __init__(self, config: h3_config.H3Config):
        super().__init__()
        self.config = config
        self.video_patch_proj = nn.Linear(config.video_patch_dim,
                                          config.hidden_size, dtype=_FP32)
        self.audio_patch_proj = nn.Linear(config.audio_channels,
                                          config.hidden_size, dtype=_FP32)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size,
                                        dtype=_BF16)
        self.time_embedder = TimeEmbedder(config)
        self.rope = Rope(config)
        self.token_refiner = TokenRefiner(config)
        self.blocks = nn.ModuleList(Block(config)
                                    for _ in range(config.num_layers))
        self.final_layer = FinalLayer(config)
        self.gradient_checkpointing = False
        self.dense_backend = 'auto'

    def set_mlp_chunk_rows(self, rows: int | None) -> None:
        for block in self.blocks:
            block.mlp.chunk_rows = rows

    def drop_adaln_projections(self) -> None:
        """Removes the per-block AdaLN projections (13B parameters).

        Only valid with a frozen trunk and precomputed tables (see
        miowtion.train.adaln); forward then requires `adaln_table`.
        """
        for block in self.blocks:
            block.adaln_proj = None

    @torch.no_grad()
    def refine_text(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """[L, text_dim] encoder states -> [L, hidden] bf16 (once per clip)."""
        x = self.condition_proj(text_hidden.to(_BF16))
        return self.token_refiner(x, self.dense_backend)

    def clip_inputs(self, layout: h3_layout.PackedLayout,
                    refined_text: torch.Tensor,
                    device: torch.device) -> ClipInputs:
        return ClipInputs(
            text=refined_text.to(device),
            img_pos=layout.img_pos.to(device),
            audio_pos=layout.audio_pos.to(device),
            target_img_pos=layout.target_img_pos.to(device),
            target_audio_pos=layout.target_audio_pos.to(device),
            rope=self.rope.cos_sin(layout.position_ids),
            seq_len=layout.seq_len,
            used=layout.used,
        )

    def adaln_input(self, timesteps: torch.Tensor) -> torch.Tensor:
        """[M] timesteps -> SiLU(t_emb) [M, t_dim] bf16."""
        return F.silu(self.time_embedder(timesteps)).to(_BF16)

    def embed(self, clip: ClipInputs, video_rows: torch.Tensor,
              audio_rows: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(clip.seq_len, self.config.hidden_size, dtype=_BF16,
                        device=clip.text.device)
        x[:clip.text.shape[0]] = clip.text
        x.index_copy_(0, clip.img_pos,
                      self.video_patch_proj(video_rows.to(_FP32)).to(_BF16))
        x.index_copy_(0, clip.audio_pos,
                      self.audio_patch_proj(audio_rows.to(_FP32)).to(_BF16))
        return x

    def forward(self, clip: ClipInputs, video_rows: torch.Tensor,
                audio_rows: torch.Tensor,
                timestep: h3_schedule.TimestepState,
                attention_fn: AttentionFn | None = None,
                adaln_table: Sequence[Sequence[torch.Tensor]] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """One velocity evaluation.

        Args:
            clip: Trajectory-static inputs.
            video_rows: [Nv, 96] fp32 video-latent rows in img_pos order
                (conditions and target).
            audio_rows: [Na, 32] fp32 audio rows in audio_pos order.
            timestep: Per-row timestep state (on the model device).
            attention_fn: Trunk attention; dense when None.
            adaln_table: Optional precomputed per-block AdaLN tables for
                `timestep.timesteps` (see precompute_adaln).

        Returns:
            video_v: [Nt, 96] fp32 velocity of the target video rows.
            audio_v: [Nta, 32] fp32 velocity of the target audio rows.
        """
        if attention_fn is None:
            attention_fn = DenseAttention(clip.used, self.dense_backend)
        x = self.embed(clip, video_rows, audio_rows)
        adaln_input = self.adaln_input(timestep.timesteps)
        for index, block in enumerate(self.blocks):
            adaln = None if adaln_table is None else adaln_table[index]
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = torch_checkpoint.checkpoint(
                    block, x, adaln, adaln_input, timestep.adaln_index,
                    clip.rope, attention_fn, index, use_reentrant=False)
            else:
                x = block(x, adaln, adaln_input, timestep.adaln_index,
                          clip.rope, attention_fn, index)
        return self.final_layer(x, adaln_input, timestep.slot,
                                clip.target_img_pos, clip.target_audio_pos)

    @torch.no_grad()
    def precompute_adaln(self, timesteps: torch.Tensor
                         ) -> list[tuple[torch.Tensor, ...]]:
        """Per-block AdaLN tables for one distinct-timestep set.

        With a frozen trunk these tables depend only on the timesteps, so a
        trajectory's tables can be computed once and the 13B adaln_proj
        parameters need not be gathered on every forward.
        """
        adaln_input = self.adaln_input(timesteps)
        return [block.adaln_proj(adaln_input) for block in self.blocks]
