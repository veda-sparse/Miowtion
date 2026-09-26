"""Architecture constants and configuration of the MiniMax-H3 DiT."""

from __future__ import annotations

import dataclasses
import json
import os

# Rows of the packed sequence carry one of these modality tags. Padding rows
# are tagged -1 by the packer; the DiT clamps them to 0 before indexing the
# AdaLN table, so padding rows are modulated as video rows.
TAG_VIDEO = 0
TAG_TEXT = 1
TAG_AUDIO = 2
TAG_PAD = -1

# One AdaLN modulation row per (timestep slot, modality).
MODALITY_NUM = 3

# The packed sequence length is rounded up to this multiple.
PACKED_SEQUENCE_ALIGNMENT = 64

# Checkpoint tensors stored in fp32; every other tensor is bf16.
FP32_PARAM_PREFIXES = (
    'video_patch_proj.',
    'audio_patch_proj.',
    'time_embedder.',
    'final_layer.video_out.',
    'final_layer.audio_out.',
)
FP32_BUFFER_NAMES = ('rope.inv_freq',)


# config.json key of every H3Config field, most recent spelling first. Two
# releases are in circulation: the first one ("MiniMaxH3DiTModel",
# diffusers 0.32) and the current diffusers port
# ("MiniMaxH3Transformer3DModel", diffusers 0.36) which renamed most keys.
# Both describe the same 33B architecture.
CONFIG_KEYS = {
    'hidden_size': ('hidden_size',),
    'num_layers': ('num_layers',),
    'num_refiner_layers': ('num_refiner_layers',
                           'token_refiner_num_layers'),
    'num_heads': ('num_attention_heads',),
    'head_dim': ('attention_head_dim',),
    'ffn_dim': ('ffn_dim', 'ffn_hidden_size'),
    'video_channels': ('in_channels', 'latents_dim'),
    'audio_channels': ('audio_in_channels', 'audio_latents_dim'),
    'patch_size': ('patch_size',),
    'text_dim': ('text_dim',),
    'freq_dim': ('freq_dim', 'timestep_input_dim'),
    'time_embed_hidden': ('time_embed_hidden_dim', 'time_embed_hidden_size'),
    'time_embed_dim': ('time_embed_dim',),
    'rope_freqs_per_axis': ('rope_freq_dim', 'rope_inv_freq_len'),
    'norm_eps': ('norm_eps',),
    'qk_norm_eps': ('qk_norm_eps',),
    'final_norm_eps': ('final_norm_eps',),
}
# The first release does not store rope_theta; both use 10000.0.
OPTIONAL_CONFIG_KEYS = {'rope_theta': ('rope_theta',)}


@dataclasses.dataclass(frozen=True)
class H3Config:
    """Shape hyper-parameters of the H3 DiT (defaults: the released 33B)."""

    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    num_heads: int = 56
    head_dim: int = 128
    ffn_dim: int = 14336
    video_channels: int = 24
    audio_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden: int = 5376
    time_embed_dim: int = 2688
    rope_freqs_per_axis: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.video_channels * pt * ph * pw

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def rope_dim(self) -> int:
        """Rotated channels per head: (t, h, w) frequencies, cos and sin."""
        return 2 * 3 * self.rope_freqs_per_axis

    @classmethod
    def from_pretrained(cls, transformer_dir: str) -> H3Config:
        """Reads `<checkpoint>/<variant>/transformer/config.json`.

        Two releases are in circulation and they spell the same fields
        differently (see CONFIG_KEYS); every field must be present under one
        of its spellings, so a third schema fails loudly instead of silently
        falling back to the defaults of this class.

        Raises:
            KeyError: When a field is missing under every known spelling.
            ValueError: When the AdaLN width contradicts hidden_size.
        """
        with open(os.path.join(transformer_dir, 'config.json')) as f:
            raw = json.load(f)
        kwargs = {}
        for field, keys in CONFIG_KEYS.items():
            present = [k for k in keys if k in raw]
            if not present:
                raise KeyError(f'{transformer_dir}/config.json has no key '
                               f'for {field} (tried {list(keys)})')
            kwargs[field] = raw[present[0]]
        for field, keys in OPTIONAL_CONFIG_KEYS.items():
            for key in keys:
                if key in raw:
                    kwargs[field] = raw[key]
                    break
        kwargs['patch_size'] = tuple(kwargs['patch_size'])
        config = cls(**kwargs)
        expected_adaln = 6 * MODALITY_NUM * config.hidden_size
        if raw.get('adaln_out_features', expected_adaln) != expected_adaln:
            raise ValueError(
                f'adaln_out_features {raw["adaln_out_features"]} != '
                f'{expected_adaln}')
        return config

    @classmethod
    def tiny(cls, num_layers: int = 2, num_heads: int = 4) -> H3Config:
        """A small config with the real structure, for CPU tests."""
        return cls(
            hidden_size=64,
            num_layers=num_layers,
            num_refiner_layers=1,
            num_heads=num_heads,
            head_dim=32,
            ffn_dim=96,
            text_dim=48,
            freq_dim=32,
            time_embed_hidden=64,
            time_embed_dim=32,
            rope_freqs_per_axis=4,
        )
