"""Route Wan's self-attention through Veda, by swapping its processor.

Step two of P0-3. Wan is a stock diffusers DiT, so this needs no fork:
`WanAttnProcessor.__call__` applies RoPE and then calls
`dispatch_attention_fn(query, key, value, ...)` on `[B, S, H, D]`
tensors, which is exactly the shape our `AttentionFn` protocol takes
once the batch dimension is dropped. Subclassing and overriding that one
call keeps the projections, the RoPE, the image cross-attention branch
and the output projection untouched.

Only *self*-attention is routed. Wan's cross-attention to the text
encoder has a different key length and no tile structure, so it stays
dense; the processor tells the two apart by whether
`encoder_hidden_states` was passed.
"""

from __future__ import annotations

from typing import Protocol

import torch


class AttentionFn(Protocol):
    """q, k, v [S, H, D] -> out [S, H, D]; the same contract as h3."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 layer_index: int) -> torch.Tensor:
        ...


def make_processor(attention: AttentionFn, layer_index: int):
    """A Wan attention processor whose self-attention calls `attention`.

    Args:
        attention: The sparse (or dense) attention to route through.
        layer_index: Which DiT block this processor belongs to; passed
            through so a per-layer plan can select its tile shape.

    Returns:
        A processor instance to assign to one `WanAttention` module.

    Raises:
        RuntimeError: If diffusers does not expose WanAttnProcessor,
            which would mean the attribute this subclasses moved.
    """
    try:
        from diffusers.models.transformers.transformer_wan import (  # noqa: PLC0415
            WanAttnProcessor)
    except ImportError as error:                           # pragma: no cover
        raise RuntimeError(
            'diffusers no longer exposes WanAttnProcessor at '
            'models.transformers.transformer_wan; the processor swap has '
            'to be re-pointed') from error

    class _VedaWanProcessor(WanAttnProcessor):
        """Self-attention through `attention`, everything else untouched."""

        def __init__(self) -> None:
            super().__init__()
            self._attention = attention
            self._layer_index = layer_index

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None, **kwargs):
            if encoder_hidden_states is not None:
                # Cross-attention to the text encoder: different key
                # length, no tiling, so leave it dense.
                return super().__call__(
                    attn, hidden_states, encoder_hidden_states,
                    attention_mask, rotary_emb, **kwargs)
            return self._self_attention(attn, hidden_states, rotary_emb)

        def _self_attention(self, attn, hidden_states, rotary_emb):
            # Upstream normalises on the flat [B, S, H*D] projection and
            # only then splits the heads; doing it the other way round
            # changes the RMS denominator from H*D to D.
            from diffusers.models.transformers.transformer_wan import (  # noqa: PLC0415
                _get_qkv_projections)
            query, key, value = _get_qkv_projections(attn, hidden_states,
                                                     None)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            if rotary_emb is not None:
                query, key = _apply_rope(query, key, rotary_emb)
            if query.shape[0] != 1:
                raise ValueError(
                    f'batch {query.shape[0]}: the tile layout describes one '
                    'clip, so the batch has to be split before this point')
            out = self._attention(query[0], key[0], value[0],
                                  self._layer_index)
            return attn.to_out[1](attn.to_out[0](
                out[None].flatten(2, 3).type_as(query)))

    return _VedaWanProcessor()


def _apply_rope(query: torch.Tensor, key: torch.Tensor,
                rotary_emb) -> tuple[torch.Tensor, torch.Tensor]:
    """Wan's rotary embedding, transcribed from its own processor.

    The interleaving matters and is easy to get wrong: the frequencies
    arrive at full width and the cosine takes the even lanes while the
    sine takes the odd ones, so multiplying the whole tensor by `cos`
    and a rotated copy by `sin` -- the usual formulation -- is a
    different function. tests/unit/test_wan_attention.py pins this
    against the upstream processor, which is the only reason the error
    surfaced instead of quietly skewing every layer.
    """
    freqs_cos, freqs_sin = rotary_emb

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(x)

    return rotate(query), rotate(key)
