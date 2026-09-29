"""GPU tests: the fused elementwise kernels are bitwise the eager chains."""

import pytest
import torch
import torch.nn.functional as F

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule as h3_schedule
from miowtion.kernels import elementwise_triton

pytestmark = pytest.mark.gpu


def _bf16(*shape, seed=0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to('cuda',
                                                           torch.bfloat16)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@pytest.mark.parametrize('heads,rope_dim', [(56, 96), (5, 96), (4, 128)])
def test_rope_is_bitwise_eager(heads, rope_dim):
    x = _bf16(333, heads, 128, seed=1, scale=3.0)
    cos, sin = _bf16(333, 1, rope_dim, seed=2), _bf16(333, 1, rope_dim,
                                                      seed=3)
    x_rot = x[..., :rope_dim]
    expected = torch.cat((x_rot * cos + _rotate_half(x_rot) * sin,
                          x[..., rope_dim:]), -1)
    assert torch.equal(elementwise_triton.rope(x, cos, sin), expected)


@pytest.mark.parametrize('cols', [5376, 1000, 2048])
def test_modulate_and_gated_residual_are_bitwise_eager(cols):
    x, h = _bf16(517, cols, seed=1, scale=4.0), _bf16(517, cols, seed=2)
    scale, shift = _bf16(3, cols, seed=3), _bf16(3, cols, seed=4)
    gate = _bf16(3, cols, seed=5)
    index = torch.randint(0, 3, (517,), device='cuda')
    one_plus_scale = 1.0 + scale
    assert torch.equal(
        elementwise_triton.modulate(x, one_plus_scale, shift, index),
        x * one_plus_scale.index_select(0, index)
        + shift.index_select(0, index))
    assert torch.equal(
        elementwise_triton.gated_residual(x, gate, index, h),
        x + gate.index_select(0, index) * h)


def test_swiglu_on_strided_fc1_halves_is_bitwise_eager():
    fc1 = _bf16(300, 2 * 3000, seed=1, scale=4.0)
    gate, up = fc1.chunk(2, dim=-1)
    assert torch.equal(elementwise_triton.swiglu(gate, up),
                       F.silu(gate) * up)


def test_silu_over_every_bf16_value():
    bits = torch.arange(-2**15, 2**15, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16).cuda()[None, :]
    up = torch.ones_like(gate)
    got, expected = elementwise_triton.swiglu(gate, up), F.silu(gate) * up
    finite = ~expected.isnan()
    assert torch.equal(got.isnan(), expected.isnan())
    assert torch.equal(got[finite], expected[finite])


def test_block_forward_is_bitwise_eager(monkeypatch):
    cfg = h3_config.H3Config.tiny(num_layers=2, num_heads=4)
    torch.manual_seed(0)
    model = h3_model.H3DiT(cfg).cuda()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.05)
    model.dense_backend = 'sdpa'
    model.set_mlp_chunk_rows(64)
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = h3_layout.pack(torch.ones(12, dtype=torch.long), geo)
    state = h3_schedule.build_timestep_state(
        lay, *h3_schedule.Schedule.build(9, h3_schedule.ShiftScales(
            12., 3.)).timesteps(2)).to('cuda')
    text = torch.randn(12, cfg.text_dim, device='cuda')
    video = torch.randn(geo.num_video_tokens, 96, device='cuda')
    audio = torch.randn(geo.num_audio_rows, 32, device='cuda')

    def run():
        with torch.no_grad():
            clip = model.clip_inputs(lay, model.refine_text(text),
                                     torch.device('cuda'))
            return model(clip, video, audio, state)

    fused = run()
    monkeypatch.setattr(h3_model, '_fused', lambda *t: False)
    eager = run()
    assert torch.equal(fused[0], eager[0])
    assert torch.equal(fused[1], eager[1])


def test_gradients_keep_the_eager_path():
    x = _bf16(4, 64).requires_grad_()
    assert not h3_model._fused(x)  # pylint: disable=protected-access
    with torch.no_grad():
        assert h3_model._fused(x)  # pylint: disable=protected-access
