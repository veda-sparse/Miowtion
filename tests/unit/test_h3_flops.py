"""Tests for miowtion.h3.flops.

The important one is `test_step_matches_a_traced_forward`: the analytic
count is only useful if it equals what the model actually issues, so it is
compared against torch's own operator-level counter on a tiny model instead
of against a second copy of the same formula.
"""

import torch
from torch.utils import flop_counter

from miowtion.h3 import config as h3_config
from miowtion.h3 import flops as h3_flops
from miowtion.h3 import geometry
from miowtion.h3 import layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import noise
from miowtion.h3 import schedule

_TEXT_LEN = 12


def _clip(cfg):
    g = geometry.geometry_from_latent_t('16:9', 7, short_edge=64)
    return g, layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), g)


def _random_model(cfg, seed=0):
    torch.manual_seed(seed)
    m = h3_model.H3DiT(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(std=0.05)
    m.dense_backend = 'math'
    return m


def test_attention_is_quadratic_in_the_real_rows():
    inner = 7168
    assert h3_flops.attention_flops(1000, inner) == 4 * 1000 ** 2 * inner
    # Padding rows never attend, so only `used` enters.
    assert (h3_flops.attention_flops(2000, inner)
            == 4 * h3_flops.attention_flops(1000, inner))


def test_keep_ratio_scales_only_attention():
    cfg = h3_config.H3Config()
    dense = h3_flops.block_flops(cfg, 4096, 4000)
    sparse = h3_flops.block_flops(cfg, 4096, 4000, keep_ratio=0.1)
    assert sparse.attention == dense.attention // 10
    assert sparse.linear == dense.linear
    assert sparse.total < dense.total


def test_attention_share_grows_with_the_clip():
    """The whole point of Veda: attention dominates the long geometries."""
    cfg = h3_config.H3Config()
    shares = []
    for latent_t in (37, 72, 102):
        g = geometry.geometry_from_latent_t('16:9', latent_t)
        lay = layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), g)
        step = h3_flops.step_flops(cfg, lay)
        shares.append(step.attention / step.total)
    assert shares == sorted(shares)
    assert shares[0] > 0.3 and shares[-1] > 0.6


def test_trajectory_counts_the_text_tower_once():
    cfg = h3_config.H3Config()
    g, lay = _clip(cfg)
    steps = 8
    traj = h3_flops.trajectory_flops(cfg, lay, steps)
    expected = (h3_flops.refiner_flops(cfg, lay.text_len)
                + h3_flops.step_flops(cfg, lay) * steps)
    assert traj.as_dict() == expected.as_dict()


def test_dense_steps_are_priced_dense():
    cfg = h3_config.H3Config()
    g, lay = _clip(cfg)
    mixed = h3_flops.trajectory_flops(cfg, lay, 8, keep_ratio=0.1,
                                      dense_steps=2)
    all_dense = h3_flops.trajectory_flops(cfg, lay, 8)
    all_sparse = h3_flops.trajectory_flops(cfg, lay, 8, keep_ratio=0.1)
    assert all_sparse.total < mixed.total < all_dense.total


def test_mfu_is_the_fraction_of_peak():
    peak = h3_flops.PEAK_BF16_FLOPS['NVIDIA GeForce RTX 4090']
    assert h3_flops.mfu(int(peak), 1.0, peak) == 1.0
    assert abs(h3_flops.mfu(int(peak / 2), 1.0, peak) - 0.5) < 1e-6


def test_step_matches_a_traced_forward():
    """Analytic count == what torch sees the tiny model issue.

    The counter sees mm/bmm only, which is exactly the set flops.py claims
    to count, so the two must agree to the last FLOP. Attention runs on the
    'math' backend so its two einsums show up as bmm; the AdaLN projections
    are counted because this forward projects them per block.
    """
    cfg = h3_config.H3Config.tiny()
    m = _random_model(cfg)
    g, lay = _clip(cfg)
    video, audio = noise.initial_noise(g, 0)
    state = schedule.build_timestep_state(lay, 0.3, 0.1)
    text = torch.randn(_TEXT_LEN, cfg.text_dim)
    with torch.no_grad():
        refined = m.refine_text(text)
        clip = m.clip_inputs(lay, refined, torch.device('cpu'))
        counter = flop_counter.FlopCounterMode(display=False)
        with counter:
            m(clip, video, audio, state)
        traced = counter.get_total_flops()
    num_slots = int(state.timesteps.numel())
    predicted = h3_flops.step_flops(cfg, lay, adaln_tables=False,
                                    num_slots=num_slots)
    assert predicted.total == traced


def test_refiner_matches_a_traced_forward():
    cfg = h3_config.H3Config.tiny()
    m = _random_model(cfg)
    text = torch.randn(_TEXT_LEN, cfg.text_dim)
    with torch.no_grad():
        counter = flop_counter.FlopCounterMode(display=False)
        with counter:
            m.refine_text(text)
        traced = counter.get_total_flops()
    assert h3_flops.refiner_flops(cfg, _TEXT_LEN).total == traced


def test_device_peak_refuses_an_unknown_device():
    try:
        h3_flops.device_peak('NVIDIA Imaginary 5090')
    except KeyError:
        return
    raise AssertionError('an unknown device must not get a silent default')
