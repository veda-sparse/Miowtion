"""Tests for miowtion.train.muon."""

import math

import pytest
import torch

from miowtion.train import muon


def _singular_values(x):
    return torch.linalg.svdvals(x.float())


def test_orthogonalize_drives_singular_values_to_one():
    torch.manual_seed(0)
    # Badly conditioned on purpose: this is what the optimizer must fix.
    x = torch.randn(384, 128) @ torch.diag(
        torch.logspace(0, -3, 128))
    before = _singular_values(x)
    after = _singular_values(muon.orthogonalize(x))
    assert before.max() / before.min() > 100
    assert after.max() / after.min() < 2.0
    assert 0.6 < after.min() and after.max() < 1.3


def test_five_newton_schulz_steps_would_not_be_enough():
    """Why the default is 8: the reference's 5 leave a badly conditioned
    matrix under-corrected, and the iteration converges by 8."""
    torch.manual_seed(0)
    x = torch.randn(384, 128) @ torch.diag(torch.logspace(0, -3, 128))
    five = _singular_values(muon.orthogonalize(x, steps=5))
    eight = _singular_values(muon.orthogonalize(x, steps=8))
    twelve = _singular_values(muon.orthogonalize(x, steps=12))
    assert five.min() < 0.2
    assert eight.min() > 0.6
    # By 8 the spread has stopped moving: four more steps change the
    # extremes by under 1e-3, so the extra iterations buy nothing.
    assert abs(eight.min() - twelve.min()) < 1e-3
    assert abs(eight.max() - twelve.max()) < 1e-3


def test_orthogonalize_is_per_head_not_over_the_stack():
    torch.manual_seed(0)
    heads = torch.randn(4, 384, 128)
    batched = muon.orthogonalize(heads)
    for h in range(heads.shape[0]):
        alone = muon.orthogonalize(heads[h])
        assert torch.equal(batched[h], alone), h


def test_orthogonalize_handles_wide_and_tall_alike():
    torch.manual_seed(0)
    tall = torch.randn(384, 128)
    wide = tall.T.contiguous()
    assert torch.allclose(muon.orthogonalize(tall).T,
                          muon.orthogonalize(wide), atol=1e-5)


def test_orthogonalize_survives_a_zero_gradient():
    out = muon.orthogonalize(torch.zeros(8, 16))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('shape', [(56, 384, 128), (4, 128, 384), (64, 64)])
def test_update_rms_matches_the_target(shape):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(*shape) * 1e-4)
    opt = muon.Muon([p], lr=1.0, rms_target=0.2, momentum=0.0,
                    nesterov=False)
    before = p.detach().clone()
    p.grad = torch.randn(*shape)
    opt.step()
    step_rms = (p.detach() - before).pow(2).mean().sqrt().item()
    # lr = 1, so the parameter step is the scaled update itself.
    assert step_rms == pytest.approx(0.2, rel=0.15)
    assert opt.last_update_rms() == pytest.approx(0.2, rel=0.15)


def test_lr_scales_the_step_linearly():
    steps = {}
    for lr in (1e-3, 5e-3):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(2, 32, 16))
        opt = muon.Muon([p], lr=lr, rms_target=0.2, momentum=0.0,
                        nesterov=False)
        p.grad = torch.randn(2, 32, 16)
        opt.step()
        steps[lr] = p.detach().pow(2).mean().sqrt().item()
    assert steps[5e-3] == pytest.approx(5 * steps[1e-3], rel=1e-4)
    # A full-rank gradient hits the target: the step is lr * rms_target.
    assert steps[5e-3] == pytest.approx(5e-3 * 0.2, rel=0.15)


def test_a_rank_deficient_gradient_falls_short_of_the_target():
    """Muon cannot invent directions the gradient does not contain."""
    p = torch.nn.Parameter(torch.zeros(2, 32, 16))
    opt = muon.Muon([p], lr=1.0, rms_target=0.2, momentum=0.0,
                    nesterov=False)
    p.grad = torch.full((2, 32, 16), 0.1)   # rank 1
    opt.step()
    rms = p.detach().pow(2).mean().sqrt().item()
    # sqrt(rank / min(m, n)) = sqrt(1 / 16) of the target, roughly.
    assert 0.01 < rms < 0.1
    assert opt.last_update_rms() == pytest.approx(rms, rel=1e-5)


def test_step_size_ignores_the_gradient_magnitude():
    """The point of orthogonalization: only the direction survives."""
    sizes = []
    for gain in (1e-6, 1.0, 1e6):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(2, 32, 16))
        opt = muon.Muon([p], lr=1e-2, rms_target=0.2, momentum=0.0,
                        nesterov=False)
        p.grad = torch.randn(2, 32, 16) * gain
        opt.step()
        sizes.append(p.detach().pow(2).mean().sqrt().item())
    assert max(sizes) / min(sizes) < 1.01


def test_one_dimensional_parameters_are_refused():
    with pytest.raises(ValueError, match='matrix parameters'):
        muon.Muon([torch.nn.Parameter(torch.zeros(8))])


def test_bad_hyperparameters_are_refused():
    p = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match='lr must be positive'):
        muon.Muon([p], lr=0.0)
    with pytest.raises(ValueError, match='rms_target must be positive'):
        muon.Muon([p], rms_target=-1.0)


def test_update_scale_formula():
    assert muon.update_scale(torch.Size([56, 384, 128]), 0.2) == (
        pytest.approx(0.2 * math.sqrt(384)))
    assert muon.update_scale(torch.Size([128, 384]), 0.2) == (
        pytest.approx(0.2 * math.sqrt(384)))


def test_alignment_reads_1_without_momentum_and_falls_with_it():
    """The diagnostic for a step spread across stale directions."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(2, 32, 16))
    plain = muon.Muon([p], lr=1e-3, momentum=0.0, nesterov=False)
    p.grad = torch.randn(2, 32, 16)
    plain.step()
    # No momentum: the step is the orthogonalized current gradient, which
    # keeps its dominant direction, so the cosine is solidly positive.
    assert plain.last_alignment() > 0.3
    # With momentum, a gradient orthogonal to the buffer's contents gets
    # only part of the step.
    q = torch.nn.Parameter(torch.zeros(2, 32, 16))
    heavy = muon.Muon([q], lr=1e-3, momentum=0.95, nesterov=False)
    first = torch.randn(2, 32, 16)
    q.grad = first
    heavy.step()
    q.grad = torch.randn(2, 32, 16)      # a fresh, unrelated batch
    heavy.step()
    assert heavy.last_alignment() < plain.last_alignment()
