"""Teacher parameter residency never changes predictions or train policy."""

import pytest
import torch
from torch import nn

from miowtion.train import teacher_weight_cache


def test_plain_frozen_forward_matches_and_grad_mode_is_guarded():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(3, 3)])
    x = torch.randn(4, 3)
    with pytest.raises(RuntimeError, match='no_grad'):
        with teacher_weight_cache.retain_teacher_weights(model):
            pass
    with torch.no_grad():
        expected = model.blocks[0](x)
        with teacher_weight_cache.retain_teacher_weights(model):
            actual = model.blocks[0](x)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('raise_during_forward', [False, True])
def test_train_policy_and_shards_restored_on_normal_or_exception_exit(
        monkeypatch, raise_during_forward):
    class ShardedBlock(nn.Linear):
        def __init__(self):
            super().__init__(3, 3)
            self.reshard_after_forward = True
            self.retained = False

        def set_reshard_after_forward(self, value, recurse):
            assert not recurse
            self.reshard_after_forward = value

        def forward(self, x):
            self.retained = not self.reshard_after_forward
            return super().forward(x)

        def reshard(self):
            self.retained = False

    monkeypatch.setattr(teacher_weight_cache, 'FSDPModule', ShardedBlock)
    model = nn.Module()
    model.blocks = nn.ModuleList([ShardedBlock(), ShardedBlock()])
    x = torch.randn(4, 3)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        try:
            with teacher_weight_cache.retain_teacher_weights(model):
                for block in model.blocks:
                    x = block(x)
                    assert block.retained
                if raise_during_forward:
                    raise ValueError('teacher forward failed')
        except ValueError:
            assert raise_during_forward
    assert all(block.reshard_after_forward and not block.retained
               for block in model.blocks)
    assert all(torch.equal(p, original[n]) for n, p in model.named_parameters())
