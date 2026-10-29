"""Retain frozen FSDP weights during consecutive teacher forwards only."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule


@contextlib.contextmanager
def retain_teacher_weights(model: nn.Module) -> Iterator[None]:
    """Avoid gathering the same frozen block weights for every rollout step.

    This context changes parameter residency, never the teacher's states or
    predictions. It must run inside no_grad and end before student forward.
    Blocks must use the training path's reshard_after_forward=True policy
    and GPU-resident local shards. The policy and sharded parameters are
    restored even when a teacher forward raises. Non-FSDP models are a no-op.

    Args:
        model: H3 model whose blocks were sharded by parallel.shard_trunk.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('teacher weight retention requires no_grad')
    units = [module for block in model.blocks for module in block.modules()
             if isinstance(module, FSDPModule)]
    enabled = []
    try:
        for module in units:
            module.set_reshard_after_forward(False, recurse=False)
            enabled.append(module)
        yield
    finally:
        for module in reversed(enabled):
            module.set_reshard_after_forward(True, recurse=False)
            module.reshard()
