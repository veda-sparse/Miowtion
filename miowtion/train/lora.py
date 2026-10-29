"""LoRA on trunk projections, and merging external LoRA adapters.

LoRA parameters are registered on the existing nn.Linear modules
(`...qkv_proj.lora_a`, `...qkv_proj.lora_b`) and applied by a forward hook,
so base parameter names stay equal to the checkpoint keys. They must be
added before FSDP sharding so they shard with their block.

The frozen teacher of stage 2 is the same module with LoRA disabled
(`set_enabled(model, False)`); otherwise the teacher moves with the student
and "sparse output matches dense output" can be satisfied by drifting away
from H3.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import torch
from safetensors import safe_open
from torch import nn
from torch.distributed import tensor as dtensor
from torch.distributed.tensor import _utils as dtensor_utils

DEFAULT_TARGETS = ('attn.qkv_proj', 'attn.out_proj')
# Explicit opt-in; existing Stage-2 and sparse configs keep their defaults.
TRUNK_TARGETS = DEFAULT_TARGETS + ('mlp.fc1', 'mlp.fc2')


def _hook(module: nn.Linear, inputs, output):
    if not module.lora_enabled:
        return output
    x = inputs[0]
    chunk_rows = module.lora_chunk_rows
    if chunk_rows is not None and x.numel() // x.shape[-1] > chunk_rows:
        flat = x.reshape(-1, x.shape[-1])
        updates = []
        for chunk in flat.split(chunk_rows):
            delta = ((chunk.to(module.lora_a.dtype) @ module.lora_a.t())
                     @ module.lora_b.t())
            # Scaling out of place would create another full FP32 delta.
            delta.mul_(module.lora_scale)
            updates.append(delta.to(output.dtype))
        return output + torch.cat(updates).reshape(output.shape)
    delta = (x.to(module.lora_a.dtype) @ module.lora_a.t()) @ module.lora_b.t()
    return output + (delta * module.lora_scale).to(output.dtype)


def add_lora(model: nn.Module, rank: int, alpha: float | None = None,
             targets: Sequence[str] = DEFAULT_TARGETS,
             prefix: str = 'blocks.',
             layers: Sequence[int] | None = None,
             chunk_rows: int | None = None) -> list[str]:
    """Adds LoRA to every Linear named `<prefix>*.<target>`.

    A is Kaiming-initialized, B is zero, so the model is unchanged at init.
    With layers set, every exact layer/target pair must exist before any
    module is modified. None retains the legacy all-blocks behavior.
    chunk_rows bounds FP32 temporary rows; None retains the full GEMMs.

    Returns:
        Names of the wrapped modules.
    """
    if (type(rank) is not int or rank <= 0 or (alpha is not None and
            (not math.isfinite(alpha) or alpha <= 0))):
        raise ValueError('rank and alpha must be positive')
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('targets must be nonempty and unique')
    if chunk_rows is not None and (type(chunk_rows) is not int
                                   or chunk_rows < 1):
        raise ValueError('chunk_rows must be a positive integer or None')
    if layers is not None:
        if (not layers or len(set(layers)) != len(layers)
                or any(type(i) is not int or i < 0 for i in layers)):
            raise ValueError('layers must be unique nonnegative integers')
        expected = {f'{prefix}{i}.{t}' for i in layers for t in targets}
    else:
        expected = None
    candidates = [(name, module) for name, module in model.named_modules()
                  if name.startswith(prefix)
                  and name.endswith(tuple(targets))
                  and isinstance(module, nn.Linear)
                  and (expected is None or name in expected)]
    missing = expected - {n for n, _ in candidates} if expected else set()
    if missing or not candidates:
        raise ValueError(f'LoRA modules missing: {sorted(missing)}; '
                         f'targets={targets}')
    if any(hasattr(m, 'lora_a') for _, m in candidates):
        raise ValueError('LoRA already installed; refusing duplicate hooks')
    wrapped = []
    for name, module in candidates:
        a = torch.empty(rank, module.in_features, dtype=torch.float32,
                        device=module.weight.device)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        module.lora_a = nn.Parameter(a)
        module.lora_b = nn.Parameter(torch.zeros(
            module.out_features, rank, dtype=torch.float32,
            device=module.weight.device))
        module.lora_scale = (rank if alpha is None else alpha) / rank
        module.lora_enabled = True
        module.lora_chunk_rows = chunk_rows
        module.register_forward_hook(_hook)
        wrapped.append(name)
    if not wrapped:
        raise ValueError(f'no module matched {targets}')
    return wrapped


@torch.no_grad()
def reset_parameters(model: nn.Module, seed: int) -> None:
    """A ~ Kaiming uniform, B = 0 (also for FSDP-local shards).

    Needed after `to_empty`, which leaves meta-initialized LoRA parameters
    uninitialized. B = 0 makes the model exactly the base model.
    """
    gen = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if not hasattr(module, 'lora_enabled'):
            continue
        a = module.lora_a
        a_local = a.to_local() if isinstance(a, dtensor.DTensor) else a
        bound = 1.0 / math.sqrt(module.in_features)
        # Draw the global A before slicing. Identically seeded draws of
        # each rank's local shape would repeat A rows across shards and
        # collapse the initial rank. A is small even on the full model.
        values = (torch.rand(a.shape, generator=gen) * 2 - 1) * bound
        if isinstance(a, dtensor.DTensor):
            if any(p.is_shard() and p.dim != 0 for p in a.placements):
                raise ValueError('LoRA initialization supports dim-0 shards')
            shape, offset = dtensor_utils.compute_local_shape_and_global_offset(
                a.shape, a.device_mesh, a.placements)
            values = values[offset[0]:offset[0] + shape[0]]
        a_local.copy_(values)
        b = module.lora_b
        (b.to_local() if isinstance(b, dtensor.DTensor) else b).zero_()


def set_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, 'lora_enabled'):
            module.lora_enabled = enabled


def lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if '.lora_' in n]


Adapter = dict[str, tuple[torch.Tensor, torch.Tensor]]


def load_adapter(path: str) -> Adapter:
    """{module name: (A [r, in], B [out, r])} from `<name>.lora_A/B.weight`.

    Raises:
        KeyError: On unpaired tensors or keys of another format.
    """
    with safe_open(path, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        adapter = {}
        for key in sorted(keys):
            match = re.fullmatch(r'(.+)\.lora_A\.weight', key)
            if match is None:
                continue
            name = match.group(1)
            b_key = f'{name}.lora_B.weight'
            if b_key not in keys:
                raise KeyError(f'{key} has no lora_B')
            adapter[name] = (f.get_tensor(key), f.get_tensor(b_key))
        paired = {f'{n}.lora_{x}.weight' for n in adapter for x in 'AB'}
        if keys - paired:
            raise KeyError(f'unexpected adapter tensors: '
                           f'{sorted(keys - paired)[:5]}')
    return adapter


@torch.no_grad()
def merged_weight(weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                  scale: float = 1.0,
                  compute_device: torch.device | None = None
                  ) -> torch.Tensor:
    """(W + scale * B @ A) computed in fp32, returned in W's dtype/device.

    `b` may already be restricted to the rows held by W (sharded weights).
    """
    device = compute_device or weight.device
    delta = (b.to(device, torch.float32) @ a.to(device, torch.float32))
    out = weight.to(device, torch.float32) + scale * delta
    return out.to(weight.device, weight.dtype)


@torch.no_grad()
def merge_adapter(model: nn.Module, adapter: Adapter, scale: float = 1.0,
                  skip: Callable[[str], bool] = lambda name: False,
                  compute_device: torch.device | None = None) -> set[str]:
    """Merges an adapter into frozen weights: W += scale * B @ A.

    The few-step teacher adapter is merged with scale 1.0 before the first
    trajectory. FSDP-sharded weights receive their local rows.

    Args:
        model: Model whose weights are updated in place.
        adapter: From load_adapter.
        scale: Merge scale.
        skip: Names handled elsewhere (e.g. dropped AdaLN projections whose
            deltas go into the AdaLN tables).
        compute_device: Where the delta is computed (default: the weight's
            device; pass the GPU for host-offloaded shards).

    Returns:
        Names merged here.

    Raises:
        KeyError: If a non-skipped adapter entry has no destination.
        ValueError: On shape mismatches.
    """
    modules = dict(model.named_modules())
    merged = set()
    for name, (a, b) in adapter.items():
        if skip(name):
            continue
        if name not in modules or not hasattr(modules[name], 'weight'):
            raise KeyError(f'adapter entry {name} has no destination')
        weight = modules[name].weight
        if (b.shape[0], a.shape[1]) != tuple(weight.shape):
            raise ValueError(f'{name}: delta {(b.shape[0], a.shape[1])} vs '
                             f'weight {tuple(weight.shape)}')
        local = weight
        if isinstance(weight, dtensor.DTensor):
            shape, offset = dtensor_utils.compute_local_shape_and_global_offset(
                weight.shape, weight.device_mesh, weight.placements)
            b = b[offset[0]:offset[0] + shape[0]]
            local = weight.to_local()
        local.copy_(merged_weight(local, a, b, scale, compute_device))
        merged.add(name)
    return merged
