"""Self-contained predictor bundle: live weights plus the tile plans.

A training checkpoint is not a deployment artifact. It carries the EMA
shadow next to the live weights (twice the size for something inference
never reads), it is a `torch.load` pickle, and it says nothing about which
tile shapes the predictor was trained against -- those live in a separate
plan directory that has to be shipped and named correctly alongside it.
Getting that pairing wrong is silent: the predictor still produces scores,
they are just scores for a tiling it never saw.

The bundle is one safetensors file holding the live predictor weights, with
every plan of the plan table serialized into its `__metadata__` (which
safetensors defines as str -> str, so the plans go in as one JSON blob).
Geometry, tile shapes and the weights therefore travel together and a
mismatch is impossible by construction.

EMA is deliberately dropped: which of the two the run should deploy is a
decision made once, at export time, and recorded in `source_weights`.

Weights are stored in bf16 by default. `LayerPredictor.embed` upcasts the
projection it selects (`proj.index_select(0, heads).float()`), so the
scoring arithmetic is fp32 whatever the storage dtype is; keeping fp32 on
disk only doubles the file and, more importantly, doubles the resident copy
each inference replica holds on its own card.
"""

from __future__ import annotations

import dataclasses
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor

FORMAT = 'miowtion-veda-predictor-v1'

_PREFIX = 'predictor.'

# Storage dtypes. fp8 is deliberately absent: e4m3 carries 3 mantissa bits,
# which perturbs the block logits enough to reorder the top-k, and would
# need a per-head scale to be safe.
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16}

# Bundles written before the dtype was recorded are fp32.
_LEGACY_DTYPE = 'float32'


@dataclasses.dataclass
class Bundle:
    """A loaded bundle.

    Attributes:
        predictor: The predictor, on the requested device, in eval mode.
        plans: Tile plans the predictor was trained against.
        keep_ratio: Keep ratio the run trained with; a default for
            inference, not a constraint.
        metadata: The file's raw string metadata (provenance).
    """

    predictor: veda_predictor.TileScorePredictor
    plans: veda_plan.PlanTable
    keep_ratio: float
    metadata: dict[str, str]


def save(path: str, weights: dict[str, torch.Tensor],
         plans: veda_plan.PlanTable, *, num_layers: int, num_heads: int,
         head_dim: int, keep_ratio: float, source: str,
         source_weights: str, step: int,
         dtype: torch.dtype = torch.bfloat16) -> None:
    """Writes a bundle.

    Args:
        path: Destination `.safetensors` file.
        weights: Predictor state dict, keys either bare (`layers.0.proj_q`)
            or checkpoint style (`predictor.layers.0.proj_q`).
        plans: Plans to embed; every plan of the table is written.
        num_layers: Layers the predictor was built with.
        num_heads: Heads per layer.
        head_dim: Head dimension.
        keep_ratio: Keep ratio the run trained with.
        source: Checkpoint directory the weights came from (provenance).
        source_weights: 'live' or 'ema' (which of the two was exported).
        step: Training update the checkpoint was written at.
        dtype: Storage dtype; must be one of `DTYPES`.

    Raises:
        ValueError: If `weights` is empty, mixes prefixed and bare keys, or
            `dtype` is not a supported storage dtype.
    """
    if not weights:
        raise ValueError('no predictor weights to save')
    names = {v: k for k, v in DTYPES.items()}
    if dtype not in names:
        raise ValueError(f'unsupported storage dtype {dtype}; '
                         f'expected one of {sorted(DTYPES)}')
    prefixed = [k for k in weights if k.startswith(_PREFIX)]
    if prefixed and len(prefixed) != len(weights):
        raise ValueError('weights mix prefixed and bare keys')
    tensors = {k[len(_PREFIX):] if prefixed else k: v.to(dtype).contiguous()
               for k, v in weights.items()}
    if not plans.plans:
        raise ValueError('bundle without plans; pass the run\'s plan table')
    metadata = {
        'format': FORMAT,
        'num_layers': str(num_layers),
        'num_heads': str(num_heads),
        'head_dim': str(head_dim),
        'dtype': names[dtype],
        'keep_ratio': repr(float(keep_ratio)),
        'source': source,
        'source_weights': source_weights,
        'step': str(step),
        'plans': json.dumps({name: p.to_json()
                             for name, p in sorted(plans.plans.items())}),
    }
    save_file(tensors, path, metadata=metadata)


def read_metadata(path: str) -> dict[str, str]:
    """The file's string metadata, without reading a single tensor.

    Raises:
        ValueError: If the file is not a bundle of this format.
    """
    with safe_open(path, framework='pt', device='cpu') as f:
        metadata = dict(f.metadata() or {})
    if metadata.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} bundle '
                         f'(format {metadata.get("format")!r})')
    return metadata


def load(path: str, device: torch.device | str = 'cpu') -> Bundle:
    """Loads a bundle, strictly: every stored tensor must land on a parameter.

    Args:
        path: A `.safetensors` file written by `save()`.
        device: Where to put the predictor.

    Returns:
        The bundle.

    Raises:
        ValueError: If the file is not a bundle of this format, or its
            tensors do not match the predictor the metadata describes.
    """
    metadata = read_metadata(path)
    with safe_open(path, framework='pt', device='cpu') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    stored = metadata.get('dtype', _LEGACY_DTYPE)
    if stored not in DTYPES:
        raise ValueError(f'{path}: unknown storage dtype {stored!r}')
    model = veda_predictor.TileScorePredictor(int(metadata['num_layers']),
                                              int(metadata['num_heads']),
                                              int(metadata['head_dim']))
    # Before the load, not after: load_state_dict copies into the existing
    # parameter, so an fp32 module would silently upcast a bf16 file back to
    # fp32 and hold twice the memory the bundle was exported to save.
    model = model.to(DTYPES[stored])
    shape = (f'{metadata["num_layers"]} layers x {metadata["num_heads"]} '
             f'heads x {metadata["head_dim"]}')
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as error:
        raise ValueError(f'{path}: predictor weights do not match the '
                         f'{shape} the metadata declares: {error}') from error
    plans = veda_plan.PlanTable(
        [veda_plan.TilePlan.from_json(p)
         for p in json.loads(metadata['plans']).values()])
    return Bundle(predictor=model.to(device).eval(), plans=plans,
                  keep_ratio=float(metadata['keep_ratio']),
                  metadata=metadata)
