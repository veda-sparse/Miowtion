"""Random stand-in for a released checkpoint, for weight-free benchmarks.

The cost of a DiT forward does not depend on the values of its weights:
every GEMM, attention call and host-to-device copy has the same shape and
dtype whatever the numbers are, and Veda keeps a fixed fraction of blocks
(the budget) whatever the scores are. A benchmark can therefore run on a
machine that only has the release's config files (the `third_party/
MiniMax-H3` submodule) instead of 66 GB of weights.

`RandomCheckpoint` exposes the same `keys()` / `read_rows()` interface as
`miowtion.h3.weights.Checkpoint`, so the real loading path (meta build,
offload placement, pinning, the strict name/shape/dtype checks, AdaLN
tables) runs unchanged and only the byte source differs. The key set is the
release's own index, so the release-schema check still sees real names.

Only timing is meaningful on these weights. The generated samples are
noise, and the values are not a stand-in for anything numerical.
"""

from __future__ import annotations

import json
import math
import os
import zlib

import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.h3 import weights as h3_weights


class RandomCheckpoint:
    """Deterministic random tensors under the release's tensor names.

    Values are chosen so that a forward stays finite, not to be realistic:
    norm scales are 1, biases 0, weight matrices N(0, 1/fan_in) (unit gain
    per layer, so 50 residual blocks do not overflow bf16), and the RoPE
    frequencies are the model's own. Every key has its own generator seeded
    from (seed, name), so a tensor does not depend on the read order or on
    which rows are read.
    """

    def __init__(self, transformer_dir: str, seed: int = 0,
                 device: torch.device | str = 'cpu'):
        """Reads the tensor names and shapes of a release.

        Args:
            transformer_dir: `<root>/<variant>/transformer`; only its
                config.json and model.safetensors.index.json are read.
            seed: Base seed of the values.
            device: Where tensors are generated (a GPU generates the 33B
                parameters in seconds, a CPU in minutes).

        Raises:
            KeyError: If the index names a tensor the model does not have
                (a release this module cannot shape).
        """
        index_path = os.path.join(transformer_dir,
                                  'model.safetensors.index.json')
        with open(index_path) as f:
            self._keys = set(json.load(f)['weight_map'])
        self.config = h3_config.H3Config.from_pretrained(transformer_dir)
        with torch.device('meta'):
            model = h3_model.H3DiT(self.config)
        shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        shapes.update((n, tuple(b.shape)) for n, b in model.named_buffers())
        unknown = sorted(self._keys - set(shapes))
        if unknown:
            raise KeyError(f'{transformer_dir}: tensors without a model '
                           f'shape: {unknown[:5]} ({len(unknown)})')
        self._shapes = {k: shapes[k] for k in self._keys}
        self._inv_freq = h3_model.Rope(self.config).inv_freq
        self._seed = seed
        self._device = torch.device(device)

    def keys(self) -> set[str]:
        return set(self._keys)

    def num_parameters(self) -> int:
        return sum(math.prod(s) for s in self._shapes.values())

    def read_rows(self, key: str, rows: torch.Tensor | None) -> torch.Tensor:
        """Random `key`, restricted to dim-0 `rows` (None = everything).

        The full tensor is drawn and then indexed, so a row has the same
        value whether it is read alone (an FSDP shard) or with the rest.
        """
        shape = self._shapes[key]
        dtype = h3_weights.expected_dtype(key)
        if key in h3_config.FP32_BUFFER_NAMES:
            value = self._inv_freq.clone()
        elif key.endswith('.bias'):
            value = torch.zeros(shape, dtype=dtype, device=self._device)
        elif len(shape) == 1:
            # Every 1-D weight of the release is a norm scale.
            value = torch.ones(shape, dtype=dtype, device=self._device)
        else:
            generator = torch.Generator(self._device).manual_seed(
                self._seed * 1_000_003 + zlib.crc32(key.encode()))
            value = torch.randn(shape, generator=generator,
                                device=self._device, dtype=torch.float32)
            value = value.mul_(shape[1] ** -0.5).to(dtype)
        return value if rows is None else value[rows.to(value.device)]
