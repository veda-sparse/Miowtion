"""Training data: encoded samples, prompt corpus rules and geometry sampling.

There is no video dataset. A training sample is a text presentation already
encoded by the text encoder (plus, for fl2va / ref2va, clean condition
latents). Trajectories start from pure noise and are rolled out by the
teacher itself (see trajectory.py).

Sample cache directory:
    index.json          list of sample records (see Sample)
    text.safetensors    'hidden' [T, 5120] bf16, 'tags' [T] int8
    cond.safetensors    'video' [Nv, 96] fp32, 'audio' [Na, 32] fp32
                        (clean condition rows, keyframes first, then
                        reference visuals, in layout order)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miowtion.h3 import geometry as h3_geometry

TASKS = ('t2va', 'fl2va', 'ref2va')
# Checkpoint variant serving each task.
VARIANT_OF_TASK = {'t2va': 'FL2VA', 'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}

# Section headers of the structured prompt, in order (see the prompt-writing
# guides shipped in the MiniMax-H3 repository, skills/h3-prompt-writing).
# t2va / fl2va use the base format (fl2va adds a keyframe alignment line
# before it); ref2va uses the six-section reference format.
_PROMPT_FIELDS = {
    'base': ('integrated_multimodal_description:', 'overall_soundscape:',
             'non_diegetic_music:'),
    'ref2va': ('subject_definitions:', 'summary:', 'retention_analysis:',
               'detailed_description:', 'overall_soundscape:',
               'non_diegetic_music:'),
}


def validate_prompt(prompt: str, task: str = 't2va') -> None:
    """Rejects prompts not in the structured H3 format of `task`.

    Base format (t2va / fl2va): the three base fields in order and a
    '[Shot 1]' marker in the multimodal description. ref2va: the six
    reference-format sections in order.

    Raises:
        ValueError: With the reason.
    """
    fields = _PROMPT_FIELDS['ref2va' if task == 'ref2va' else 'base']
    positions = [prompt.find(field) for field in fields]
    if any(p < 0 for p in positions):
        missing = [f for f, p in zip(fields, positions) if p < 0]
        raise ValueError(f'missing fields {missing}')
    if positions != sorted(positions):
        raise ValueError('fields out of order')
    if task != 'ref2va' and '[Shot 1]' not in prompt[positions[0]:
                                                     positions[1]]:
        raise ValueError('no [Shot 1] marker in the description')


def load_prompts(path: str, task: str = 't2va') -> list[str]:
    """Loads a .jsonl ({'prompt': ...} per line) or .json list corpus.

    Malformed prompts raise instead of being dropped silently.
    """
    with open(path) as f:
        if path.endswith('.jsonl'):
            prompts = [json.loads(line)['prompt'] for line in f
                       if line.strip()]
        else:
            prompts = [p if isinstance(p, str) else p['prompt']
                       for p in json.load(f)]
    for i, prompt in enumerate(prompts):
        try:
            validate_prompt(prompt, task)
        except ValueError as e:
            raise ValueError(f'{path}: prompt {i}: {e}') from e
    return prompts


def split_ids(ids: Sequence[str], num_test: int = 20,
              seed: int = 0) -> dict[str, str]:
    """Fixed-seed train/test split shared by training and evaluation."""
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(
        seed)).tolist()
    test = {ids[i] for i in order[:num_test]}
    return {i: ('test' if i in test else 'train') for i in ids}


@dataclasses.dataclass(frozen=True)
class Sample:
    """One encoded presentation.

    Attributes:
        id: Unique id.
        task: 't2va', 'fl2va' or 'ref2va'.
        split: 'train' or 'test'.
        text: [start, stop) rows in text.safetensors.
        keyframes: Keyframe signature (fl2va), () otherwise.
        references: ref2va layout blocks (see layout.pack).
        aspect: Canvas aspect the condition latents were encoded for
            (keyframes live on the target canvas); None when any aspect fits.
        latent_t: Target latent frames the prompt was written for (its
            shot timing depends on the duration); None when any fits.
        cond_video: [start, stop) rows in cond.safetensors 'video'.
        cond_audio: [start, stop) rows in cond.safetensors 'audio'.
    """

    id: str
    task: str
    split: str
    text: tuple[int, int]
    keyframes: tuple[int, ...] = ()
    references: tuple[dict, ...] = ()
    aspect: str | None = None
    latent_t: int | None = None
    cond_video: tuple[int, int] = (0, 0)
    cond_audio: tuple[int, int] = (0, 0)

    @property
    def text_len(self) -> int:
        return self.text[1] - self.text[0]


class SampleCache:
    """Read-only, lazily sliced view of a sample cache directory."""

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, 'index.json')) as f:
            records = json.load(f)['samples']
        self.samples = [Sample(**{'latent_t': None, **r,
                                  'text': tuple(r['text']),
                                  'keyframes': tuple(r['keyframes']),
                                  'references': tuple(r['references']),
                                  'cond_video': tuple(r['cond_video']),
                                  'cond_audio': tuple(r['cond_audio'])})
                        for r in records]
        self._text = safe_open(os.path.join(directory, 'text.safetensors'),
                               framework='pt', device='cpu')
        cond_path = os.path.join(directory, 'cond.safetensors')
        self._cond = (safe_open(cond_path, framework='pt', device='cpu')
                      if os.path.exists(cond_path) else None)

    def select(self, split: str, tasks: Sequence[str]) -> list[Sample]:
        return [s for s in self.samples
                if s.split == split and s.task in tasks]

    def text(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """(hidden [L, 5120] bf16, tags [L] int64)."""
        start, stop = sample.text
        hidden = self._text.get_slice('hidden')[start:stop]
        tags = self._text.get_slice('tags')[start:stop].to(torch.long)
        return hidden, tags

    def conditions(self, sample: Sample
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean condition rows (video [Nc, 96] fp32, audio [Na, 32] fp32)."""
        if self._cond is None:
            return torch.empty(0, 96), torch.empty(0, 32)
        video = self._cond.get_slice('video')[slice(*sample.cond_video)]
        audio = self._cond.get_slice('audio')[slice(*sample.cond_audio)]
        return video, audio


class SyntheticSampleCache:
    """A SampleCache stand-in holding one random t2va sample.

    For benchmarks without an encoded prompt: a DiT step only sees the
    prompt through its row count (the text rows of the packed layout and
    the token refiner), so random rows of the right length cost the same
    as encoded ones.
    """

    def __init__(self, text_len: int, text_dim: int, seed: int = 0):
        """Draws the rows once.

        Args:
            text_len: Text rows of the prompt (e.g. 589 for a typical
                structured prompt of the holdout set).
            text_dim: Width of the text encoder's hidden states.

        Raises:
            ValueError: On a non-positive `text_len`.
        """
        if text_len <= 0:
            raise ValueError(f'text_len must be positive, got {text_len}')
        generator = torch.Generator().manual_seed(seed)
        self._hidden = torch.randn(text_len, text_dim,
                                   generator=generator).to(torch.bfloat16)
        self.samples = [Sample(id=f'synthetic_text{text_len}', task='t2va',
                               split='train', text=(0, text_len))]

    def select(self, split: str, tasks: Sequence[str]) -> list[Sample]:
        return [s for s in self.samples
                if s.split == split and s.task in tasks]

    def text(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """(hidden [L, D] bf16, tags [L] int64); every row is text."""
        del sample
        return self._hidden, torch.ones(self._hidden.shape[0],
                                        dtype=torch.long)

    def conditions(self, sample: Sample
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """t2va has no condition rows."""
        del sample
        return torch.empty(0, 96), torch.empty(0, 32)


class SampleCacheWriter:
    """Accumulates encoded samples and writes a cache atomically."""

    def __init__(self, directory: str):
        self.directory = directory
        self.records = []
        self._hidden, self._tags, self._video, self._audio = [], [], [], []
        self._rows = {'text': 0, 'video': 0, 'audio': 0}

    def add(self, sample_id: str, task: str, split: str,
            hidden: torch.Tensor, tags: torch.Tensor,
            keyframes: Sequence[int] = (),
            references: Sequence[dict] = (), aspect: str | None = None,
            latent_t: int | None = None, cond_video: torch.Tensor | None = None,
            cond_audio: torch.Tensor | None = None) -> None:
        if task not in TASKS:
            raise ValueError(f'unknown task {task}')
        if hidden.shape[0] != tags.shape[0]:
            raise ValueError('hidden and tags disagree in length')
        spans = {}
        for key, value, store in (('text', hidden, self._hidden),
                                  ('video', cond_video, self._video),
                                  ('audio', cond_audio, self._audio)):
            start = self._rows[key]
            if value is not None:
                store.append(value)
                self._rows[key] += value.shape[0]
            spans[key] = [start, self._rows[key]]
        self._tags.append(tags.to(torch.int8))
        self.records.append({
            'id': sample_id, 'task': task, 'split': split,
            'text': spans['text'], 'keyframes': list(keyframes),
            'references': list(references), 'aspect': aspect,
            'latent_t': latent_t,
            'cond_video': spans['video'], 'cond_audio': spans['audio']})

    def finalize(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        tmp = os.path.join(self.directory, '.tmp')
        os.makedirs(tmp, exist_ok=True)
        save_file({'hidden': torch.cat(self._hidden).to(torch.bfloat16)
                   .contiguous(), 'tags': torch.cat(self._tags)},
                  os.path.join(tmp, 'text.safetensors'))
        if self._video or self._audio:
            video = (torch.cat(self._video) if self._video
                     else torch.empty(0, 96))
            audio = (torch.cat(self._audio) if self._audio
                     else torch.empty(0, 32))
            save_file({'video': video.float().contiguous(),
                       'audio': audio.float().contiguous()},
                      os.path.join(tmp, 'cond.safetensors'))
        with open(os.path.join(tmp, 'index.json'), 'w') as f:
            json.dump({'samples': self.records}, f, indent=1)
        for name in os.listdir(tmp):
            os.replace(os.path.join(tmp, name),
                       os.path.join(self.directory, name))
        os.rmdir(tmp)


def parse_geometry(spec: str) -> h3_geometry.Geometry:
    """'16:9@37' -> geometry with latent_t 37 (or '16:9@5.0s' seconds)."""
    match = re.fullmatch(r'(\d+:\d+)@(\d+(?:\.\d+)?)(s?)', spec)
    if match is None:
        raise ValueError(f'bad geometry spec {spec!r}')
    aspect, value, seconds = match.groups()
    if seconds:
        return h3_geometry.resolve_geometry(aspect, float(value))
    return h3_geometry.geometry_from_latent_t(aspect, int(value))


GEOMETRY_SAMPLING_MODES = ('uniform', 'cycle')


class GeometrySampler:
    """Geometry choice shared by all ranks.

    'uniform' draws every trajectory's geometry independently. 'cycle' visits
    each geometry once per round in a shuffled order, so every geometry is
    covered after len(specs) trajectories (balanced mixing; smoke tests use it
    to reach all geometries quickly).

    Nothing depends on the rank: every rank must draw the same geometry,
    otherwise ranks run different sequence lengths and fall out of step.
    """

    def __init__(self, specs: Sequence[str], seed: int,
                 mode: str = 'uniform'):
        if mode not in GEOMETRY_SAMPLING_MODES:
            raise ValueError(f'geometry sampling must be one of '
                             f'{GEOMETRY_SAMPLING_MODES}, got {mode!r}')
        self.geometries = [parse_geometry(s) for s in specs]
        self.mode = mode
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.draws = 0

    def next(self) -> h3_geometry.Geometry:
        if self.mode == 'uniform':
            index = torch.randint(len(self.geometries), (1,),
                                  generator=self.generator).item()
            return self.geometries[index]
        # The order of a round depends only on (seed, round), so the draw
        # count alone is the resumable state.
        count = len(self.geometries)
        round_index, position = divmod(self.draws, count)
        order = torch.randperm(count, generator=torch.Generator().manual_seed(
            self.seed * 7919 + round_index))
        self.draws += 1
        return self.geometries[order[position].item()]

    def state(self) -> torch.Tensor:
        """Resumable state (generator state, or the cycle's draw count)."""
        if self.mode == 'uniform':
            return self.generator.get_state()
        return torch.tensor([self.draws], dtype=torch.int64)

    def load_state(self, state: torch.Tensor) -> None:
        if self.mode == 'uniform':
            self.generator.set_state(state)
        else:
            self.draws = int(state[0])


class SampleSampler:
    """Per-rank sample order (epoch permutations) with resumable state."""

    def __init__(self, samples: Sequence[Sample], seed: int, rank: int):
        if not samples:
            raise ValueError('no samples to train on')
        self.samples = list(samples)
        self.seed = seed
        self.rank = rank
        self.epoch = 0
        self.position = 0
        self._order = self._permutation()

    def _permutation(self) -> list[int]:
        gen = torch.Generator().manual_seed(
            self.seed * 100003 + self.rank * 1009 + self.epoch)
        return torch.randperm(len(self.samples), generator=gen).tolist()

    def next(self, aspect: str | None = None,
             latent_t: int | None = None) -> Sample:
        """Next sample compatible with the geometry.

        A sample fits when its canvas-bound conditions match `aspect` and its
        prompt was written for `latent_t` (or it declares neither).
        """
        for _ in range(2 * len(self.samples)):
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._permutation()
            sample = self.samples[self._order[self.position]]
            self.position += 1
            if ((aspect is None or sample.aspect in (None, aspect)) and
                    (latent_t is None or sample.latent_t in (None, latent_t))):
                return sample
        raise ValueError(f'no sample compatible with aspect {aspect}, '
                         f'latent_t {latent_t}')

    def state(self) -> dict:
        return {'epoch': self.epoch, 'position': self.position}

    def load_state(self, state: dict) -> None:
        self.epoch = state['epoch']
        self.position = state['position']
        self._order = self._permutation()
