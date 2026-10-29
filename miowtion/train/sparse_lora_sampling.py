"""Exact prompt coverage with synchronous geometry batches for Sparse LoRA."""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Sequence

import torch

from miowtion.train import data

SAMPLING_VERSION = 'geometry-batches-zero-weight-tail-v2'
BALANCED_SAMPLING_VERSION = SAMPLING_VERSION + '-duration-balanced-v1'


def balance_duration_window(batches: list[list[int]],
                            samples: Sequence[data.Sample], window: int,
                            prefix: int = 0) -> list[list[int]]:
    """Balances a finite batch window, preserving an already-used prefix.

    Whole same-geometry batches move together. No prompt, geometry, tail
    validity or within-duration order changes. Remaining batches retain
    their original order after the balanced window.
    """
    if (type(window) is not int or not 1 <= window <= len(batches)
            or type(prefix) is not int or not 0 <= prefix <= window):
        raise ValueError('invalid duration balance window/prefix')
    queues = collections.defaultdict(collections.deque)
    used = collections.Counter()
    for index, batch in enumerate(batches):
        duration = samples[batch[0]].latent_t
        if index < prefix:
            used[duration] += 1
        else:
            queues[duration].append(index)
    durations = sorted({s.latent_t for s in samples})
    base, extra = divmod(window, len(durations))
    quota = {t: base + int(i < extra) for i, t in enumerate(durations)}
    if any(used[t] > quota[t] or used[t] + len(queues[t]) < quota[t]
           for t in durations):
        raise ValueError('duration quota incompatible with prefix/data')
    selected = list(range(prefix))
    while len(selected) < window:
        duration = min((t for t in durations if used[t] < quota[t]),
                       key=lambda t: (used[t], t))
        selected.append(queues[duration].popleft())
        used[duration] += 1
    chosen = set(selected)
    return ([batches[i] for i in selected]
            + [batch for i, batch in enumerate(batches) if i not in chosen])


class CoverageSampler:
    """Visits each real prompt once per epoch with zero-weight tail slots.

    All ranks construct identical, shuffled same-geometry batches. A short
    batch uses its first prompt for otherwise idle ranks, with last_valid
    false: these ranks must still execute forward/backward for FSDP, but
    contribute zero loss and do not count towards dataset coverage.
    """

    def __init__(self, samples: Sequence[data.Sample], seed: int,
                 rank: int = 0, world_size: int = 1,
                 duration_balance_window: int | None = None,
                 duration_balance_prefix: int = 0):
        if (type(world_size) is not int or world_size < 1
                or type(rank) is not int or not 0 <= rank < world_size):
            raise ValueError('invalid sampler rank/world_size')
        self.samples = list(samples)
        if (not self.samples or len({s.id for s in samples}) != len(samples)
                or any(s.aspect is None or s.latent_t is None
                       for s in samples)):
            raise ValueError('unique samples with explicit geometry required')
        self.seed, self.rank, self.world_size = seed, rank, world_size
        if duration_balance_window is None and duration_balance_prefix:
            raise ValueError('duration prefix requires a balance window')
        self.duration_balance_window = duration_balance_window
        self.duration_balance_prefix = duration_balance_prefix
        self.buckets = collections.defaultdict(list)
        for index, sample in enumerate(samples):
            self.buckets[sample.aspect, sample.latent_t].append(index)
        self.epoch, self.position = 0, 0
        self._order = self._permutation(self.epoch)

    @property
    def batches_per_epoch(self) -> int:
        """Number of synchronized prompt batches, including short ones."""
        return sum(math.ceil(len(v) / self.world_size)
                   for v in self.buckets.values())

    @property
    def issued(self) -> int:
        """Batch slots issued per rank, including an unfinished trajectory."""
        return self.epoch * self.batches_per_epoch + self.position

    @property
    def issued_real_global(self) -> int:
        """Real prompt visits issued across ranks, excluding padding."""
        return self.epoch * len(self.samples) + sum(
            len(batch) for batch in self._order[:self.position])

    @property
    def last_batch_size(self) -> int:
        """Real prompts in the last issued synchronized batch."""
        if not self.position:
            raise ValueError('no batch has been issued in this epoch')
        return len(self._order[self.position - 1])

    @property
    def last_valid(self) -> bool:
        """Whether this rank's last prompt is real rather than padding."""
        return self.rank < self.last_batch_size

    def _permutation(self, epoch: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed * 100003 + epoch)
        batches = []
        for _, indices in sorted(self.buckets.items()):
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            batches.extend(shuffled[i:i + self.world_size]
                           for i in range(0, len(shuffled), self.world_size))
        order = torch.randperm(len(batches), generator=generator).tolist()
        result = [batches[i] for i in order]
        if self.duration_balance_window is not None:
            result = balance_duration_window(
                result, self.samples, self.duration_balance_window,
                self.duration_balance_prefix if epoch == 0 else 0)
        return result

    def next(self) -> data.Sample:
        """Returns a real prompt or a same-geometry zero-weight stand-in."""
        if self.position == len(self._order):
            self.epoch += 1
            self.position = 0
            self._order = self._permutation(self.epoch)
        batch = self._order[self.position]
        index = batch[self.rank] if self.rank < len(batch) else batch[0]
        self.position += 1
        return self.samples[index]

    def peek_batch_sizes(self, count: int) -> list[int]:
        """Looks ahead without consuming the sampler or any global RNG."""
        order, epoch, position = self._order, self.epoch, self.position
        result = []
        for _ in range(count):
            if position == len(order):
                epoch += 1
                position = 0
                order = self._permutation(epoch)
            result.append(len(order[position]))
            position += 1
        return result

    def valid_state_count(self, micro_steps: int, states_per_visit: int,
                          active_remaining: int = 0) -> int:
        """Real global states in the next update, including active rollout."""
        active = min(micro_steps, active_remaining)
        total = active * self.last_batch_size if active else 0
        remaining = micro_steps - active
        for size in self.peek_batch_sizes(
                math.ceil(remaining / states_per_visit)):
            take = min(remaining, states_per_visit)
            total += size * take
            remaining -= take
        return total

    def state(self) -> dict:
        """Returns the cursor; input identity is in the resume contract."""
        state = {'version': SAMPLING_VERSION, 'epoch': self.epoch,
                 'position': self.position}
        if self.duration_balance_window is not None:
            state.update(version=BALANCED_SAMPLING_VERSION,
                         duration_balance_window=self.duration_balance_window,
                         duration_balance_prefix=self.duration_balance_prefix)
        return state

    def load_state(self, state: dict) -> None:
        """Rejects older sampling policies instead of reinterpreting cursors."""
        epoch, position = state.get('epoch'), state.get('position')
        identity = self.state()
        keys = ('version', 'duration_balance_window', 'duration_balance_prefix')
        if (any(state.get(key) != identity.get(key) for key in keys)
                or type(epoch) is not int or epoch < 0
                or type(position) is not int
                or not 0 <= position <= self.batches_per_epoch):
            raise ValueError('incompatible exact-coverage sampler state')
        self.epoch, self.position = epoch, position
        self._order = self._permutation(epoch)


def timestep_generator(seed: int) -> torch.Generator:
    """Shared rank-independent RNG so FSDP teacher forward counts agree."""
    return torch.Generator().manual_seed(seed * 104729)


def planned_updates(batches_per_epoch: int, states_per_visit: int,
                    accum: int, steps: int | None, epochs: int | None) -> int:
    """Resolves the actual batched workload, allowing a short final update."""
    if (steps is None) == (epochs is None):
        raise ValueError('set exactly one of steps or epochs')
    if epochs is None:
        return steps
    return math.ceil(batches_per_epoch * states_per_visit * epochs / accum)


@dataclasses.dataclass(frozen=True)
class UpdatePlan:
    """Normalization shared across ranks for one optimizer update."""

    micro_steps: int
    real_states: int
    slots: int
    loss_scale: float


def update_plan(sampler: CoverageSampler, states_per_visit: int, accum: int,
                step: int, epochs: int | None,
                active_remaining: int = 0) -> UpdatePlan:
    """Scales local losses so rank-averaged gradients mean real samples.

    FSDP/DDP averages across world_size. Multiplying each real local loss
    by world_size / real_states before backward compensates that average;
    padding multiplies by zero. The count spans the complete accumulation
    window, not separately normalized micro-batches of unequal sizes.
    """
    count = accum
    if epochs is not None:
        remaining = (sampler.batches_per_epoch * states_per_visit * epochs
                     - step * accum)
        count = min(accum, remaining)
    if count < 1:
        raise ValueError('no states remain in the epoch workload')
    real = sampler.valid_state_count(count, states_per_visit, active_remaining)
    return UpdatePlan(count, real, count * sampler.world_size,
                      sampler.world_size / real)
