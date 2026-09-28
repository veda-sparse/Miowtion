"""The frozen teacher shared by training, tile search and smoke runs.

Teacher = released DiT (FL2VA or Ref2VA) + optional few-step LoRA merged
into the weights (W += B @ A, scale 1.0, before the first trajectory) +
the schedule it is sampled with:
  * 'base':  the release's 50-point grid (49 steps), no adapter;
  * 'turbo': the few-step grid of the Turbo LoRA (4 or 8 steps), with the
    adapter. A few-step teacher must be rolled out on its own grid; on the
    50-step grid it would be queried at noise levels it never saw.
Every adapter entry must land somewhere: trunk / refiner / final-layer
weights are merged in place, the per-block AdaLN deltas go into the AdaLN
tables (the projections themselves are dropped).
"""

from __future__ import annotations

import dataclasses
import os

from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule as h3_schedule
from miowtion.train import adaln
from miowtion.train import lora
from miowtion.train import parallel
from miowtion.train import trajectory
from miowtion.utils import progress

SCHEDULES = ('base', 'turbo')


def make_schedule(kind: str, num_steps: int,
                  scales: h3_schedule.ShiftScales) -> h3_schedule.Schedule:
    if kind == 'base':
        return h3_schedule.Schedule.build(num_steps + 1, scales)
    if kind == 'turbo':
        return h3_schedule.turbo_schedule(num_steps, scales)
    raise ValueError(f'schedule must be one of {SCHEDULES}, got {kind!r}')


def _is_block_adaln(name: str) -> bool:
    return name.startswith('blocks.') and name.endswith('.adaln_proj.linear')


@dataclasses.dataclass
class Teacher:
    model: h3_model.H3DiT
    schedule: h3_schedule.Schedule
    tables: adaln.AdalnTables
    transformer_dir: str


def build_teacher(checkpoint_root: str, variant: str, schedule: str,
                  num_steps: int, adapter_path: str | None,
                  env: parallel.DistEnv, visual_conditions: bool,
                  audio_references: bool, offload_blocks: int = 0,
                  prefetch: int = 1, mlp_chunk_rows: int | None = None,
                  before_shard=None,
                  stages: dict[str, float] | None = None) -> Teacher:
    """Loads, merges and tabulates the frozen teacher.

    Args:
        stages: Optional dict; the wall time of the three startup phases
            ('load_weights', 'merge_adapter', 'adaln_tables') is written
            into it, for scripts/benchmark.py.

    Raises:
        KeyError: If an adapter entry has no destination.
    """
    variant_dir = os.path.join(checkpoint_root, variant)
    transformer_dir = os.path.join(variant_dir, 'transformer')
    with progress.Timer('load the teacher weights') as timer:
        model = parallel.build_model(
            transformer_dir, env, drop_adaln=True,
            offload_blocks=offload_blocks, prefetch=prefetch,
            mlp_chunk_rows=mlp_chunk_rows, before_shard=before_shard)
    if stages is not None:
        stages['load_weights'] = timer.seconds
    model.requires_grad_(False)
    model.eval()
    sched = make_schedule(schedule, num_steps,
                          h3_schedule.ShiftScales.from_checkpoint(variant_dir))
    progress.log(f'teacher schedule {schedule} ({sched.num_steps} steps): '
                 f'video sigmas {[round(s, 4) for s in sched.video]}')
    adapter = None
    merged = set()
    if adapter_path:
        with progress.Timer(f'merge few-step adapter {adapter_path}') as timer:
            adapter = lora.load_adapter(adapter_path)
            merged = lora.merge_adapter(model, adapter, skip=_is_block_adaln,
                                        compute_device=env.device)
        if stages is not None:
            stages['merge_adapter'] = timer.seconds
    tables = adaln.AdalnTables()
    with progress.Timer('precompute AdaLN tables') as timer:
        consumed = tables.build(
            model, transformer_dir,
            trajectory.schedule_timestep_sets(sched, visual_conditions,
                                              audio_references),
            env.device, adapter=adapter)
    if stages is not None:
        stages['adaln_tables'] = timer.seconds
    if adapter is not None:
        missing = set(adapter) - merged - consumed
        if missing:
            raise KeyError(f'adapter entries not applied: {sorted(missing)[:5]}')
        progress.log(f'adapter applied: {len(merged)} weights merged, '
                     f'{len(consumed)} AdaLN projections tabulated')
    return Teacher(model, sched, tables, transformer_dir)
