"""Four-rank CUDA/FSDP trainer fixture; launch with torchrun directly.

Synthetic weights exercise the real fifty-layer sparse backward, rank
averaging, zero-weight tail and restart. Native weights are tested apart.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.distributed as dist

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule
from miowtion.train import adaln
from miowtion.train import data
from miowtion.train import lora
from miowtion.train import parallel
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_sampling as sampling
from miowtion.train import sparse_lora_trainer as trainer
from miowtion.train import trajectory
from miowtion.veda import plan
from miowtion.veda import predictor
from miowtion.veda import tiling


def run_fixture(output: Path, resume: str | None = None,
                memory_optimized: bool = False,
                sequential: bool = False) -> None:
    """Runs exactly two updates, or one replay after the first update."""
    env = parallel.init_distributed(timeout_minutes=5)
    if env.world_size != 4:
        raise ValueError('this fixture requires exactly four CUDA ranks')
    torch.manual_seed(19)
    cfg = dataclasses.replace(h3_config.H3Config.tiny(
        num_layers=50, num_heads=2), head_dim=128)
    model = h3_model.H3DiT(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    chunk_rows = 31 if memory_optimized else None
    lora.add_lora(model, 32, 32, layers=[0, 1, 49], chunk_rows=chunk_rows)
    model.to(env.device)
    model.dense_backend = 'fa4'
    model.gradient_checkpointing = True
    model.checkpoint_group_size = 5 if memory_optimized else 1
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    cache_path = output.parent / 'cache'
    if env.is_main and not (cache_path / 'index.json').is_file():
        writer = data.SampleCacheWriter(str(cache_path))
        generator = torch.Generator().manual_seed(71)
        for index in range(17):
            writer.add(str(index), 't2va', 'train', torch.randn(
                12, cfg.text_dim, generator=generator),
                torch.ones(12, dtype=torch.long), aspect='16:9', latent_t=7)
        writer.finalize()
    dist.barrier()
    cache = data.SampleCache(str(cache_path))
    sched = schedule.turbo_schedule(8, schedule.ShiftScales(12., 3.))
    traj = trajectory.Trajectory(model, cache, cache.samples[0], geo, sched,
                                  0, env.device)
    tables = adaln.AdalnTables()
    with torch.no_grad():
        for timesteps in traj.timestep_sets():
            tables._tables[adaln.timestep_key(timesteps)] = (
                model.precompute_adaln(timesteps.to(env.device)))
    del traj
    model.drop_adaln_projections()
    parallel.shard_trunk(model, env.mesh)
    # Match the native build: freeze after sharding, before its first call.
    model.requires_grad_(False)
    lora.reset_parameters(model, 20261020)
    pred = predictor.TileScorePredictor(50, 2, 128).to(env.device)
    obj = trainer.SparseLoraTrainer.__new__(trainer.SparseLoraTrainer)
    obj.model, obj.predictor = model, pred
    obj.named = core.freeze_except_lora(model, pred)
    obj.optimizer = torch.optim.AdamW([p for _, p in obj.named], lr=1e-4,
                                      betas=(.9, .95), weight_decay=0.)
    obj.config = core.SparseLoraConfig(
        output.name, '', '', '', '', '', str(cache_path), ['16:9@7'],
        steps=2 if sequential else None,
        epochs=None if sequential else 1, accum=4, warmup=5,
        timestep_sampling='trajectory' if sequential else 'random',
        verify_invariants=True, ordered_dq=True, save_every=1,
        lora_chunk_rows=chunk_rows,
        checkpoint_group_size=model.checkpoint_group_size)
    obj.env, obj.cache = env, cache
    obj.samples = sampling.CoverageSampler(cache.samples, 20261020, env.rank, 4)
    obj.geometries = data.GeometrySampler(['16:9@7'], 53)
    obj.geometries.geometries = [geo]
    obj.noise_gen = torch.Generator().manual_seed(59 + env.rank)
    obj.timestep_gen = sampling.timestep_generator(20261020)
    obj.schedule, obj.tables = sched, tables
    tile_plan = plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(1, 8, 16),
                               tiling.TileShape(2, 8, 8)],
                              [[0, 1] for _ in range(50)])
    obj.plans = plan.PlanTable([tile_plan])
    obj.run_dir = output
    output.mkdir(parents=True, exist_ok=True)
    obj.step, obj.traj, obj.probe = 0, None, None
    obj.contract = {'fixture': 'fifty-layer-four-rank-fsdp-tail',
                    'sampling': sampling.SAMPLING_VERSION, 'world_size': 4}
    if memory_optimized:
        obj.contract['execution'] = {
            'lora_chunk_rows': chunk_rows,
            'checkpoint_group_size': model.checkpoint_group_size,
            'checkpoint_inputs_to_cpu': model.checkpoint_inputs_to_cpu}
    if sequential:
        obj.contract['timestep_sampling'] = 'trajectory'
    obj.total_steps = (2 if sequential else
                       sampling.planned_updates(5, 1, 4, None, 1))
    obj.frozen_before = core.frozen_digest(model, pred)
    if resume:
        obj._restore(resume)
    original_micro = obj._micro_step

    def record_micro(loss_scale=None):
        record = original_micro(loss_scale)
        with (output / f'rank_{env.rank}_micro.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        return record

    obj._micro_step = record_micro
    obj.train()
    dist.barrier()
    if env.is_main:
        logs = [json.loads(line) for line in
                (output / 'log.jsonl').read_text().splitlines()]
        if sequential:
            assert [r['micro_steps_in_update'] for r in logs] == (
                [4] if resume else [4, 4])
            states = [r['state'] for log in logs for r in log['micro_steps']]
            assert states == (list(range(4, 8)) if resume else list(range(8)))
            assert all(r['teacher_preparation_steps'] == 0
                       for log in logs for r in log['micro_steps'])
        else:
            assert [r['micro_steps_in_update'] for r in logs] == (
                [1] if resume else [4, 1])
            assert logs[-1]['global_states_per_update'] == 1
            assert logs[-1]['global_padding_states_per_update'] == 3
            assert logs[-1]['completed_prompt_visits_global'] == 17
            assert logs[-1]['completed_epochs'] == 1
        (output / 'fixture_passed.json').write_text(json.dumps({
            'world_size': 4, 'real_cuda_fsdp': True,
            'train_prompts': 17, 'accum': 4, 'updates': obj.step,
            'synthetic_weights': True, 'status': 'passed',
            'memory_optimized': memory_optimized,
            'sequential_sampling': sequential,
            'checkpoint_group_size': model.checkpoint_group_size,
            'checkpoint_inputs_to_cpu': model.checkpoint_inputs_to_cpu},
            indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--memory-optimized', action='store_true')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    run_fixture(args.output, args.resume, args.memory_optimized,
                args.sequential)
