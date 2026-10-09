"""Denoising with the dense teacher or the Veda sparse student.

The loop is the training trajectory (miowtion.train.trajectory): the same
packed layout, initial noise, AdaLN tables and dual-clock Euler steps. A
dense run is therefore exactly the teacher rollout the predictor was trained
on, and a Veda run differs from it only in the attention function: the
sparse student (predictor + tile plan + FA4 block sparsity), except on the
steps listed as dense.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Collection, Sequence

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import model as h3_model
from miowtion.kernels import sol
from miowtion.train import checkpoint
from miowtion.train import data
from miowtion.train import trajectory as traj_lib
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor

ATTENTION_MODES = ('dense', 'veda', 'sol',
                   'oracle_mass', 'oracle_max', 'oracle_mean')


@dataclasses.dataclass
class Generated:
    """Final latents of one generation (normalized DiT rows, on CPU).

    Attributes:
        video_rows: [N_video, 96] fp32 target video rows.
        audio_rows: [2 * audio_t, 32] fp32 target audio rows.
        step_seconds: Wall time of every denoising step (synchronized).
        attention_seconds: GPU time of all attention calls of every step
            (for Veda including predictor scoring, mask selection, gathers).
        sparse_calls: Attention calls that ran block-sparse.
        dense_calls: Attention calls that ran dense.
    """

    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    step_seconds: list[float]
    attention_seconds: list[float]
    sparse_calls: int
    dense_calls: int

    @property
    def seconds(self) -> float:
        return sum(self.step_seconds)


class _TimedAttention:
    """Records CUDA events around every call of an attention function."""

    def __init__(self, fn):
        self.fn = fn
        self.events = []

    def __call__(self, q, k, v, layer_index):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.fn(q, k, v, layer_index)
        end.record()
        self.events.append((start, end))
        return out

    def seconds(self) -> float:
        """Total GPU time; call after synchronizing."""
        return sum(s.elapsed_time(e) for s, e in self.events) / 1000.0


def load_predictor(checkpoint_dir: str, num_layers: int, num_heads: int,
                   head_dim: int, device: torch.device,
                   use_ema: bool = True
                   ) -> veda_predictor.TileScorePredictor:
    """Predictor weights of a training checkpoint (EMA by default), strict."""
    predictor = veda_predictor.TileScorePredictor(num_layers, num_heads,
                                                  head_dim)
    named = [(f'predictor.{n}', p) for n, p in predictor.named_parameters()]
    payload = checkpoint.load(checkpoint_dir)
    # Stage-2 checkpoints also hold LoRA tensors; inference here only needs
    # the predictor.
    other = [k for k in payload['ema'] if not k.startswith('predictor.')]
    checkpoint.init_weights(named, payload, use_ema=use_ema,
                            drop_prefixes=sorted({k.split('.')[0] + '.'
                                                  for k in other}))
    return predictor.to(device).eval()


@torch.no_grad()
def generate(model: h3_model.H3DiT, schedule, tables,
             cache: data.SampleCache, sample: data.Sample,
             geometry: h3_geometry.Geometry, seed: int,
             device: torch.device, attention: str = 'dense',
             plan: veda_plan.TilePlan | None = None,
             predictor: veda_predictor.TileScorePredictor | None = None,
             veda_config: veda_attention.VedaConfig | None = None,
             dense_steps: Collection[int] = (),
             sol_tau: float = 1.0) -> Generated:
    """Rolls one sample from noise to the end of the schedule.

    Args:
        model: The (few-step LoRA merged) DiT.
        schedule: Sigma schedule of the teacher.
        tables: AdaLN tables of the schedule's timestep sets.
        cache: Sample cache holding the encoded prompt.
        sample: The sample to generate.
        geometry: Target geometry.
        seed: Noise seed.
        device: Model device.
        attention: 'dense' or 'veda'.
        plan: Tile plan of `geometry` (veda only).
        predictor: Trained predictor (veda only).
        veda_config: Budgets and dense layers (veda only).
        dense_steps: Denoising steps that stay dense in a veda run.
        sol_tau: Threshold coefficient of the sol mode. Larger routes
            fewer key blocks exactly. It is not a budget, so a run that
            has to match a fixed-budget router must calibrate it.

    Raises:
        ValueError: On an unknown mode or missing veda components.
    """
    if attention not in ATTENTION_MODES:
        raise ValueError(f'attention must be one of {ATTENTION_MODES}')
    if attention == 'veda' and None in (plan, predictor, veda_config):
        raise ValueError('veda needs a plan, a predictor and a VedaConfig')
    if attention.startswith('oracle_') and None in (plan, veda_config):
        raise ValueError(f'{attention} needs a plan and a VedaConfig; it '
                         'needs no predictor, which is the point')
    if attention == 'sol' and not sol.available(device):
        raise RuntimeError('attention sol is unavailable: '
                           f'{sol.unavailable_reason() or "device is not CUDA"}')
    traj = traj_lib.Trajectory(model, cache, sample, geometry, schedule,
                               seed, device)
    tiled = attention == 'veda' or attention.startswith('oracle_')
    clip = (veda_attention.ClipTiling(traj.layout, veda_config, device)
            if tiled else None)
    steps = progress.Progress(f'generate {sample.id} ({attention})',
                              schedule.num_steps, every=1)
    calls = {'sparse': 0, 'dense': 0}
    step_seconds, attention_seconds = [], []
    while not traj.done:
        torch.cuda.synchronize(device)
        start = time.time()
        inputs = traj.inputs()
        if attention == 'veda' and inputs.step not in dense_steps:
            fn = veda_attention.SparseStudent(clip, plan, predictor)
        elif (attention.startswith('oracle_')
              and inputs.step not in dense_steps):
            fn = veda_attention.OracleStudent(
                clip, plan, attention.removeprefix('oracle_'))
        elif attention == 'sol' and inputs.step not in dense_steps:
            fn = h3_model.SolAttention(traj.layout.used, tau=sol_tau)
        else:
            fn = h3_model.DenseAttention(traj.layout.used)
        timed = _TimedAttention(fn)
        video_v, audio_v = model(traj.clip, inputs.video_rows,
                                 inputs.audio_rows, inputs.timestep, timed,
                                 tables.get(inputs.timestep.timesteps))
        sparse = isinstance(fn, (veda_attention.SparseStudent,
                                 veda_attention.OracleStudent,
                                 h3_model.SolAttention))
        if isinstance(fn, (veda_attention.SparseStudent,
                           veda_attention.OracleStudent)):
            calls['sparse'] += fn.calls['sparse']
            calls['dense'] += fn.calls['dense']
        elif isinstance(fn, h3_model.SolAttention):
            calls['sparse'] += model.config.num_layers
        else:
            calls['dense'] += model.config.num_layers
        traj.advance(video_v, audio_v)
        torch.cuda.synchronize(device)
        step_seconds.append(time.time() - start)
        attention_seconds.append(timed.seconds())
        steps.update(f'step {inputs.step} {"sparse" if sparse else "dense"}: '
                     f'{step_seconds[-1]:.1f} s, attention '
                     f'{attention_seconds[-1]:.1f} s')
    return Generated(traj.video_rows[traj.n_cond_video:].cpu(),
                     traj.audio_rows[traj.n_cond_audio:].cpu(), step_seconds,
                     attention_seconds, calls['sparse'], calls['dense'])


def assign_jobs(costs: Sequence[float], num_devices: int) -> list[list[int]]:
    """Greedy longest-first assignment of job indices to devices.

    Deterministic: ties go to the lowest device index. Each device's list
    keeps the longest-first order, so the slow jobs start early.
    """
    if num_devices < 1:
        raise ValueError(f'num_devices must be >= 1, got {num_devices}')
    loads = [0.0] * num_devices
    assigned: list[list[int]] = [[] for _ in range(num_devices)]
    for job in sorted(range(len(costs)), key=lambda j: (-costs[j], j)):
        device = min(range(num_devices), key=lambda d: (loads[d], d))
        assigned[device].append(job)
        loads[device] += costs[job]
    return assigned


def geometry_cost(geometry: h3_geometry.Geometry) -> float:
    """Relative generation cost: attention grows with tokens squared."""
    t, h, w = geometry.video_grid
    return float(t * h * w) ** 2


def run_on_devices(devices: Sequence[torch.device],
                   work: Callable[[int, torch.device], None]) -> None:
    """Runs work(rank, device) in one thread per GPU; re-raises failures.

    One process drives all GPUs: the model is loaded, merged and tabulated
    once, offloaded blocks live once in pinned host memory, and each GPU
    streams them over its own PCIe link; there is no inter-GPU traffic.
    """
    errors: list[BaseException] = []

    def target(rank: int, device: torch.device) -> None:
        try:
            torch.cuda.set_device(device)
            work(rank, device)
        except BaseException as e:  # pylint: disable=broad-except
            errors.append(e)

    threads = [threading.Thread(target=target, args=(r, d), name=f'gpu{r}')
               for r, d in enumerate(devices)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
