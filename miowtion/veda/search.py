"""Tile-plan search: oracle-mask relative output error per (layer, head).

For every scored (clip, step, layer), candidate shape and head:
  1. permute q/k/v with the candidate's tile layout;
  2. sample video query tiles (same seed for every candidate, so all shapes
     are compared on paired samples); global query rows are identical under
     every shape and are not scored;
  3. compute the true probabilities p and the dense output of those rows;
  4. block score = max of p inside each (query tile, key tile) block;
  5. keep blocks with exactly the training rules (budget, Bresenham,
     diagonal, segmented blocks; global columns always kept) -- the best
     mask this permutation can offer;
  6. renormalize p on the kept blocks -> sparse output;
     rel_mse = sum ||o_sparse - o_dense||^2 / sum ||o_dense||^2 over real
     rows.
The oracle (not a trained predictor) isolates the locality of a tile shape
from the predictor's accuracy.

Plans are built from the scores by padding filter -> per-(layer, head)
majority vote over (clip, step) argmins -> at most two shapes per layer.
The model must run dense while scoring; scoring a sparse model would grade
the plan with itself.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import os
from collections.abc import Iterable, Sequence

import torch

from miowtion.h3 import attention as h3_attention
from miowtion.h3 import layout as h3_layout
from miowtion.kernels import block_heat_triton
from miowtion.kernels import fa4
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE


def oracle_rel_mse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   tile_layout: tiling.TileLayout,
                   blocks: list[veda_mask.ColumnBlock], rows: torch.Tensor,
                   dense_out: torch.Tensor | None = None,
                   lse: torch.Tensor | None = None) -> torch.Tensor:
    """Oracle relative MSE per head of one head group.

    Uses the kernel path (block heat kernel for the oracle mask, FA4 block
    sparsity for the sparse output of the sampled rows) when `dense_out` and
    `lse` are given and the kernels are available; else the fp32 reference.

    Args:
        q: [S, H', D] bf16 packed queries of the heads to score.
        k: [S, H', D] bf16.
        v: [S, H', D] bf16.
        tile_layout: Candidate permutation.
        blocks: Column blocks (budgets).
        rows: [R] int64 sampled video query tiles.
        dense_out: [S, H', D] dense attention output of these heads.
        lse: [S, H'] fp32 dense log-sum-exp of these heads.

    Returns:
        [H'] fp32 relative MSE per head.
    """
    if (dense_out is not None and lse is not None and q.is_cuda
            and fa4.available(q.device) and block_heat_triton.available()):
        return _oracle_rel_mse_kernels(q, k, v, dense_out, lse, tile_layout,
                                       blocks, rows)
    return oracle_rel_mse_reference(q, k, v, tile_layout, blocks, rows)


@torch.no_grad()
def _oracle_rel_mse_kernels(q, k, v, dense_out, lse, tile_layout, blocks,
                            rows):
    """All heads at once; never materializes [rows, N] probabilities."""
    q_t, k_t, v_t = (tiling.gather_tiles(t, tile_layout, None)
                     for t in (q, k, v))
    lse_t = lse.index_select(0, tile_layout.gather_index)
    if tile_layout.pad_slots.numel():
        lse_t.index_fill_(0, tile_layout.pad_slots, 0.0)
    heat = block_heat_triton.teacher_heat(q_t, k_t, lse_t.contiguous(),
                                          tile_layout, rows)
    selection = veda_mask.select_video_blocks(heat, tile_layout, blocks, rows)
    indices = veda_mask.kernel_indices(selection, tile_layout,
                                       global_rows=False)
    slots = (rows[:, None] * _TILE + torch.arange(
        _TILE, device=q.device)[None]).view(-1)
    o_sparse = fa4.block_sparse_attention(q_t.index_select(0, slots), k_t,
                                          v_t, indices, tile_layout)
    o_dense = dense_out.index_select(
        0, tile_layout.gather_index.index_select(0, slots))
    valid = tile_layout.slot_valid.bool().index_select(0, slots)
    diff = (o_sparse.float() - o_dense.float()).square().sum(-1)[valid]
    norm = o_dense.float().square().sum(-1)[valid]
    return diff.sum(0) / norm.sum(0).clamp(min=torch.finfo(torch.float32).tiny)


@torch.no_grad()
def oracle_rel_mse_reference(q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, tile_layout: tiling.TileLayout,
                             blocks: list[veda_mask.ColumnBlock],
                             rows: torch.Tensor) -> torch.Tensor:
    """fp32 reference of oracle_rel_mse, head by head (see module doc)."""
    heads, head_dim = q.shape[1], q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    valid = tile_layout.slot_valid.bool()
    slots = (rows[:, None] * _TILE + torch.arange(
        _TILE, device=q.device)[None]).view(-1)
    row_valid = valid.index_select(0, slots)
    n_tiles = tile_layout.n_tiles
    out = torch.empty(heads, dtype=torch.float32, device=q.device)
    for h in range(heads):
        qt = q[tile_layout.gather_index.index_select(0, slots), h]
        kt = k[tile_layout.gather_index, h]
        vt = v[tile_layout.gather_index, h].float()
        s = (qt @ kt.transpose(0, 1)).float() * scale  # [R*128, N]
        s.masked_fill_(~valid[None, :], float('-inf'))
        # Padding query slots gathered row 0; drop them before any use.
        p = torch.softmax(s, dim=-1).masked_fill_(~row_valid[:, None], 0.0)
        del s
        o_dense = p @ vt
        block = p.view(rows.numel(), _TILE, n_tiles, _TILE).amax(dim=(1, 3))
        selection = veda_mask.select_video_blocks(block[None], tile_layout,
                                                  blocks, rows)
        keep = torch.zeros(rows.numel(), n_tiles, dtype=torch.bool,
                           device=q.device)
        keep.scatter_(1, selection.index[0], selection.keep[0])
        keep[:, tile_layout.n_video_tiles:] = True
        p_sparse = p.view(rows.numel(), _TILE, n_tiles, _TILE) * keep[
            :, None, :, None]
        p_sparse = p_sparse.view_as(p)
        p_sparse = p_sparse / p_sparse.sum(-1, keepdim=True).clamp(
            min=torch.finfo(torch.float32).tiny)
        o_sparse = p_sparse @ vt
        del p, p_sparse
        err = ((o_sparse - o_dense)**2).sum(-1)[row_valid].sum()
        norm = (o_dense**2).sum(-1)[row_valid].sum()
        out[h] = err / norm.clamp(min=torch.finfo(torch.float32).tiny)
    return out


class OracleScorer:
    """AttentionFn that returns dense attention and scores candidates.

    Only the layers of the steps being scored should use this function; the
    trajectory itself stays on the dense teacher.
    """

    def __init__(self, layout: h3_layout.PackedLayout,
                 candidates: Sequence[tiling.TileShape],
                 config: veda_attention.VedaConfig, query_tiles: int,
                 seed: int, device: torch.device,
                 dense_backend: str = 'auto'):
        self.layout = layout
        self.candidates = list(candidates)
        self.clip = veda_attention.ClipTiling(layout, config, device)
        self.query_tiles = query_tiles
        self.seed = seed
        self.dense_backend = dense_backend
        self.scores: dict[int, torch.Tensor] = {}
        self.progress = None

    def __call__(self, q, k, v, layer_index):
        if self.progress is None:
            self.progress = progress.Progress(
                f'  scoring {len(self.candidates)} shapes x {q.shape[1]} '
                'heads: layers', total=_num_layers_hint(self), every=5)
        out, lse = h3_attention.dense_attention(
            q, k, v, self.layout.used, return_lse=True,
            backend=self.dense_backend)
        table = torch.empty(len(self.candidates), q.shape[1],
                            dtype=torch.float32)
        for ci, shape in enumerate(self.candidates):
            tile_layout = self.clip.get(shape)
            gen = torch.Generator().manual_seed(self.seed * 1000 + layer_index)
            count = min(self.query_tiles, tile_layout.n_video_tiles)
            rows = torch.randperm(tile_layout.n_video_tiles, generator=gen)
            rows = rows[:count].sort().values.to(q.device)
            table[ci] = oracle_rel_mse(q, k, v, tile_layout,
                                       self.clip.blocks(tile_layout), rows,
                                       dense_out=out, lse=lse).cpu()
        self.scores[layer_index] = table
        best = [str(self.candidates[i]) for i in
                table.mean(1).argsort()[:1].tolist()]
        self.progress.update(f'layer {layer_index}: mean rel-mse '
                             f'{table.mean().item():.4f}, best {best[0]}')
        return out


def _num_layers_hint(scorer: OracleScorer) -> int:
    return getattr(scorer, 'num_layers', 50)


@dataclasses.dataclass(frozen=True)
class ScoreEntry:
    """rel-MSE of every candidate for one (clip, step, layer): [C, H]."""

    clip: str
    step: int
    layer: int
    table: torch.Tensor


def save_entries(path: str, entries: Iterable[ScoreEntry],
                 candidates: Sequence[tiling.TileShape],
                 meta: dict) -> None:
    """Atomically writes entries, then a completion marker next to them."""
    entries = list(entries)
    payload = {
        'candidates': [str(c) for c in candidates],
        'meta': meta,
        'entries': [{'clip': e.clip, 'step': e.step, 'layer': e.layer,
                     'table': e.table.tolist()} for e in entries],
    }
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    with open(path + '.done', 'w') as f:
        json.dump({'entries': len(entries)}, f)


def load_entries(paths: Iterable[str]
                 ) -> tuple[list[tiling.TileShape], list[ScoreEntry], dict]:
    """Loads completed score files (files without a marker are rejected)."""
    candidates, entries, meta = None, [], {}
    for path in paths:
        if not os.path.exists(path + '.done'):
            raise FileNotFoundError(f'{path} has no completion marker')
        with open(path) as f:
            payload = json.load(f)
        shapes = [tiling.TileShape.parse(s) for s in payload['candidates']]
        if candidates is not None and shapes != candidates:
            raise ValueError(f'{path}: candidate menu differs')
        candidates = shapes
        meta = payload['meta']
        entries += [ScoreEntry(e['clip'], e['step'], e['layer'],
                               torch.tensor(e['table']))
                    for e in payload['entries']]
    return candidates, entries, meta


def build_plan(geometry_name: str, grid: tuple[int, int, int],
               candidates: Sequence[tiling.TileShape],
               entries: Sequence[ScoreEntry], num_layers: int,
               max_padding: float = 0.2, max_shapes_per_layer: int = 2,
               meta: dict | None = None) -> veda_plan.TilePlan:
    """Votes a timestep-independent plan from oracle scores.

    Args:
        geometry_name: Geometry key of the plan.
        grid: Target token grid.
        candidates: Candidate menu (columns of every entry table).
        entries: Scores of every (clip, step, layer).
        num_layers: Trunk layers.
        max_padding: Candidates padding the grid more than this are dropped.
        max_shapes_per_layer: Shapes kept per layer (each extra shape costs
            one more permutation and top-k per layer).
        meta: Extra provenance.

    Returns:
        The plan; provenance records the plan / best-single-shape MSEs.
    """
    allowed = [i for i, c in enumerate(candidates)
               if c.padding_ratio(grid) <= max_padding + 1e-12]
    if not allowed:
        raise ValueError('no candidate passes the padding limit')
    allowed_t = torch.tensor(allowed)
    by_layer = collections.defaultdict(list)
    for e in entries:
        by_layer[e.layer].append(e.table[allowed_t])  # [A, H]
    missing = [l for l in range(num_layers) if l not in by_layer]
    if missing:
        raise ValueError(f'no scores for layers {missing[:5]}')
    head_shape, plan_mse, per_layer = [], [], []
    for layer in range(num_layers):
        tables = torch.stack(by_layer[layer])  # [E, A, H]
        mean = tables.mean(0)  # [A, H]
        winners = tables.argmin(1)  # [E, H]
        votes = torch.zeros_like(mean)
        votes.scatter_add_(0, winners, torch.ones_like(winners,
                                                       dtype=mean.dtype))
        # Majority per head; ties -> lower mean MSE.
        head_pick = (votes - 1e-6 * mean).argmax(0)  # [H]
        counts = torch.bincount(head_pick, minlength=mean.shape[0]).float()
        ranking = sorted(range(mean.shape[0]),
                         key=lambda a: (-counts[a].item(),
                                        mean[a].mean().item()))
        kept = [a for a in ranking if counts[a] > 0][:max_shapes_per_layer]
        kept_t = torch.tensor(kept)
        # Heads of dropped shapes take the kept shape best for that head.
        final = kept_t[mean[kept_t].argmin(0)]
        head_shape.append([allowed[a] for a in final.tolist()])
        plan_mse.append(mean.gather(0, final[None])[0])
        per_layer.append([str(candidates[allowed[a]]) for a in kept])
    used = sorted({i for row in head_shape for i in row})
    remap = {old: new for new, old in enumerate(used)}
    all_tables = torch.stack([torch.stack(by_layer[l]).mean(0)
                              for l in range(num_layers)])  # [L, A, H]
    best_single = all_tables.mean(dim=(0, 2)).min().item()
    provenance = {
        'entries': len(entries),
        'clips': sorted({e.clip for e in entries}),
        'steps': sorted({e.step for e in entries}),
        'max_padding': max_padding,
        'max_shapes_per_layer': max_shapes_per_layer,
        'plan_mse': torch.stack(plan_mse).mean().item(),
        'best_single_shape_mse': best_single,
        'layer_shapes': per_layer,
        **(meta or {}),
    }
    return veda_plan.TilePlan(
        geometry=geometry_name, grid=tuple(grid),
        shapes=[candidates[i] for i in used],
        head_shape=[[remap[i] for i in row] for row in head_shape],
        provenance=provenance)


@dataclasses.dataclass
class SearchConfig:
    """Tile-search run (YAML keys have the same names).

    Attributes:
        run_name: Output directory name under out_dir.
        checkpoint_root: Checkpoint root containing FL2VA/ and Ref2VA/.
        variant: 'FL2VA' or 'Ref2VA'.
        sample_cache: Encoded prompts to score (test split excluded).
        geometries: Geometry specs scored one after another with the model
            loaded once, e.g. ['1:1@37', '16:9@102']. Each uses the cached
            samples whose latent_t (and aspect, if set) match.
        num_clips: Prompts to score per geometry (spread over ranks).
        steps: Denoising steps to score, e.g. [0, 12, 25, 40] of 49.
        schedule / num_steps: 'base' + 49 or 'turbo' + 4 / 8 (few-step
            teacher, see miowtion.train.teacher).
        teacher_adapter: Few-step LoRA merged into the teacher (turbo).
        candidates: Tile shapes to compare.
        keep_ratio: Budget used by the oracle masks.
        query_tiles: Sampled video query tiles per (layer, candidate).
        max_padding: Padding filter for plan building.
        mlp_chunk_rows / offload_blocks / prefetch: Memory knobs (see
            TrainConfig).
    """

    run_name: str
    checkpoint_root: str
    sample_cache: str
    geometries: list[str]
    variant: str = 'FL2VA'
    tasks: list[str] = dataclasses.field(default_factory=lambda: ['t2va'])
    num_clips: int = 6
    steps: list[int] = dataclasses.field(
        default_factory=lambda: [0, 12, 25, 40])
    schedule: str = 'base'
    num_steps: int = 49
    teacher_adapter: str | None = None
    candidates: list[str] = dataclasses.field(default_factory=lambda: [
        '8x4x4', '2x8x8', '4x4x8', '4x8x4', '8x2x8', '8x8x2'])
    keep_ratio: float = 0.1
    query_tiles: int = 40
    max_padding: float = 0.2
    seed: int = 0
    out_dir: str = 'runs'
    mlp_chunk_rows: int | None = None
    offload_blocks: int = 0
    prefetch: int = 1
    dense_backend: str = 'auto'


def score_clip(model, cache, sample, geometry, schedule, steps: Sequence[int],
               candidates: Sequence[tiling.TileShape],
               veda_config: veda_attention.VedaConfig, query_tiles: int,
               seed: int, device: torch.device, tables,
               dense_backend: str = 'auto') -> list[ScoreEntry]:
    """Rolls one clip with the dense teacher, scoring the chosen steps."""
    from miowtion.h3 import model as h3_model  # pylint: disable=import-outside-toplevel
    from miowtion.train import trajectory  # pylint: disable=import-outside-toplevel
    traj = trajectory.Trajectory(model, cache, sample, geometry, schedule,
                                 seed, device)
    entries = []
    last = max(steps)
    progress.log(f'clip {sample.id}: {traj.layout.used} tokens, rolling '
                 f'steps 0..{last} (scoring {list(steps)})')
    step_progress = progress.Progress(f'clip {sample.id}: denoise steps',
                                      last + 1)
    while not traj.done and traj.step <= last:
        inputs = traj.inputs()
        if traj.step in steps:
            attention_fn = OracleScorer(
                traj.layout, candidates, veda_config, query_tiles,
                seed=seed + traj.step, device=device,
                dense_backend=dense_backend)
            attention_fn.num_layers = model.config.num_layers
        else:
            attention_fn = h3_model.DenseAttention(traj.layout.used,
                                                   dense_backend)
        with torch.no_grad():
            video_v, audio_v = model(traj.clip, inputs.video_rows,
                                     inputs.audio_rows, inputs.timestep,
                                     attention_fn,
                                     tables.get(inputs.timestep.timesteps))
        if isinstance(attention_fn, OracleScorer):
            entries += [ScoreEntry(sample.id, traj.step, layer, table)
                        for layer, table in sorted(
                            attention_fn.scores.items())]
        step_progress.update(
            f'step {traj.step} '
            f'{"scored" if traj.step in steps else "dense"}')
        traj.advance(video_v, audio_v)
    return entries


def run_search(config: SearchConfig) -> None:
    """Scores `num_clips` prompts per geometry (all ranks in lockstep).

    Clip rounds whose score files are already complete on every rank are
    skipped, so an interrupted run resumes where it stopped.
    """
    from miowtion.train import data  # pylint: disable=import-outside-toplevel
    from miowtion.train import parallel  # pylint: disable=import-outside-toplevel
    from miowtion.train import teacher  # pylint: disable=import-outside-toplevel

    env = parallel.init_distributed()
    teacher_ = teacher.build_teacher(
        config.checkpoint_root, config.variant, config.schedule,
        config.num_steps, config.teacher_adapter, env,
        visual_conditions=config.variant == 'Ref2VA'
        or 'fl2va' in config.tasks,
        audio_references=config.variant == 'Ref2VA',
        offload_blocks=config.offload_blocks, prefetch=config.prefetch,
        mlp_chunk_rows=config.mlp_chunk_rows)
    model, schedule, tables = (teacher_.model, teacher_.schedule,
                               teacher_.tables)
    model.dense_backend = config.dense_backend
    geometries = [data.parse_geometry(s) for s in config.geometries]
    candidates = [tiling.TileShape.parse(s) for s in config.candidates]
    veda_config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=config.keep_ratio))
    cache = data.SampleCache(config.sample_cache)
    progress.log(f'tile search over {len(geometries)} geometries: '
                 f'{[g.name for g in geometries]}')
    for geometry in geometries:
        _search_geometry(config, geometry, model, schedule, tables, cache,
                         candidates, veda_config, env)


def _all_ranks(flag: bool, env) -> bool:
    """True iff `flag` is true on every rank."""
    if not env.distributed:
        return flag
    value = torch.tensor([int(flag)], device=env.device)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
    return bool(value.item())


def _search_geometry(config: SearchConfig, geometry, model, schedule, tables,
                     cache, candidates: Sequence[tiling.TileShape],
                     veda_config: veda_attention.VedaConfig, env) -> None:
    """Scores `config.num_clips` cached samples on one geometry."""
    samples = [s for s in cache.select('train', config.tasks)
               if s.aspect in (None, geometry.aspect)
               and s.latent_t in (None, geometry.latent_t)][:config.num_clips]
    if len(samples) < config.num_clips:
        raise ValueError(f'only {len(samples)} samples for {geometry.name}')
    out_dir = os.path.join(config.out_dir, config.run_name, 'scores',
                           geometry.name)
    os.makedirs(out_dir, exist_ok=True)
    per_rank = -(-len(samples) // env.world_size)
    progress.log(f'tile search {geometry.name} grid {geometry.video_grid}: '
                 f'{len(samples)} clips over {env.world_size} ranks, steps '
                 f'{config.steps}, candidates {config.candidates}, keep '
                 f'{config.keep_ratio}, output {out_dir}')
    clips = progress.Progress(f'tile search {geometry.name}: clip rounds',
                              per_rank)
    for i in range(per_rank):
        index = i * env.world_size + env.rank
        # Ranks without a clip re-score clip 0 to stay in lockstep with the
        # sharded forwards of the others, and discard the result.
        sample = samples[index] if index < len(samples) else samples[0]
        path = os.path.join(out_dir, f'{sample.id}.json')
        # Skipping must be collective: a rank that skipped alone would leave
        # the others waiting in the sharded forward.
        done = index >= len(samples) or os.path.exists(path + '.done')
        if _all_ranks(done, env):
            progress.log(f'{geometry.name}: clip round {i} already scored, '
                         'skipped')
            clips.update()
            continue
        entries = score_clip(model, cache, sample, geometry, schedule,
                             config.steps, candidates, veda_config,
                             config.query_tiles, config.seed + index,
                             env.device, tables, config.dense_backend)
        if index < len(samples):
            save_entries(path, entries, candidates, {
                'geometry': geometry.name, 'grid': list(geometry.video_grid),
                'keep_ratio': config.keep_ratio,
                'schedule': config.schedule,
                'num_steps': config.num_steps,
                'teacher_adapter': config.teacher_adapter,
                'query_tiles': config.query_tiles,
                'variant': config.variant})
            progress.log(f'saved {path} ({len(entries)} entries)',
                         all_ranks=True)
        clips.update()
