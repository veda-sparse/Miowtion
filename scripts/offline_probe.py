"""Every remaining GPU-dependent offline measurement, in one model load.

The experiment plan has four items that need real activations but no
generation: the routing decomposition by denoising step and by layer
(P1-3), calibration robustness (P1-4), the sub-tile log-sum-exp readout
(P2-1), and cross-layer selection overlap (P2-3). Run separately each
pays the same 8 minutes of adapter merge and AdaLN tabulation, so they
are all driven from one teacher here.

Nothing in this script generates video; it hooks the dense teacher,
captures the tile-ordered q/k of every (layer, head group), and reduces.

Example:
    python scripts/offline_probe.py --config configs/offline_probe_t37.yaml
"""

import argparse
import collections
import dataclasses
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.h3 import schedule as h3_schedule           # noqa: E402
from miowtion.train import data                           # noqa: E402
from miowtion.train import parallel                       # noqa: E402
from miowtion.train import teacher as teacher_lib         # noqa: E402
from miowtion.train import trajectory as traj_lib         # noqa: E402
from miowtion.utils import progress                       # noqa: E402
from miowtion.veda import attention as veda_attention     # noqa: E402
from miowtion.veda import bundle as veda_bundle           # noqa: E402
from miowtion.veda import heatmap                         # noqa: E402
from miowtion.veda import mask as veda_mask               # noqa: E402
from miowtion.veda import plan as veda_plan               # noqa: E402
from miowtion.veda import predictor as veda_predictor     # noqa: E402
from miowtion.veda import solattn                         # noqa: E402


@dataclasses.dataclass
class ProbeConfig:
    """Launch parameters of an offline probe run.

    A dataclass rather than a raw dict so that the configs/ guard can
    reject a typo'd key before a run pays eight minutes of model load to
    find out (tests/unit/test_configs.py).
    """

    root: str
    variant: str
    schedule: str
    num_steps: int
    adapter: str
    sample_cache: str
    sample_id: list[str]
    geometry: list[str]
    predictor: str
    seed: int = 0
    keep_ratio: float = 0.05
    offload_blocks: int = 30
    prefetch: int = 1
    mlp_chunk_rows: int = 8192
    subtile_m: tuple[int, ...] = (1, 2, 4)
    layer_every: int = 10
    row_stride: int = 7
    heads_per_group: int = 8

    @classmethod
    def from_yaml(cls, path: str) -> 'ProbeConfig':
        """Loads a config, rejecting unknown keys.

        Raises:
            ValueError: On a key that is not a field.
        """
        with open(path) as handle:
            raw = yaml.safe_load(handle)
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f'unknown config keys {sorted(unknown)}')
        return cls(**raw)


class _Probe:
    """Dense attention that also records what a router would have done."""

    def __init__(self, clip, plan, predictor, step, out):
        self.clip = clip
        self.plan = plan
        self.predictor = predictor
        self.step = step
        self.out = out
        # layer -> the selected (head, row, tile) set, for P2-3.
        self.selection = {}

    def __call__(self, q, k, v, layer_index):
        from miowtion.h3 import attention as h3_attention
        used = self.clip.layout.used
        dense, lse = h3_attention.dense_attention(q, k, v, used,
                                                  return_lse=True)
        with torch.no_grad():
            self._record(q, k, lse, layer_index)
        return dense

    def _record(self, q, k, lse, layer_index):
        want = self.predictor.second_order_rank
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            heads = group.heads[:8]           # a fixed subset bounds cost
            if heads.numel() == 0:
                continue
            q_t, feats_q, sq_q = (
                veda_attention._gather_and_pool(
                    q, tile_layout, heads, veda_predictor.SECOND_RAW)
                if want else
                (*veda_attention._gather_and_pool(q, tile_layout, heads),
                 None))
            k_t, feats_k, var_k = (
                veda_attention._gather_and_pool(
                    k, tile_layout, heads, veda_predictor.SECOND_CENTRAL)
                if want else
                (*veda_attention._gather_and_pool(k, tile_layout, heads),
                 None))
            rows = torch.arange(tile_layout.n_video_tiles,
                                device=q.device)[::7]       # subsample rows
            heat = heatmap.teacher_heat(
                q_t, k_t,
                veda_attention._gather_lse(lse, tile_layout, heads),
                tile_layout, rows, 'sum')
            logits = self.predictor.layers[layer_index](
                feats_q, feats_k, heads,
                **veda_attention._extra_features(
                    self.predictor, tile_layout, sq_q, var_k))
            blocks = self.clip.blocks(tile_layout)
            got = heatmap.mask_diagnostics(
                logits[:, rows], heat, tile_layout, blocks, rows)
            self.out.append({
                'step': self.step, 'layer': layer_index,
                'shape': f'{group.shape.t}x{group.shape.h}x{group.shape.w}',
                'heads': int(heads.numel()),
                'recall': float(got['recall']),
                'retained': float(got['retained']),
                'heat_kept': float(got['heat_kept']),
                'heat_ceiling': float(got['heat_ceiling']),
            })
            # P2-1: how much of the gap to the true block mass a
            # log-sum-exp over m x m sub-blocks closes. m = 1 is the
            # present model exactly, so the m = 1 row is the control.
            if layer_index % 10 == 0:
                truth = torch.log(heat.clamp(min=1e-12))
                entry = self.out[-1]
                for m in (1, 2, 4):
                    got = solattn.subtile_log_mass(q_t, k_t, tile_layout, m)
                    entry[f'subtile_m{m}_spearman'] = solattn.rank_correlation(
                        got[:, rows], truth)
                entry['predictor_spearman'] = solattn.rank_correlation(
                    logits[:, rows], truth)

            # P2-3: the predicted set, so adjacent layers can be compared.
            sel = veda_mask.select_video_blocks(
                logits[:, rows, :tile_layout.n_video_tiles], tile_layout,
                blocks, rows)
            dense_sel = torch.zeros(*sel.index.shape[:2],
                                    tile_layout.n_video_tiles,
                                    dtype=torch.bool, device=q.device)
            dense_sel.scatter_(2, sel.index, sel.keep)
            self.selection[(layer_index, str(group.shape))] = dense_sel.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = ProbeConfig.from_yaml(args.config)

    env = parallel.init_distributed()
    tch = teacher_lib.build_teacher(
        cfg.root, cfg.variant, cfg.schedule, cfg.num_steps,
        cfg.adapter, env, visual_conditions=False, audio_references=False,
        offload_blocks=cfg.offload_blocks, prefetch=cfg.prefetch,
        mlp_chunk_rows=cfg.mlp_chunk_rows)
    loaded = veda_bundle.load(cfg.predictor)
    predictor = loaded.predictor.to(env.device).eval()
    cache = data.SampleCache(cfg.sample_cache)
    rows, overlaps = [], []
    for sample_id in cfg.sample_id:
        sample = cache.sample(sample_id)
        for geometry in cfg.geometry:
            geo = data.parse_geometry(geometry)
            traj = traj_lib.Trajectory(tch.model, cache, sample, geo,
                                       tch.schedule, cfg.seed,
                                       env.device)
            plan = loaded.plans.select(geo)
            veda_cfg = veda_attention.VedaConfig(
                keep_ratio=cfg.keep_ratio)
            clip = veda_attention.ClipTiling(traj.layout, veda_cfg,
                                             env.device)
            steps = progress.Progress(f'probe {sample_id} {geometry}',
                                      tch.schedule.num_steps, every=1)
            while not traj.done:
                inputs = traj.inputs()
                probe = _Probe(clip, plan, predictor, inputs.step, rows)
                with torch.no_grad():
                    vv, av = tch.model(
                        traj.clip, inputs.video_rows, inputs.audio_rows,
                        inputs.timestep, probe,
                        tch.tables.get(inputs.timestep.timesteps))
                # P2-3: overlap of the selected sets of adjacent layers.
                keys = sorted(probe.selection, key=lambda x: (x[1], x[0]))
                by_shape = collections.defaultdict(list)
                for layer, shape in keys:
                    by_shape[shape].append((layer,
                                            probe.selection[(layer, shape)]))
                for shape, items in by_shape.items():
                    for (l0, a), (l1, b) in zip(items, items[1:]):
                        if a.shape != b.shape:
                            continue
                        inter = (a & b).sum().item()
                        union = a.sum().item()
                        overlaps.append({
                            'sample': sample_id, 'geometry': geometry,
                            'step': inputs.step, 'shape': shape,
                            'layer_a': l0, 'layer_b': l1,
                            'overlap': inter / max(1, union)})
                traj.advance(vv, av)
                steps.update(f'step {inputs.step}')
    json.dump({'config': dataclasses.asdict(cfg), 'per_call': rows, 'layer_overlap': overlaps},
              open(args.out, 'w'), indent=1)
    progress.log(f'wrote {args.out}: {len(rows)} calls, '
                 f'{len(overlaps)} layer pairs')


if __name__ == '__main__':
    main()
