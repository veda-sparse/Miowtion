"""Tests for miowtion.h3.synthetic."""

import os

import pytest
import torch

from miowtion.h3 import geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule as h3_schedule
from miowtion.h3 import synthetic
from miowtion.h3 import weights
from miowtion.train import parallel
from miowtion.train import teacher

from tests.unit import test_train_pipeline as pipeline_test

_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party',
                     'MiniMax-H3')


def _tdir(root):
    return os.path.join(root, 'FL2VA', 'transformer')


def test_matches_release_names_shapes_and_dtypes(tmp_path):
    root = str(tmp_path)
    pipeline_test._write_release(root)
    real = weights.Checkpoint(_tdir(root))
    fake = synthetic.RandomCheckpoint(_tdir(root))
    assert fake.keys() == real.keys()
    for key in sorted(real.keys()):
        expected = real.read_rows(key, None)
        value = fake.read_rows(key, None)
        assert value.shape == expected.shape, key
        assert value.dtype == expected.dtype, key
    assert torch.equal(fake.read_rows('rope.inv_freq', None),
                       real.read_rows('rope.inv_freq', None))


def test_rows_and_seed_are_deterministic(tmp_path):
    root = str(tmp_path)
    pipeline_test._write_release(root)
    key = 'blocks.1.attn.qkv_proj.weight'
    fake = synthetic.RandomCheckpoint(_tdir(root), seed=3)
    full = fake.read_rows(key, None)
    rows = torch.tensor([5, 0, 7])
    assert torch.equal(fake.read_rows(key, rows), full[rows])
    again = synthetic.RandomCheckpoint(_tdir(root), seed=3)
    assert torch.equal(again.read_rows(key, None), full)
    other = synthetic.RandomCheckpoint(_tdir(root), seed=4)
    assert not torch.equal(other.read_rows(key, None), full)


def test_random_teacher_runs_finite(tmp_path):
    root = str(tmp_path)
    pipeline_test._write_release(root)
    env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
    tch = teacher.build_teacher(root, 'FL2VA', 'turbo', 4, None, env,
                                visual_conditions=False,
                                audio_references=False,
                                random_weights_seed=0)
    tch.model.dense_backend = 'math'
    cfg = tch.model.config
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = h3_layout.pack(torch.ones(12, dtype=torch.long), geo)
    state = h3_schedule.build_timestep_state(lay, *tch.schedule.timesteps(0))
    with torch.no_grad():
        video, audio = tch.model(
            tch.model.clip_inputs(lay, tch.model.refine_text(
                torch.randn(12, cfg.text_dim)), torch.device('cpu')),
            torch.randn(geo.num_video_tokens, 96),
            torch.randn(geo.num_audio_rows, 32), state,
            adaln_table=tch.tables.get(state.timesteps))
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_random_teacher_rejects_adapter(tmp_path):
    root = str(tmp_path)
    pipeline_test._write_release(root)
    env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
    with pytest.raises(ValueError):
        teacher.build_teacher(root, 'FL2VA', 'turbo', 4, 'lora.safetensors',
                              env, visual_conditions=False,
                              audio_references=False, random_weights_seed=0)


@pytest.mark.parametrize('variant', ['FL2VA', 'Ref2VA'])
def test_shapes_the_full_release(variant):
    """The submodule's index alone is enough to shape all 33B parameters."""
    fake = synthetic.RandomCheckpoint(
        os.path.join(_REPO, variant, 'transformer'))
    with torch.device('meta'):
        model = h3_model.H3DiT(fake.config)
    expected = set(dict(model.named_parameters())) | {'rope.inv_freq'}
    assert fake.keys() == expected
    assert 32e9 < fake.num_parameters() < 34e9
