"""Tests for miowtion.veda.bundle."""

import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miowtion.h3 import geometry as h3_geometry
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

_LAYERS, _HEADS, _DIM = 2, 3, 8


def _plans() -> veda_plan.PlanTable:
    geo = h3_geometry.geometry_from_latent_t('16:9', 7)
    plan = veda_plan.TilePlan.uniform(geo, tiling.TileShape(2, 4, 16),
                                      _LAYERS, _HEADS)
    return veda_plan.PlanTable([plan, plan.mirrored()])


def _write(tmp_path, **overrides) -> tuple[str, dict]:
    os.makedirs(tmp_path, exist_ok=True)
    model = veda_predictor.TileScorePredictor(_LAYERS, _HEADS, _DIM)
    weights = {f'predictor.{k}': v for k, v in model.state_dict().items()}
    path = str(tmp_path / 'bundle.safetensors')
    kwargs = dict(num_layers=_LAYERS, num_heads=_HEADS, head_dim=_DIM,
                  keep_ratio=0.1, source='runs/x/ckpt/step_0000600',
                  source_weights='live', step=600)
    kwargs.update(overrides)
    veda_bundle.save(path, weights, _plans(), **kwargs)
    return path, model.state_dict()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_round_trip_is_bit_wise_exact(tmp_path, dtype):
    """Storage may round, but a load must reproduce the stored bits."""
    path, state = _write(tmp_path, dtype=dtype)
    loaded = veda_bundle.load(path)
    got = loaded.predictor.state_dict()
    assert sorted(got) == sorted(state)
    for name, value in state.items():
        assert got[name].dtype == dtype
        assert torch.equal(got[name], value.to(dtype))
    assert loaded.keep_ratio == 0.1
    assert loaded.metadata['source_weights'] == 'live'
    assert loaded.metadata['step'] == '600'


def test_bf16_is_the_default_and_is_not_upcast_on_load(tmp_path):
    """load_state_dict copies into the parameter, so an fp32 module would
    swallow a bf16 file and hold twice the memory it was exported to save."""
    path, _ = _write(tmp_path)
    assert veda_bundle.read_metadata(path)['dtype'] == 'bfloat16'
    for value in veda_bundle.load(path).predictor.parameters():
        assert value.dtype == torch.bfloat16


def test_a_bundle_without_a_dtype_is_read_as_fp32(tmp_path):
    """Bundles written before the dtype was recorded are fp32."""
    path, state = _write(tmp_path, dtype=torch.float32)
    with safe_open(path, framework='pt', device='cpu') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        metadata = dict(f.metadata())
    del metadata['dtype']
    save_file(tensors, path, metadata=metadata)
    got = veda_bundle.load(path).predictor.state_dict()
    for name, value in state.items():
        assert torch.equal(got[name], value)


def test_an_unsupported_storage_dtype_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='unsupported storage dtype'):
        _write(tmp_path, dtype=torch.float16)


def test_bf16_storage_keeps_the_block_ranking(tmp_path):
    """What the kernel reads is the top-k, not the logits: bf16 rounding of
    the projection must not reorder which blocks a query tile keeps."""
    torch.manual_seed(0)
    fp32, _ = _write(tmp_path / 'a', dtype=torch.float32)
    bf16, _ = _write(tmp_path / 'b', dtype=torch.bfloat16)
    # Same weights in both: _write rebuilds the model, so seed each save.
    torch.manual_seed(0)
    model = veda_predictor.TileScorePredictor(_LAYERS, _HEADS, _DIM)
    weights = {f'predictor.{k}': v for k, v in model.state_dict().items()}
    kwargs = dict(num_layers=_LAYERS, num_heads=_HEADS, head_dim=_DIM,
                  keep_ratio=0.1, source='s', source_weights='live', step=0)
    veda_bundle.save(fp32, weights, _plans(), dtype=torch.float32, **kwargs)
    veda_bundle.save(bf16, weights, _plans(), dtype=torch.bfloat16, **kwargs)

    n_tiles = 64
    feats_q = torch.randn(_HEADS, n_tiles, 3 * _DIM)
    feats_k = torch.randn(_HEADS, n_tiles, 3 * _DIM)
    heads = torch.arange(_HEADS)
    a = veda_bundle.load(fp32).predictor.layers[0](feats_q, feats_k, heads)
    b = veda_bundle.load(bf16).predictor.layers[0](feats_q, feats_k, heads)
    assert not torch.equal(a, b), 'bf16 rounded nothing; test is vacuous'
    keep = max(1, n_tiles // 10)
    top_a = a.topk(keep, dim=-1).indices.sort(-1).values
    top_b = b.topk(keep, dim=-1).indices.sort(-1).values
    assert torch.equal(top_a, top_b)


def test_plans_travel_with_the_weights(tmp_path):
    """The point of the bundle: tile shapes cannot be lost or mispaired."""
    path, _ = _write(tmp_path)
    plans = veda_bundle.load(path).plans
    assert sorted(plans.plans) == ['16x9_t7', '9x16_t7']
    for name, want in _plans().plans.items():
        got = plans.plans[name]
        assert got.grid == want.grid
        assert got.shapes == want.shapes
        assert got.head_shape == want.head_shape
    # And they are selectable by geometry, like a plan directory.
    geo = h3_geometry.geometry_from_latent_t('9:16', 7)
    assert plans.select(geo).geometry == '9x16_t7'


def test_read_metadata_does_not_need_the_tensors(tmp_path):
    path, _ = _write(tmp_path)
    assert veda_bundle.read_metadata(path)['keep_ratio'] == '0.1'


def test_a_foreign_safetensors_file_is_rejected(tmp_path):
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
    path = str(tmp_path / 'other.safetensors')
    save_file({'x': torch.zeros(2)}, path)
    with pytest.raises(ValueError, match='not a miowtion-veda-predictor'):
        veda_bundle.load(path)


def test_mismatched_shape_metadata_is_rejected(tmp_path):
    path, _ = _write(tmp_path, num_heads=_HEADS + 1)
    with pytest.raises(ValueError, match='do not match the'):
        veda_bundle.load(path)


def test_empty_weights_and_empty_plans_are_rejected(tmp_path):
    path = str(tmp_path / 'b.safetensors')
    with pytest.raises(ValueError, match='no predictor weights'):
        veda_bundle.save(path, {}, _plans(), num_layers=1, num_heads=1,
                         head_dim=1, keep_ratio=0.1, source='x',
                         source_weights='live', step=0)
    model = veda_predictor.TileScorePredictor(_LAYERS, _HEADS, _DIM)
    with pytest.raises(ValueError, match='without plans'):
        veda_bundle.save(path, dict(model.state_dict()),
                         veda_plan.PlanTable([]), num_layers=_LAYERS,
                         num_heads=_HEADS, head_dim=_DIM, keep_ratio=0.1,
                         source='x', source_weights='live', step=0)


def test_fp8_round_trip_reproduces_the_stored_values(tmp_path):
    """fp8 loads back as bf16, exactly the dequantized stored bits."""
    path, state = _write(tmp_path, dtype=torch.float8_e4m3fn)
    got = veda_bundle.load(path).predictor.state_dict()
    with safe_open(path, framework='pt', device='cpu') as f:
        keys = sorted(f.keys())
        stored = {k: f.get_tensor(k) for k in keys}
    assert sorted(state) == [k for k in keys if not k.endswith('.__scale')]
    for name, value in state.items():
        assert stored[name].dtype == torch.float8_e4m3fn
        assert got[name].dtype == torch.bfloat16  # bmm has no e4m3 path
        expect = veda_bundle._dequantize_fp8(
            stored[name], stored[name + '.__scale'], torch.bfloat16)
        assert torch.equal(got[name], expect)
        # A per-head amax scale: 3 mantissa bits of the head's own range.
        rel = (got[name].float() - value).abs().amax(dim=(1, 2))
        assert (rel <= 2.0 ** -4 * value.abs().amax(dim=(1, 2))).all()


def test_fp8_halves_the_file_against_bf16(tmp_path):
    fp8, _ = _write(tmp_path / 'a', dtype=torch.float8_e4m3fn)
    bf16, _ = _write(tmp_path / 'b', dtype=torch.bfloat16)
    # The per-head scales and the json header ride along and dominate at
    # this toy size, so the ratio is well above the asymptotic 0.5.
    assert os.path.getsize(fp8) < 0.75 * os.path.getsize(bf16)


def test_fp8_scale_is_per_head_and_maps_the_amax_onto_the_format():
    weight = torch.randn(_HEADS, 3 * _DIM, _DIM)
    weight[1] *= 1000.0  # one head far out of the others' range
    values, scale = veda_bundle._quantize_fp8(weight)
    assert scale.shape == (_HEADS,)
    back = veda_bundle._dequantize_fp8(values, scale, torch.float32)
    assert values.float().abs().amax().item() == pytest.approx(448.0, rel=0.3)
    for head in range(_HEADS):
        amax = weight[head].abs().amax()
        assert (back[head] - weight[head]).abs().max() <= 2.0 ** -4 * amax


def test_fp8_keeps_an_all_zero_head_at_zero():
    weight = torch.randn(_HEADS, 3 * _DIM, _DIM)
    weight[0] = 0.0
    values, scale = veda_bundle._quantize_fp8(weight)
    back = veda_bundle._dequantize_fp8(values, scale, torch.float32)
    assert torch.equal(back[0], torch.zeros_like(back[0]))
    assert not back.isnan().any()


def test_fp8_bundle_without_a_scale_is_rejected(tmp_path):
    path, _ = _write(tmp_path, dtype=torch.float8_e4m3fn)
    with safe_open(path, framework='pt', device='cpu') as f:
        metadata = dict(f.metadata())
        tensors = {k: f.get_tensor(k) for k in f.keys()
                   if not k.endswith('.__scale')}
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError, match='without a .* scale'):
        veda_bundle.load(path)
