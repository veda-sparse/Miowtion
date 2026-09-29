"""Tests for the end-to-end benchmark aggregation helpers."""

import math

from scripts import bench_e2e


def _sample(value):
    return {
        'kernel_ms': value,
        'prep_ms': value + 1,
        'max_err': 0.1,
        'efficiency': None,
        'efficiency_prep': None,
        'tflops': float('nan'),
        'mfu': 0.5,
        'gemm_frac': 0.2,
    }


def test_aggregate_ignores_missing_and_nonfinite_values():
    got = bench_e2e._aggregate([_sample(1.0), _sample(3.0)])

    assert got['kernel_ms']['median'] == 2.0
    assert got['kernel_ms']['p95'] == 2.9
    assert got['efficiency'] is None
    assert got['tflops'] is None


def test_result_dict_converts_nan_to_json_null():
    # Use the real dataclass so this test also pins the JSON boundary.
    from miowtion.kernels.bench import Result

    payload = bench_e2e._result_dict(Result('x', kernel_ms=math.nan))
    assert payload['kernel_ms'] is None
