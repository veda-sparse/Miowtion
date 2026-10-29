"""Compare chunked LoRA numerics and peak temporary CUDA storage."""
import json

import pytest
import torch

from miowtion.train import lora

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason='CUDA required')]


def test_chunked_lora_forward_gradient_and_peak_memory():
    torch.manual_seed(151)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].attn = torch.nn.Module()
    module = torch.nn.Linear(1024, 3072, bias=False, dtype=torch.bfloat16,
                             device='cuda')
    model.blocks[0].attn.out_proj = module
    lora.add_lora(model, 32, alpha=17, layers=[0],
                  targets=['attn.out_proj'])
    model.requires_grad_(False)
    for _, parameter in lora.lora_parameters(model):
        parameter.requires_grad_(True)
        parameter.data.normal_(std=.01)
    source = torch.randn(1025, 1024, device='cuda', dtype=torch.bfloat16)
    upstream = torch.randn(1025, 3072, device='cuda', dtype=torch.bfloat16)
    results = []
    for chunk_rows in (None, 128):
        module.lora_chunk_rows = chunk_rows
        model.zero_grad(set_to_none=True)
        x = source.clone().requires_grad_()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = module(x)
        torch.cuda.synchronize()
        temporary_peak = torch.cuda.max_memory_allocated() - baseline
        output.backward(upstream)
        results.append({
            'output': output.detach().float().cpu(),
            'grads': {'x': x.grad.float().cpu(), **{
                n: p.grad.cpu() for n, p in lora.lora_parameters(model)}},
            'peak': temporary_peak})
        del output, x
    reference, chunked = results
    error = chunked['output'] - reference['output']
    ulp = (torch.nextafter(reference['output'].to(torch.bfloat16),
                           torch.full_like(reference['output'],
                                           float('inf')).to(torch.bfloat16))
           .float() - reference['output']).abs()
    # A different GEMM row count can change FP32 reduction rounding before
    # the BF16 addition. Record it instead of claiming bitwise equivalence.
    assert torch.all(error.abs() <= ulp + 1e-7)
    gradient_errors = {}
    for name, expected in reference['grads'].items():
        actual = chunked['grads'][name]
        relative = ((actual - expected).norm() / expected.norm().clamp_min(
            1e-20)).item()
        assert relative < (1e-2 if name == 'x' else 1e-4), name
        gradient_errors[name] = relative
    assert chunked['peak'] < reference['peak']
    print(json.dumps({
        'forward_max_abs': error.abs().max().item(),
        'forward_relative_mse': (error.square().mean()
                                 / reference['output'].square().mean()).item(),
        'forward_bitwise_equal': torch.equal(
            reference['output'], chunked['output']),
        'gradient_relative_l2': gradient_errors,
        'forward_extra_peak_bytes': {
            'full': reference['peak'], 'chunked': chunked['peak']}}))
