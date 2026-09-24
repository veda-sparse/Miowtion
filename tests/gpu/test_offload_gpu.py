"""GPU tests of miowtion.train.parallel.BlockStreamer."""

import copy

import pytest
import torch
from torch import nn

from miowtion.train import parallel

pytestmark = pytest.mark.gpu


class _Stack(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _stack(seed=0):
    torch.manual_seed(seed)
    return _Stack([nn.Sequential(nn.Linear(256, 512), nn.GELU(),
                                 nn.Linear(512, 256)) for _ in range(6)])


@pytest.mark.parametrize('prefetch', [1, 2])
def test_streamed_blocks_match_resident_blocks(prefetch):
    resident = _stack().cuda().requires_grad_(False)
    streamed = copy.deepcopy(resident).cpu()
    offload = {1, 2, 4, 5}
    for i, block in enumerate(streamed.blocks):
        if i not in offload:
            block.cuda()
    parallel.BlockStreamer(streamed.blocks, offload, torch.device('cuda'),
                           prefetch)
    x = torch.randn(64, 256, device='cuda')
    with torch.no_grad():
        for _ in range(3):  # wrap-around prefetch across forwards
            assert torch.equal(streamed(x), resident(x))
    for i in offload:
        for p in streamed.blocks[i].parameters():
            assert p.device.type == 'cpu' and p.is_pinned()


def test_trainable_offloaded_block_is_rejected():
    model = _stack().cpu()
    parallel.BlockStreamer(model.blocks, {0}, torch.device('cuda'))
    for block in list(model.blocks)[1:]:
        block.cuda()
    with pytest.raises(RuntimeError, match='frozen'):
        model(torch.randn(4, 256, device='cuda'))


def test_replica_shares_host_slabs_and_matches():
    resident = _stack().cuda().requires_grad_(False)
    streamed = copy.deepcopy(resident).cpu()
    offload = {0, 2, 3, 5}
    for i, block in enumerate(streamed.blocks):
        if i not in offload:
            block.cuda()
    streamed.block_streamer = parallel.BlockStreamer(
        streamed.blocks, offload, torch.device('cuda'), prefetch=1)
    replica = parallel.replicate(streamed, torch.device('cuda'))
    x = torch.randn(64, 256, device='cuda')
    with torch.no_grad():
        for _ in range(2):
            assert torch.equal(streamed(x), resident(x))
            assert torch.equal(replica(x), resident(x))
    for i in offload:  # one host copy: the replica views the same slab
        for a, b in zip(streamed.blocks[i].parameters(),
                        replica.blocks[i].parameters()):
            assert a.data_ptr() == b.data_ptr() and a.is_pinned()
    assert replica.block_streamer.host is streamed.block_streamer.host
