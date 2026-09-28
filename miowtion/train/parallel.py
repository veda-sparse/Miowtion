"""Process groups, FSDP2 sharding of the trunk and replicated-gradient sync.

Parallelism is data parallel only: one clip per rank, one attention document
per clip. Sequence parallelism is not supported because tile permutations
need every row of a sequence on one device.

  * Trunk blocks are FSDP2-sharded one unit per block (parameter dtypes
    unchanged, fp32 gradient reduction, reshard after forward). Across nodes
    the mesh is HSDP: shard inside a node, replicate across nodes.
  * The embedding / output parts (patch projections, condition projection,
    refiner, time embedder, final layer; 0.84B, mixed precision) and the
    predictor are replicated; their gradients are all-reduced manually. The
    predictor must not be sharded: its dim 0 is the head axis and per-head
    indexing of a sharded DTensor does not work.
"""

from __future__ import annotations

import collections
import copy as copy_lib
import dataclasses
import datetime
import functools
import os
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import device_mesh as dm
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import fully_shard

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.h3 import weights as h3_weights
from miowtion.utils import progress


@dataclasses.dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    mesh: dm.DeviceMesh | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(timeout_minutes: int = 60) -> DistEnv:
    """Initializes NCCL from torchrun variables (or a single process)."""
    if 'RANK' not in os.environ:
        device = torch.device('cuda', 0) if torch.cuda.is_available() else (
            torch.device('cpu'))
        return DistEnv(0, 1, 0, device, None)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        'nccl', timeout=datetime.timedelta(minutes=timeout_minutes))
    world = dist.get_world_size()
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', world))
    nodes = world // local_world
    if nodes > 1:
        mesh = dm.init_device_mesh('cuda', (nodes, local_world),
                                   mesh_dim_names=('replicate', 'shard'))
    else:
        mesh = dm.init_device_mesh('cuda', (world,),
                                   mesh_dim_names=('shard',))
    return DistEnv(dist.get_rank(), world, local_rank,
                   torch.device('cuda', local_rank), mesh)


def offloaded_blocks(num_layers: int, count: int) -> set[int]:
    """`count` trunk blocks spread evenly over the stack (Bresenham).

    Interleaving offloaded and resident blocks lets the host-to-device copy
    of an offloaded block overlap the compute of its resident neighbour
    instead of stalling a run of consecutive offloaded blocks.
    """
    if not 0 <= count <= num_layers:
        raise ValueError(f'offload count {count} not in [0, {num_layers}]')
    ratio = count / num_layers
    return {i for i in range(num_layers)
            if int((i + 1) * ratio) > int(i * ratio)}


def shard_trunk(model: h3_model.H3DiT, mesh: dm.DeviceMesh,
                offload: set[int] = frozenset(), prefetch: int = 1) -> None:
    """Applies fully_shard to every trunk block (and its AdaLN projection).

    Args:
        model: The meta-initialized model.
        mesh: Device mesh (1D shard or 2D HSDP).
        offload: Blocks whose sharded parameters live in pinned host memory.
        prefetch: Blocks all-gathered ahead of the running one.
    """
    policy = MixedPrecisionPolicy(param_dtype=None,
                                  reduce_dtype=torch.float32)
    for index, block in enumerate(model.blocks):
        kwargs = {'mesh': mesh, 'mp_policy': policy,
                  'reshard_after_forward': True}
        if index in offload:
            kwargs['offload_policy'] = CPUOffloadPolicy(pin_memory=True)
        if block.adaln_proj is not None:
            # Its own unit, so it is never gathered when tables are used.
            fully_shard(block.adaln_proj, **kwargs)
        fully_shard(block, **kwargs)
    # The root must be an FSDP module too: it owns the shared communication
    # streams that cross-block prefetching needs. Parameters outside the
    # trunk are ignored, i.e. stay plain replicated tensors.
    fully_shard(model, mesh=mesh, mp_policy=policy,
                ignored_params=set(replicated_parameters(model)))
    blocks = list(model.blocks)
    for index, block in enumerate(blocks):
        ahead = blocks[index + 1:index + 1 + prefetch]
        if ahead:
            block.set_modules_to_forward_prefetch(ahead)


class HostSlabs:
    """Offloaded blocks' parameters packed into pinned host slabs.

    One contiguous pinned buffer per (block, dtype): a block reaches the GPU
    with one DMA per dtype instead of one copy per parameter. The parameters
    become views into their slab, so the host holds exactly one copy, which
    any number of devices (BlockStreamer per GPU) can stream from.
    """

    def __init__(self, blocks: Sequence[nn.Module], offload: set[int]):
        self.order = sorted(offload)
        self.slabs: dict[int, dict[torch.dtype, torch.Tensor]] = {}
        # Per block: (name, dtype, offset, shape) of every parameter.
        self.entries: dict[int, list[tuple[str, torch.dtype, int,
                                           torch.Size]]] = {}
        self.max_numel: dict[torch.dtype, int] = {}
        for i in self.order:
            named = list(blocks[i].named_parameters())
            sizes: dict[torch.dtype, int] = {}
            entries = []
            for name, p in named:
                if p.device.type != 'cpu':
                    raise ValueError(f'block {i} is not on the host')
                entries.append((name, p.dtype, sizes.get(p.dtype, 0),
                                p.shape))
                sizes[p.dtype] = sizes.get(p.dtype, 0) + p.numel()
            slabs = {dt: torch.empty(n, dtype=dt, pin_memory=True)
                     for dt, n in sizes.items()}
            for (name, dt, offset, shape), (_, p) in zip(entries, named):
                view = slabs[dt][offset:offset + p.numel()].view(shape)
                view.copy_(p.data)
                p.data = view
            self.slabs[i] = slabs
            self.entries[i] = entries
            for dt, n in sizes.items():
                self.max_numel[dt] = max(self.max_numel.get(dt, 0), n)

    def host_views(self, index: int) -> dict[str, torch.Tensor]:
        """Parameter name -> its view in the block's host slab."""
        return {name: self.slabs[index][dt][offset:offset + shape.numel()]
                .view(shape)
                for name, dt, offset, shape in self.entries[index]}


class BlockStreamer:
    """Streams CPU-offloaded blocks to one GPU ahead of use.

    FSDP2 skips its all-gather path at world size 1: its prefetch does
    nothing and an offloaded block is copied to the GPU on the compute
    stream when the block starts (2.4 s of a 16 s denoising step on RTX
    4090). Single-process runs therefore keep offloaded blocks as plain
    modules whose parameters are views into pinned HostSlabs. A forward
    pre-hook on block i waits for the copy event of its own weights, then
    issues the copies of the next `prefetch` offloaded blocks (wrapping around
    to the next forward) on a dedicated stream, one DMA per slab into a ring
    of `prefetch + 1` preallocated device buffers, so copies overlap compute
    and the caching allocator never churns. A ring slot is reused only after
    the compute stream has finished the block that used it.

    Several streamers (one per GPU, each on its own module replica) can share
    one HostSlabs. Offloaded blocks must be frozen.
    """

    def __init__(self, blocks: Sequence[nn.Module], offload: set[int],
                 device: torch.device, prefetch: int = 1,
                 host: HostSlabs | None = None):
        if prefetch < 1:
            raise ValueError(f'prefetch must be >= 1, got {prefetch}')
        self.device = device
        self.prefetch = prefetch
        self.order = sorted(offload)
        self.host = host if host is not None else HostSlabs(blocks, offload)
        if self.host.order != self.order:
            raise ValueError('HostSlabs cover other blocks')
        self.stream = torch.cuda.Stream(device)
        self.params = {i: dict(blocks[i].named_parameters())
                       for i in self.order}
        for i in self.order:
            for name, view in self.host.host_views(i).items():
                self.params[i][name].data = view
        slots = prefetch + 1
        self.ring = [{dt: torch.empty(n, dtype=dt, device=device)
                      for dt, n in self.host.max_numel.items()}
                     for _ in range(slots)]
        self._released: list[torch.cuda.Event | None] = [None] * slots
        self._inflight: dict[int, tuple[int, torch.cuda.Event]] = {}
        self._running: dict[int, int] = {}
        for i in self.order:
            blocks[i].register_forward_pre_hook(
                functools.partial(self._before, i))
            blocks[i].register_forward_hook(functools.partial(self._after, i))

    def _free_slot(self) -> int:
        busy = set(self._running.values()) | {
            slot for slot, _ in self._inflight.values()}
        for slot in range(len(self.ring)):
            if slot not in busy:
                return slot
        raise RuntimeError('no free ring slot (prefetch accounting bug)')

    def _fetch(self, index: int) -> None:
        if index in self._inflight or index in self._running:
            return
        slot = self._free_slot()
        with torch.cuda.stream(self.stream):
            if self._released[slot] is not None:
                self.stream.wait_event(self._released[slot])
            for dt, host in self.host.slabs[index].items():
                self.ring[slot][dt][:host.numel()].copy_(host,
                                                         non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self._inflight[index] = (slot, event)

    def _following(self, index: int) -> list[int]:
        k = self.order.index(index)
        count = min(self.prefetch, len(self.order) - 1)
        return [self.order[(k + d) % len(self.order)]
                for d in range(1, count + 1)]

    def _before(self, index: int, module: nn.Module, args) -> None:
        del module, args
        params = self.params[index]
        if torch.is_grad_enabled() and any(p.requires_grad
                                           for p in params.values()):
            raise RuntimeError(f'offloaded block {index} must be frozen')
        self._fetch(index)
        slot, event = self._inflight.pop(index)
        torch.cuda.current_stream(self.device).wait_event(event)
        for name, dt, offset, shape in self.host.entries[index]:
            params[name].data = self.ring[slot][dt][
                offset:offset + shape.numel()].view(shape)
        self._running[index] = slot
        for following in self._following(index):
            self._fetch(following)

    def _after(self, index: int, module: nn.Module, args, output) -> None:
        del module, args, output
        slot = self._running.pop(index)
        released = torch.cuda.Event()
        released.record(torch.cuda.current_stream(self.device))
        self._released[slot] = released
        for name, view in self.host.host_views(index).items():
            self.params[index][name].data = view


def replicate(model: h3_model.H3DiT, device: torch.device,
              prefetch: int = 1) -> h3_model.H3DiT:
    """A copy of a single-process model on another GPU.

    Resident parameters and buffers are copied to `device`; offloaded blocks
    keep pointing at the source model's pinned HostSlabs (no second host
    copy) and get their own BlockStreamer on `device`. The source must not
    be running a forward.
    """
    source = getattr(model, 'block_streamer', None)
    offload = set(source.order) if source is not None else set()
    offloaded = {id(p) for i in offload for p in model.blocks[i].parameters()}
    memo = {}
    for p in model.parameters():
        data = p.data if id(p) in offloaded else p.data.to(device)
        memo[id(p)] = nn.Parameter(data, requires_grad=p.requires_grad)
    for b in model.buffers():
        memo[id(b)] = b.to(device)
    # The streamer's hooks and stream belong to the source device: detach
    # them for the copy.
    hooks = [(block._forward_pre_hooks, block._forward_hooks)  # pylint: disable=protected-access
             for block in model.blocks]
    for block in model.blocks:
        block._forward_pre_hooks = collections.OrderedDict()  # pylint: disable=protected-access
        block._forward_hooks = collections.OrderedDict()  # pylint: disable=protected-access
    if source is not None:
        del model.block_streamer
    try:
        copy = copy_lib.deepcopy(model, memo)
    finally:
        for block, (pre, post) in zip(model.blocks, hooks):
            block._forward_pre_hooks = pre  # pylint: disable=protected-access
            block._forward_hooks = post  # pylint: disable=protected-access
        if source is not None:
            model.block_streamer = source
    if source is not None:
        copy.block_streamer = BlockStreamer(copy.blocks, offload, device,
                                            prefetch, host=source.host)
    return copy


def build_model(transformer_dir: str, env: DistEnv, drop_adaln: bool,
                offload_blocks: int = 0, prefetch: int = 1,
                mlp_chunk_rows: int | None = None,
                before_shard: Callable[[h3_model.H3DiT], object] | None = None,
                checkpoint=None) -> h3_model.H3DiT:
    """Meta-initializes, shards and loads the DiT shard-locally.

    Args:
        transformer_dir: `<root>/<FL2VA|Ref2VA>/transformer`.
        env: Distributed environment.
        drop_adaln: Remove per-block AdaLN projections (frozen trunk with
            precomputed tables, see miowtion.train.adaln).
        offload_blocks: Trunk blocks kept in pinned host memory (0 = none,
            num_layers = all); see offloaded_blocks for the placement.
        prefetch: Blocks all-gathered ahead during forward.
        mlp_chunk_rows: Row chunking of the MLP intermediate.
        before_shard: Called on the meta model before sharding (e.g. to add
            LoRA parameters, which must shard with their block). Parameters
            it creates are uninitialized afterwards and must be reset.
        checkpoint: Tensor source (see h3_weights.load_dit_weights);
            defaults to the safetensors under `transformer_dir`.

    Returns:
        The model on env.device, trunk sharded when distributed.
    """
    config = h3_config.H3Config.from_pretrained(transformer_dir)
    with torch.device('meta'):
        model = h3_model.H3DiT(config)
    if drop_adaln:
        model.drop_adaln_projections()
    if before_shard is not None:
        before_shard(model)
    offload = offloaded_blocks(config.num_layers, offload_blocks)
    progress.log(f'build DiT: {config.num_layers} layers, world '
                 f'{env.world_size}, offload {len(offload)} blocks, prefetch '
                 f'{prefetch}, drop_adaln={drop_adaln}')
    # FSDP only across several ranks; a single process streams offloaded
    # blocks itself (see BlockStreamer).
    use_fsdp = env.mesh is not None and env.world_size > 1
    if use_fsdp:
        shard_trunk(model, env.mesh, offload, prefetch)
    elif offload and env.device.type != 'cuda':
        raise ValueError('block offloading needs a CUDA device')
    # Offloaded parameters are materialized on the host; everything else
    # lives on the device.
    for name, child in model.named_children():
        if name != 'blocks':
            child.to_empty(device=env.device)
    for index, block in enumerate(model.blocks):
        block.to_empty(device='cpu' if index in offload else env.device)
    skip = ()
    if drop_adaln:
        skip = tuple(f'blocks.{i}.adaln_proj.'
                     for i in range(config.num_layers))
    with progress.Timer(f'load weights from {transformer_dir}'):
        h3_weights.load_dit_weights(model, transformer_dir,
                                    skip_prefixes=skip,
                                    checkpoint=checkpoint)
    model.set_mlp_chunk_rows(mlp_chunk_rows)
    if offload and not use_fsdp:
        model.block_streamer = BlockStreamer(model.blocks, offload,
                                             env.device, prefetch)
    return model


def replicated_parameters(model: h3_model.H3DiT) -> list[torch.nn.Parameter]:
    """Parameters outside the sharded trunk."""
    return [p for name, p in model.named_parameters()
            if not name.startswith('blocks.')]


@torch.no_grad()
def all_reduce_gradients(params: Iterable[torch.nn.Parameter],
                         env: DistEnv) -> None:
    """Averages gradients of replicated parameters over all ranks.

    Gradients are flattened into one fp32 buffer so a single collective runs
    regardless of the parameter count. Missing gradients are zero-filled:
    every rank must contribute a buffer of the same layout, or the
    collective hangs.
    """
    if not env.distributed:
        return
    params = [p for p in params if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= env.world_size
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def broadcast_object(obj, env: DistEnv):
    """Broadcasts a picklable object from rank 0."""
    if not env.distributed:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
