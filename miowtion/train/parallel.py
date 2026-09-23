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

import dataclasses
import datetime
import os
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
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


def build_model(transformer_dir: str, env: DistEnv, drop_adaln: bool,
                offload_blocks: int = 0, prefetch: int = 1,
                mlp_chunk_rows: int | None = None,
                before_shard: Callable[[h3_model.H3DiT], object] | None = None
                ) -> h3_model.H3DiT:
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
    if env.mesh is not None:
        shard_trunk(model, env.mesh, offload, prefetch)
    elif offload:
        raise ValueError('block offloading needs the FSDP (torchrun) path')
    # Offloaded FSDP parameters must be materialized on the host; everything
    # else lives on the device.
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
                                    skip_prefixes=skip)
    model.set_mlp_chunk_rows(mlp_chunk_rows)
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
