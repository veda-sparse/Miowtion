"""Compares miowtion.h3.model with the diffusers reference, module by module.

`miowtion.h3.model.H3DiT` is a re-implementation of MiniMax-H3's DiT; every
check we have so far compares *our* torch model with *our* MLX port, so a
misunderstanding shared by both survives them all. This script puts the
released weights into both our model and diffusers'
`MiniMaxH3Transformer3DModel` and compares the two forwards stage by stage
(text refiner, embedding, timestep embedding, every trunk block, the output
heads), which is what localizes a wrong module.

Only `--layers` trunk blocks are built, so the check fits in a few GB; the
point is the per-stage numbers, not a full 50-layer forward.

Needs a diffusers with MiniMaxH3Transformer3DModel (>= 0.36) and the
diffusers-format release (`<variant>/transformer`).

Its verdict is about *that* release only. Anything the two releases store
differently - today the fused mlp.fc1 half order
(miowtion.h3.release.MLP_GATE_FIRST), tomorrow whatever the next port
renames - is checked here for the diffusers release and stays unchecked for
the first release, which is what the CUDA inference path loads. A green run
is not permission to change a release-dependent constant for both.

Examples:
    python scripts/check_vs_diffusers.py --transformer weights/h3/transformer
"""

import argparse
import dataclasses

import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule as h3_schedule
from miowtion.h3 import weights as h3_weights
from miowtion.utils import progress

_TEXT_LEN = 24


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 of a against b, both fp32."""
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _report(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> None:
    progress.log(f'{name:28s} rel {_relative(ours, theirs):.3e}  '
                 f'max |d| {float((ours.float() - theirs.float()).abs().max()):.3e}  '
                 f'|ours| {float(ours.float().norm()):.3f}  '
                 f'|theirs| {float(theirs.float().norm()):.3f}')


def _load_ours(model: h3_model.H3DiT, transformer_dir: str,
               config: h3_config.H3Config) -> None:
    """Loads the release into our model through the MLX name mapping.

    That mapping (`miowtion.mlx.convert`) is the one the MLX inference path
    uses, so loading through it also checks the mapping itself.
    """
    from miowtion.mlx import convert as mlx_convert
    from miowtion.mlx import interop

    perm = h3_weights.qkv_row_permutation(config.num_heads, config.head_dim)
    state = {}

    def take(tensors, prefix=''):
        for name, value in tensors.items():
            tensor = interop.to_torch(value)
            if name.endswith('attn.qkv_proj.weight'):
                tensor = tensor[perm]
            state[prefix + name] = tensor

    with mlx_convert.ReleaseReader(transformer_dir) as reader:
        take(mlx_convert.non_trunk_tensors(reader))
        for index in range(config.num_refiner_layers):
            take(mlx_convert.refiner_tensors(reader, index),
                 f'token_refiner.blocks.{index}.')
        for index in range(config.num_layers):
            take(mlx_convert.trunk_tensors(reader, index),
                 f'blocks.{index}.')
            for part, value in mlx_convert.adaln_tensors(reader,
                                                         index).items():
                state[f'blocks.{index}.adaln_proj.linear.{part}'] = (
                    interop.to_torch(value))
    own = dict(model.named_parameters())
    missing = sorted(set(own) - set(state))
    extra = sorted(set(state) - set(own))
    if missing or extra:
        raise KeyError(f'missing={missing[:5]} extra={extra[:5]}')
    for name, param in own.items():
        param.data.copy_(state[name].to(param.dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', default='weights/h3/transformer')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import diffusers  # pylint: disable=import-outside-toplevel

    config = dataclasses.replace(
        h3_config.H3Config.from_pretrained(args.transformer),
        num_layers=args.layers)
    geometry = h3_geometry.resolve_geometry('16:9', 21 / h3_geometry.FPS,
                                            short_edge=256)
    layout = h3_layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), geometry)
    progress.log(f'{geometry.name}: seq {layout.seq_len}, used {layout.used}')

    torch.manual_seed(args.seed)
    text = torch.randn(_TEXT_LEN, config.text_dim, dtype=torch.bfloat16)
    video_rows = torch.randn(layout.img_pos.numel(), config.video_patch_dim)
    audio_rows = torch.randn(layout.audio_pos.numel(), config.audio_channels)
    timestep = h3_schedule.build_timestep_state(layout, 0.8, 0.6)

    ours = h3_model.H3DiT(config).eval()
    _load_ours(ours, args.transformer, config)

    reference = diffusers.MiniMaxH3Transformer3DModel.from_pretrained(
        args.transformer, torch_dtype=torch.bfloat16, num_layers=args.layers,
        ignore_mismatched_sizes=False, low_cpu_mem_usage=True).eval()

    with torch.no_grad():
        # 1. Text tower.
        our_text = ours.refine_text(text)
        their_text = reference.token_refiner(
            reference.context_embedder(text.to(torch.bfloat16).unsqueeze(0)))
        _report('token refiner', our_text, their_text[0])

        # 2. Packed embedding.
        clip = ours.clip_inputs(layout, our_text, torch.device('cpu'))
        our_x = ours.embed(clip, video_rows, audio_rows)
        video_embeds = reference.proj_in(video_rows.float())
        audio_embeds = reference.audio_proj_in(audio_rows.float())
        their_x = their_text.new_zeros((1, layout.seq_len,
                                        config.hidden_size))
        # Text rows are exactly [0, text_len) (see PackedLayout).
        their_x[:, :layout.text_len] = their_text.reshape(1, _TEXT_LEN, -1)
        their_x = their_x.index_copy(
            1, layout.img_pos, video_embeds.to(their_x.dtype).unsqueeze(0))
        their_x = their_x.index_copy(
            1, layout.audio_pos, audio_embeds.to(their_x.dtype).unsqueeze(0))
        _report('packed embedding', our_x, their_x[0])

        # 3. Timestep embedding and the AdaLN table addressing.
        our_temb = ours.time_embedder(timestep.timesteps)
        their_temb = reference.time_embedder(
            reference.time_proj(timestep.timesteps).float())
        _report('time embedding', our_temb, their_temb)
        their_adaln = (timestep.slot * h3_config.MODALITY_NUM
                       + layout.token_tags.clamp(min=0))
        same = torch.equal(timestep.adaln_index, their_adaln)
        progress.log(f'adaln index equal: {same}')

        # 4. Trunk blocks, fed the *reference's* input so the errors do not
        # compound: a block that matches here is a block that is right.
        rope = reference.rope(layout.position_ids.float())
        x_ours = their_x[0].clone()
        for index in range(args.layers):
            x_theirs = reference.transformer_blocks[index](
                x_ours.unsqueeze(0), their_temb, their_adaln, rope)
            x_ours = ours.blocks[index](x_ours, None, ours.adaln_input(
                timestep.timesteps), timestep.adaln_index, clip.rope,
                h3_model.DenseAttention(layout.used), index)
            _report(f'block {index}', x_ours, x_theirs[0])
            x_ours = x_theirs[0].clone()

        # 5. Output heads.
        our_video, our_audio = ours.final_layer(
            x_ours, ours.adaln_input(timestep.timesteps), timestep.slot,
            clip.target_img_pos, clip.target_audio_pos)
        h = reference.norm_out(x_ours.unsqueeze(0), their_temb,
                               timestep.slot).float()
        their_video = reference.proj_out(h).index_select(
            1, clip.target_img_pos)
        their_audio = reference.audio_proj_out(h).index_select(
            1, clip.target_audio_pos)
        _report('video head', our_video, their_video[0])
        _report('audio head', our_audio, their_audio[0])


if __name__ == '__main__':
    main()
