"""Offline encoders: text presentations and condition latents.

Reuses the MiniMax-H3 release as-is:
  * text encoder: `<root>/FL2VA/text_encoder` (Qwen3-VL, shared by FL2VA and
    Ref2VA) loaded with transformers, truncated to 50 layers with the final
    RMSNorm removed (the DiT consumes the raw layer-50 residual stream);
    tokenizer / processor from the same release;
  * video / audio VAE: the release's own `video_vae` / `audio_vae` packages
    (`MiniMaxH3VideoVAE` / `MiniMaxH3AudioVAE`).

Presentation (token order fed to the text encoder):
  t2va:   prompt, verbatim, no special tokens, no chat template.
  fl2va:  per keyframe i: '<Picture i>: ' + <|vision_start|>
          <|image_pad|> x n <|vision_end|>; then the prompt.
  ref2va: per reference in request order: image -> '<Picture i>: ' + image
          block; audio -> '<Audio j>: ' label only; then the prompt.
Vision blocks are tagged 0 (video) for the DiT's AdaLN, text 1.

Condition latents: the VAE posterior is *sampled*, with the global RNG
seeded to 42 right before each encode (the seed is part of the contract),
then normalized with the VAE's latents_mean / latents_std and patchified.
Audio references use the deterministic posterior mean.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from miowtion.h3 import noise

TEXT_LAYERS = 50
CONDITION_ENCODE_SEED = 42
REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32
_VISION_START = '<|vision_start|>'
_VISION_END = '<|vision_end|>'
_IMAGE_PAD = '<|image_pad|>'
_TEXT_TAG = 1
_VISION_TAG = 0


class TextEncoder:
    """Qwen3-VL text tower -> [L, 5120] bf16 hidden states + DiT tags."""

    def __init__(self, variant_dir: str, device_map: str | dict = 'auto',
                 max_memory: dict | None = None):
        from transformers import AutoConfig  # pylint: disable=import-outside-toplevel
        from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
        from transformers import Qwen3VLForConditionalGeneration  # pylint: disable=import-outside-toplevel
        encoder_dir = os.path.join(variant_dir, 'text_encoder')
        config = AutoConfig.from_pretrained(encoder_dir)
        config.text_config.num_hidden_layers = TEXT_LAYERS
        # Layers past TEXT_LAYERS are never instantiated; their checkpoint
        # tensors are skipped by the loader.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            encoder_dir, config=config, torch_dtype=torch.bfloat16,
            device_map=device_map, max_memory=max_memory)
        self.model.model.language_model.norm = nn.Identity()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(variant_dir, 'tokenizer'))
        self._variant_dir = variant_dir
        self._processor = None

    @property
    def processor(self):
        """The release's Qwen3-VL processor, loaded on first image.

        Built lazily because it is only used to patchify images: the
        processor drags in the whole image stack (PIL, torchvision), which a
        t2va corpus -- text only -- would otherwise have to install to
        encode a prompt.
        """
        if self._processor is None:
            from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel
            self._processor = AutoProcessor.from_pretrained(
                os.path.join(self._variant_dir, 'processor'))
        return self._processor

    def _ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)[
            'input_ids'])

    def _vision_block(self, count: int) -> list[int]:
        convert = self.tokenizer.convert_tokens_to_ids
        return ([convert(_VISION_START)] + [convert(_IMAGE_PAD)] * count
                + [convert(_VISION_END)])

    @torch.no_grad()
    def encode(self, prompt: str, images: Sequence = (),
               image_labels: Sequence[str] = (),
               audio_labels_after: dict[int, list[str]] | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes one presentation.

        Args:
            prompt: Structured prompt text (verbatim).
            images: PIL images, each preceded by its label.
            image_labels: Labels such as '<Picture 1>: ', one per image.
            audio_labels_after: {i: labels} emitted after image i (i = -1:
                before the first image); ref2va audio references are labels
                only.

        Returns:
            (hidden [L, 5120] bf16 on CPU, tags [L] int64).
        """
        audio_labels_after = audio_labels_after or {}
        ids, tags = [], []

        def text(value):
            new = self._ids(value)
            ids.extend(new)
            tags.extend([_TEXT_TAG] * len(new))

        for label in audio_labels_after.get(-1, []):
            text(label)
        pixel_values, grids = None, None
        if images:
            processed = self.processor.image_processor(images=list(images),
                                                       return_tensors='pt')
            pixel_values = processed['pixel_values']
            grids = processed['image_grid_thw']
            merge = self.processor.image_processor.merge_size
            for i, (label, grid) in enumerate(zip(image_labels, grids)):
                text(label)
                block = self._vision_block(int(grid.prod()) // merge**2)
                ids.extend(block)
                tags.extend([_VISION_TAG] * len(block))
                for audio_label in audio_labels_after.get(i, []):
                    text(audio_label)
        text(prompt)
        device = self.model.device
        kwargs = {}
        if pixel_values is not None:
            kwargs = {'pixel_values': pixel_values.to(device),
                      'image_grid_thw': grids.to(device)}
        out = self.model.model(input_ids=torch.tensor([ids], device=device),
                               use_cache=False, **kwargs)
        hidden = out.last_hidden_state[0].to('cpu', torch.bfloat16)
        return hidden, torch.tensor(tags, dtype=torch.long)

    def encode_t2va(self, prompt: str):
        return self.encode(prompt)

    def encode_fl2va(self, prompt: str, keyframes: Sequence):
        labels = [f'<Picture {i + 1}>: ' for i in range(len(keyframes))]
        return self.encode(prompt, keyframes, labels)


@contextlib.contextmanager
def _seeded(seed: int):
    """Seeds the global CPU/CUDA RNGs for one encode, then restores them."""
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def import_release_package(variant_dir: str, package: str):
    if variant_dir not in sys.path:
        sys.path.insert(0, variant_dir)
    return importlib.import_module(package)


class ConditionEncoder:
    """The release's video / audio VAEs producing normalized DiT rows."""

    def __init__(self, variant_dir: str, device: torch.device):
        self.device = device
        video_pkg = import_release_package(
            variant_dir, 'video_vae.minimax_h3_video_vae')
        video_dir = os.path.join(variant_dir, 'video_vae')
        self.video_vae = video_pkg.MiniMaxH3VideoVAE.from_pretrained(
            video_dir).to(device)
        with open(os.path.join(video_dir, 'config.json')) as f:
            stats = json.load(f)
        self.video_mean = torch.tensor(stats['latents_mean']).view(-1, 1, 1, 1)
        self.video_std = torch.tensor(stats['latents_std']).view(-1, 1, 1, 1)

    def _normalize_rows(self, z: torch.Tensor) -> torch.Tensor:
        """[24, T, H, W] latent -> normalized [T*H/2*W/2, 96] fp32 rows."""
        z = z.float().cpu()
        z = (z - self.video_mean) / self.video_std
        return noise.patchify(z[None])

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """PIL image (already on its final canvas) -> rows."""
        with _seeded(CONDITION_ENCODE_SEED):
            z = self.video_vae.encode_images(image, use_fp16_latent=True)[0]
        return self._normalize_rows(z)

    @torch.no_grad()
    def encode_video(self, frames: np.ndarray) -> torch.Tensor:
        """[T, H, W, 3] uint8 frames (17n+5, on canvas) -> rows."""
        with _seeded(CONDITION_ENCODE_SEED):
            z = self.video_vae.encode_videos([frames],
                                             use_fp16_latent=True)[0]
        return self._normalize_rows(z)


def cover_crop(image, width: int, height: int):
    """Aspect-preserving resize (LANCZOS) + center crop onto a canvas."""
    from PIL import Image  # pylint: disable=import-outside-toplevel
    image = image.convert('RGB')
    if image.size == (width, height):
        return image
    scale = max(width / image.size[0], height / image.size[1])
    resized = (max(width, round(image.size[0] * scale)),
               max(height, round(image.size[1] * scale)))
    image = image.resize(resized, Image.Resampling.LANCZOS)
    left, top = (resized[0] - width) // 2, (resized[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def reference_image_size(width: int, height: int) -> tuple[int, int]:
    """Reference images: short edge 2048 (upscaling allowed), multiples of
    32 on both sides."""
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    snap = lambda v: max(REFERENCE_IMAGE_MULTIPLE, int(round(
        v * scale / REFERENCE_IMAGE_MULTIPLE)) * REFERENCE_IMAGE_MULTIPLE)
    return snap(width), snap(height)
