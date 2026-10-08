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
  ref2va: references in request order: image -> '<Picture i>: ' + image
          block; video -> '<Video i>: ' + timestamped vision blocks;
          audio -> '<Audio j>: ' label only; then the prompt.
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
import re
import subprocess
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
_VIDEO_PAD = '<|video_pad|>'
_TEXT_TAG = 1
_VISION_TAG = 0
REFERENCE_VIDEO_FPS = 24
REFERENCE_VIDEO_QWEN_FPS = 2
# H3 permits references up to 15 seconds at 24 FPS. The earlier 124-frame
# cap silently shortened longer benchmark references to about five seconds.
REFERENCE_VIDEO_MAX_FRAMES = 15 * REFERENCE_VIDEO_FPS
REFERENCE_VIDEO_SHORT_EDGE = 768
REFERENCE_VIDEO_MAX_PIXELS = 768 * 1344
REFERENCE_AUDIO_SAMPLE_RATE = 32000
REFERENCE_AUDIO_CHANNELS = 2


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

    def _video_block(self, count: int) -> list[int]:
        convert = self.tokenizer.convert_tokens_to_ids
        return ([convert(_VISION_START)] + [convert(_VIDEO_PAD)] * count
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
            kwargs['mm_token_type_ids'] = torch.tensor(
                self.processor.create_mm_token_type_ids([ids]),
                device=device)
        out = self.model.model(input_ids=torch.tensor([ids], device=device),
                               use_cache=False, **kwargs)
        hidden = out.last_hidden_state[0].to('cpu', torch.bfloat16)
        return hidden, torch.tensor(tags, dtype=torch.long)

    def encode_t2va(self, prompt: str):
        return self.encode(prompt)

    def encode_fl2va(self, prompt: str, keyframes: Sequence):
        labels = [f'<Picture {i + 1}>: ' for i in range(len(keyframes))]
        return self.encode(prompt, keyframes, labels)

    @torch.no_grad()
    def encode_ref2va(
            self, prompt: str, images: Sequence,
            videos: Sequence[np.ndarray],
            condition_labels: Sequence[tuple[str, int]],
            video_timestamps: Sequence[Sequence[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the official mixed-reference Qwen presentation.

        The presentation follows vLLM-Omni's MiniMax-H3 implementation at
        commit 4af28f33: references keep request order, images use one vision
        block, videos use timestamped 2-fps temporal blocks, and audio enters
        Qwen as a label only.

        Args:
            prompt: Six-section Ref2VA prompt.
            images: Reference PIL images in picture-number order.
            videos: Reference videos as [T, H, W, 3] uint8 arrays in
                video-number order, sampled to 2 fps for Qwen.
            condition_labels: Ordered ``(modality, one-based ordinal)`` pairs.
            video_timestamps: One timestamp list per video, after temporal
                merge by pairs.

        Returns:
            Hidden states [L, 5120] bf16 and tags [L] int64 on CPU.
        """
        ids, tags = [], []

        def text(value: str) -> None:
            new = self._ids(value)
            ids.extend(new)
            tags.extend([_TEXT_TAG] * len(new))

        image_grids = None
        video_grids = None
        kwargs = {}
        if images:
            processed = self.processor.image_processor(
                images=list(images), return_tensors='pt')
            kwargs['pixel_values'] = processed['pixel_values']
            image_grids = processed['image_grid_thw']
            kwargs['image_grid_thw'] = image_grids
        if videos:
            processed = self.processor.video_processor(
                videos=list(videos), do_sample_frames=False,
                cap_pixels_per_frame=True,
                return_tensors='pt')
            kwargs['pixel_values_videos'] = processed['pixel_values_videos']
            video_grids = processed['video_grid_thw']
            kwargs['video_grid_thw'] = video_grids

        merge_length = self.processor.image_processor.merge_size ** 2
        image_counts = ([int(grid.prod()) // merge_length
                         for grid in image_grids]
                        if image_grids is not None else [])
        video_counts = []
        if video_grids is not None:
            for grid in video_grids:
                blocks = int(grid[0])
                per_block = int(grid[1:].prod()) // merge_length
                video_counts.append([per_block] * blocks)

        image_seen = video_seen = 0
        for modality, ordinal in condition_labels:
            if modality == 'image':
                count = image_counts[image_seen]
                image_seen += 1
                text(f'<Picture {ordinal}>: ')
                block = self._vision_block(count)
                ids.extend(block)
                tags.extend([_VISION_TAG] * len(block))
            elif modality == 'video':
                counts = video_counts[video_seen]
                timestamps = list(video_timestamps[video_seen])
                video_seen += 1
                if len(counts) != len(timestamps):
                    raise ValueError('video token blocks and timestamps differ: '
                                     f'{len(counts)} != {len(timestamps)}')
                text(f'<Video {ordinal}>: ')
                for count, timestamp in zip(counts, timestamps):
                    text(f'<{timestamp:.1f} seconds>')
                    block = self._video_block(count)
                    ids.extend(block)
                    tags.extend([_VISION_TAG] * len(block))
            elif modality == 'audio':
                text(f'<Audio {ordinal}>: ')
            else:
                raise ValueError(f'unsupported reference modality {modality}')
        if image_seen != len(image_counts) or video_seen != len(video_counts):
            raise ValueError('unused image or video text-encoder inputs')
        text(prompt)
        device = self.model.device
        kwargs = {key: value.to(device) for key, value in kwargs.items()}
        if image_grids is not None or video_grids is not None:
            kwargs['mm_token_type_ids'] = torch.tensor(
                self.processor.create_mm_token_type_ids([ids]),
                device=device)
        out = self.model.model(
            input_ids=torch.tensor([ids], device=device), use_cache=False,
            **kwargs)
        hidden = out.last_hidden_state[0].to('cpu', torch.bfloat16)
        return hidden, torch.tensor(tags, dtype=torch.long)


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
        self._variant_dir = variant_dir
        self.audio_vae = None
        self.audio_mean = None
        self.audio_std = None

    def _load_audio_vae(self) -> None:
        """Keep the audio VAE out of image-only FL2VA encoding."""
        if self.audio_vae is not None:
            return
        variant_dir = self._variant_dir
        audio_pkg = import_release_package(
            variant_dir, 'audio_vae.minimax_h3_audio_vae')
        audio_dir = os.path.join(variant_dir, 'audio_vae')
        self.audio_vae = audio_pkg.MiniMaxH3AudioVAE.from_pretrained(
            audio_dir).to(self.device)
        with open(os.path.join(audio_dir, 'config.json')) as f:
            audio_stats = json.load(f)
        self.audio_mean = torch.tensor(audio_stats['latents_mean']).view(
            1, 1, -1)
        self.audio_std = torch.tensor(audio_stats['latents_std']).view(
            1, 1, -1)

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

    @torch.no_grad()
    def encode_reference_video(
            self, frames: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Encode reference frames to rows and their latent ``(T,H,W)``."""
        with _seeded(CONDITION_ENCODE_SEED):
            z = self.video_vae.encode_videos(
                [frames], use_fp16_latent=True)[0]
        if z.ndim != 4:
            raise ValueError(f'unexpected reference video latent {z.shape}')
        return self._normalize_rows(z), tuple(int(v) for v in z.shape[-3:])

    @torch.no_grad()
    def encode_audio(
            self, waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Encode stereo 32-kHz waveform [2, samples] to [2*T, 32]."""
        if waveform.ndim != 2 or waveform.shape[0] != REFERENCE_AUDIO_CHANNELS:
            raise ValueError('waveform must be stereo [2, samples]')
        self._load_audio_vae()
        model = self.audio_vae.model
        waveform = waveform.float().to(self.device)
        saved = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.enabled,
        )
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
            audio = model.preprocess(
                waveform.unsqueeze(1), REFERENCE_AUDIO_SAMPLE_RATE)
            latent = model.encoder(audio)
            if bool(getattr(model, 'attn_proj', False)):
                latent = model.pre_block(
                    latent.transpose(1, 2)).transpose(1, 2)
            latent = model.mean_proj(latent).float().cpu()
        finally:
            (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic,
             torch.backends.cudnn.enabled) = saved
        channels = self.audio_mean.shape[-1]
        if latent.shape[-1] != channels:
            if latent.shape[1] != channels:
                raise ValueError(f'cannot canonicalize audio latent {latent.shape}')
            latent = latent.transpose(1, 2).contiguous()
        rows = ((latent - self.audio_mean) / self.audio_std).reshape(
            -1, channels)
        return rows.float(), int(latent.shape[1])


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
    """Preserve image resolution while snapping both axes to 32 pixels."""
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise ValueError(f'reference image ratio outside [0.4, 2.5]: '
                         f'{width}x{height}')
    if min(width, height) < 256 or max(width, height) > 5760:
        raise ValueError(f'reference image dimensions outside [256, 5760]: '
                         f'{width}x{height}')
    snap = lambda v: max(REFERENCE_IMAGE_MULTIPLE, int(round(
        v / REFERENCE_IMAGE_MULTIPLE)) * REFERENCE_IMAGE_MULTIPLE)
    return snap(width), snap(height)


def _probe_video_size(path: str) -> tuple[int, int]:
    """Read video dimensions from ffmpeg without requiring ffprobe."""
    result = subprocess.run(
        [os.environ.get('MIOWTION_FFMPEG', 'ffmpeg'), '-hide_banner', '-i', path],
        capture_output=True,
        text=True, check=False)
    match = re.search(r'Video:.*?\b(\d{2,5})x(\d{2,5})\b', result.stderr)
    if match is None:
        raise ValueError(f'cannot determine video dimensions: {path}')
    return int(match.group(1)), int(match.group(2))


def reference_video_size(width: int, height: int) -> tuple[int, int]:
    """Resolve the official 768p reference-video canvas without upscaling."""
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise ValueError(f'reference video ratio outside [0.4, 2.5]: '
                         f'{width}x{height}')
    if ratio >= 1:
        target_w, target_h = REFERENCE_VIDEO_SHORT_EDGE * ratio, 768.0
    else:
        target_w, target_h = 768.0, REFERENCE_VIDEO_SHORT_EDGE / ratio
    area = target_w * target_h
    if area > REFERENCE_VIDEO_MAX_PIXELS:
        scale = (REFERENCE_VIDEO_MAX_PIXELS / area) ** 0.5
        target_w, target_h = target_w * scale, target_h * scale
    snap = lambda v: max(32, int(round(v / 32)) * 32)
    canvas = (snap(target_w), snap(target_h))
    if width * height < canvas[0] * canvas[1]:
        return snap(width), snap(height)
    return canvas


def load_reference_video(
        path: str, max_frames: int = REFERENCE_VIDEO_MAX_FRAMES,
) -> np.ndarray:
    """Decode a reference video at 24 fps on the official reference canvas."""
    width, height = _probe_video_size(path)
    width, height = reference_video_size(width, height)
    command = [
        os.environ.get('MIOWTION_FFMPEG', 'ffmpeg'), '-loglevel', 'error',
        '-i', path, '-map', '0:v:0', '-an',
        '-vf', f'fps={REFERENCE_VIDEO_FPS},scale={width}:{height}:flags=lanczos,setsar=1',
        '-frames:v', str(max_frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        'pipe:1',
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise ValueError(f'invalid decoded byte count for {path}')
    count = len(result.stdout) // frame_bytes
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        count, height, width, 3).copy()


def sample_qwen_video(
        frames: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Sample 24-fps frames to Qwen's 2-fps temporal blocks."""
    stride = REFERENCE_VIDEO_FPS // REFERENCE_VIDEO_QWEN_FPS
    sampled = frames[::stride]
    if len(sampled) == 0:
        raise ValueError('reference video contains no frames')
    timestamps = [i / REFERENCE_VIDEO_QWEN_FPS
                  for i in range(len(sampled))]
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    block_timestamps = [
        (timestamps[i] + timestamps[i + 1]) / 2
        for i in range(0, len(timestamps), 2)
    ]
    return sampled, block_timestamps


def load_reference_audio(path: str) -> torch.Tensor:
    """Decode any ffmpeg-supported audio as stereo 32-kHz float32."""
    result = subprocess.run(
        [os.environ.get('MIOWTION_FFMPEG', 'ffmpeg'), '-loglevel', 'error',
         '-i', path, '-vn', '-ac', '2',
         '-ar', str(REFERENCE_AUDIO_SAMPLE_RATE), '-f', 'f32le', 'pipe:1'],
        check=True, capture_output=True)
    values = np.frombuffer(result.stdout, dtype=np.float32)
    if not len(values) or len(values) % REFERENCE_AUDIO_CHANNELS:
        raise ValueError(f'invalid decoded audio for {path}')
    return torch.from_numpy(values.copy()).view(-1, 2).t().contiguous()
