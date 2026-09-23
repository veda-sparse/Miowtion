"""Packed-sequence layout of one H3 request (t2va / fl2va / ref2va).

Row order:
    [text | keyframes | references | target audio | target video | pad]

* t2va / fl2va use the FL2VA checkpoint; fl2va adds first/last keyframe
  conditions.
* ref2va uses the Ref2VA checkpoint; references are images, audio clips or
  video(+audio) clips, optionally together with keyframes.

Invariants that the tile plans and the predictor depend on:
  * RoPE coordinates are fp64 and never rounded to integers (spatial axes are
    fractional; rounding merges columns).
  * `used` is rounded up to a multiple of 64 (or a fixed bucket length); the
    padding rows form their own attention segment, cu_seqlens =
    [0, used, seq_len], so they neither read nor are read by real rows.
  * Padding rows are tagged -1 here; the DiT clamps the tag to 0.
  * Text rows keep their own tags: vision-language rows inside the text
    segment (image/video pads of the text encoder) are tagged video (0).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry as h3_geometry

# Spatial RoPE axes span [left, left + ratio) * _INTERP.
_INTERP = 32
# Latent frames come in groups of 5 covering 1, 4, 4, 4, 4 pixel frames; each
# covered pixel frame advances the temporal coordinate by 5/3.
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH = h3_geometry.PATCH_HW
_AUDIO_CHANNELS = h3_geometry.AUDIO_CHANNELS
# Accepted keyframe signatures: first frame, last frame, first + last.
KEYFRAME_SIGNATURES = ((0,), (-1,), (0, -1))


@dataclasses.dataclass(frozen=True)
class VideoSpan:
    """A contiguous run of video-like rows on a (T, H, W) token grid.

    Attributes:
        start: First packed row of the span.
        grid: Token grid (T, H, W); rows are t-major, then h, then w.
        role: 'target', 'keyframe', 'ref_image' or 'ref_video'.
    """

    start: int
    grid: tuple[int, int, int]
    role: str

    @property
    def num_rows(self) -> int:
        t, h, w = self.grid
        return t * h * w

    @property
    def stop(self) -> int:
        return self.start + self.num_rows


@dataclasses.dataclass
class PackedLayout:
    """Structural description of one packed sequence (CPU tensors).

    Attributes:
        seq_len: Packed length (a multiple of 64 or a fixed bucket length).
        used: Number of non-padding rows.
        text_len: Text rows are exactly [0, text_len).
        img_pos: [Nv] int64 packed rows of all video-latent rows (keyframes,
            reference visuals, target) in model input order.
        audio_pos: [Na] int64 packed rows of all audio rows (references,
            target) in model input order.
        update_mask: [Nv] bool, True for target video rows.
        audio_update_mask: [Na] bool, True for target audio rows.
        position_ids: [seq_len, 3] float64 (t, h, w) RoPE coordinates.
        token_tags: [seq_len] int64 (0 video, 1 text, 2 audio, -1 pad).
        cu_seqlens: [3] int32, [0, used, seq_len].
        spans: Video-like spans in packed order; the target is last.
        visual_cond_shapes: Latent (T, H, W) of every visual condition
            (keyframes then reference visuals), in img_pos order.
        audio_cond_lengths: Audio latent T of every audio condition.
    """

    seq_len: int
    used: int
    text_len: int
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    update_mask: torch.Tensor
    audio_update_mask: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    cu_seqlens: torch.Tensor
    spans: tuple[VideoSpan, ...]
    visual_cond_shapes: tuple[tuple[int, int, int], ...]
    audio_cond_lengths: tuple[int, ...]

    @property
    def target(self) -> VideoSpan:
        return self.spans[-1]

    @property
    def target_img_pos(self) -> torch.Tensor:
        return self.img_pos[self.update_mask]

    @property
    def cond_img_pos(self) -> torch.Tensor:
        return self.img_pos[~self.update_mask]

    @property
    def target_audio_pos(self) -> torch.Tensor:
        return self.audio_pos[self.audio_update_mask]

    @property
    def cond_audio_pos(self) -> torch.Tensor:
        return self.audio_pos[~self.audio_update_mask]


def _align_up(value: int) -> int:
    alignment = h3_config.PACKED_SEQUENCE_ALIGNMENT
    return (value + alignment - 1) // alignment * alignment


def _spatial_axis(dim: int, sqrt_area: float) -> torch.Tensor:
    """Fractional coordinates of one spatial token axis, fp64."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // _PATCH, endpoint=False)
    return torch.from_numpy(grid * _INTERP).to(torch.float64)


def _frame_coords(latent_h: int, latent_w: int) -> torch.Tensor:
    """[H*W, 2] (h, w) coordinates of one latent frame, fp64."""
    sqrt_area = np.sqrt(latent_h * latent_w)
    hh, ww = torch.meshgrid(_spatial_axis(latent_h, sqrt_area),
                            _spatial_axis(latent_w, sqrt_area), indexing='ij')
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def _temporal_coords(num_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % 5] for k in range(num_frames)],
        dtype=torch.float64)
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_extent(num_frames: int) -> float:
    """Temporal length of a video span (sequential fp64 sum)."""
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[k % 5]
               for k in range(num_frames))


def _last_frame_t(num_frames: int, origin: float) -> float:
    """Temporal coordinate of the last pixel frame (pairwise fp64 sum)."""
    spans = np.ones(num_frames, dtype=np.float64) * _FRAME_RESCALE
    for k in range(5):
        spans[k::5] *= _FRAME_PER_TOKEN[k]
    return origin + float(spans.sum()) - _FRAME_RESCALE


def _fill_audio(g: torch.Tensor, start: int, audio_t: int, t_origin: float,
                w_min: float, w_max: float) -> None:
    """Channel-major stereo rows, each channel pinned to one width extreme."""
    stop = start + _AUDIO_CHANNELS * audio_t
    g[start:stop, 0] = (t_origin + torch.arange(
        audio_t, dtype=torch.float64)).repeat(_AUDIO_CHANNELS)
    if audio_t:
        g[start:start + audio_t, 2] = w_min
        g[start + audio_t:stop, 2] = w_max


def _fill_video(g: torch.Tensor, start: int, latent_t: int, latent_h: int,
                latent_w: int, t_origin: float) -> None:
    frame = _frame_coords(latent_h, latent_w)
    view = g[start:start + latent_t * frame.shape[0]].view(
        latent_t, frame.shape[0], 3)
    view[:, :, 0] = _temporal_coords(latent_t, t_origin)[:, None]
    view[:, :, 1:] = frame[None]


def _int_field(ref: Mapping[str, object], key: str, path: str,
               allow_zero: bool = False) -> int:
    value = ref.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{path}.{key} must be an integer')
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{path}.{key} must be positive')
    return value


def pack(text_tags: torch.Tensor, geometry: h3_geometry.Geometry,
         keyframes: Sequence[int] = (),
         references: Sequence[Mapping[str, object]] = (),
         seq_len: int | None = None) -> PackedLayout:
    """Packs one request into a single sequence.

    Args:
        text_tags: [text_len] int64 per-row tags of the text segment (1 for
            text tokens, 0 for vision-language rows of keyframes/references).
        geometry: Resolved target geometry.
        keyframes: () or one of KEYFRAME_SIGNATURES (0 = first pixel frame,
            -1 = last pixel frame).
        references: ref2va reference blocks in request order:
            {'kind': 'image', 'latent_h': H, 'latent_w': W},
            {'kind': 'audio', 'audio_t': T},
            {'kind': 'video', 'audio_t': T, 'latent_t': RT, 'latent_h': RH,
             'latent_w': RW} (its audio rows precede its video rows;
            audio_t may be 0 for a silent clip).
        seq_len: Optional fixed bucket length; must be >= the 64-aligned
            length. Extra rows are padding.

    Returns:
        The packed layout.

    Raises:
        ValueError: On malformed keyframes/references or a short bucket.
    """
    keyframes = tuple(int(k) for k in keyframes)
    if keyframes and keyframes not in KEYFRAME_SIGNATURES:
        raise ValueError(f'unsupported keyframes {keyframes}')
    text_len = int(text_tags.numel())
    latent_t, latent_h, latent_w = (geometry.latent_t, geometry.latent_h,
                                    geometry.latent_w)
    frame_rows = (latent_h // _PATCH) * (latent_w // _PATCH)

    # 1. Size every block.
    cursor = text_len + len(keyframes) * frame_rows
    blocks = []  # (kind, ref, audio_start, audio_rows, visual_start, rows)
    for index, ref in enumerate(references):
        path = f'references[{index}]'
        kind = ref.get('kind')
        if kind == 'image':
            rh = _int_field(ref, 'latent_h', path)
            rw = _int_field(ref, 'latent_w', path)
            rows = (rh // _PATCH) * (rw // _PATCH)
            blocks.append((kind, ref, cursor, 0, cursor, rows))
            cursor += rows
        elif kind == 'audio':
            audio_rows = _AUDIO_CHANNELS * _int_field(ref, 'audio_t', path)
            blocks.append((kind, ref, cursor, audio_rows, cursor, 0))
            cursor += audio_rows
        elif kind == 'video':
            audio_rows = _AUDIO_CHANNELS * _int_field(ref, 'audio_t', path,
                                                      allow_zero=True)
            rows = _int_field(ref, 'latent_t', path) * (
                _int_field(ref, 'latent_h', path) // _PATCH) * (
                    _int_field(ref, 'latent_w', path) // _PATCH)
            blocks.append((kind, ref, cursor, audio_rows, cursor + audio_rows,
                           rows))
            cursor += audio_rows + rows
        else:
            raise ValueError(f'{path}.kind unsupported: {kind!r}')
    audio_start = cursor
    video_start = audio_start + geometry.num_audio_rows
    used = video_start + latent_t * frame_rows
    aligned = _align_up(used)
    if seq_len is None:
        seq_len = aligned
    elif seq_len < aligned:
        raise ValueError(f'seq_len {seq_len} < aligned length {aligned}')

    # 2. RoPE coordinates. References advance the temporal origin.
    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    target_frame = _frame_coords(latent_h, latent_w)
    w_min, w_max = float(target_frame[0, 1]), float(target_frame[-1, 1])
    spans = []
    visual_cond_shapes = []
    audio_cond_lengths = []
    ref_img_pos, ref_audio_pos = [], []
    t_cursor = float(text_len)
    for kind, ref, a_start, a_rows, v_start, v_rows in blocks:
        if kind == 'image':
            rh, rw = ref['latent_h'], ref['latent_w']
            ref_frame = _frame_coords(rh, rw)
            g[v_start:v_start + v_rows, 0] = t_cursor
            g[v_start:v_start + v_rows, 1:] = ref_frame
            spans.append(VideoSpan(v_start, (1, rh // _PATCH, rw // _PATCH),
                                   'ref_image'))
            visual_cond_shapes.append((1, rh, rw))
            t_cursor += 1.0
        elif kind == 'audio':
            _fill_audio(g, a_start, ref['audio_t'], t_cursor, w_min, w_max)
            audio_cond_lengths.append(ref['audio_t'])
            t_cursor += float(ref['audio_t'])
        else:
            rt, rh, rw = ref['latent_t'], ref['latent_h'], ref['latent_w']
            ref_frame = _frame_coords(rh, rw)
            _fill_audio(g, a_start, ref['audio_t'], t_cursor,
                        float(ref_frame[0, 1]), float(ref_frame[-1, 1]))
            _fill_video(g, v_start, rt, rh, rw, t_cursor)
            spans.append(VideoSpan(v_start, (rt, rh // _PATCH, rw // _PATCH),
                                   'ref_video'))
            visual_cond_shapes.append((rt, rh, rw))
            if a_rows:
                audio_cond_lengths.append(ref['audio_t'])
            t_cursor += max(float(ref['audio_t']), _temporal_extent(rt))
        ref_img_pos.append(torch.arange(v_start, v_start + v_rows))
        ref_audio_pos.append(torch.arange(a_start, a_start + a_rows))

    _fill_audio(g, audio_start, geometry.audio_t, t_cursor, w_min, w_max)
    _fill_video(g, video_start, latent_t, latent_h, latent_w, t_cursor)
    keyframe_start = text_len
    keyframe_spans = []
    for index, keyframe in enumerate(keyframes):
        start = keyframe_start + index * frame_rows
        g[start:start + frame_rows, 0] = (
            t_cursor if keyframe == 0 else _last_frame_t(latent_t, t_cursor))
        g[start:start + frame_rows, 1:] = target_frame
        keyframe_spans.append(VideoSpan(start, (1, latent_h // _PATCH,
                                                latent_w // _PATCH),
                                        'keyframe'))
    spans = keyframe_spans + spans + [
        VideoSpan(video_start, geometry.video_grid, 'target')]
    visual_cond_shapes = [(1, latent_h, latent_w)] * len(keyframes) + (
        visual_cond_shapes)

    # 3. Row index sets, tags and segments.
    keyframe_pos = torch.arange(keyframe_start,
                                keyframe_start + len(keyframes) * frame_rows)
    img_pos = torch.cat([keyframe_pos] + ref_img_pos +
                        [torch.arange(video_start, used)])
    audio_pos = torch.cat(ref_audio_pos + [torch.arange(audio_start,
                                                        video_start)])
    num_cond_img = img_pos.numel() - latent_t * frame_rows
    num_cond_audio = audio_pos.numel() - geometry.num_audio_rows
    token_tags = torch.full((seq_len,), h3_config.TAG_PAD, dtype=torch.long)
    token_tags[:text_len] = text_tags.to(torch.long)
    token_tags[audio_pos] = h3_config.TAG_AUDIO
    token_tags[img_pos] = h3_config.TAG_VIDEO
    return PackedLayout(
        seq_len=seq_len,
        used=used,
        text_len=text_len,
        img_pos=img_pos,
        audio_pos=audio_pos,
        update_mask=torch.arange(img_pos.numel()) >= num_cond_img,
        audio_update_mask=torch.arange(audio_pos.numel()) >= num_cond_audio,
        position_ids=g,
        token_tags=token_tags,
        cu_seqlens=torch.tensor([0, used, seq_len], dtype=torch.int32),
        spans=tuple(spans),
        visual_cond_shapes=tuple(visual_cond_shapes),
        audio_cond_lengths=tuple(audio_cond_lengths),
    )
