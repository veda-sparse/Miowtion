"""Tests for miowtion.h3.geometry and miowtion.h3.layout."""

import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import layout


def _text(n):
    return torch.ones(n, dtype=torch.long)


def test_canvas_and_grids():
    assert geometry.resolve_canvas(16, 9) == (1344, 768)
    assert geometry.resolve_canvas(9, 16) == (768, 1344)
    assert geometry.resolve_canvas(4, 3) == (1024, 768)
    g = geometry.resolve_geometry('16:9', 5.0)
    assert (g.frame_count, g.video_grid, g.num_video_tokens) == (
        124, (37, 24, 42), 37296)
    g = geometry.resolve_geometry('16:9', 14.375)
    assert (g.frame_count, g.video_grid, g.num_video_tokens) == (
        345, (102, 24, 42), 102816)
    assert geometry.resolve_geometry('9:16', 5.0).video_grid == (37, 42, 24)
    assert geometry.resolve_geometry('4:3', 5.0).video_grid == (37, 24, 32)


def test_duration_ladder():
    ladder = geometry.latent_t_ladder()
    assert ladder == list(range(37, 103, 5))
    assert len(ladder) == 14
    # 15.0 s aligns to latent_t 107 (15.083 s), beyond the maximum.
    assert geometry.resolve_geometry('16:9', 15.0).latent_t == 107


def test_t2va_self_check():
    g = geometry.resolve_geometry('16:9', 5.0)
    lay = layout.pack(_text(353), g)
    assert (lay.target.start, lay.target.stop) == (767, 38063)
    assert lay.target.grid == (37, 24, 42)
    assert (lay.used, lay.seq_len) == (38063, 38080)
    assert lay.cu_seqlens.tolist() == [0, 38063, 38080]
    assert (lay.token_tags[38063:] == h3_config.TAG_PAD).all()
    assert (lay.token_tags[353:767] == h3_config.TAG_AUDIO).all()
    assert lay.position_ids.dtype == torch.float64
    # Spatial coordinates are fractional; rounding them would merge columns.
    w = lay.position_ids[767:767 + 42, 2]
    assert torch.unique(w).numel() == 42
    assert not torch.equal(w, w.round())


def test_audio_rows_pinned_to_width_extremes():
    g = geometry.resolve_geometry('16:9', 5.0)
    lay = layout.pack(_text(10), g)
    audio = lay.position_ids[lay.audio_pos]
    video_w = lay.position_ids[lay.target_img_pos, 2]
    assert torch.equal(audio[:g.audio_t, 2],
                       torch.full((g.audio_t,), video_w.min().item(),
                                  dtype=torch.float64))
    assert (audio[g.audio_t:, 2] == video_w.max()).all()
    assert torch.equal(audio[:g.audio_t, 0], audio[g.audio_t:, 0])


def test_temporal_coordinates():
    g = geometry.resolve_geometry('16:9', 5.0)
    lay = layout.pack(_text(100), g)
    frame_rows = 24 * 42
    t = lay.position_ids[lay.target_img_pos, 0].view(37, frame_rows)[:, 0]
    steps = (t[1:] - t[:-1]) / (5.0 / 3.0)
    expected = torch.tensor([1, 4, 4, 4, 4] * 8, dtype=torch.float64)[:36]
    assert torch.allclose(steps, expected)
    assert t[0].item() == 100.0


def test_fl2va_keyframes():
    g = geometry.resolve_geometry('16:9', 5.0)
    frame_rows = 24 * 42
    lay = layout.pack(_text(50), g, keyframes=(0, -1))
    assert [s.role for s in lay.spans] == ['keyframe', 'keyframe', 'target']
    assert lay.visual_cond_shapes == ((1, 48, 84), (1, 48, 84))
    assert lay.update_mask.sum().item() == g.num_video_tokens
    first = lay.position_ids[50:50 + frame_rows]
    last = lay.position_ids[50 + frame_rows:50 + 2 * frame_rows]
    target_t = lay.position_ids[lay.target_img_pos, 0]
    assert (first[:, 0] == 50.0).all()
    # The last latent frame covers 4 pixel frames of 5/3 each; the last
    # keyframe sits on the last of them.
    assert last[0, 0].item() == pytest.approx(target_t.max().item() + 5.0)
    assert (last[:, 0] == last[0, 0]).all()
    assert torch.equal(first[:, 1:], lay.position_ids[
        lay.target.start:lay.target.start + frame_rows, 1:])
    with pytest.raises(ValueError):
        layout.pack(_text(50), g, keyframes=(-1, 0))


def test_ref2va_blocks():
    g = geometry.resolve_geometry('16:9', 5.0)
    refs = [{'kind': 'image', 'latent_h': 128, 'latent_w': 224},
            {'kind': 'video', 'audio_t': 200, 'latent_t': 12,
             'latent_h': 48, 'latent_w': 84},
            {'kind': 'audio', 'audio_t': 100}]
    lay = layout.pack(_text(300), g, references=refs)
    roles = [s.role for s in lay.spans]
    assert roles == ['ref_image', 'ref_video', 'target']
    assert lay.visual_cond_shapes == ((1, 128, 224), (12, 48, 84))
    assert lay.audio_cond_lengths == (200, 100)
    image, video = lay.spans[0], lay.spans[1]
    assert image.start == 300 and image.grid == (1, 64, 112)
    assert video.start == image.stop + 400
    # The target starts after the image (1) and the longer of the video's
    # audio (200) and video spans, then the standalone audio (100).
    target_t0 = lay.position_ids[lay.target.start, 0].item()
    assert target_t0 == pytest.approx(300 + 1 + 200 + 100)
    assert (~lay.audio_update_mask).sum().item() == 600
    assert (lay.token_tags[lay.cond_img_pos] == h3_config.TAG_VIDEO).all()


def test_text_tags_and_bucket():
    g = geometry.resolve_geometry('4:3', 5.0)
    tags = torch.ones(40, dtype=torch.long)
    tags[5:20] = 0
    lay = layout.pack(tags, g, seq_len=40000)
    assert lay.seq_len == 40000
    assert torch.equal(lay.token_tags[:40], tags)
    with pytest.raises(ValueError):
        layout.pack(tags, g, seq_len=100)
