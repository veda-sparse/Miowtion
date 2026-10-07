"""Pure preprocessing tests for offline MiniMax-H3 encoders."""

import numpy as np
import pytest

from miowtion.train import encode


def test_reference_image_size_preserves_resolution_grid():
    assert encode.reference_image_size(1280, 720) == (1280, 704)
    assert encode.reference_image_size(512, 512) == (512, 512)


def test_reference_image_size_rejects_official_limit_violations():
    with pytest.raises(ValueError, match='dimensions'):
        encode.reference_image_size(128, 128)
    with pytest.raises(ValueError, match='ratio'):
        encode.reference_image_size(2560, 256)


def test_reference_video_size_caps_area_without_upscaling():
    assert encode.reference_video_size(1280, 720) == (1280, 704)
    assert encode.reference_video_size(3840, 2160) == (1344, 768)
    assert encode.reference_video_size(512, 512) == (512, 512)


def test_sample_qwen_video_uses_two_fps_and_temporal_pairs():
    frames = np.zeros((124, 8, 8, 3), dtype=np.uint8)
    sampled, timestamps = encode.sample_qwen_video(frames)
    assert sampled.shape == (11, 8, 8, 3)
    assert timestamps == [0.25, 1.25, 2.25, 3.25, 4.25, 5.0]


def test_image_only_condition_encoder_does_not_require_audio_vae(
        tmp_path, monkeypatch):
    """Adding Ref2VA audio must preserve the image-only encoder contract."""
    import json
    from types import SimpleNamespace

    video_dir = tmp_path / 'video_vae'
    video_dir.mkdir()
    (video_dir / 'config.json').write_text(json.dumps({
        'latents_mean': [0.0] * 24, 'latents_std': [1.0] * 24,
    }))
    loaded = []

    class VideoVAE:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def to(self, device):
            return self

    def load_package(variant_dir, name):
        loaded.append(name)
        assert name == 'video_vae.minimax_h3_video_vae'
        return SimpleNamespace(MiniMaxH3VideoVAE=VideoVAE)

    monkeypatch.setattr(encode, 'import_release_package', load_package)
    encoder = encode.ConditionEncoder(str(tmp_path), 'cpu')
    assert encoder.audio_vae is None
    assert loaded == ['video_vae.minimax_h3_video_vae']
