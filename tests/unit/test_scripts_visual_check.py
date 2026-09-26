"""Command building of scripts/visual_check.py (no ffmpeg needed)."""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _module():
    path = os.path.join(_ROOT, 'scripts', 'visual_check.py')
    spec = importlib.util.spec_from_file_location('visual_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panes_keep_their_order():
    parse = _module().parse_videos
    assert list(parse(['Dense=a.mp4', 'Veda=b.mp4'])) == ['Dense', 'Veda']


@pytest.mark.parametrize('specs,match', [
    (['Dense'], 'LABEL=PATH'),
    (['=a.mp4'], 'empty label'),
    (['A=a.mp4', 'A=b.mp4'], 'duplicate'),
])
def test_bad_pane_specs_raise(specs, match):
    with pytest.raises(ValueError, match=match):
        _module().parse_videos(specs)


def test_stack_titles_every_pane_and_hstacks_them():
    module = _module()
    videos = {'Dense': 'a.mp4', 'Veda 90%': 'b.mp4', 'fp8': 'c.mp4'}
    args = module.stack_command(videos, 'out.mp4')
    assert args[:2] == ['ffmpeg', '-y']
    assert args.count('-i') == 3
    chain = args[args.index('-filter_complex') + 1]
    assert chain.count('drawtext') == 3
    assert '[p0][p1][p2]hstack=inputs=3[v]' in chain
    # Percent and colon are drawtext syntax; a label must survive them.
    assert 'Veda 90\\%' in chain
    assert args[-1] == 'out.mp4'


def test_labels_with_ffmpeg_syntax_are_escaped():
    escape = _module().escape_text
    assert escape('a:b') == 'a\\:b'
    assert escape("it's") == "it\\'s"


def test_metrics_are_read_out_of_ffmpeg_stderr():
    module = _module()
    stderr = ('[Parsed_psnr_0 @ 0x1] PSNR y:31.94 u:44.1 v:43.7 '
              'average:33.02 min:28.1 max:39.0\n'
              '[Parsed_ssim_1 @ 0x2] SSIM Y:0.9310 U:0.98 V:0.98 '
              'All:0.945612 (12.6)\n')
    assert module.parse_metrics(stderr) == {'psnr_db': 33.02,
                                            'ssim': 0.945612}


def test_identical_inputs_report_infinite_psnr():
    module = _module()
    got = module.parse_metrics('PSNR y:inf average:inf min:inf max:inf\n')
    assert got['psnr_db'] == float('inf')


def test_metrics_without_a_report_raise():
    with pytest.raises(ValueError, match='neither PSNR nor SSIM'):
        _module().parse_metrics('ffmpeg version 4.4.2\n')


def test_heatmap_is_a_difference_blend_against_the_reference():
    args = _module().heatmap_command('ref.mp4', 'other.mp4', 'd.mp4', 8.0)
    chain = args[args.index('-filter_complex') + 1]
    assert 'blend=all_mode=difference' in chain
    assert 'pseudocolor' in chain
    assert args[args.index('-i') + 1] == 'ref.mp4'
