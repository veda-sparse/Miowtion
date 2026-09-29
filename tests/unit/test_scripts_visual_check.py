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
    titles = {label: f'{label}.png' for label in videos}
    args = module.stack_command(videos, titles, 'out.mp4')
    assert args[:2] == ['ffmpeg', '-y']
    # Three panes and three title images, panes first.
    assert args.count('-i') == 6
    assert args[2:8:2] == ['-i'] * 3
    chain = args[args.index('-filter_complex') + 1]
    # Each pane gets a bar and its own title overlaid centered on it.
    assert chain.count('overlay=(W-w)/2:0') == 3
    assert '[b0][3:v]overlay' in chain and '[b2][5:v]overlay' in chain
    assert '[p0][p1][p2]hstack=inputs=3[v]' in chain
    # No drawtext: the titles are pre-rendered, so no label needs escaping
    # and a label may contain any character.
    assert 'drawtext' not in chain
    assert args[-1] == 'out.mp4'


def test_title_png_is_written_and_sized_to_the_text(tmp_path):
    module = _module()
    short = module.write_title('A', str(tmp_path / 'a.png'))
    long = module.write_title('A much longer label', str(tmp_path / 'b.png'))
    from PIL import Image
    with Image.open(short) as a, Image.open(long) as b:
        assert a.height == b.height
        assert b.width > a.width


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


def test_stack_direction_switches_the_filter():
    module = _module()
    videos = {'A': 'a.mp4', 'B': 'b.mp4'}
    titles = {label: f'{label}.png' for label in videos}
    for direction in ('h', 'v'):
        args = module.stack_command(videos, titles, 'out.mp4',
                                    stack=direction)
        chain = args[args.index('-filter_complex') + 1]
        assert f'{direction}stack=inputs=2[v]' in chain
    with pytest.raises(ValueError, match='stack must be'):
        module.stack_command(videos, titles, 'out.mp4', stack='diagonal')


def test_probe_stack_opposes_the_clip_orientation(monkeypatch):
    module = _module()
    import subprocess as sp

    def fake(args, **kwargs):
        wh = {'wide.mp4': '1344,768', 'tall.mp4': '768,1344',
              'square.mp4': '768,768'}[args[-1]]
        return sp.CompletedProcess(args, 0, stdout=wh + '\n', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', fake)
    # Landscape and square stack vertically, portrait horizontally.
    assert module.probe_stack('wide.mp4') == 'v'
    assert module.probe_stack('square.mp4') == 'v'
    assert module.probe_stack('tall.mp4') == 'h'


def test_probe_stack_refuses_to_guess_without_ffprobe(monkeypatch):
    module = _module()

    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(module.subprocess, 'run', missing)
    with pytest.raises(RuntimeError, match='--stack'):
        module.probe_stack('a.mp4')
