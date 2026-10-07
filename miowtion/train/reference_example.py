"""Prepare the official MiniMax-H3 Ref2VA example for Miowtion (CPU only)."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request


REVISION = 'd21241f0a4b3acbb34c97dae47fa417b7065e438'
SOURCE_URL = (
    'https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/'
    f'{REVISION}/scripts/readme/reproducible-768p-ref2va-request.sh'
)
PROMPT_FIELDS = (
    'subject_definitions:', 'summary:', 'retention_analysis:',
    'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:',
)


def download(url: str, path: Path) -> None:
    """Download to a temporary file before publishing a complete asset."""
    if path.is_file() and path.stat().st_size:
        print(f'Reuse {path}', flush=True)
        return
    print(f'Download {path.name}', flush=True)
    temporary = path.with_name(path.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'Miowtion'})
    try:
        with urllib.request.urlopen(request, timeout=60) as source:
            with temporary.open('wb') as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
        if temporary.stat().st_size == 0:
            raise ValueError(f'Empty asset: {url}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def probe(path: Path) -> dict:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-show_format',
         '-of', 'json', str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def prepare_example(output_directory: str) -> Path:
    """Download official references and write a Miowtion Ref2VA manifest.

    The official H3-Context-IR prompt is preserved verbatim. Miowtion
    treats a video and its soundtrack as separate references, so the
    soundtrack is exposed as Audio 1 before the external Audio 2 voice.
    Returns the JSONL manifest path; no encoder weights are loaded.
    """
    for binary in ('ffmpeg', 'ffprobe'):
        if shutil.which(binary) is None:
            raise RuntimeError(f'{binary} is required')
    output = Path(output_directory)
    for relative in ('manifests', 'media/videos', 'media/audio'):
        (output / relative).mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        SOURCE_URL, headers={'User-Agent': 'Miowtion'})
    with urllib.request.urlopen(request, timeout=30) as response:
        source = response.read().decode('utf-8')
    # Read the request as data. Never execute the official shell script.
    match = re.search(r"<<'JSON'\s*\n(.*?)\nJSON", source, re.DOTALL)
    if match is None:
        raise ValueError('Official JSON request not found')
    official = json.loads(match.group(1))
    if official['task'] != 'ref2va':
        raise ValueError('Expected the official Ref2VA request')
    prompt = official['prompt']
    positions = [prompt.find(field) for field in PROMPT_FIELDS]
    if min(positions) < 0 or positions != sorted(positions):
        raise ValueError('Official prompt is not in H3 reference format')
    conditions = official['conditions']
    if [c['type'] for c in conditions] != ['video', 'audio']:
        raise ValueError('Expected one video and one external voice reference')
    if '<Audio 1>' not in prompt or '<Audio 2>' not in prompt:
        raise ValueError('Expected video soundtrack and voice-reference labels')

    video = output / 'media/videos/reference.mp4'
    voice = output / 'media/audio/voice_reference.mp3'
    soundtrack = output / 'media/audio/video_soundtrack.wav'
    download(conditions[0]['uri'], video)
    download(conditions[1]['uri'], voice)
    video_info, voice_info = probe(video), probe(voice)
    if not any(s['codec_type'] == 'video' for s in video_info['streams']):
        raise ValueError('Reference video has no video stream')
    for path, info in ((video, video_info), (voice, voice_info)):
        if not any(s['codec_type'] == 'audio' for s in info['streams']):
            raise ValueError(f'No audio stream: {path}')
    print('Extract video soundtrack as <Audio 1>', flush=True)
    temporary = soundtrack.with_name('video_soundtrack.partial.wav')
    try:
        subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(video), '-map', '0:a:0', '-vn', '-ac', '2',
             '-ar', '32000', '-c:a', 'pcm_s16le', str(temporary)],
            check=True,
        )
        temporary.replace(soundtrack)
    finally:
        temporary.unlink(missing_ok=True)

    references = [
        {'modality': 'video', 'label': '<Video 1>',
         'path': 'media/videos/reference.mp4'},
        {'modality': 'audio', 'label': '<Audio 1>',
         'path': 'media/audio/video_soundtrack.wav'},
        {'modality': 'audio', 'label': '<Audio 2>',
         'path': 'media/audio/voice_reference.mp3'},
    ]
    media_info = {}
    for reference in references:
        path = output / reference['path']
        info = probe(path)
        duration = float(info['format']['duration'])
        if not 2 <= duration <= 15:
            raise ValueError(f'Reference duration outside H3 limits: {path}')
        media_info[reference['label']] = {
            'path': reference['path'], 'duration_seconds': duration,
            'bytes': path.stat().st_size,
        }
    audio_total = sum(v['duration_seconds'] for k, v in media_info.items()
                      if k.startswith('<Audio'))
    if audio_total > 15:
        raise ValueError('Combined audio references exceed 15 seconds')
    record = {
        'id': 'demo', 'task': 'ref2va', 'prompt': prompt,
        'latent_t': 37, 'references': references,
    }
    manifest = output / 'manifests/prompts.jsonl'
    manifest.write_text(json.dumps(record, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    provenance = {
        'source_url': SOURCE_URL, 'source_revision': REVISION,
        'official_prompt_unchanged': True,
        'adaptation': 'Expose video soundtrack separately as <Audio 1>; '
                      'external voice reference is <Audio 2>.',
        'official_target': official['target'],
        'miowtion_target': {'latent_t': 37, 'example_geometry': '16:9@37'},
        'media': media_info,
    }
    (output / 'source.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(f'Ready: {manifest}', flush=True)
    print('Official PE preserved; no API key or GPU used.', flush=True)

    return manifest
