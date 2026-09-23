"""Prompt expansion: short video requests -> structured H3 T2VA prompts.

The rewrite rules are the official MiniMax-H3 prompt-writing skill
(`third_party/MiniMax-H3/skills/h3-prompt-writing`: SKILL.md and
references/base-en.txt). The system prompt is assembled from those files at
runtime so that upgrading the submodule upgrades the rules; nothing of the
skill is copied into this module.

The LLM is DeepSeek's OpenAI-compatible chat completions API
(https://api-docs.deepseek.com/), called with the standard library only.
The API key is read from the environment variable `DEEPSEEK_API_KEY`; it is
never stored in records, logs or error messages.

Every expansion is accepted only after `data.validate_prompt` (the same check
the encoder applies) plus the stricter `check_expansion` below. Rejected
outputs are retried with the checker's error fed back to the model; prompts
that still fail are returned as failures, never dropped.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import http.client
import json
import logging
import os
import pathlib
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from miowtion.h3 import geometry as h3_geometry
from miowtion.train import data

_LOG = logging.getLogger(__name__)

API_KEY_ENV = 'DEEPSEEK_API_KEY'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
_CHAT_COMPLETIONS_PATH = '/chat/completions'
# 'deepseek-flash' is DeepSeek-V4.1-Flash (api-docs.deepseek.com, pricing).
DEFAULT_MODEL = 'deepseek-flash'
DEFAULT_REASONING_EFFORT = 'medium'
# Documented values are none/low/high/max; minimal -> low and
# medium/xhigh -> high are accepted as compatibility aliases (API reference,
# `reasoning_effort`). 'none' disables thinking.
REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh',
                     'max')
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 3
# Exponential backoff after API errors: 2, 4, 8, ... seconds, capped.
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0
# HTTP statuses worth retrying (DeepSeek error codes: 429 rate limit, 500
# server error, 503 overloaded); 400/401/402/422 are caller errors.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_H3_REPO = pathlib.Path(__file__).resolve().parents[2] / 'third_party' / (
    'MiniMax-H3')
# Same skill vendored at three places in the H3 repo; `skills/` is the
# canonical copy (the other two lack the "Tips for Better Results" section).
DEFAULT_SKILL_DIRS = (
    _H3_REPO / 'skills' / 'h3-prompt-writing',
    _H3_REPO / '.claude' / 'skills' / 'h3-prompt-writing',
    _H3_REPO / '.agents' / 'skills' / 'h3-prompt-writing',
)
_BASE_REFERENCE = pathlib.Path('references') / 'base-en.txt'

TASK = 't2va'
_FIELDS = ('integrated_multimodal_description:', 'overall_soundscape:',
           'non_diegetic_music:')

_PREAMBLE = (
    'You rewrite short video requests into generation prompts for the '
    'MiniMax H3 audio-video model. Follow the h3-prompt-writing skill '
    'below. Its base-mode reference guide is included in full, so there '
    'are no files to read.')

_T2VA_CONTRACT = """\
Every user message is a T2VA task (text only, no reference images). It \
gives the target duration, the aspect ratio and the request text. Reply \
with the final H3 T2VA prompt under this contract:
1. Output only the final prompt: no preamble, notes, headings or Markdown \
code fences.
2. T2VA has no instruction line. Start with \
"integrated_multimodal_description: [Shot 1]", then "overall_soundscape:", \
then "non_diegetic_music:", each field exactly once and separated by one \
blank line. Each field is a single paragraph with no line breaks inside it; \
later shots follow the previous shot inline, as in the skill's examples.
3. Write in English; keep dialogue, lyrics and on-screen text in their \
original language, as the skill requires.
4. Timing must match the target duration. [Shot 1] has no timestamp; every \
later shot starts with "At MM:SS.mmm," and these cut times strictly \
increase and stay strictly below the duration; no timestamp may exceed the \
duration. Describe only as much action as fits in real time within each \
shot, so the timeline fills the whole duration with no unfinished action \
and no dead time; a short video holds few shots and simple actions.
5. Compose for the given aspect ratio, but do not state the aspect ratio, \
the orientation or the total duration in the prompt."""

_REPAIR = (
    'The format checker rejected your prompt: {error}. Rewrite it to fix '
    'this while keeping every rule. Output only the corrected final prompt.')

_SHOT_RE = re.compile(r'\[Shot (\d+)\]')
_CUT_RE = re.compile(r'\s*At (\d{2}):(\d{2})\.(\d{3})\b')
_TIMESTAMP_RE = re.compile(r'\b(\d{2}):(\d{2})\.(\d{3})\b')
_SECRET_RE = re.compile(r'sk-[A-Za-z0-9*._-]+')
# Timestamps carry milliseconds; the duration is compared at that precision.
_TIME_EPS = 5e-4


class ApiError(Exception):
    """A failed chat completion call.

    Attributes:
        status: HTTP status, or None for transport / malformed responses.
        body: Response body or error description, with secrets redacted.
    """

    def __init__(self, status: int | None, body: str):
        self.status = status
        self.body = _redact(body)
        super().__init__(f'HTTP {status}: {self.body}' if status is not None
                         else self.body)

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in _RETRYABLE_STATUS


class ExpansionError(Exception):
    """A prompt that failed every attempt.

    Attributes:
        errors: One message per failed attempt.
        usage: Token usage summed over all attempts.
        last_output: Content of the last completion, '' if none.
    """

    def __init__(self, request_id: str, errors: list[str],
                 usage: dict[str, Any], last_output: str):
        self.errors = errors
        self.usage = usage
        self.last_output = last_output
        super().__init__(f'{request_id}: {len(errors)} failed attempts; '
                         f'last: {errors[-1] if errors else "none"}')


@dataclasses.dataclass(frozen=True)
class Completion:
    """The parts of a chat completion response we consume."""

    content: str
    finish_reason: str
    usage: dict[str, Any]
    model: str


@dataclasses.dataclass(frozen=True)
class ExpansionRequest:
    """One source prompt with its target geometry.

    Attributes:
        id: Sample id, e.g. 'moviegen_0000'.
        source_prompt: Short request text.
        aspect: Aspect ratio string, e.g. '16:9'.
        duration_seconds: Actual (frame-aligned) duration, frame_count / FPS.
        latent_t: Video latent frames of that duration.
    """

    id: str
    source_prompt: str
    aspect: str
    duration_seconds: float
    latent_t: int


# (url, headers, json body, timeout seconds) -> parsed JSON response.
# Raises ApiError on HTTP / transport failure.
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float],
                     dict[str, Any]]


def _redact(text: str) -> str:
    return _SECRET_RE.sub('sk-***', text)


def http_transport(url: str, headers: Mapping[str, str],
                   body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    """POSTs `body` as JSON with urllib and returns the decoded response.

    DeepSeek may send blank keep-alive lines before a non-streaming body;
    json.loads skips that leading whitespace.

    Raises:
        ApiError: On HTTP errors, network errors or a non-JSON body.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode('utf-8'), headers=dict(headers),
        method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode('utf-8', 'replace')) from None
    except (OSError, http.client.HTTPException) as e:
        raise ApiError(None, f'{type(e).__name__}: {e}') from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError(None, 'non-JSON response: '
                       f'{raw[:200].decode("utf-8", "replace")!r}') from None


def find_skill_dir(
        candidates: Sequence[pathlib.Path] = DEFAULT_SKILL_DIRS
) -> pathlib.Path:
    """Returns the first candidate holding SKILL.md and the base reference.

    Raises:
        FileNotFoundError: If no candidate is complete (submodule missing?).
    """
    for candidate in candidates:
        candidate = pathlib.Path(candidate)
        if ((candidate / 'SKILL.md').is_file()
                and (candidate / _BASE_REFERENCE).is_file()):
            return candidate
    raise FileNotFoundError(
        'h3-prompt-writing skill not found in '
        f'{[str(c) for c in candidates]}; run '
        '`git submodule update --init third_party/MiniMax-H3`')


def _strip_front_matter(text: str) -> str:
    """Drops a leading YAML front matter block (agent metadata only)."""
    if not text.startswith('---\n'):
        return text
    end = text.find('\n---\n', 4)
    if end < 0:
        raise ValueError('unterminated front matter in SKILL.md')
    return text[end + len('\n---\n'):]


def build_system_prompt(skill_dir: pathlib.Path | str) -> str:
    """Assembles the T2VA system prompt from the skill files.

    The result is identical for every request of a run, so DeepSeek's
    prefix cache bills it at the cache-hit rate after the first call.

    Args:
        skill_dir: Directory with SKILL.md and references/base-en.txt.

    Returns:
        Preamble, SKILL.md body, base-en.txt verbatim, then the T2VA output
        contract.
    """
    skill_dir = pathlib.Path(skill_dir)
    skill = _strip_front_matter(
        (skill_dir / 'SKILL.md').read_text(encoding='utf-8')).strip()
    reference = (skill_dir / _BASE_REFERENCE).read_text(
        encoding='utf-8').strip()
    return (f'{_PREAMBLE}\n\n'
            f'<skill name="h3-prompt-writing" file="SKILL.md">\n{skill}\n'
            '</skill>\n\n'
            f'<reference file="{_BASE_REFERENCE.as_posix()}">\n{reference}\n'
            '</reference>\n\n'
            f'{_T2VA_CONTRACT}')


def format_timestamp(seconds: float) -> str:
    """Formats seconds as the skill's MM:SS.mmm, e.g. 5.1667 -> '00:05.167'."""
    millis = int(round(seconds * 1000))
    minutes, millis = divmod(millis, 60000)
    return f'{minutes:02d}:{millis // 1000:02d}.{millis % 1000:03d}'


def build_user_message(request: ExpansionRequest) -> str:
    """The per-request part: duration, aspect and the source text."""
    return (f'Target duration: {request.duration_seconds:.3f} seconds '
            f'(timeline 00:00.000 to '
            f'{format_timestamp(request.duration_seconds)})\n'
            f'Aspect ratio: {request.aspect} (width:height)\n'
            f'Request: {request.source_prompt}')


def build_request_body(model: str, messages: Sequence[Mapping[str, str]],
                       reasoning_effort: str,
                       max_tokens: int | None = None) -> dict[str, Any]:
    """Chat completions body with DeepSeek's thinking controls.

    `thinking` is a top-level field on the raw HTTP API (the OpenAI SDK
    needs it in extra_body). temperature / top_p are not sent: thinking
    mode ignores temperature and clamps top_p to [0.95, 1].

    Raises:
        ValueError: On an unknown reasoning effort.
    """
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f'reasoning_effort {reasoning_effort!r} not in '
                         f'{REASONING_EFFORTS}')
    body: dict[str, Any] = {
        'model': model,
        'messages': [dict(m) for m in messages],
        'thinking': {
            'type': 'disabled' if reasoning_effort == 'none' else 'enabled'},
        'reasoning_effort': reasoning_effort,
        'stream': False,
    }
    if max_tokens is not None:
        body['max_tokens'] = max_tokens
    return body


def parse_completion(response: Mapping[str, Any]) -> Completion:
    """Extracts content, finish reason, usage and served model.

    Raises:
        ApiError: (retryable) if the response lacks the expected fields.
    """
    try:
        choice = response['choices'][0]
        content = choice['message']['content'] or ''
        finish_reason = choice['finish_reason'] or ''
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(None, f'malformed response ({e!r}): '
                       f'{json.dumps(response)[:300]}') from None
    return Completion(content=content, finish_reason=finish_reason,
                      usage=dict(response.get('usage') or {}),
                      model=str(response.get('model', '')))


class DeepSeekClient:
    """Minimal DeepSeek chat completions client."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 base_url: str = DEEPSEEK_BASE_URL,
                 transport: Transport = http_transport,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 max_tokens: int | None = None):
        if not api_key:
            raise ValueError('empty API key')
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f'reasoning_effort {reasoning_effort!r} not in '
                             f'{REASONING_EFFORTS}')
        self._api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.url = base_url.rstrip('/') + _CHAT_COMPLETIONS_PATH
        self._transport = transport
        self.timeout = timeout
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls, **kwargs: Any) -> DeepSeekClient:
        """Client with the key from $DEEPSEEK_API_KEY.

        Raises:
            KeyError: If the variable is unset or empty.
        """
        api_key = os.environ.get(API_KEY_ENV, '')
        if not api_key:
            raise KeyError(f'set {API_KEY_ENV} to the DeepSeek API key')
        return cls(api_key, **kwargs)

    def __repr__(self) -> str:
        return (f'DeepSeekClient(model={self.model!r}, '
                f'reasoning_effort={self.reasoning_effort!r}, '
                f'url={self.url!r})')

    def complete(self, messages: Sequence[Mapping[str, str]]) -> Completion:
        """One chat completion call.

        Raises:
            ApiError: On HTTP / transport failure or a malformed response.
        """
        body = build_request_body(self.model, messages, self.reasoning_effort,
                                  self.max_tokens)
        headers = {'Content-Type': 'application/json',
                   'Accept': 'application/json',
                   'Authorization': f'Bearer {self._api_key}'}
        try:
            response = self._transport(self.url, headers, body, self.timeout)
        except ApiError as e:
            # Transports are pluggable; scrub the exact key as well.
            raise ApiError(e.status, e.body.replace(self._api_key,
                                                    'sk-***')) from None
        return parse_completion(response)


def _parse_time(match: re.Match[str]) -> float:
    minutes, seconds, millis = (int(g) for g in match.groups())
    return minutes * 60 + seconds + millis / 1000


def check_expansion(prompt: str, duration_seconds: float,
                    aspect: str | None = None,
                    allow_silent_soundscape: bool = False) -> None:
    """Rejects expansions that break the skill's T2VA output rules.

    On top of `data.validate_prompt(prompt, 't2va')`: the prompt starts with
    the description field and '[Shot 1]', each field occurs once, shots are
    numbered 1..N, [Shot 1] has no cut time and every later shot opens with
    'At MM:SS.mmm,' strictly increasing and below the duration, no
    timestamp exceeds the duration, and fields are non-empty single
    paragraphs (every skill example writes shots inline; soundscape 'N/A'
    only when allowed, the skill reserves it for requested silence). The
    aspect string (e.g. '16:9') must not appear: geometry reaches the model
    through the latent grid, not the text.

    Args:
        prompt: Candidate prompt, already stripped.
        duration_seconds: Target duration.
        aspect: Target aspect ratio, checked for leaks when given.
        allow_silent_soundscape: Accept 'overall_soundscape: N/A'.

    Raises:
        ValueError: With the first violated rule.
    """
    if '```' in prompt:
        raise ValueError('Markdown code fence in the prompt')
    data.validate_prompt(prompt, TASK)
    if not prompt.startswith(_FIELDS[0]):
        raise ValueError(f'prompt must start with "{_FIELDS[0]}"')
    for field in _FIELDS:
        if prompt.count(field) != 1:
            raise ValueError(f'"{field}" occurs {prompt.count(field)} times')
    starts = [prompt.index(f) for f in _FIELDS]
    ends = starts[1:] + [len(prompt)]
    sections = [prompt[s + len(f):e].strip()
                for f, s, e in zip(_FIELDS, starts, ends)]
    for field, text in zip(_FIELDS, sections):
        if not text:
            raise ValueError(f'"{field}" is empty')
        if '\n' in text:
            raise ValueError(f'"{field}" spans several paragraphs; write it '
                             'as one paragraph with shots inline')
    description, soundscape = sections[0], sections[1]
    if soundscape.rstrip('.') == 'N/A' and not allow_silent_soundscape:
        raise ValueError('overall_soundscape is N/A but the request does '
                         'not ask for silence')
    if not description.startswith('[Shot 1]'):
        raise ValueError('description must start with "[Shot 1]"')
    if aspect is not None and aspect in prompt:
        raise ValueError(f'the prompt states the aspect ratio "{aspect}"')

    shots = list(_SHOT_RE.finditer(description))
    numbers = [int(m.group(1)) for m in shots]
    if numbers != list(range(1, len(shots) + 1)):
        raise ValueError(f'shot numbers {numbers} are not 1..N')
    limit = duration_seconds + _TIME_EPS
    previous = 0.0
    for shot in shots:
        cut = _CUT_RE.match(description, shot.end())
        if shot.group(1) == '1':
            if cut is not None:
                raise ValueError('[Shot 1] must not have a cut time')
            continue
        if cut is None:
            raise ValueError(f'[Shot {shot.group(1)}] does not start with '
                             '"At MM:SS.mmm,"')
        seconds = _parse_time(cut)
        if not previous < seconds < duration_seconds:
            raise ValueError(
                f'[Shot {shot.group(1)}] cut at {cut.group(0).strip()} is not '
                f'in ({format_timestamp(previous)}, '
                f'{format_timestamp(duration_seconds)})')
        previous = seconds
    for stamp in _TIMESTAMP_RE.finditer(prompt):
        if _parse_time(stamp) > limit:
            raise ValueError(f'timestamp {stamp.group(0)} exceeds the '
                             f'duration {format_timestamp(duration_seconds)}')


def add_usage(total: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Sums numeric usage fields (recursing into *_details) into `total`."""
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, Mapping):
            add_usage(total.setdefault(key, {}), value)


def _backoff_seconds(attempt: int) -> float:
    return min(_BACKOFF_BASE_SECONDS ** attempt, _BACKOFF_MAX_SECONDS)


def expand_one(client: DeepSeekClient, system_prompt: str,
               request: ExpansionRequest,
               max_retries: int = DEFAULT_MAX_RETRIES,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Expands one request, retrying on API errors and rejected outputs.

    A rejected output is sent back together with the checker's error (the
    next attempt is a repair turn); an API error resends the original
    messages after a backoff. Non-retryable HTTP errors (400/401/402/422)
    fail immediately.

    Args:
        client: DeepSeek client.
        system_prompt: From build_system_prompt.
        request: The request to expand.
        max_retries: Extra attempts after the first one.
        sleep: Injected for tests.

    Returns:
        Output record: id, task, source_prompt, prompt, aspect,
        duration_seconds, latent_t, model, served_model, reasoning_effort,
        attempts, usage (summed over attempts).

    Raises:
        ExpansionError: If no attempt produced an accepted prompt.
    """
    if max_retries < 0:
        raise ValueError(f'max_retries {max_retries} < 0')
    base = [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': build_user_message(request)}]
    messages = base
    errors: list[str] = []
    usage: dict[str, Any] = {}
    last_output = ''
    for attempt in range(1, max_retries + 2):
        try:
            completion = client.complete(messages)
        except ApiError as e:
            errors.append(f'attempt {attempt}: api error: {e}')
            _LOG.warning('%s: %s', request.id, errors[-1])
            if not e.retryable:
                break
            if attempt <= max_retries:
                sleep(_backoff_seconds(attempt))
            continue
        add_usage(usage, completion.usage)
        last_output = completion.content
        prompt = completion.content.strip()
        try:
            if completion.finish_reason != 'stop':
                raise ValueError(
                    f'finish_reason {completion.finish_reason!r}')
            check_expansion(prompt, request.duration_seconds,
                            request.aspect)
        except ValueError as e:
            errors.append(f'attempt {attempt}: rejected: {e}')
            _LOG.warning('%s: %s', request.id, errors[-1])
            messages = base
            if prompt:
                messages = base + [
                    {'role': 'assistant', 'content': prompt},
                    {'role': 'user', 'content': _REPAIR.format(error=e)}]
            continue
        return {
            'id': request.id,
            'task': TASK,
            'source_prompt': request.source_prompt,
            'prompt': prompt,
            'aspect': request.aspect,
            'duration_seconds': request.duration_seconds,
            'latent_t': request.latent_t,
            'model': client.model,
            'served_model': completion.model,
            'reasoning_effort': client.reasoning_effort,
            'attempts': attempt,
            'usage': usage,
        }
    raise ExpansionError(request.id, errors, usage, last_output)


def expand_all(client: DeepSeekClient, system_prompt: str,
               requests: Sequence[ExpansionRequest], concurrency: int = 4,
               max_retries: int = DEFAULT_MAX_RETRIES,
               sleep: Callable[[float], None] = time.sleep
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expands requests concurrently.

    Returns:
        (records, failures), both in request order. A failure holds id,
        source_prompt, duration_seconds, errors, usage and last_output.
    """
    if concurrency < 1:
        raise ValueError(f'concurrency {concurrency} < 1')
    ids = [r.id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate request ids')
    results: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(expand_one, client, system_prompt, r,
                               max_retries, sleep): r for r in requests}
        for future in concurrent.futures.as_completed(futures):
            request = futures[future]
            try:
                results[request.id] = future.result()
                _LOG.info('%s: accepted (%d attempts)', request.id,
                          results[request.id]['attempts'])
            except ExpansionError as e:
                _LOG.error('%s', e)
                failed[request.id] = {
                    'id': request.id,
                    'source_prompt': request.source_prompt,
                    'duration_seconds': request.duration_seconds,
                    'errors': e.errors,
                    'usage': e.usage,
                    'last_output': e.last_output,
                }
    return ([results[i] for i in ids if i in results],
            [failed[i] for i in ids if i in failed])


def read_source_prompts(source: str, count: int | None = None) -> list[str]:
    """Reads one prompt per non-empty line from a file path or http(s) URL.

    Args:
        source: Local path or URL of a UTF-8 text file.
        count: Take the first `count` prompts; None takes all.

    Raises:
        ValueError: If fewer than `count` prompts exist.
    """
    if source.startswith(('http://', 'https://')):
        with urllib.request.urlopen(source, timeout=60) as response:
            text = response.read().decode('utf-8')
    else:
        text = pathlib.Path(source).read_text(encoding='utf-8')
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if count is None:
        return prompts
    if not 0 < count <= len(prompts):
        raise ValueError(f'count {count} not in [1, {len(prompts)}]')
    return prompts[:count]


def default_duration_choices() -> tuple[float, float]:
    """Shortest and longest trainable durations (5.167 s and 14.375 s).

    These are the ends of `geometry.latent_t_ladder()` (latent_t 37 and
    102), i.e. frame-aligned durations in [5, 15] seconds.
    """
    ladder = h3_geometry.latent_t_ladder()
    return tuple(h3_geometry.frame_count_from_latent_t(t) / h3_geometry.FPS
                 for t in (ladder[0], ladder[-1]))


def assign_durations(count: int, choices: Sequence[float],
                     mode: str = 'alternate', seed: int = 0) -> list[float]:
    """Per-prompt target durations.

    Args:
        count: Number of prompts.
        choices: Candidate durations in seconds.
        mode: 'alternate' cycles through `choices` in order; 'random' draws
            uniformly with `seed`.
        seed: RNG seed for 'random'.

    Raises:
        ValueError: On an unknown mode or empty choices.
    """
    if not choices:
        raise ValueError('no duration choices')
    if mode == 'alternate':
        return [choices[i % len(choices)] for i in range(count)]
    if mode == 'random':
        rng = random.Random(seed)
        return [rng.choice(list(choices)) for _ in range(count)]
    raise ValueError(f'unknown duration mode {mode!r}')


def make_requests(prompts: Sequence[str], durations: Sequence[float],
                  aspect: str = '16:9', id_prefix: str = 'p',
                  first_index: int = 0) -> list[ExpansionRequest]:
    """Pairs prompts with frame-aligned geometries.

    Each duration is snapped to the 17n+5 frame grid by
    geometry.resolve_geometry (5.17 -> 124 frames = 5.1667 s); the snapped
    value is what the LLM is told and what is recorded.

    Raises:
        ValueError: On length mismatch or a duration outside
            (0, MAX_DURATION_SECONDS] after snapping.
    """
    if len(prompts) != len(durations):
        raise ValueError(f'{len(prompts)} prompts vs {len(durations)} '
                         'durations')
    requests = []
    for i, (prompt, duration) in enumerate(zip(prompts, durations)):
        geo = h3_geometry.resolve_geometry(aspect, duration)
        if not 0 < geo.duration_seconds <= h3_geometry.MAX_DURATION_SECONDS:
            raise ValueError(f'duration {duration} snaps to '
                             f'{geo.duration_seconds:.3f} s, outside (0, '
                             f'{h3_geometry.MAX_DURATION_SECONDS}]')
        requests.append(ExpansionRequest(
            id=f'{id_prefix}_{first_index + i:04d}', source_prompt=prompt,
            aspect=aspect, duration_seconds=geo.duration_seconds,
            latent_t=geo.latent_t))
    return requests


def write_jsonl(path: pathlib.Path | str,
                records: Sequence[Mapping[str, Any]]) -> None:
    """Writes one JSON object per line (UTF-8, non-ASCII kept)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
