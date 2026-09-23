"""CPU tests of prompt expansion: no network, fake transports only."""

import json
import re
import threading

import pytest

from miowtion.train import data
from miowtion.train import prompt_expansion as pe

_KEY = 'sk-test0123456789'
_SHORT = 124 / 24  # 5.1667 s, latent_t 37
_LONG = 14.375  # latent_t 102

_GOOD = ('integrated_multimodal_description: [Shot 1] Live-action, '
         'cinematic, a medium-wide shot frames a baker at the counter. '
         '[Shot 2] At 00:03.000, the camera cuts to a close-up of bread.\n\n'
         'overall_soundscape: Trays clink softly.\n\n'
         'non_diegetic_music: Sparse piano notes at a slow tempo.')


def _response(content, finish_reason='stop', prompt_tokens=100,
              completion_tokens=20, reasoning_tokens=15):
    return {
        'id': 'x', 'object': 'chat.completion', 'model': 'deepseek-flash',
        'choices': [{'index': 0, 'finish_reason': finish_reason,
                     'message': {'role': 'assistant', 'content': content,
                                 'reasoning_content': 'thinking...'}}],
        'usage': {'prompt_tokens': prompt_tokens,
                  'completion_tokens': completion_tokens,
                  'total_tokens': prompt_tokens + completion_tokens,
                  'prompt_cache_hit_tokens': 0,
                  'prompt_cache_miss_tokens': prompt_tokens,
                  'completion_tokens_details': {
                      'reasoning_tokens': reasoning_tokens}},
    }


class _FakeTransport:
    """Replays scripted responses (dicts) or raises scripted ApiErrors."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self._lock:
            self.calls.append({'url': url, 'headers': dict(headers),
                               'body': json.loads(json.dumps(body)),
                               'timeout': timeout})
            item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(script, **kwargs):
    transport = _FakeTransport(script)
    return pe.DeepSeekClient(_KEY, transport=transport, **kwargs), transport


def _request(duration=_SHORT, request_id='p_0000'):
    return pe.make_requests(['A baker opens a bakery.'], [duration],
                            id_prefix=request_id.split('_')[0])[0]


@pytest.fixture(name='skill_dir')
def _skill_dir(tmp_path):
    (tmp_path / 'references').mkdir()
    (tmp_path / 'SKILL.md').write_text(
        '---\nname: h3-prompt-writing\ndescription: x\n---\n\n'
        '# H3 Prompt Writing\nSKILL BODY\n', encoding='utf-8')
    (tmp_path / 'references' / 'base-en.txt').write_text(
        '# Guide\nBASE REFERENCE\n', encoding='utf-8')
    return tmp_path


def test_system_prompt_is_built_from_skill_files(skill_dir):
    prompt = pe.build_system_prompt(skill_dir)
    assert 'SKILL BODY' in prompt and 'BASE REFERENCE' in prompt
    assert 'name: h3-prompt-writing' not in prompt  # front matter dropped
    assert prompt.index('SKILL BODY') < prompt.index('BASE REFERENCE')
    assert 'integrated_multimodal_description: [Shot 1]' in prompt
    assert pe.find_skill_dir([skill_dir.parent / 'missing', skill_dir]) == (
        skill_dir)
    with pytest.raises(FileNotFoundError):
        pe.find_skill_dir([skill_dir / 'references'])


def test_vendored_skill_loads():
    try:
        skill_dir = pe.find_skill_dir()
    except FileNotFoundError:
        pytest.skip('MiniMax-H3 submodule not checked out')
    prompt = pe.build_system_prompt(skill_dir)
    assert '## 2. Final Prompt Structure' in prompt  # base-en.txt
    assert '# H3 Prompt Writing' in prompt  # SKILL.md


def test_user_message_carries_geometry():
    message = pe.build_user_message(_request(_SHORT))
    assert '5.167 seconds' in message and '00:05.167' in message
    assert '16:9 (width:height)' in message
    assert 'landscape' not in message  # leaked into prompts as a word
    assert message.endswith('Request: A baker opens a bakery.')
    assert pe.format_timestamp(_LONG) == '00:14.375'
    assert pe.format_timestamp(75.5) == '01:15.500'


def test_request_body_and_headers():
    client, transport = _client([_response(_GOOD)])
    completion = client.complete([{'role': 'user', 'content': 'hi'}])
    call = transport.calls[0]
    assert call['url'] == 'https://api.deepseek.com/chat/completions'
    assert call['headers']['Authorization'] == f'Bearer {_KEY}'
    assert call['body'] == {
        'model': 'deepseek-flash',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'thinking': {'type': 'enabled'},
        'reasoning_effort': 'medium',
        'stream': False,
    }
    assert completion.content == _GOOD
    assert completion.finish_reason == 'stop'
    assert completion.model == 'deepseek-flash'
    assert completion.usage['completion_tokens_details'] == {
        'reasoning_tokens': 15}

    body = pe.build_request_body('m', [], 'none', max_tokens=100)
    assert body['thinking'] == {'type': 'disabled'}
    assert body['max_tokens'] == 100
    with pytest.raises(ValueError):
        pe.build_request_body('m', [], 'extreme')


def test_api_key_stays_out_of_repr_and_errors(monkeypatch):
    client, _ = _client([pe.ApiError(401, f'bad key {_KEY}')])
    assert _KEY not in repr(client)
    with pytest.raises(pe.ApiError) as info:
        client.complete([])
    assert _KEY not in str(info.value) and info.value.status == 401
    assert not info.value.retryable
    monkeypatch.delenv(pe.API_KEY_ENV, raising=False)
    with pytest.raises(KeyError):
        pe.DeepSeekClient.from_env()
    monkeypatch.setenv(pe.API_KEY_ENV, _KEY)
    assert pe.DeepSeekClient.from_env().model == pe.DEFAULT_MODEL


def test_parse_completion_rejects_malformed_response():
    with pytest.raises(pe.ApiError) as info:
        pe.parse_completion({'error': 'nope'})
    assert info.value.retryable
    assert pe.parse_completion(_response(None)).content == ''


def test_check_expansion_accepts_skill_shaped_prompt():
    pe.check_expansion(_GOOD, _SHORT)
    data.validate_prompt(_GOOD, 't2va')
    single = _GOOD.split(' [Shot 2]')[0] + _GOOD[_GOOD.index('\n\n'):]
    pe.check_expansion(single, _SHORT)
    silent = _GOOD.replace('Trays clink softly.', 'N/A')
    pe.check_expansion(silent, _SHORT, allow_silent_soundscape=True)
    with pytest.raises(ValueError, match='aspect ratio'):
        pe.check_expansion(_GOOD.replace('medium-wide', 'wide 16:9'), _SHORT,
                           '16:9')


@pytest.mark.parametrize('prompt, reason', [
    (_GOOD.replace('[Shot 1] ', ''), '[Shot 1]'),
    ('Here is the prompt:\n' + _GOOD, 'must start with'),
    ('```\n' + _GOOD + '\n```', 'code fence'),
    (_GOOD + '\n\noverall_soundscape: again.', 'occurs 2 times'),
    (_GOOD.replace('[Shot 2]', '[Shot 3]'), 'not 1..N'),
    (_GOOD.replace('At 00:03.000, ', ''), 'does not start with'),
    (_GOOD.replace(' [Shot 2]', '\n\n[Shot 2]'), 'several paragraphs'),
    (_GOOD.replace('00:03.000', '00:05.200'), 'is not in'),
    (_GOOD.replace('[Shot 1] ', '[Shot 1] At 00:00.500, '), 'must not'),
    (_GOOD.replace('bread.', 'bread. [Shot 3] At 00:02.000, a hand.'),
     'is not in'),
    (_GOOD.replace('softly.', 'softly until 00:06.000.'), 'exceeds'),
    (_GOOD.replace('Trays clink softly.', 'N/A'), 'N/A'),
    (_GOOD.replace('Sparse piano notes at a slow tempo.', ''), 'is empty'),
])
def test_check_expansion_rejects(prompt, reason):
    with pytest.raises(ValueError, match=re.escape(reason)):
        pe.check_expansion(prompt, _SHORT)


def test_expand_one_repairs_rejected_output():
    bad = _GOOD.replace('00:03.000', '00:09.000')
    client, transport = _client([_response(bad), _response(_GOOD)])
    sleeps = []
    record = pe.expand_one(client, 'SYSTEM', _request(), sleep=sleeps.append)
    assert record['prompt'] == _GOOD and record['attempts'] == 2
    assert sleeps == []  # rejections are not rate-limit problems
    assert record['usage']['prompt_tokens'] == 200
    assert record['usage']['completion_tokens_details'] == {
        'reasoning_tokens': 30}
    first, second = (c['body']['messages'] for c in transport.calls)
    assert [m['role'] for m in first] == ['system', 'user']
    assert first[0]['content'] == 'SYSTEM'
    assert second[:2] == first
    assert second[2] == {'role': 'assistant', 'content': bad}
    assert second[3]['role'] == 'user' and '00:09.000' in second[3]['content']
    assert {k: record[k] for k in ('id', 'task', 'aspect', 'latent_t',
                                   'model', 'reasoning_effort')} == {
        'id': 'p_0000', 'task': 't2va', 'aspect': '16:9', 'latent_t': 37,
        'model': 'deepseek-flash', 'reasoning_effort': 'medium'}


def test_expand_one_retries_api_errors_with_backoff():
    client, transport = _client([
        pe.ApiError(503, 'overloaded'), pe.ApiError(None, 'timeout'),
        _response(_GOOD, finish_reason='length'), _response(_GOOD)])
    sleeps = []
    record = pe.expand_one(client, 'S', _request(), sleep=sleeps.append)
    assert record['attempts'] == 4 and sleeps == [2.0, 4.0]
    # API errors resend the original messages.
    assert all(len(c['body']['messages']) == 2 for c in transport.calls[:3])
    # A truncated completion is rejected, then repaired.
    assert len(transport.calls[3]['body']['messages']) == 4


def test_expand_one_gives_up():
    client, transport = _client([_response('garbage')] * 4)
    with pytest.raises(pe.ExpansionError) as info:
        pe.expand_one(client, 'S', _request(), max_retries=3,
                      sleep=lambda s: None)
    assert len(transport.calls) == 4 and len(info.value.errors) == 4
    assert info.value.last_output == 'garbage'
    assert info.value.usage['prompt_tokens'] == 400

    client, transport = _client([pe.ApiError(401, 'auth'), _response(_GOOD)])
    with pytest.raises(pe.ExpansionError, match='HTTP 401'):
        pe.expand_one(client, 'S', _request(), sleep=lambda s: None)
    assert len(transport.calls) == 1  # not retryable


def test_expand_all_keeps_order_and_reports_failures():
    requests = pe.make_requests(['a', 'b', 'c'], [_SHORT, _LONG, _SHORT],
                                id_prefix='m')

    def transport(url, headers, body, timeout):
        del url, headers, timeout
        if 'Request: b' in body['messages'][1]['content']:
            return _response('not a prompt')
        return _response(_GOOD)

    client = pe.DeepSeekClient(_KEY, transport=transport)
    records, failures = pe.expand_all(client, 'S', requests, concurrency=3,
                                      max_retries=1, sleep=lambda s: None)
    assert [r['id'] for r in records] == ['m_0000', 'm_0002']
    assert [f['id'] for f in failures] == ['m_0001']
    assert len(failures[0]['errors']) == 2
    assert failures[0]['duration_seconds'] == _LONG
    with pytest.raises(ValueError):
        pe.expand_all(client, 'S', requests + requests[:1])


def test_durations_and_requests():
    assert pe.default_duration_choices() == (_SHORT, _LONG)
    assert pe.assign_durations(4, (1.0, 2.0)) == [1.0, 2.0, 1.0, 2.0]
    drawn = pe.assign_durations(20, (1.0, 2.0), 'random', seed=3)
    assert drawn == pe.assign_durations(20, (1.0, 2.0), 'random', seed=3)
    assert set(drawn) == {1.0, 2.0}
    with pytest.raises(ValueError):
        pe.assign_durations(2, (1.0,), 'shuffle')
    requests = pe.make_requests(['a', 'b'], [5.17, 14.375], '16:9',
                                'moviegen', first_index=10)
    assert [r.id for r in requests] == ['moviegen_0010', 'moviegen_0011']
    assert [r.latent_t for r in requests] == [37, 102]
    assert [r.duration_seconds for r in requests] == [_SHORT, _LONG]
    with pytest.raises(ValueError):
        pe.make_requests(['a'], [15.0])  # snaps to 362 frames > 15 s
    with pytest.raises(ValueError):
        pe.make_requests(['a'], [5.0, 6.0])


def test_read_source_prompts_and_write_jsonl(tmp_path):
    source = tmp_path / 'prompts.txt'
    source.write_text('first\n\n  second  \nthird\n', encoding='utf-8')
    assert pe.read_source_prompts(str(source)) == ['first', 'second', 'third']
    assert pe.read_source_prompts(str(source), 2) == ['first', 'second']
    with pytest.raises(ValueError):
        pe.read_source_prompts(str(source), 4)
    out = tmp_path / 'out' / 'records.jsonl'
    pe.write_jsonl(out, [{'id': 'a', 'task': 't2va', 'prompt': _GOOD,
                          'source_prompt': 'Big Sur’s'}])
    line = out.read_text(encoding='utf-8')
    assert 'Big Sur’s' in line
    assert data.load_prompts(str(out)) == [_GOOD]
