"""Content isolation and split-preserving encoding contracts on CPU."""

import json

import pytest
import torch

from miowtion.train import data
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_data as split_data
from miowtion.train import sparse_lora_trainer as trainer


def _record(index, split='train'):
    prompt = (f'integrated_multimodal_description: [Shot 1] case {index}.\n'
              'overall_soundscape: wind.\nnon_diegetic_music: none.')
    return {'id': str(index), 'task': 't2va', 'split': split, 'prompt': prompt,
            'aspect': '16:9', 'latent_t': 37,
            'text_family_id': f'family{index}', 'source_job_id': f'job{index}',
            'parent_id': f'parent{index}'}


def test_transitive_source_and_family_groups():
    a, b, c, d = [_record(i) for i in range(4)]
    b['source_job_id'] = a['source_job_id']
    c['text_family_id'] = b['text_family_id']
    groups = split_data.content_groups([a, b, c, d])
    assert sorted(len(g) for g in groups) == [1, 3]
    with pytest.raises(ValueError, match='share content'):
        split_data.assert_disjoint([a], [b])
    with pytest.raises(ValueError, match='share content'):
        split_data.assert_disjoint([b], [c])
    split_data.assert_disjoint([a, b, c], [d])


def test_splits_reproduce_verbatim_and_pilot_is_independent(tmp_path):
    records = [_record(i) for i in range(200)]
    source = tmp_path / 'source.jsonl'
    split_data.write_manifest(source, records)
    reports = []
    for name in ('a', 'b'):
        reports.append(split_data.prepare_splits(str(source), str(tmp_path / name)))
    assert reports[0] == reports[1]
    for path in (tmp_path / 'a').glob('*.jsonl'):
        assert path.read_bytes() == (tmp_path / 'b' / path.name).read_bytes()
    train = split_data.read_manifest(tmp_path / 'a' / 'full.train.jsonl')
    validation = split_data.read_manifest(tmp_path / 'a' / 'full.validation.jsonl')
    assert len(train) == 180 and len(validation) == 20
    assert {r['id']: r['prompt'] for r in train + validation} == {
        r['id']: r['prompt'] for r in records}
    split_data.assert_disjoint(train, validation)
    with pytest.raises(FileExistsError):
        split_data.prepare_splits(str(source), str(tmp_path / 'a'))


def test_cache_prompt_and_assignment_mismatch_is_rejected(tmp_path):
    records = [_record(0), _record(1, 'validation')]
    writer = data.SampleCacheWriter(str(tmp_path))
    for record in records:
        writer.add(record['id'], 't2va', record['split'], torch.zeros(3, 8),
                   torch.ones(3, dtype=torch.int64), aspect='16:9', latent_t=37)
    writer.finalize()
    provenance = {'variant': 'FL2VA', 'text_layers': 50,
                  'chat_template': False, 'special_tokens': False,
                  'prompts': {r['id']: core.canonical_hash(r['prompt'])
                              for r in records}}
    proof = tmp_path / 'sparse_lora_encoding.json'
    proof.write_text(json.dumps(provenance))
    cfg = core.SparseLoraConfig('x', '', '', '', '', '', str(tmp_path), ['16:9@37'])
    cache = data.SampleCache(str(tmp_path))
    selected = trainer.validate_cache(cfg, cache, records[:1], records[1:])
    assert [s.id for s in selected] == ['0']
    provenance['prompts']['0'] = 'wrong'
    proof.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='encoded prompt'):
        trainer.validate_cache(cfg, cache, records[:1], records[1:])
    proof.write_text(json.dumps({**provenance, 'prompts': {
        r['id']: core.canonical_hash(r['prompt']) for r in records}}))
    changed = {**records[0], 'aspect': '4:3'}
    with pytest.raises(ValueError, match='assignment'):
        trainer.validate_cache(cfg, cache, [changed], records[1:])


def test_explicit_split_encoder_does_not_resplit_or_drop_aspect(tmp_path,
                                                             monkeypatch):
    from miowtion.train import encode

    class TinyEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_t2va(self, prompt):
            return torch.ones(3, 8), torch.ones(3, dtype=torch.int64)

    monkeypatch.setattr(encode, 'TextEncoder', TinyEncoder)
    records = [_record(0), _record(1, 'validation')]
    source = tmp_path / 'input.jsonl'
    split_data.write_manifest(source, records)
    split_data.encode_manifest('weights', str(source), str(tmp_path / 'cache'))
    cache = data.SampleCache(str(tmp_path / 'cache'))
    assert [(s.id, s.split, s.aspect, s.latent_t) for s in cache.samples] == [
        ('0', 'train', '16:9', 37), ('1', 'validation', '16:9', 37)]
