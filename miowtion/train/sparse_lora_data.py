"""Reproducible content-group splits and split-preserving T2VA encoding."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from miowtion.train import data
from miowtion.train.sparse_lora import canonical_hash
from miowtion.train.sparse_lora import sha256_file
from miowtion.utils import progress


def read_manifest(path: str | Path) -> list[dict]:
    """Reads unique T2VA records, checking PE and the recorded prompt hash."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()
               if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('manifest must have nonempty unique ids')
    for record in records:
        if record['task'] != 't2va':
            raise ValueError('only text-only T2VA is supported')
        data.validate_prompt(record['prompt'])
        digest = hashlib.sha256(record['prompt'].encode()).hexdigest()
        if record.get('prompt_sha256', digest) != digest:
            raise ValueError(f'prompt hash mismatch: {record["id"]}')
        if record.get('keyframes') or record.get('references'):
            raise ValueError('T2VA manifest cannot contain references')
    return records


def _identities(record: dict) -> set[tuple[str, str]]:
    identities = {(key, str(record[key])) for key in
                  ('id', 'text_family_id', 'source_job_id', 'parent_id',
                   'parent_prompt_sha256') if record.get(key)}
    normalized = ' '.join(record['prompt'].casefold().split())
    identities.add(('text', hashlib.sha256(normalized.encode()).hexdigest()))
    return identities


def content_groups(records: list[dict]) -> list[list[dict]]:
    """Unions parent, source, family, and text overlaps transitively."""
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owner = {}
    for index, record in enumerate(records):
        for identity in _identities(record):
            if identity in owner:
                parents[root(index)] = root(owner[identity])
            else:
                owner[identity] = index
    groups = collections.defaultdict(list)
    for index, record in enumerate(records):
        groups[root(index)].append(record)
    return list(groups.values())


def assert_disjoint(train: list[dict], validation: list[dict]) -> None:
    """Refuses any known content identity crossing the requested split."""
    train_keys = set().union(*(_identities(r) for r in train))
    validation_keys = set().union(*(_identities(r) for r in validation))
    if train_keys & validation_keys:
        raise ValueError('train/validation share content identities')


def write_manifest(path: Path, records: list[dict]) -> None:
    """Writes verbatim prompt strings and provenance as UTF-8 JSONL."""
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                            for r in records))


def prepare_splits(source: str, destination: str, seed: int = 20261015
                   ) -> dict:
    """Keeps group isolation; reserves ~10% per aspect/duration stratum.

    All 1k contents remain in the full train/validation union. The initial
    pilot is deliberately 5s/16:9, allowing an exact published tile plan.
    Its smoke subset is training-seen, never an independent evaluation.
    """
    records = read_manifest(source)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    strata = collections.defaultdict(list)
    for group in content_groups(records):
        stratum = min((r['aspect'], r['latent_t']) for r in group)
        strata[stratum].append(group)
    train, validation = [], []
    # Largest-remainder allocation gives exactly 10% for single-row groups.
    targets = {k: len(v) // 10 for k, v in strata.items()}
    remaining = round(sum(map(len, strata.values())) * 0.1) - sum(
        targets.values())
    order = sorted(strata, key=lambda k: (-(len(strata[k]) % 10), k))
    for key in order[:remaining]:
        targets[key] += 1
    for key, groups in sorted(strata.items()):
        ordered = sorted(groups, key=lambda g: canonical_hash(
            [seed, sorted(r['id'] for r in g)]))
        for index, group in enumerate(ordered):
            split = 'validation' if index < targets[key] else 'train'
            destination_rows = validation if split == 'validation' else train
            destination_rows.extend({**r, 'split': split} for r in group)
    assert_disjoint(train, validation)
    for name, rows in [('train', train), ('validation', validation)]:
        write_manifest(output / f'full.{name}.jsonl', rows)
    pilot = {}
    for name, rows, count in [('train', train, 24),
                              ('validation', validation, 12)]:
        eligible = [r for r in rows if r['aspect'] == '16:9'
                    and r['latent_t'] == 37]
        eligible.sort(key=lambda r: canonical_hash([seed, r['id']]))
        if len(eligible) < count:
            raise ValueError(f'not enough independent pilot {name} samples')
        pilot[name] = eligible[:count]
        write_manifest(output / f'pilot.{name}.jsonl', pilot[name])
    assert_disjoint(pilot['train'], pilot['validation'])
    write_manifest(output / 'smoke.train.jsonl', pilot['train'][:2])
    write_manifest(output / 'pilot.encode.jsonl',
                   pilot['train'] + pilot['validation'])
    report = {
        'source_sha256': sha256_file(source), 'seed': seed,
        'full_train': len(train), 'full_validation': len(validation),
        'pilot_train': 24, 'pilot_validation': 12, 'smoke_train_subset': 2,
        'content_group_disjoint': True,
        'group_keys': ['parent_id', 'source_job_id', 'text_family_id',
                       'parent_prompt_sha256', 'normalized_prompt'],
        'counts': {name: dict(collections.Counter(
            f'{r["aspect"]}|t{r["latent_t"]}' for r in rows))
                   for name, rows in [('train', train),
                                      ('validation', validation)]},
        'limits': [
            'Independence is relative to this Sparse LoRA split only.',
            'User identity is not a split key; prompts may share users.',
            'No claim of unseen historical predictor/Turbo training content.',
            'Pilot contents are hash-selected, not visually quality reviewed.',
            'Known cached flash cases remain separate regression evaluation.',
        ],
    }
    (output / 'split_report.json').write_text(json.dumps(report, indent=2))
    return report


def encode_manifest(root: str, manifest: str, destination: str,
                    max_memory: dict | None = None) -> None:
    """Encodes explicit splits/aspects without randomly splitting again."""
    from miowtion.train import encode
    records = read_manifest(manifest)
    if Path(destination).exists():
        raise FileExistsError('refusing to overwrite an existing sample cache')
    if any(r['split'] not in ('train', 'validation') for r in records):
        raise ValueError('explicit train/validation labels required')
    encoder = encode.TextEncoder(f'{root}/FL2VA', max_memory=max_memory)
    writer = data.SampleCacheWriter(destination)
    counter = progress.Progress('encode Sparse LoRA text', len(records))
    for record in records:
        hidden, tags = encoder.encode_t2va(record['prompt'])
        writer.add(record['id'], 't2va', record['split'], hidden, tags,
                   aspect=record['aspect'], latent_t=record['latent_t'])
        counter.update(record['id'])
    writer.finalize()
    provenance = {
        'manifest_sha256': sha256_file(manifest),
        'prompts': {r['id']: canonical_hash(r['prompt']) for r in records},
        'variant': 'FL2VA', 'text_layers': 50,
        'chat_template': False, 'special_tokens': False,
    }
    Path(destination, 'sparse_lora_encoding.json').write_text(
        json.dumps(provenance, indent=2))
