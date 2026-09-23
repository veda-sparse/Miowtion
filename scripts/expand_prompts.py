"""Expands short video prompts into structured H3 T2VA prompts (DeepSeek).

Reads one prompt per non-empty line, assigns each a target duration, asks
DeepSeek to rewrite it following the MiniMax-H3 h3-prompt-writing skill and
writes a manifest for scripts/encode_samples.py. Prompts that fail every
attempt go to <output>.failures.jsonl and the exit status is 1.

The API key is taken from $DEEPSEEK_API_KEY only.

Example:
    DEEPSEEK_API_KEY=... python scripts/expand_prompts.py \
        --source https://raw.githubusercontent.com/guandeh17/Self-Forcing/main/prompts/MovieGenVideoBench.txt \
        --count 10 --id-prefix moviegen \
        --output artifacts/prompts/moviegen_smoke10.jsonl
"""

import argparse
import logging
import pathlib
import sys
import time

from miowtion.train import prompt_expansion as pe


def _durations(text):
    return [float(v) for v in text.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--source', required=True,
                        help='prompt file path or http(s) URL')
    parser.add_argument('--count', type=int, default=None,
                        help='first N non-empty lines (default: all)')
    parser.add_argument('--output', required=True, help='output .jsonl')
    parser.add_argument('--id-prefix', default='p')
    parser.add_argument('--model', default=pe.DEFAULT_MODEL)
    parser.add_argument('--reasoning-effort',
                        default=pe.DEFAULT_REASONING_EFFORT,
                        choices=pe.REASONING_EFFORTS,
                        help='thinking level; "none" disables thinking')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max-retries', type=int,
                        default=pe.DEFAULT_MAX_RETRIES)
    parser.add_argument('--aspect', default='16:9')
    parser.add_argument('--durations', type=_durations, default=None,
                        help='comma-separated seconds, snapped to the frame '
                        'grid (default: shortest and longest trainable)')
    parser.add_argument('--duration-mode', choices=('alternate', 'random'),
                        default='alternate')
    parser.add_argument('--seed', type=int, default=0,
                        help='duration assignment seed (random mode)')
    parser.add_argument('--skill-dir', default=None,
                        help='h3-prompt-writing directory (default: the '
                        'third_party/MiniMax-H3 submodule)')
    parser.add_argument('--base-url', default=pe.DEEPSEEK_BASE_URL)
    parser.add_argument('--timeout', type=float,
                        default=pe.DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    client = pe.DeepSeekClient.from_env(
        model=args.model, reasoning_effort=args.reasoning_effort,
        base_url=args.base_url, timeout=args.timeout)
    skill_dir = (pathlib.Path(args.skill_dir) if args.skill_dir
                 else pe.find_skill_dir())
    system_prompt = pe.build_system_prompt(skill_dir)
    prompts = pe.read_source_prompts(args.source, args.count)
    durations = pe.assign_durations(
        len(prompts), args.durations or pe.default_duration_choices(),
        args.duration_mode, args.seed)
    requests = pe.make_requests(prompts, durations, args.aspect,
                                args.id_prefix)
    logging.info('%d prompts, skill %s, %r', len(requests), skill_dir, client)

    start = time.monotonic()
    records, failures = pe.expand_all(client, system_prompt, requests,
                                      args.concurrency, args.max_retries)
    wall = time.monotonic() - start
    pe.write_jsonl(args.output, records)
    usage = {}
    for r in records + failures:
        pe.add_usage(usage, r['usage'])
    print(f'accepted {len(records)}/{len(requests)} -> {args.output}; '
          f'wall {wall:.1f} s; usage {usage}')
    if failures:
        failures_path = f'{args.output}.failures.jsonl'
        pe.write_jsonl(failures_path, failures)
        print(f'{len(failures)} failed -> {failures_path}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
