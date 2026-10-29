"""Exact mixed-geometry coverage, rank partitioning and resumable cursors."""

import collections

import pytest

from miowtion.train import data
from miowtion.train import sparse_lora_sampling as sampling


def _samples(counts):
    return [data.Sample(f'{aspect}-{t}-{i}', 't2va', 'train', (0, 0),
                        aspect=aspect, latent_t=t)
            for (aspect, t), count in counts.items() for i in range(count)]


def test_balanced_window_keeps_consumed_batches_and_full_epoch_coverage():
    samples = _samples({('16:9', 37): 120, ('1:1', 72): 120,
                        ('4:3', 102): 200})
    original = sampling.CoverageSampler(samples, 20261015, 0, 4)
    balanced = sampling.CoverageSampler(
        samples, 20261015, 0, 4, duration_balance_window=50,
        duration_balance_prefix=5)
    assert balanced._order[:5] == original._order[:5]
    assert sorted(map(tuple, balanced._order)) == sorted(
        map(tuple, original._order))
    counts = collections.Counter(samples[b[0]].latent_t
                                 for b in balanced._order[:50])
    assert counts == {37: 17, 72: 17, 102: 16}
    for duration in counts:
        before = [b for b in original._order[5:]
                  if samples[b[0]].latent_t == duration]
        after = [b for b in balanced._order[5:]
                 if samples[b[0]].latent_t == duration]
        assert before == after


def test_balanced_four_rank_replay_preserves_geometry_padding_and_coverage():
    samples = _samples({('16:9', 37): 43, ('1:1', 72): 45,
                        ('4:3', 102): 81})
    ranks = [sampling.CoverageSampler(
        samples, 21, rank, 4, duration_balance_window=21)
             for rank in range(4)]
    seen = collections.Counter()
    for _ in range(2 * ranks[0].batches_per_epoch):
        batch = [rank.next() for rank in ranks]
        assert len({(s.aspect, s.latent_t) for s in batch}) == 1
        seen.update(s.id for s, rank in zip(batch, ranks) if rank.last_valid)
    assert seen == collections.Counter({s.id: 2 for s in samples})
    restored = [sampling.CoverageSampler(
        samples, 21, rank, 4, duration_balance_window=21)
                for rank in range(4)]
    for old, new in zip(ranks, restored):
        new.load_state(old.state())
        for _ in range(old.batches_per_epoch + 3):
            assert old.next() == new.next()
            assert old.last_valid == new.last_valid


def test_balanced_sampler_rejects_implicit_policy_changes():
    samples = _samples({('16:9', 37): 24, ('1:1', 72): 24,
                        ('4:3', 102): 24})
    old = sampling.CoverageSampler(samples, 5, 0, 4)
    new = sampling.CoverageSampler(
        samples, 5, 0, 4, duration_balance_window=12)
    with pytest.raises(ValueError, match='sampler state'):
        new.load_state(old.state())
    with pytest.raises(ValueError, match='sampler state'):
        old.load_state(new.state())
    with pytest.raises(ValueError, match='sampler state'):
        other = sampling.CoverageSampler(
            samples, 5, 0, 4, duration_balance_window=15)
        other.load_state(new.state())
    with pytest.raises(ValueError, match='window/prefix'):
        sampling.CoverageSampler(
            samples, 5, 0, 4, duration_balance_window=100)


def test_each_mixed_prompt_exactly_once_per_epoch_and_cursor_resume():
    # Unequal bucket sizes reproduce the old filter-and-skip failure.
    samples = _samples({('16:9', 37): 139, ('16:9', 72): 95,
                        ('16:9', 102): 144, ('1:1', 37): 54,
                        ('1:1', 72): 54, ('1:1', 102): 54,
                        ('4:3', 37): 54, ('4:3', 72): 54,
                        ('4:3', 102): 54, ('9:16', 37): 43,
                        ('9:16', 72): 63, ('9:16', 102): 92})
    sampler = sampling.CoverageSampler(samples, 20261015)
    for _ in range(2):
        seen = [sampler.next().id for _ in samples]
        assert collections.Counter(seen) == collections.Counter(
            s.id for s in samples)
    assert sampler.issued == 1800
    for prefix in (0, 1, 899, 900, 901, 1799):
        first = sampling.CoverageSampler(samples, 31)
        for _ in range(prefix):
            first.next()
        restored = sampling.CoverageSampler(samples, 31)
        restored.load_state(first.state())
        assert [first.next().id for _ in range(905)] == [
            restored.next().id for _ in range(905)]
        assert first.state() == restored.state()


@pytest.mark.parametrize('world_size', [2, 4, 8])
def test_rank_batches_share_geometry_without_duplicate_or_dropped_prompts(
        world_size):
    samples = _samples({('16:9', 37): 16, ('1:1', 72): 8,
                        ('4:3', 102): 24})
    ranks = [sampling.CoverageSampler(samples, 17, rank, world_size)
             for rank in range(world_size)]
    for _ in range(2):
        seen = []
        for _ in range(len(samples) // world_size):
            batch = [rank.next() for rank in ranks]
            assert len({(s.aspect, s.latent_t) for s in batch}) == 1
            assert len({s.id for s in batch}) == world_size
            seen.extend(s.id for s in batch)
        assert collections.Counter(seen) == collections.Counter(
            s.id for s in samples)


def test_invalid_and_legacy_cursors_are_rejected():
    samples = _samples({('16:9', 37): 3, ('1:1', 72): 5})
    sampler = sampling.CoverageSampler(samples, 5)
    for state in ({'epoch': 0, 'position': 0},
                  {**sampler.state(), 'position': 9},
                  {**sampler.state(), 'epoch': -1},
                  {**sampler.state(), 'position': True}):
        with pytest.raises(ValueError, match='sampler state'):
            sampler.load_state(state)


def test_workload_includes_short_final_optimizer_update():
    assert sampling.planned_updates(900, 1, 4, None, 2) == 450
    assert sampling.planned_updates(900, 8, 8, None, 2) == 1800
    assert sampling.planned_updates(24, 8, 8, 50, None) == 50
    assert sampling.planned_updates(112, 8, 4, None, 2) == 448
    assert sampling.planned_updates(5, 8, 3, None, 2) == 27
    assert sampling.planned_updates(116, 1, 4, None, 2) == 58


@pytest.mark.parametrize('world_size', [2, 4, 8])
def test_short_geometry_batches_have_zero_weight_slots_and_exact_coverage(
        world_size):
    samples = _samples({('16:9', 37): 3, ('1:1', 72): 5,
                        ('4:3', 102): 1})
    ranks = [sampling.CoverageSampler(samples, 19, rank, world_size)
             for rank in range(world_size)]
    counts = collections.Counter()
    pads = 0
    for _ in range(2 * ranks[0].batches_per_epoch):
        batch = [sampler.next() for sampler in ranks]
        assert len({(s.aspect, s.latent_t) for s in batch}) == 1
        valid = [s.id for s, sampler in zip(batch, ranks) if sampler.last_valid]
        assert len(valid) == ranks[0].last_batch_size
        assert len(valid) == len(set(valid))
        counts.update(valid)
        pads += world_size - len(valid)
        assert len({s.issued_real_global for s in ranks}) == 1
    assert counts == collections.Counter({s.id: 2 for s in samples})
    assert ranks[0].issued_real_global == 2 * len(samples)
    assert pads == 2 * (ranks[0].batches_per_epoch * world_size - len(samples))
    for sampler in ranks:
        copy = sampling.CoverageSampler(samples, 19, sampler.rank, world_size)
        copy.load_state(sampler.state())
        for _ in range(2 * sampler.batches_per_epoch + 1):
            assert sampler.next().id == copy.next().id
            assert sampler.last_valid == copy.last_valid
            assert sampler.issued_real_global == copy.issued_real_global


def test_update_lookahead_spans_epochs_and_unfinished_trajectories():
    samples = _samples({('16:9', 37): 3, ('1:1', 72): 1})
    sampler = sampling.CoverageSampler(samples, 7, 1, 2)
    original = sampler.state()
    sizes = sampler.peek_batch_sizes(7)
    assert sampler.state() == original
    actual = []
    for _ in range(7):
        sampler.next()
        actual.append(sampler.last_batch_size)
    assert actual == sizes
    size = sampler.last_batch_size
    next_size = sampler.peek_batch_sizes(1)[0]
    cursor = sampler.state()
    assert sampler.valid_state_count(10, 8, 3) == size * 3 + next_size * 7
    assert sampler.state() == cursor
    restored = sampling.CoverageSampler(samples, 7, 1, 2)
    restored.load_state(cursor)
    assert restored.last_valid == sampler.last_valid
    assert restored.valid_state_count(10, 8, 3) == size * 3 + next_size * 7


def test_real_900_distribution_two_epochs_on_eight_ranks():
    samples = _samples({('16:9', 37): 139, ('16:9', 72): 95,
                        ('16:9', 102): 144, ('1:1', 37): 54,
                        ('1:1', 72): 54, ('1:1', 102): 54,
                        ('4:3', 37): 54, ('4:3', 72): 54,
                        ('4:3', 102): 54, ('9:16', 37): 43,
                        ('9:16', 72): 63, ('9:16', 102): 92})
    ranks = [sampling.CoverageSampler(samples, 20261015, rank, 8)
             for rank in range(8)]
    assert ranks[0].batches_per_epoch == 116
    seen, real, slots = collections.Counter(), 0, 0
    for step in range(58):
        plans = [sampling.update_plan(s, 1, 4, step, 2) for s in ranks]
        assert all(p == plans[0] for p in plans)
        plan = plans[0]
        real += plan.real_states
        slots += plan.slots
        for _ in range(plan.micro_steps):
            for sampler in ranks:
                sample = sampler.next()
                if sampler.last_valid:
                    seen[sample.id] += 1
    assert seen == collections.Counter({s.id: 2 for s in samples})
    assert real == 1800 and slots == 1856
    assert all(s.issued_real_global == 1800 for s in ranks)
    with pytest.raises(ValueError, match='no states remain'):
        sampling.update_plan(ranks[0], 1, 4, 58, 2)


def _gloo_weighted_tail_worker(rank, store, output):
    import copy
    import datetime
    from pathlib import Path

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{store}',
                            rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        model = torch.nn.Linear(2, 1).double()
        with torch.no_grad():
            model.weight.fill_(0.125)
            model.bias.fill_(0.25)
        actual = DistributedDataParallel(model)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        samples = _samples({('16:9', 37): 3, ('1:1', 72): 1,
                            ('4:3', 102): 1})
        ids = {s.id: i for i, s in enumerate(samples)}
        sampler = sampling.CoverageSampler(samples, 73, rank, 2)
        references = [sampling.CoverageSampler(samples, 73, r, 2)
                      for r in range(2)]
        updates = sampling.planned_updates(
            sampler.batches_per_epoch, 1, 3, None, 2)
        pads = 0

        def inputs(sample):
            i = ids[sample.id]
            return (torch.tensor([[i / 2, (i + 1) / 4]], dtype=torch.float64),
                    torch.tensor([[i / 8]], dtype=torch.float64))

        for step in range(updates):
            plan = sampling.update_plan(sampler, 1, 3, step, 2)
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(plan.micro_steps):
                sample = sampler.next()
                x, target = inputs(sample)
                prediction = actual(x)
                prediction.retain_grad()
                loss = (prediction - target).square().mean()
                (loss * plan.loss_scale * int(sampler.last_valid)).backward()
                if not sampler.last_valid:
                    assert bool((prediction.grad == 0).all())
                    pads += 1
                for ref_sampler in references:
                    ref_sample = ref_sampler.next()
                    if ref_sampler.last_valid:
                        x, target = inputs(ref_sample)
                        losses.append((reference(x) - target).square().mean())
            assert len(losses) == plan.real_states
            torch.stack(losses).mean().backward()
            for a, b in zip(actual.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-12)
            optimizer.step()
            ref_optimizer.step()
            for a, b in zip(actual.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
        assert sampler.issued_real_global == 10
        torch.save({'weights': model.state_dict(), 'updates': updates,
                    'padded_slots': pads}, Path(output) / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_rank_average_matches_unpadded_reference(
        tmp_path, monkeypatch):
    """Real two-process DDP; no GPU/FSDP emulation claim."""
    import torch
    import torch.multiprocessing as mp

    monkeypatch.setenv('GLOO_SOCKET_IFNAME', 'lo')
    mp.spawn(_gloo_weighted_tail_worker,
             args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f'rank{r}.pt', weights_only=True)
               for r in range(2)]
    assert all(r['updates'] == 3 for r in results)
    assert sum(r['padded_slots'] for r in results) == 6
    assert all(torch.equal(results[0]['weights'][n], results[1]['weights'][n])
               for n in results[0]['weights'])
