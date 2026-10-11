"""Block masks from tile scores.

Rules (the tile search's oracle uses exactly the same rules):
  1. Global tiles are dense in both directions: every query tile sees every
     global key tile, and global query tiles see every tile.
  2. Only the video -> video quadrant is sparse (top-k per query tile).
  3. Budgets are equal-kernel-cost: with n_ideal = ceil(real tokens / 128),
     a query tile keeps ratio * n_ideal^2 / n_tiles key tiles. The kernel
     pays for every kept tile as a full 128x128 block, so a padded plan must
     not get a larger budget than an unpadded one.
  4. The fractional part of the budget is spread by a Bresenham pattern
     indexed by the query tile number, so the mean kept count per row equals
     the budget exactly (rounding the budget moves real sparsity by 20-50%
     at high sparsity).
  5. A query tile's own tile is forced in (+inf score) and consumes budget.
  6. Empty key tiles (valid_count == 0) are never selected.
  7. With condition spans tiled, the video columns split into a reference
     block [0, n_ref_tiles) and a target block [n_ref_tiles, n_video_tiles).
     Each block runs its own top-k with its own budget; the diagonal is only
     forced inside the block that contains it. A single pooled top-k starves
     the reference block (spatial neighbours in the target always win).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from miowtion.veda import tiling

_NEG_INF = float('-inf')
_POS_INF = float('inf')


@dataclasses.dataclass(frozen=True)
class Budget:
    """Keep budget of one column block: a ratio or an absolute tile count.

    Attributes:
        ratio: Keep ratio of the equal-cost budget; >= 1 keeps everything.
        tiles: Absolute key tiles per query tile (overrides ratio).
    """

    ratio: float | None = None
    tiles: float | None = None

    def __post_init__(self):
        if (self.ratio is None) == (self.tiles is None):
            raise ValueError('set exactly one of ratio / tiles')
        value = self.ratio if self.tiles is None else self.tiles
        if value <= 0:
            raise ValueError(f'budget must be positive: {self}')

    @property
    def keeps_all(self) -> bool:
        return self.ratio is not None and self.ratio >= 1.0

    def per_row(self, real_tokens: int, n_tiles: int) -> float:
        if self.tiles is not None:
            return float(self.tiles)
        n_ideal = math.ceil(real_tokens / tiling.TILE_SIZE)
        return self.ratio * n_ideal * n_ideal / n_tiles


def split_budget(budget: float, n_cols: int) -> tuple[int, int, float]:
    """(k_lo, k_hi, frac) with 1 <= k_lo <= k_hi <= n_cols."""
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    k_lo = min(max(1, math.floor(budget)), n_cols)
    # Rounded so that e.g. 2.3 - 2 gives 0.3, not 0.29999999999999982,
    # which would drop one extra tile per 1/frac rows.
    frac = round(min(max(budget - k_lo, 0.0), 1.0), 12)
    return k_lo, min(k_lo + 1, n_cols), frac


@functools.lru_cache(maxsize=256)
def _bresenham(n_rows: int, frac: float, device: str) -> torch.Tensor:
    # Built on the host in fp64 and cached per device: a fresh host-to-device
    # copy per call would be a pageable copy, which waits for the GPU.
    ramp = torch.floor(torch.arange(n_rows + 1, dtype=torch.float64) * frac)
    return (ramp[1:] > ramp[:-1]).to(device)


def bresenham_extra(n_rows: int, frac: float,
                    device: torch.device | str) -> torch.Tensor:
    """[n_rows] bool: row r keeps k_hi iff floor((r+1)f) > floor(r f).

    The returned tensor is cached and shared; callers must not modify it.
    """
    return _bresenham(n_rows, float(frac), str(torch.device(device)))


@dataclasses.dataclass(frozen=True)
class ColumnBlock:
    """A sparse column block of the video quadrant."""

    start: int
    stop: int
    budget: Budget
    real_tokens: int


def column_blocks(layout: tiling.TileLayout, target_budget: Budget,
                  ref_budget: Budget | None = None) -> list[ColumnBlock]:
    """Target block, preceded by a reference block when references are tiled.

    Raises:
        ValueError: If references are tiled but no ref budget is given.
    """
    target = ColumnBlock(layout.n_ref_tiles, layout.n_video_tiles,
                         target_budget, layout.target_tokens)
    if layout.n_ref_tiles == 0:
        return [target]
    if ref_budget is None:
        raise ValueError('tiled references need their own budget')
    return [ColumnBlock(0, layout.n_ref_tiles, ref_budget, layout.ref_tokens),
            target]


@dataclasses.dataclass
class Selection:
    """Selected video key tiles per video query tile.

    Attributes:
        index: [H', n_video, K] int64 global key tile ids (all blocks
            concatenated).
        keep: [H', n_video, K] bool, which entries are selected.
    """

    index: torch.Tensor
    keep: torch.Tensor


# How many times the cap below bound, i.e. a row wanted more tiles than the
# threshold rule was allowed to give it. Read by the probe scripts: a rule
# that saturates often is no longer the rule it claims to be.
SATURATED = [0, 0]  # [rows that hit the cap, rows seen]


def threshold_allowed(sorted_scores: torch.Tensor, valid: torch.Tensor,
                      mean_k: torch.Tensor, cap: int) -> torch.Tensor:
    """Sol's per-row threshold, recalibrated to our mean budget.

    Sol-Attn routes a key block when its score clears `mean + tau * std`
    of that query row's scores, so the number of blocks a row keeps is
    decided by the data; our top-k gives every row the same number. The
    two differ in the *shape* of the allocation, not in its mean -- which
    is why a comparison at a matched mean density isolates exactly that
    difference. Here tau is not a constant but is solved for per head so
    that the realised mean equals `mean_k`: the kernel then costs the
    same as the top-k arm it is compared against, to the tile.

    Rows are standardised before the threshold is applied (z = (s - mean)
    / std over that row's valid columns), which is what makes one tau
    comparable across rows; a raw threshold would just reproduce top-k
    with extra steps on rows whose scores are uniformly large.

    Args:
        sorted_scores: [H', R, C] row scores, descending along C, as
            returned by topk. Forced columns carry +inf.
        valid: [H', R, C] bool, False where the column is padding.
        mean_k: [] float target mean kept columns per row.
        cap: C, the largest number a row can be given.

    Returns:
        [H', R] int64 kept count per row, summing to `mean_k * R` per
        head up to the tie at the threshold.
    """
    counts = valid.sum(-1, keepdim=True).clamp(min=2).float()
    finite = torch.where(valid, sorted_scores.float(), 0.0)
    # Forced (+inf) columns would poison mean/std, so they are excluded
    # from the statistics and re-forced afterwards.
    forced = torch.isinf(finite) & (finite > 0)
    finite = torch.where(forced, 0.0, finite)
    stats_n = (counts - forced.sum(-1, keepdim=True)).clamp(min=2)
    mean = finite.sum(-1, keepdim=True) / stats_n
    var = (torch.where(valid & ~forced, (finite - mean) ** 2, 0.0)
           ).sum(-1, keepdim=True) / (stats_n - 1)
    z = (sorted_scores.float() - mean) / var.sqrt().clamp(min=1e-12)
    z = torch.where(valid, z, torch.tensor(_NEG_INF, device=z.device))
    z = torch.where(forced, torch.tensor(_POS_INF, device=z.device), z)

    heads, rows, _ = z.shape
    total = int(round(float(mean_k) * rows))
    total = max(1, min(total, z.shape[-1] * rows))
    flat = z.reshape(heads, -1)
    # The threshold is the total-th largest z of the whole head, so the
    # head keeps exactly the budget it would have kept under top-k while
    # being free to spend it unevenly across rows.
    tau = flat.topk(total, dim=-1).values[:, -1:]
    allowed = (z >= tau[:, :, None]).sum(-1)
    SATURATED[0] += int((allowed >= cap).sum())
    SATURATED[1] += heads * rows
    return allowed.clamp(min=1)


@torch.no_grad()
def select_video_blocks(scores: torch.Tensor, layout: tiling.TileLayout,
                        blocks: list[ColumnBlock],
                        rows: torch.Tensor | None = None,
                        rule: str = 'topk',
                        cap_mult: float = 4.0) -> Selection:
    """Top-k over the video quadrant of `scores` (rules 2-7).

    Args:
        scores: [H', R, >= n_video] block scores of query tiles `rows` (any
            score monotone in importance: logits or heat).
        layout: Tile layout.
        blocks: Column blocks from column_blocks().
        rows: [R] int64 video query tile ids; None means all n_video tiles.
            The Bresenham pattern and the diagonal follow the tile id, not
            the position within `rows`.
        rule: 'topk' gives every row the same count (rules 3-4);
            'threshold' spends the same per-head total unevenly, by
            Sol-Attn's standardised per-row threshold. See
            threshold_allowed().
        cap_mult: 'threshold' only -- the largest multiple of the mean
            budget one row may take. Saturation is counted in SATURATED.

    Returns:
        The selection for the R query tiles.
    """
    if rule not in ('topk', 'threshold'):
        raise ValueError(f"rule must be 'topk' or 'threshold': {rule!r}")
    heads = scores.shape[0]
    n_video = layout.n_video_tiles
    device = scores.device
    if rows is None:
        rows = torch.arange(n_video, device=device)
    num_rows = rows.numel()
    # `scores` must already be restricted to `rows`. Passing the full
    # [H', n_tiles, .] logits with a shorter `rows` used to die deep
    # inside a broadcast ("size of tensor a (330) must match b (341)"),
    # which says nothing about which argument is wrong.
    if scores.shape[1] != num_rows:
        raise ValueError(
            f'scores has {scores.shape[1]} query rows but rows has '
            f'{num_rows}; index the query axis by `rows` before calling')
    indices, keeps = [], []
    for block in blocks:
        n_cols = block.stop - block.start
        if block.budget.keeps_all:
            idx = torch.arange(block.start, block.stop, device=device)
            indices.append(idx.expand(heads, num_rows, n_cols))
            keeps.append(layout.kv_ok[block.start:block.stop].expand(
                heads, num_rows, n_cols))
            continue
        s = scores[:, :, block.start:block.stop].float().clone()
        s.masked_fill_(~layout.kv_ok[None, None, block.start:block.stop],
                       _NEG_INF)
        # A query tile always keeps its own key tile. Written as a gather /
        # where / scatter over every row instead of indexing the rows that
        # fall in the block: that indexing needs torch.nonzero, whose
        # data-dependent size synchronizes the host once per call.
        own = (rows >= block.start) & (rows < block.stop)
        col = (rows - block.start).clamp(0, n_cols - 1)
        col = col[None, :, None].expand(heads, num_rows, 1)
        s.scatter_(-1, col, torch.where(own[None, :, None], _POS_INF,
                                        s.gather(-1, col)))
        budget = block.budget.per_row(block.real_tokens, n_cols)
        k_lo, k_hi, frac = split_budget(budget, n_cols)
        if rule == 'threshold':
            # The rule can hand a row more than the mean, so the topk has
            # to be widened before it can be applied; `cap_mult` bounds
            # how uneven the allocation is allowed to get.
            k_hi = min(n_cols, max(k_hi, int(math.ceil(cap_mult * budget))))
        vals, idx = torch.topk(s, k_hi, dim=-1, sorted=True)
        if rule == 'threshold':
            allowed = threshold_allowed(
                vals, vals > _NEG_INF,
                torch.tensor(budget, device=device), k_hi)
            keep = (torch.arange(k_hi, device=device)[None, None, :]
                    < allowed[:, :, None]) & (vals > _NEG_INF)
        else:
            extra = bresenham_extra(n_video, frac,
                                    device).index_select(0, rows)
            allowed = k_lo + extra.to(torch.long)
            keep = (torch.arange(k_hi, device=device)[None, None, :]
                    < allowed[None, :, None]) & (vals > _NEG_INF)
        indices.append(idx + block.start)
        keeps.append(keep)
    return Selection(torch.cat(indices, -1), torch.cat(keeps, -1))


@torch.no_grad()
def dense_block_mask(selection: Selection, layout: tiling.TileLayout,
                     global_rows: bool = True) -> torch.Tensor:
    """Block mask of a selection (rules 1-7), the kernels' input.

    Args:
        selection: Video query tiles' selected key tiles (all n_video rows,
            or a subset of R rows).
        layout: Tile layout.
        global_rows: Append the global query rows, which see every non-empty
            tile (the full-sequence case). False yields the selection's R
            rows only, e.g. to run sampled query tiles.

    Returns:
        [H', n_tiles, n_tiles] bool, or [H', R, n_tiles] without global rows.
        Global key columns are always kept.
    """
    heads, num_rows, _ = selection.index.shape
    n, n_video = layout.n_tiles, layout.n_video_tiles
    if global_rows and num_rows != n_video:
        raise ValueError('global_rows needs a selection of all video rows')
    mask = torch.zeros(heads, n if global_rows else num_rows, n,
                       dtype=torch.bool, device=selection.index.device)
    # Global columns start after the layout's video tiles, whatever the
    # number of selected rows.
    mask[:, :num_rows].scatter_(2, selection.index, selection.keep)
    mask[:, :, n_video:] = layout.kv_ok[n_video:]
    if global_rows:
        mask[:, n_video:, :] = layout.kv_ok
    return mask


def kept_tiles_per_row(selection: Selection) -> torch.Tensor:
    """[H', n_video] number of selected video tiles (budget accounting)."""
    return selection.keep.sum(-1)
