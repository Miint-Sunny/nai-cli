"""Pixel snapping as the official Director tool runs it: grid of scaled pixel art, a colour per cell, palette."""

import math
from dataclasses import dataclass

import numpy as np

from nai_cli import errors

# The browser's CPU backends work on at most this many pixels and shrink larger images to fit
MAX_PIXELS = 4_000_000
# Defaults the official client sends
AUTO_TOLERANCE = 6.5
MAX_DETAIL = 0.2
_LAM = 1.2
_MARGIN = 0.27
_MIN_PITCH = 3.0
_MAX_PITCH = 24.0
_FAR = 1e18
_PALETTE_LEAVES = 256


@dataclass(frozen=True)
class Snapped:
    # Hc x Wc x 3
    rgb: np.ndarray
    alpha: np.ndarray | None
    pitch: tuple[float, float]
    palette_size: int | None
    # Set when the source was shrunk to fit MAX_PIXELS
    note: str = ""


def snap(rgba: np.ndarray, colors: int | str = "auto", max_detail: float | None = MAX_DETAIL) -> Snapped:
    """colors is "auto", 0 for none or a palette size; max_detail None keeps finer grids from being chosen."""
    height, width = rgba.shape[:2]
    alpha_full = rgba[..., 3].astype(np.uint8)
    if not alpha_full.any():
        raise errors.NaiError(errors.BAD_ARGS, "the image is fully transparent")
    rgb = np.where(alpha_full[..., None] > 0, rgba[..., :3], 0).astype(np.float64)
    alpha = alpha_full if (alpha_full < 255).any() else None
    note = ""
    if width * height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (width * height))
        w, h = max(1, math.floor(width * scale)), max(1, math.floor(height * scale))
        rows = np.minimum(height - 1, np.arange(h) * height // h)
        cols = np.minimum(width - 1, np.arange(w) * width // w)
        rgb = rgb[rows][:, cols]
        alpha = alpha[rows][:, cols] if alpha is not None else None
        width, height, note = w, h, f"shrunk to {w}x{h} to fit"
    col_profile, row_profile = _profiles(rgb)
    engine = _Spread(rgb)
    # A single row or column has no neighbours to compare, so no grid
    candidates = _candidates([col_profile, row_profile]) if width > 1 and height > 1 else []
    if not candidates:
        raise errors.NaiError(
            errors.BAD_ARGS, "no repeating pixel grid found; the image does not look like scaled pixel art"
        )
    best = _coarse_to_fine(engine, width, height, candidates)
    pitch_x, pitch_y, phase_x, phase_y = _refine(
        engine, rgb, col_profile, row_profile, best[1], best[2], best[3], max_detail
    )
    xb = _lines(col_profile, pitch_x, _mod(phase_x * pitch_x, pitch_x))
    yb = _lines(row_profile, pitch_y, _mod(phase_y * pitch_y, pitch_y))
    cells, cell_alpha = _sample(rgb, xb, yb, alpha)
    palette_size = None
    if colors == "auto":
        cells, palette_size = _palette(cells, lambda dist, count: dist > AUTO_TOLERANCE)
    elif isinstance(colors, int) and colors > 0:
        cells, palette_size = _palette(cells, lambda dist, count: count <= colors, max(_PALETTE_LEAVES, colors))
    return Snapped(
        _bytes(cells), _bytes(cell_alpha) if cell_alpha is not None else None, (pitch_x, pitch_y), palette_size, note
    )


def _round(value: float) -> int:
    """Half to even, as the official code rounds here."""
    return int(np.round(value))


def _mod(value: float, by: float) -> float:
    r = value % by
    return r + by if r < 0 else r


def _bytes(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _profiles(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How much the colour changes between neighbouring columns, and between neighbouring rows."""
    cols = np.abs(np.diff(rgb, axis=1)).sum(axis=(0, 2))
    rows = np.abs(np.diff(rgb, axis=0)).sum(axis=(1, 2))
    return cols, rows


def _candidates(profiles: list[np.ndarray]) -> list[float]:
    """Pitches whose period shows in the profiles, with their whole fractions."""
    found: list[float] = []
    step = (_MIN_PITCH + 0.02) - _MIN_PITCH
    periods = _MIN_PITCH + np.arange(max(0, math.ceil((_MAX_PITCH - _MIN_PITCH) / 0.02))) * step
    for profile in profiles:
        peak = profile.max()
        strong = profile[profile > 0.02 * peak + 1e-12]
        clipped = np.minimum(profile, np.percentile(strong, 60) if strong.size else peak)
        scores = np.array([_periodicity(clipped, p) for p in periods])
        peaks = []
        for i in range(len(periods)):
            window = scores[max(0, i - 15) : min(len(periods), i + 16)]
            if scores[i] >= window.max() and scores[i] > 1.12:
                peaks.append((scores[i], periods[i]))
        peaks.sort(key=lambda sp: (-sp[0], -sp[1]))
        found += [p for _, p in peaks[:8]]
    pitches: list[float] = []
    for pitch in sorted(found):
        divisor = 1
        while pitch / divisor >= _MIN_PITCH - 1e-9:
            part = pitch / divisor
            if all(abs(part - q) > 0.12 for q in pitches):
                pitches.append(part)
            divisor += 1
    return sorted(pitches)


def _periodicity(profile: np.ndarray, period: float) -> float:
    """How much of the profile's change falls in one narrow window of the period, block by block."""
    length = len(profile)
    block = math.trunc(max(80, 14 * period))
    bins = max(8, _round(period / 0.25))
    width = period / bins
    window = max(1, _round(0.7 / width))
    blocks = (length - 1) // block + 1
    index = np.arange(length)
    where = np.minimum(np.floor(np.mod(index + 1, period) / width).astype(int), bins - 1)
    hist = np.zeros(blocks * bins)
    np.add.at(hist, (index // block) * bins + where, profile)
    hist = hist.reshape(blocks, bins)
    totals = hist.sum(axis=1)
    keep = totals > 1e-9
    if not keep.any():
        return 0.0
    hist, totals = hist[keep], totals[keep]
    wrapped = np.concatenate([hist, hist[:, :window]], axis=1)
    sums = np.concatenate([np.zeros((len(hist), 1)), np.cumsum(wrapped, axis=1)], axis=1)
    best = (sums[:, window : window + bins] - sums[:, :bins]).max(axis=1)
    share = best / totals / (window * width / period)
    return float((share * totals).sum() / totals.sum())


@dataclass(frozen=True)
class _Segments:
    """Cell bounds along one axis: lo and hi of K cells for each of P phases in each of B bands."""

    lo: np.ndarray
    hi: np.ndarray
    sel: np.ndarray


def _segments(end: int, pitches: list[float], phases: list[list[float]], every: int) -> _Segments:
    """Cells of each pitch and phase, inset by the margin, every `every`th whole cell selected."""
    count = math.ceil(end / min(pitches)) + 3
    shape = (len(pitches), len(phases[0]), count)
    lo, hi, sel = np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, bool)
    k = np.arange(count)
    for b, pitch in enumerate(pitches):
        inset = max(1.0, _MARGIN * pitch)
        for p, phase in enumerate(phases[b]):
            start = _mod(phase, 1) * pitch - pitch + k * pitch
            valid = (start >= -1e-9) & (start + pitch <= end + 1e-9)
            a, z = np.floor(start + inset).astype(int), np.ceil(start + pitch - inset).astype(int)
            narrow = z - a < 2
            mid = np.floor(start + pitch / 2).astype(int)
            lo[b, p], hi[b, p] = np.where(narrow, mid - 1, a), np.where(narrow, mid + 1, z)
            sel[b, p] = valid & ((np.cumsum(valid) - 1) % every == 0)
    return _Segments(lo, hi, sel)


class _Spread:
    """Mean colour spread inside the cells of a candidate grid, from summed area tables."""

    def __init__(self, rgb: np.ndarray) -> None:
        values = np.concatenate([rgb, rgb**2], axis=2)
        sat = np.zeros((rgb.shape[0] + 1, rgb.shape[1] + 1, 6))
        sat[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
        self.height, self.width = rgb.shape[:2]
        self.by_row = sat
        # The same table with columns first, so either axis can be fixed with one gather
        self.by_col = np.ascontiguousarray(sat.transpose(1, 0, 2))

    def score(self, varied: _Segments, fixed: _Segments, varied_is_col: bool) -> np.ndarray:
        """B x P: for each band and phase of the varied axis, against the fixed axis's one phase."""
        table = self.by_row if varied_is_col else self.by_col
        bands, phases = varied.lo.shape[:2]
        out = np.full((bands, phases), _FAR)
        for b in range(bands):
            keep = fixed.sel[b, 0]
            counts = varied.sel[b].sum(axis=1)
            if keep.sum() < 3 or not (counts >= 3).any():
                continue
            f_lo, f_hi = fixed.lo[b, 0][keep], fixed.hi[b, 0][keep]
            # Sums over each fixed span along the varied axis, which comes first so gathering cells takes whole rows
            strips = np.ascontiguousarray((table[f_hi] - table[f_lo]).transpose(1, 0, 2))
            phase, cell = np.nonzero(varied.sel[b])
            v_lo, v_hi = varied.lo[b][phase, cell], varied.hi[b][phase, cell]
            sums = strips[v_hi]
            sums -= strips[v_lo]
            n = (v_hi - v_lo)[:, None] * (f_hi - f_lo)[None]
            var = sums[..., 3:] * n[..., None]
            var -= np.square(sums[..., :3])
            np.maximum(var, 0, out=var)
            np.sqrt(var, out=var)
            spread = (var.sum(axis=-1) / n).sum(axis=1) / 3
            totals = np.bincount(phase, spread, minlength=phases) / (keep.sum() * np.maximum(counts, 1))
            out[b] = np.where(counts >= 3, totals, _FAR)
        return out


def _phases(count: int) -> list[float]:
    return [i / count for i in range(count)]


def _best_phases(
    engine: _Spread, pitches: list[float], phase_count: int, every: int
) -> tuple[list[float], list[float]]:
    """The best column phase against rows at 0.37, then the best row phase against those columns."""
    grid = _phases(phase_count)
    w, h = engine.width, engine.height
    rows = _segments(h, pitches, [[0.37]] * len(pitches), every)
    cols = _segments(w, pitches, [grid] * len(pitches), every)
    x = [grid[int(np.argmin(r))] for r in engine.score(cols, rows, True)]
    cols = _segments(w, pitches, [[v] for v in x], every)
    rows = _segments(h, pitches, [grid] * len(pitches), every)
    y = [grid[int(np.argmin(r))] for r in engine.score(rows, cols, False)]
    return x, y


def _search(
    engine: _Spread, pitches: list[float], count: int, phase_count: int, every: int, jitter: bool
) -> list[tuple[float, float, float, float]]:
    """For each pitch, the best of `count` nearby pitches with their phases: score, pitch, x phase, y phase."""
    tried = [float(v) for p in pitches for v in _linspace(0.965 * p, 1.035 * p, count)]
    x, y = _best_phases(engine, tried, phase_count, every)
    nudges = [-0.04, 0.0, 0.04] if jitter else [0.0]
    g = len(nudges)
    expanded = [i for i in range(len(tried)) for _ in nudges]
    cols = _segments(engine.width, [tried[i] for i in expanded], [[x[i] + n for n in nudges] for i in expanded], every)
    rows = _segments(
        engine.height, [tried[i] for i in expanded], [[y[i] + nudges[j % g]] for j, i in enumerate(expanded)], every
    )
    scores = engine.score(cols, rows, True).reshape(len(pitches), count * g * g)
    out = []
    for e, row in enumerate(scores):
        at = int(np.argmin(row))
        if row[at] >= _FAR:
            out.append((_FAR, pitches[e], 0.0, 0.0))
            continue
        flat = e * count * g * g + at
        band, xi = flat // g, flat % g
        t, yi = band // g, band % g
        out.append((float(row[at]), tried[t], x[t] + nudges[xi], y[t] + nudges[yi]))
    return out


def _coarse_to_fine(engine: _Spread, width: int, height: int, candidates: list[float]) -> tuple[float, ...]:
    coarse = _search(engine, candidates, 7, 10, 2, False)
    low = min(c[0] for c in coarse)
    limit = 1.35 * max(1.25 * low, low + 0.5)
    kept = [p for p, c in zip(candidates, coarse, strict=True) if c[0] <= limit]
    ranked = [p for _, p in sorted(zip((c[0] for c in coarse), candidates, strict=True))]
    kept += [p for p in ranked[:4] if p not in kept]
    if len(kept) > 12:
        kept = sorted(kept, key=ranked.index)[:12]
    fine = _search(engine, sorted(kept), 15, 14, 1, True)
    low = min(c[0] for c in fine)
    limit = max(1.25 * low, low + 0.5)
    # The coarsest grid that is about as good as the best
    best = None
    for c in fine:
        if c[0] <= limit and (best is None or c[1] > best[1]):
            best = c
    assert best is not None
    return best


def _refine(
    engine: _Spread,
    rgb: np.ndarray,
    col_profile: np.ndarray,
    row_profile: np.ndarray,
    pitch: float,
    phase_x: float,
    phase_y: float,
    max_detail: float | None,
) -> tuple[float, float, float, float]:
    """Each axis's own pitch near the one found; a half or third of it wins when it fits the image clearly
    better."""
    pitches, phases = [pitch], [(phase_x, phase_y)]
    if max_detail is not None:
        for divisor in (2, 3):
            if pitch / divisor < 1.5:
                break
            pitches.append(pitch / divisor)
    if len(pitches) > 1:
        x, y = _best_phases(engine, pitches[1:], 14, 1)
        phases += list(zip(x, y, strict=True))
    tried = [_fit_axes(engine, rgb, col_profile, row_profile, p, ph) for p, ph in zip(pitches, phases, strict=True)]
    grid, error = tried[0]
    for other, other_error in tried[1:]:
        if max_detail is not None and other_error < error * (1 - max_detail):
            grid, error = other, other_error
    return grid


def _fit_axes(
    engine: _Spread,
    rgb: np.ndarray,
    col_profile: np.ndarray,
    row_profile: np.ndarray,
    pitch: float,
    phase: tuple[float, float],
) -> tuple[tuple[float, float, float, float], float]:
    grid16 = _phases(16)
    tried = [float(v) for v in _linspace(0.96 * pitch, 1.04 * pitch, 33)]
    w, h = engine.width, engine.height
    cols = _segments(w, tried, [grid16] * 33, 1)
    rows = _segments(h, [pitch] * 33, [[phase[1]]] * 33, 1)
    pitch_x, phase_x = _pick(engine.score(cols, rows, True), tried, grid16, pitch, phase[0])
    rows = _segments(h, tried, [grid16] * 33, 1)
    cols = _segments(w, [pitch] * 33, [[phase[0]]] * 33, 1)
    pitch_y, phase_y = _pick(engine.score(rows, cols, False), tried, grid16, pitch, phase[1])
    xb = _lines(col_profile, pitch_x, _mod(phase_x * pitch_x, pitch_x))
    yb = _lines(row_profile, pitch_y, _mod(phase_y * pitch_y, pitch_y))
    cells, _ = _sample(rgb, xb, yb, None)
    return (pitch_x, pitch_y, phase_x, phase_y), _mean_error(rgb, cells, xb, yb)


def _pick(scores: np.ndarray, tried: list[float], grid: list[float], pitch: float, phase: float) -> tuple[float, float]:
    flat = scores.reshape(-1)
    at = int(np.argmin(flat))
    if flat[at] >= _FAR:
        return pitch, phase
    return tried[at // len(grid)], grid[at % len(grid)]


def _mean_error(rgb: np.ndarray, cells: np.ndarray, xb: list[int], yb: list[int]) -> float:
    rows = np.repeat(np.arange(len(yb) - 1), np.diff(yb))
    cols = np.repeat(np.arange(len(xb) - 1), np.diff(xb))
    return float(np.abs(cells[rows][:, cols] - rgb).sum() / rgb.size)


def _linspace(start: float, end: float, count: int) -> np.ndarray:
    values = start + np.arange(count) * ((end - start) / (count - 1)) if count > 1 else np.array([start])
    values[-1] = end
    return values


def _lines(profile: np.ndarray, pitch: float, offset: float | None) -> list[int]:
    """Cell boundaries along an axis: a path through the edges of the profile with steps near the pitch."""
    smooth = _smoothed(profile)
    scale = np.percentile(smooth, 98) + 1e-9
    edges = np.clip(smooth / scale, 0, 1.5)
    end = len(profile) + 1
    gain = np.empty(end + 1)
    gain[1:end] = edges
    gain[0] = gain[end] = 0.6
    if offset is not None:
        drift = np.mod(np.arange(end + 1) - offset + pitch / 2, pitch) - pitch / 2
        gain -= 0.35 * (drift / (pitch / 2)) ** 2
    shortest = max(2, _round(0.7 * pitch))
    longest = max(shortest + 1, _round(1.35 * pitch))
    best = np.full(end + 1, -_FAR)
    back = np.full(end + 1, -1)
    best[: min(end, longest) + 1] = gain[: min(end, longest) + 1]
    for e in range(shortest, end + 1):
        start, stop = max(0, e - longest), e - shortest
        steps = e - np.arange(start, stop + 1)
        value = best[start : stop + 1] - _LAM * ((steps - pitch) / pitch) ** 2 + gain[e]
        at = int(np.argmax(value))
        if value[at] > best[e]:
            best[e], back[e] = value[at], start + at
    tail = max(0, end - longest)
    node = tail + int(np.argmax(best[tail:]))
    path = []
    while node >= 0:
        path.append(node)
        node = back[node]
    path.reverse()
    if path[0] != 0:
        path.insert(0, 0)
    if path[-1] != end:
        path.append(end)
    while len(path) > 2 and path[1] - path[0] < 2:
        del path[1]
    while len(path) > 2 and path[-1] - path[-2] < 2:
        del path[-2]
    return path


def _smoothed(profile: np.ndarray) -> np.ndarray:
    kernel = np.exp(-0.5 * (np.arange(-2, 3) / 0.8) ** 2)
    return np.convolve(profile, kernel / kernel.sum(), mode="same")


def _sample(
    rgb: np.ndarray, xb: list[int], yb: list[int], alpha: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """The median colour of each cell's middle, past the margin, over pixels that are not transparent."""
    cells = np.zeros((len(yb) - 1, len(xb) - 1, 3))
    cell_alpha = np.zeros(cells.shape[:2]) if alpha is not None else None
    cols = [_inset(xb[i], xb[i + 1]) for i in range(len(xb) - 1)]
    for r in range(len(yb) - 1):
        y0, y1 = _inset(yb[r], yb[r + 1])
        band = rgb[y0:y1]
        for c, (x0, x1) in enumerate(cols):
            block = band[:, x0:x1].reshape(-1, 3)
            if alpha is None:
                cells[r, c] = np.median(block, axis=0)
                continue
            a = alpha[y0:y1, x0:x1].reshape(-1)
            assert cell_alpha is not None
            cell_alpha[r, c] = np.median(a)
            if (a > 0).any():
                cells[r, c] = np.median(block[a > 0], axis=0)
    return cells, cell_alpha


def _inset(lo: int, hi: int) -> tuple[int, int]:
    cut = _round((hi - lo) * _MARGIN)
    a, z = lo + cut, hi - cut
    return (lo, hi) if z - a < 1 else (a, z)


def _palette(cells: np.ndarray, stop, leaves: int = _PALETTE_LEAVES) -> tuple[np.ndarray, int]:
    """An octree palette whose closest colours are merged in Lab until stop says enough."""
    flat = cells.reshape(-1, 3)
    if not len(flat):
        return cells, 0
    entries = _octree(flat, leaves)
    colors = np.array([e[0] for e in entries])
    counts = np.array([e[1] for e in entries], dtype=float)
    total = len(flat)
    while len(colors) > 1:
        lab = _lab(colors)
        dist = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1))
        small = np.minimum(counts[:, None], counts[None])
        # Rare colours merge more readily
        weight = 1 + np.maximum(np.maximum(0, 1 - small / total / 0.02), 3 * np.maximum(0, 1 - small / 16))
        score = dist / weight
        score[np.tril_indices(len(colors))] = np.inf
        at = int(np.argmin(score))
        i, j = divmod(at, len(colors))
        if stop(score[i, j], len(colors)):
            break
        merged = counts[i] + counts[j]
        colors[i] = (colors[i] * counts[i] + colors[j] * counts[j]) / merged
        counts[i] = merged
        colors, counts = np.delete(colors, j, axis=0), np.delete(counts, j)
    lab = _lab(colors)
    nearest = np.argmin(((_lab(flat)[:, None] - lab[None]) ** 2).sum(-1), axis=1)
    return colors[nearest].reshape(cells.shape), len(np.unique(nearest))


class _Node:
    __slots__ = ("count", "kids", "leaf", "total")

    def __init__(self, leaf: bool) -> None:
        self.leaf, self.count, self.total, self.kids = leaf, 0, np.zeros(3), None


def _octree(flat: np.ndarray, leaves: int) -> list[tuple[np.ndarray, int]]:
    levels: list[list[_Node]] = [[] for _ in range(8)]
    root = _Node(False)
    count = 0
    for value in np.clip(np.round(flat), 0, 255).astype(int):
        node, depth = root, 0
        while not node.leaf and depth < 8:
            shift = 7 - depth
            index = (value[0] >> shift & 1) << 2 | (value[1] >> shift & 1) << 1 | value[2] >> shift & 1
            if node.kids is None:
                node.kids = [None] * 8
                levels[depth].append(node)
            if node.kids[index] is None:
                node.kids[index] = _Node(depth == 7)
                count += depth == 7
            node, depth = node.kids[index], depth + 1
        node.count += 1
        node.total += value
        while count > leaves:
            count += _reduce(levels)
    out: list[tuple[np.ndarray, int]] = []

    def collect(node: _Node) -> None:
        if node.kids is not None:
            for kid in node.kids:
                if kid is not None:
                    collect(kid)
        elif node.count > 0:
            out.append((node.total / node.count, node.count))

    collect(root)
    return out


def _reduce(levels: list[list[_Node]]) -> int:
    """Fold the first node of the deepest level into one leaf; returns the change in leaves."""
    level = next((nodes for nodes in reversed(levels) if nodes), None)
    if level is None:
        return 0
    smallest = min(range(len(level)), key=lambda i: level[i].count)
    node = level.pop(smallest)
    kids = [k for k in node.kids or [] if k is not None]
    for kid in kids:
        node.total = node.total + kid.total
        node.count += kid.count
    node.kids, node.leaf = None, True
    return 1 - len(kids)


def _lab(rgb: np.ndarray) -> np.ndarray:
    v = rgb / 255
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)
