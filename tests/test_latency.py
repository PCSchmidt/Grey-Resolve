"""Tests for the latency measurement core (Phase 3 latency profiler).

Deterministic by construction: fake indexes and fake clocks replace real
timing, and expectations are hand-computed from the percentile convention.
No model weights, no sleeps beyond micro-benchmark-free pure calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.evaluation.latency import (
    index_size_sweep,
    percentiles,
    time_calls,
)


# ----------------------------------------------------------------- fixtures


class CountingFn:
    """Zero-argument callable that counts invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.calls


class FakeTimer:
    """Deterministic clock: each read advances by a fixed step (seconds)."""

    def __init__(self, step: float = 0.001, start: float = 0.0) -> None:
        self.value = float(start)
        self.step = float(step)
        self.reads = 0

    def __call__(self) -> float:
        now = self.value
        self.value += self.step
        self.reads += 1
        return now


class FakeIndex:
    """Minimal index: remembers searches, returns one constant hit."""

    def __init__(self, dim: int, size: int) -> None:
        self.dim = dim
        self.size = size
        self.searches: list[tuple[tuple[float, ...], int]] = []

    def search(self, vector, k: int) -> list[tuple[int, float]]:
        self.searches.append((tuple(float(v) for v in vector), int(k)))
        return [(0, 1.0)]


# ----------------------------------------------------------------- percentiles


def test_percentiles_hand_computed_samples():
    # samples 1..4 sorted; n = 4; mean = 2.5
    # h = (n-1)*q over sorted [1, 2, 3, 4]:
    #   p50: h = 1.5 -> 2 + 0.5*(3-2) = 2.5   (exact in binary)
    #   p90: h = 2.7 -> 3 + 0.7*(4-3) = 3.7
    #   p95: h = 2.85 -> 3 + 0.85*(4-3) = 3.85
    #   p99: h = 2.97 -> 3 + 0.97*(4-3) = 3.97
    report = percentiles([1.0, 2.0, 3.0, 4.0])
    assert report["n"] == 4
    assert report["p50"] == 2.5
    assert report["mean"] == 2.5
    assert report["p90"] == pytest.approx(3.7, abs=1e-12)
    assert report["p95"] == pytest.approx(3.85, abs=1e-12)
    assert report["p99"] == pytest.approx(3.97, abs=1e-12)


def test_percentiles_hand_computed_ten_samples():
    # sorted 10..100 (step 10), n = 10; h = 9*q lands on integer indices
    # for q = 0.5 (4.5 -> interpolate 50/60) and q = 0.9 (8.1):
    #   p50: h = 4.5 -> 50 + 0.5*(60-50) = 55.0
    #   p90: h = 8.1 -> 90 + 0.1*(100-90) = 91.0
    #   p95: h = 8.55 -> 90 + 0.55*(100-90) = 95.5
    #   p99: h = 8.91 -> 90 + 0.91*(100-90) = 99.1
    #   mean = (10+...+100)/10 = 55.0
    samples = [float(v) for v in range(10, 101, 10)]
    report = percentiles(samples)
    assert report["n"] == 10
    assert report["mean"] == 55.0
    assert report["p50"] == 55.0
    assert report["p90"] == pytest.approx(91.0, abs=1e-12)
    assert report["p95"] == pytest.approx(95.5, abs=1e-12)
    assert report["p99"] == pytest.approx(99.1, abs=1e-12)


def test_percentiles_single_sample_all_equal():
    report = percentiles([2.5])
    assert report["n"] == 1
    for key in ("p50", "p90", "p95", "p99", "mean"):
        assert report[key] == 2.5


def test_percentiles_unsorted_input_sorted_first():
    assert percentiles([4.0, 1.0, 3.0, 2.0])["p50"] == 2.5


def test_percentiles_empty_raises():
    with pytest.raises(ValueError):
        percentiles([])


def test_percentiles_non_finite_raises():
    with pytest.raises(ValueError):
        percentiles([1.0, float("nan")])
    with pytest.raises(ValueError):
        percentiles([1.0, float("inf")])


# ----------------------------------------------------------------- time_calls


def test_time_calls_runs_warmup_plus_n_calls():
    fn = CountingFn()
    timer = FakeTimer(step=0.001)
    samples = time_calls(fn, n=3, warmup=2, timer=timer)
    assert fn.calls == 5  # 2 warmup + 3 timed
    assert len(samples) == 3
    # FakeTimer advances one step per read; two reads bracket each timed call.
    assert samples == [0.001, 0.001, 0.001]


def test_time_calls_returns_per_call_elapsed():
    # A timer that advances by 1.0 per read: each call costs exactly 1.0 s.
    samples = time_calls(CountingFn(), n=2, warmup=0, timer=FakeTimer(step=1.0))
    assert samples == [1.0, 1.0]


def test_time_calls_invalid_n_or_warmup_raises():
    fn = CountingFn()
    with pytest.raises(ValueError):
        time_calls(fn, n=0)
    with pytest.raises(ValueError):
        time_calls(fn, n=-1)
    with pytest.raises(ValueError):
        time_calls(fn, n=1, warmup=-1)
    assert fn.calls == 0  # rejected before any invocation


# ----------------------------------------------------------------- sweep


def test_index_size_sweep_rows_and_deterministic_latency():
    built: list[tuple[int, int]] = []
    seen_searches: list[FakeIndex] = []

    def build(dim: int, size: int) -> FakeIndex:
        built.append((dim, size))
        index = FakeIndex(dim, size)
        seen_searches.append(index)
        return index

    queries = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    timer = FakeTimer(step=0.5)  # binary-exact: every timed span = 500 ms
    rows = index_size_sweep(
        build, dims=[4], sizes=[10, 20], k=[1, 2], queries=queries, warmup=1, repeats=3, timer=timer
    )

    assert built == [(4, 10), (4, 20)]  # one build per (dim, size)
    assert [(r["dim"], r["index_size"], r["k"]) for r in rows] == [
        (4, 10, 1),
        (4, 10, 2),
        (4, 20, 1),
        (4, 20, 2),
    ]
    for row in rows:
        assert row["n_samples"] == 3
        assert row["search_p50_ms"] == 500.0
        assert row["search_p95_ms"] == 500.0
        assert row["search_mean_ms"] == 500.0
        assert row["build_ms"] == 500.0
        assert set(row) == {
            "index_size",
            "dim",
            "k",
            "n_samples",
            "build_ms",
            "search_p50_ms",
            "search_p90_ms",
            "search_p95_ms",
            "search_p99_ms",
            "search_mean_ms",
        }
    # each index served both k values; each cell ran warmup + timed searches
    for index in seen_searches:
        assert len(index.searches) == 2 * (1 + 3)  # two k cells x (warmup + repeats)
        assert {k for _, k in index.searches} == {1, 2}


def test_index_size_sweep_cycles_queries():
    built: list[FakeIndex] = []

    def build(dim: int, size: int) -> FakeIndex:
        index = FakeIndex(dim, size)
        built.append(index)
        return index

    queries = [[1.0], [2.0], [3.0]]
    index_size_sweep(
        build, dims=1, sizes=[1], k=1, queries=queries, warmup=0, repeats=5, timer=FakeTimer()
    )
    used = [vector for vector, _ in built[0].searches]
    assert used == [(1.0,), (2.0,), (3.0,), (1.0,), (2.0,)]


def test_index_size_sweep_value_error_edges():
    build = FakeIndex
    queries = [[1.0, 0.0]]

    with pytest.raises(ValueError):  # empty sizes
        index_size_sweep(build, dims=2, sizes=[], k=1, queries=queries)
    with pytest.raises(ValueError):  # non-positive size
        index_size_sweep(build, dims=2, sizes=[0], k=1, queries=queries)
    with pytest.raises(ValueError):  # non-positive k
        index_size_sweep(build, dims=2, sizes=[1], k=0, queries=queries)
    with pytest.raises(ValueError):  # empty queries
        index_size_sweep(build, dims=2, sizes=[1], k=1, queries=[])
    with pytest.raises(ValueError):  # query width != dim
        index_size_sweep(build, dims=3, sizes=[1], k=1, queries=queries)
    with pytest.raises(ValueError):  # repeats < 1
        index_size_sweep(build, dims=2, sizes=[1], k=1, queries=queries, repeats=0)
    with pytest.raises(ValueError):  # warmup < 0
        index_size_sweep(build, dims=2, sizes=[1], k=1, queries=queries, warmup=-1)


def test_index_size_sweep_accepts_int_dims_and_k():
    rows = index_size_sweep(
        FakeIndex,
        dims=2,
        sizes=[5],
        k=3,
        queries=np.zeros((1, 2)),
        warmup=0,
        repeats=2,
        timer=FakeTimer(),
    )
    assert rows[0]["dim"] == 2
    assert rows[0]["k"] == 3
    assert rows[0]["index_size"] == 5
