"""Latency measurement core: percentiles, call timing, index-size sweeps.

Pure measurement helpers for the Phase 3 latency profiler
(ARCHITECTURE.md "Evaluation pipeline"): p50/p95 ingest and search latency
plus an index-size sweep. Everything here is a pure function over
caller-supplied callables -- no model imports, no I/O, no global state.

Timing is measured with ``time.perf_counter`` by default; the timer is
injectable so tests can run deterministically. Reporting is deterministic
(same samples -> same report); timings themselves are not.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["percentiles", "time_calls", "index_size_sweep"]

Timer = Callable[[], float]
"""Monotonic-ish clock: no arguments, returns a float (e.g. seconds)."""

CallFn = Callable[[], Any]
"""Zero-argument callable being timed; its return value is discarded."""

BuildIndexFn = Callable[[int, int], Any]
"""``(dim, size) ->`` a built index exposing ``search(vector, k) -> list``."""

PERCENTILE_QUANTILES: dict[str, float] = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}

_MS_PER_S = 1000.0


def _quantile(sorted_samples: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of sorted samples at ``q`` in [0, 1].

    Same convention as numpy's default ``method="linear"``: index
    ``h = (n - 1) * q`` into the sorted samples and interpolate between
    ``floor(h)`` and ``ceil(h)``. Returns an exact sample value when ``h``
    lands on an integer index.
    """
    h = (len(sorted_samples) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return float(sorted_samples[lo])
    frac = h - lo
    return float(sorted_samples[lo]) + frac * (float(sorted_samples[hi]) - float(sorted_samples[lo]))


def percentiles(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize latency samples: p50/p90/p95/p99 plus ``n`` and ``mean``.

    Percentiles use linear interpolation between closest ranks (numpy's
    default convention). Units are the caller's sample units (seconds when
    samples come from :func:`time_calls`).

    Args:
        samples: latency samples in arbitrary (consistent) units.

    Returns:
        ``{"p50", "p90", "p95", "p99", "n", "mean"}`` -- percentiles and mean
        as floats in sample units, ``n`` as int.

    Raises:
        ValueError: ``samples`` is empty or contains a non-finite value.
    """
    values = [float(s) for s in samples]
    if not values:
        raise ValueError("samples must not be empty")
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"samples must be finite, got {v}")
    values.sort()
    report: dict[str, float | int] = {
        name: _quantile(values, q) for name, q in PERCENTILE_QUANTILES.items()
    }
    report["n"] = len(values)
    report["mean"] = math.fsum(values) / len(values)
    return report


def time_calls(
    fn: CallFn,
    n: int,
    warmup: int = 0,
    *,
    timer: Timer = time.perf_counter,
) -> list[float]:
    """Time ``n`` calls of ``fn`` after ``warmup`` untimed calls.

    Each timed call is measured individually with ``timer()`` around a single
    invocation of ``fn``; warmup calls run first and are discarded so steady
    state (JIT, caches, lazy imports) is not reported.

    Args:
        fn: zero-argument callable to time (return value discarded).
        n: number of timed calls (>= 1).
        warmup: number of untimed warmup calls (>= 0).
        timer: zero-argument clock; defaults to ``time.perf_counter``.

    Returns:
        ``n`` per-call elapsed times as floats in timer units (seconds for
        ``time.perf_counter``).

    Raises:
        ValueError: ``n < 1`` or ``warmup < 0`` (non-integer inputs included).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be an int >= 1, got {n!r}")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(n):
        start = timer()
        fn()
        samples.append(timer() - start)
    return samples


def _as_int_list(value: int | Sequence[int], name: str) -> list[int]:
    """Normalize an int-or-sequence argument to a validated list of ints."""
    values = [int(value)] if isinstance(value, int) and not isinstance(value, bool) else [int(v) for v in value]
    if not values:
        raise ValueError(f"{name} must not be empty")
    for v in values:
        if v <= 0:
            raise ValueError(f"{name} values must be positive, got {v}")
    return values


def index_size_sweep(
    build_index_fn: BuildIndexFn,
    dims: int | Sequence[int],
    sizes: Sequence[int],
    k: int | Sequence[int],
    queries: Sequence[Sequence[float]] | Any,
    *,
    warmup: int = 3,
    repeats: int = 20,
    timer: Timer = time.perf_counter,
) -> list[dict[str, float | int]]:
    """Measure search latency across index sizes, dims, and k values.

    For every ``(dim, size)`` combination the index is built once via
    ``build_index_fn(dim, size)`` (build time is reported as ``build_ms``) and
    searched with every query vector in turn, cycling through ``queries``,
    once per k value. Each timed call is one ``index.search(query, k)``.

    Args:
        build_index_fn: ``(dim, size) ->`` built index exposing
            ``search(vector, k) -> list``. The caller decides how vectors get
            in (e.g. random unit-norm vectors).
        dims: one embedding dimensionality or a sequence of them.
        sizes: index sizes (vector counts) to sweep, ascending recommended.
        k: one top-k or a sequence of top-k values.
        queries: query vectors as an array-like of shape ``[n_queries, dim]``;
            rows are passed to ``search()`` one at a time.
        warmup: untimed warmup searches per ``(dim, size, k)`` cell.
        repeats: timed searches per ``(dim, size, k)`` cell (>= 1).
        timer: injectable clock; defaults to ``time.perf_counter``.

    Returns:
        One row dict per ``(dim, size, k)`` cell::

            {"index_size", "dim", "k", "n_samples", "build_ms",
             "search_p50_ms", "search_p90_ms", "search_p95_ms",
             "search_p99_ms", "search_mean_ms"}

        Rows are ordered by dim, then size, then k. Latencies are per single
        search, in milliseconds.

    Raises:
        ValueError: empty/invalid ``dims``, ``sizes``, or ``k``; empty
            ``queries``; queries whose row width differs from a swept dim; or
            ``repeats < 1`` / ``warmup < 0``.
    """
    dim_list = _as_int_list(dims, "dims")
    size_list = _as_int_list(sizes, "sizes")
    k_list = _as_int_list(k, "k")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")

    query_rows = [list(map(float, row)) for row in queries]
    if not query_rows:
        raise ValueError("queries must not be empty")
    width = len(query_rows[0])
    if any(len(row) != width for row in query_rows):
        raise ValueError("queries must be rectangular [n_queries, dim]")
    for dim in dim_list:
        if width != dim:
            raise ValueError(f"queries have width {width}, expected dim {dim}")

    rows: list[dict[str, float | int]] = []
    for dim in dim_list:
        for size in size_list:
            start = timer()
            index = build_index_fn(dim, size)
            build_ms = (timer() - start) * _MS_PER_S
            for k_val in k_list:
                state = {"i": 0}

                def _search(index=index, k_val=k_val, state=state) -> None:
                    row = query_rows[state["i"] % len(query_rows)]
                    state["i"] += 1
                    index.search(row, k_val)

                samples = time_calls(_search, repeats, warmup, timer=timer)
                stats = percentiles(samples)
                rows.append(
                    {
                        "index_size": size,
                        "dim": dim,
                        "k": k_val,
                        "n_samples": int(stats["n"]),
                        "build_ms": build_ms,
                        "search_p50_ms": float(stats["p50"]) * _MS_PER_S,
                        "search_p90_ms": float(stats["p90"]) * _MS_PER_S,
                        "search_p95_ms": float(stats["p95"]) * _MS_PER_S,
                        "search_p99_ms": float(stats["p99"]) * _MS_PER_S,
                        "search_mean_ms": float(stats["mean"]) * _MS_PER_S,
                    }
                )
    return rows
