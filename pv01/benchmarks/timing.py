"""Timing shared by the benchmark scripts."""

from __future__ import annotations

from statistics import median
from time import perf_counter
from typing import Callable

import jax

RUNS = 20


def time_jitted(function: Callable, arguments: tuple, runs: int = RUNS) -> tuple[float, float]:
    """Return (compile seconds, median run seconds) for ``jax.jit(function)``.

    The first call, which traces and compiles, is timed on its own; the
    median is over ``runs`` further calls. Every call waits for the result
    with ``block_until_ready``, so asynchronous dispatch isn't mistaken for
    speed.
    """
    compiled = jax.jit(function)
    start = perf_counter()
    compiled(*arguments).block_until_ready()
    compile_seconds = perf_counter() - start

    durations = []
    for _ in range(runs):
        start = perf_counter()
        compiled(*arguments).block_until_ready()
        durations.append(perf_counter() - start)
    return compile_seconds, median(durations)
