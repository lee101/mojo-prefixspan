"""mojo-prefixspan versus prefixspan 0.5.2 on identical databases."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojoprefixspan import PrefixSpan  # noqa: E402
from prefixspan import PrefixSpan as UpstreamPrefixSpan  # noqa: E402


def best_time(fn, reps=3):
    best = float("inf")
    value = None
    for _ in range(reps):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def frequent_runner(cls, db, minsup, maxlen):
    def run():
        miner = cls(db)
        miner.maxlen = maxlen
        return miner.frequent(minsup)

    return run


def topk_runner(cls, db, k, maxlen):
    def run():
        miner = cls(db)
        miner.maxlen = maxlen
        return miner.topk(k)

    return run


def report(name, ours, reference):
    mojo_s, mojo_value = best_time(ours)
    ref_s, ref_value = best_time(reference)
    assert mojo_value == ref_value
    ratio = ref_s / mojo_s
    label = "faster" if ratio >= 1.0 else "slower"
    print(
        f"| {name} | {mojo_s * 1e3:.2f} ms | {ref_s * 1e3:.2f} ms | "
        f"{ratio:.2f}x {label} | {len(mojo_value)} |"
    )


def main():
    rng = np.random.default_rng(0)

    planted = []
    for row in rng.integers(0, 100, size=(10_000, 37)).tolist():
        row[5:5] = [100, 101, 102]
        planted.append(row)
    dense = rng.integers(0, 20, size=(3_000, 24)).tolist()
    random_db = rng.integers(0, 64, size=(10_000, 32)).tolist()
    topk_db = rng.integers(0, 32, size=(2_000, 24)).tolist()

    print(f"Machine: {platform.processor() or platform.machine()}; {platform.platform()}")
    print()
    print("| case | mojo-prefixspan | prefixspan 0.5.2 | result | patterns |")
    print("| --- | ---: | ---: | ---: | ---: |")
    report(
        "frequent, planted motif (10k x 40)",
        frequent_runner(PrefixSpan, planted, 5_000, 4),
        frequent_runner(UpstreamPrefixSpan, planted, 5_000, 4),
    )
    report(
        "frequent, dense (3k x 24)",
        frequent_runner(PrefixSpan, dense, 400, 3),
        frequent_runner(UpstreamPrefixSpan, dense, 400, 3),
    )
    report(
        "frequent, sparse output (10k x 32)",
        frequent_runner(PrefixSpan, random_db, 1_000, 3),
        frequent_runner(UpstreamPrefixSpan, random_db, 1_000, 3),
    )
    report(
        "topk 25 (2k x 24)",
        topk_runner(PrefixSpan, topk_db, 25, 4),
        topk_runner(UpstreamPrefixSpan, topk_db, 25, 4),
    )


if __name__ == "__main__":
    main()
