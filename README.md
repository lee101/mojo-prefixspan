# mojo-prefixspan

`mojo-prefixspan` is a standalone Mojo port of the compute-heavy core of
[`prefixspan`](https://pypi.org/project/prefixspan/), the Python package for
gap-allowing, single-item sequential pattern mining.

The public class keeps upstream's constructor, method names, signatures, result
format, ordering, callbacks, and `minlen` / `maxlen` attributes:

```python
from mojoprefixspan import PrefixSpan

db = [
    [0, 1, 2, 3, 4],
    [1, 1, 1, 3, 4],
    [2, 1, 2, 2, 0],
    [1, 1, 1, 2, 2],
]

ps = PrefixSpan(db)
ps.maxlen = 3
print(ps.frequent(3))
# [(4, [1]), (3, [1, 2]), (3, [2])]
```

## Upstream coverage

| API | Status | Execution |
| --- | --- | --- |
| `PrefixSpan(db)` | covered | Python input preparation |
| `frequent(minsup)` | covered | projected-database DFS in Mojo |
| `frequent(..., filter=, callback=)` | covered | Mojo mining, Python callable dispatch |
| `topk(k)` | covered | support-pruned traversal in Mojo |
| `topk(..., filter=, callback=)` | covered | compatibility traversal in Python |
| `closed=True` (BIDE behavior) | covered | compatibility traversal in Python |
| `generator=True` (FEAT behavior) | covered | compatibility traversal in Python |
| custom `key` and `bound` | covered | compatibility traversal in Python |
| `minlen`, `maxlen`, integer and text items | covered | upstream-compatible |

Every covered row above has a parity test against the installed upstream
package. The tests compare ordered results, not just pattern sets.

The Python compatibility paths are intentional. A Python `key`, `bound`,
`filter`, or `callback` cannot execute inside a native Mojo traversal. Default
frequent and top-k searches run in Mojo; BIDE, FEAT, and customized searches
stay in Python because their callback and pruning behavior is part of the
observable contract.

Not covered:

- upstream's command-line program;
- itemset sequences such as Spark's PrefixSpan (the targeted package mines
  ordinary sequences where each element is one item);
- upstream private implementation modules beyond the observable `PrefixSpan`
  class behavior;
- prebuilt shared libraries for platforms other than the pinned Linux Pixi
  environment.

## Install

The repository carries its own pinned Mojo nightly:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-prefixspan.so`. Importing the package
also rebuilds a missing or stale library. To use a prebuilt library elsewhere,
set `MOJO_PREFIXSPAN_LIB` to its path.

Run the usage example from the repository with `pixi run python` so Pixi's
`PYTHONPATH` activation finds the source package:

```bash
pixi run python - <<'PY'
from mojoprefixspan import PrefixSpan

ps = PrefixSpan([[0, 1, 2], [0, 2], [1, 2]])
print(ps.frequent(2))
PY
```

Expected output:

```text
[(2, [0]), (2, [0, 2]), (2, [1]), (2, [1, 2]), (3, [2])]
```

## Performance

Measured in the final review with `pixi run bench` on the machine reported by
the benchmark as `x86_64; Linux-6.8.0-136-generic-x86_64-with-glibc2.39`. Each
number is the best of three runs, includes construction and Python result
conversion, and the benchmark asserts that both implementations returned
identical ordered results.

| case | mojo-prefixspan | prefixspan 0.5.2 | result | patterns |
| --- | ---: | ---: | ---: | ---: |
| frequent, planted motif (10k x 40) | 66.14 ms | 915.51 ms | 13.84x faster | 7 |
| frequent, dense (3k x 24) | 117.41 ms | 1235.66 ms | 10.52x faster | 432 |
| frequent, sparse output (10k x 32) | 119.21 ms | 1419.63 ms | 11.91x faster | 65 |
| topk 25 (2k x 24) | 13.30 ms | 170.87 ms | 12.85x faster | 25 |

Every measured kernel remains more than 5x faster than upstream, so this
optimization review made no further kernel changes.

Default top-k uses the same dense database buffers as frequent mining, with
native support-ranked pruning and upstream-compatible tie ordering. Customized
top-k calls continue to use the Python compatibility traversal.

Run the same locked benchmark with:

```bash
pixi run bench
```

## How it works

Python first maps arbitrary hashable items to dense `int64` codes in
first-encounter order. The database becomes one contiguous item array plus an
offset array delimiting sequences. At each PrefixSpan depth, Mojo stores the
earliest matching end position for every sequence. Scanning the corresponding
suffixes produces extension supports and preserves upstream's first-encounter
order. Bulk position initialization and pattern copies use native-width SIMD
with scalar tails. Top-k projection scans parallelize only at 32,768 sequences
or more, in 4,096-sequence chunks capped at eight workers.

The miner makes two native passes. The first counts patterns and output item
cells; Python then allocates exact-size NumPy result buffers, and the second
pass fills supports, pattern offsets, and item codes. All projection, count,
ordering, and result memory is caller-owned, so the Mojo library performs no
heap allocation.

The exported frequent and top-k entry points are compiled into one shared
library. The `ctypes` declarations require aligned, contiguous, one-dimensional
NumPy `int64` buffers and keep their owners alive for the full call. Mojo
rebuilds those arguments as `UnsafePointer[Int64, AnyOrigin[mut=True]]`.
Python validates native output sizes, offsets, and item-code ranges before
converting results.

There is no GPU path. Prefix projection is an irregular, branch-heavy scan with
well below two arithmetic operations per byte moved; it does not have enough
arithmetic intensity to recover device-transfer and kernel-launch overhead.

## Development

Parity tests use the real PyPI `prefixspan==0.5.2` package on published vectors
and randomized databases. They cover exact result ordering, match positions,
callbacks, filters, custom scoring, top-k, closed patterns, and generators.

```bash
pixi run build
pixi run test
pixi run bench
```

## License

MIT
