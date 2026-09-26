"""Behavioral parity with prefixspan 0.5.2 on the same databases."""

from __future__ import annotations

import ctypes
import random

import numpy as np
import pytest

from prefixspan import PrefixSpan as UpstreamPrefixSpan

from mojoprefixspan import PrefixSpan
from mojoprefixspan._lib import lib


DB = [
    [0, 1, 2, 3, 4],
    [1, 1, 1, 3, 4],
    [2, 1, 2, 2, 0],
    [1, 1, 1, 2, 2],
]


def pair(db=DB):
    return PrefixSpan(db), UpstreamPrefixSpan(db)


def test_published_frequent_vector():
    ours, theirs = pair()
    expected = [
        (2, [0]),
        (4, [1]),
        (3, [1, 2]),
        (2, [1, 2, 2]),
        (2, [1, 3]),
        (2, [1, 3, 4]),
        (2, [1, 4]),
        (2, [1, 1]),
        (2, [1, 1, 1]),
        (3, [2]),
        (2, [2, 2]),
        (2, [3]),
        (2, [3, 4]),
        (2, [4]),
    ]
    assert ours.frequent(2) == expected == theirs.frequent(2)
    assert ours._fast_calls == 1


@pytest.mark.parametrize(
    ("closed", "generator"),
    [(True, False), (False, True), (True, True)],
)
def test_published_condensed_modes(closed, generator):
    ours, theirs = pair()
    assert ours.frequent(2, closed=closed, generator=generator) == theirs.frequent(
        2, closed=closed, generator=generator
    )


@pytest.mark.parametrize(
    ("closed", "generator"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_topk_parity(closed, generator):
    ours, theirs = pair()
    assert ours.topk(5, closed=closed, generator=generator) == theirs.topk(
        5, closed=closed, generator=generator
    )


def test_native_topk_simd_tails_stay_serial():
    db = [
        [(seq * 3 + pos * 5) % 7 for pos in range(9)]
        for seq in range(13)
    ]
    ours, theirs = pair(db)
    ours.maxlen = theirs.maxlen = 5
    assert ours.topk(11) == theirs.topk(11)
    assert ours._topk_fast_calls == 1
    assert ours._last_chunked_projections == 0


def test_native_topk_chunked_projection_threshold():
    db = [[seq % 3, (seq + 1) % 3, seq % 2] for seq in range(4_101)]
    ours, theirs = pair(db)
    ours.maxlen = theirs.maxlen = 2
    ours._projection_chunk_min_sequences = 4_096
    assert ours.topk(7) == theirs.topk(7)
    assert ours._last_chunked_projections > 0


def test_native_topk_falls_back_for_incomparable_items():
    db = [[1], [1], ["x"]]
    ours, theirs = pair(db)
    assert ours.topk(1) == theirs.topk(1)
    assert ours._topk_fast_calls == 0


def test_minlen_maxlen_and_reused_instance():
    ours, theirs = pair()
    ours.minlen = theirs.minlen = 2
    ours.maxlen = theirs.maxlen = 2
    assert ours.frequent(2) == theirs.frequent(2)
    ours.maxlen = theirs.maxlen = 3
    assert ours.frequent(3) == theirs.frequent(3)


def test_upstream_minlen_greater_than_maxlen_behavior():
    ours, theirs = pair()
    ours.minlen = theirs.minlen = 3
    ours.maxlen = theirs.maxlen = 2
    assert ours.frequent(2) == theirs.frequent(2)


def test_string_items():
    db = [
        ["red", "blue", "green"],
        ["blue", "red", "green"],
        ["red", "green"],
        [],
    ]
    ours, theirs = pair(db)
    assert ours.frequent(2) == theirs.frequent(2)
    assert ours.topk(4) == theirs.topk(4)


def test_filter_receives_upstream_matches():
    def keep(patt, matches):
        return matches[0][0] > 0 and len(patt) >= 2

    ours, theirs = pair()
    assert ours.frequent(1, filter=keep) == theirs.frequent(1, filter=keep)
    assert ours.topk(5, filter=keep) == theirs.topk(5, filter=keep)


def test_frequent_callback_and_match_positions():
    got_ours = []
    got_theirs = []
    ours, theirs = pair()
    assert (
        ours.frequent(2, callback=lambda patt, matches: got_ours.append((patt, matches)))
        is None
    )
    assert (
        theirs.frequent(
            2, callback=lambda patt, matches: got_theirs.append((patt, matches))
        )
        is None
    )
    assert got_ours == got_theirs


def test_topk_callback():
    got_ours = []
    got_theirs = []
    ours, theirs = pair()
    assert ours.topk(6, callback=lambda p, m: got_ours.append((p, m))) is None
    assert theirs.topk(6, callback=lambda p, m: got_theirs.append((p, m))) is None
    assert got_ours == got_theirs


def test_custom_key_and_bound():
    key = lambda patt, matches: len(matches) * 10 - len(patt)
    bound = lambda patt, matches: len(matches) * 10
    ours, theirs = pair()
    assert ours.frequent(18, key=key, bound=bound) == theirs.frequent(
        18, key=key, bound=bound
    )
    assert ours.topk(7, key=key, bound=bound) == theirs.topk(
        7, key=key, bound=bound
    )


def test_default_key_ignores_supplied_bound_like_upstream():
    fail_if_called = lambda patt, matches: (_ for _ in ()).throw(AssertionError())
    ours, theirs = pair()
    assert ours.frequent(2, bound=fail_if_called) == theirs.frequent(
        2, bound=fail_if_called
    )


@pytest.mark.parametrize("seed", range(8))
def test_randomized_parity(seed):
    rng = random.Random(seed)
    db = [
        [rng.randrange(5) for _ in range(rng.randrange(1, 8))]
        for _ in range(8)
    ]
    minsup = rng.randrange(1, 7)
    ours, theirs = pair(db)
    ours.minlen = theirs.minlen = rng.randrange(1, 3)
    ours.maxlen = theirs.maxlen = rng.randrange(2, 5)
    assert ours.frequent(minsup) == theirs.frequent(minsup)


def test_empty_sequences_and_empty_database():
    for db in ([[], [], []], []):
        ours, theirs = pair(db)
        assert ours.frequent(1) == theirs.frequent(1)
        assert ours.topk(3) == theirs.topk(3)


def test_database_is_not_mutated():
    db = [[1, 2, 1], [2, 1]]
    snapshot = [row.copy() for row in db]
    PrefixSpan(db).frequent(1)
    assert db == snapshot


def test_native_abi_rejects_wrong_dtype_and_strided_buffers():
    fn = lib().mps_mine
    good = np.ones(1, dtype=np.int64)
    wrong_dtype = np.ones(1, dtype=np.int32)
    strided = np.ones(4, dtype=np.int64)[::2]

    with pytest.raises(ctypes.ArgumentError):
        fn(wrong_dtype, *([good] * 15), 0)
    with pytest.raises(ctypes.ArgumentError):
        fn(strided, *([good] * 15), 0)


def test_native_scalars_do_not_narrow_silently():
    with pytest.raises(OverflowError, match="minimum support"):
        PrefixSpan([[1]]).frequent(1 << 70)
