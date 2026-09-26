"""Upstream-compatible PrefixSpan API with a Mojo frequent-mining core."""

from __future__ import annotations

from heapq import heappush, heappushpop
from typing import Any, Callable

import numpy as np

from ._lib import checked_i64, lib

Pattern = list[Any]
Matches = list[tuple[int, int]]


def _require_capacity(array: np.ndarray, size: int, name: str) -> None:
    if array.size < size:
        raise RuntimeError(
            f"{name} buffer has {array.size} cells; native call requires {size}"
        )


def _nextentries(db: list[list[Any]], matches: Matches) -> dict[Any, Matches]:
    occurs: dict[Any, Matches] = {}
    for seq, endpos in matches:
        seen: set[Any] = set()
        for pos in range(endpos + 1, len(db[seq])):
            item = db[seq][pos]
            if item not in seen:
                occurs.setdefault(item, []).append((seq, pos))
                seen.add(item)
    return occurs


def _matches(db: list[list[Any]], patt: Pattern) -> Matches:
    found: Matches = []
    for seq, row in enumerate(db):
        endpos = -1
        for item in patt:
            try:
                endpos = row.index(item, endpos + 1)
            except ValueError:
                break
        else:
            found.append((seq, endpos))
    return found


def _is_subsequence(needle: Pattern, row: list[Any]) -> bool:
    at = 0
    for item in row:
        if at < len(needle) and item == needle[at]:
            at += 1
    return at == len(needle)


def _is_generator(
    db: list[list[Any]],
    patt: Pattern,
    matches: Matches,
    occursstack: list[dict[Any, Matches]],
) -> bool:
    previous_support = (
        len(db)
        if len(patt) < 2
        else len(occursstack[len(patt) - 2][patt[-2]])
    )
    if previous_support == len(matches):
        return False
    for i in range(len(patt) - 1, 0, -1):
        item = patt[i]
        previous, current = occursstack[i - 1][item], occursstack[i][item]
        current_sequences = {seq for seq, _ in current}
        missing = (
            (seq, pos) for seq, pos in previous if seq not in current_sequences
        )
        if len(previous) == len(current) or all(
            not _is_subsequence(patt[i + 1 :], db[seq][pos + 1 :])
            for seq, pos in missing
        ):
            return False
    return True


def _can_generator_prune(
    patt: Pattern, occursstack: list[dict[Any, Matches]]
) -> bool:
    for i in range(len(patt) - 1, 0, -1):
        item = patt[i]
        if occursstack[i - 1][item] == occursstack[i][item]:
            return True
    return False


def _reverse_scan(
    db: list[list[Any]], previous_items: list[Any], scan_matches: Matches
) -> bool:
    for previous in reversed(previous_items):
        common: set[Any] = set()
        for index, (seq, endpos) in enumerate(scan_matches):
            local: set[Any] = set()
            for pos in range(endpos - 1, -1, -1):
                item = db[seq][pos]
                if item == previous:
                    scan_matches[index] = (seq, pos)
                    break
                local.add(item)
            if index == 0:
                common.update(local)
            else:
                common.intersection_update(local)
        if common:
            return True
    return False


def _is_closed(db: list[list[Any]], patt: Pattern, matches: Matches) -> bool:
    scan_matches = [(seq, len(db[seq])) for seq, _ in matches]
    return not _reverse_scan(db, [None, *patt], scan_matches)


def _can_closed_prune(db: list[list[Any]], patt: Pattern, matches: Matches) -> bool:
    return _reverse_scan(db, [None, *patt[:-1]], matches.copy())


class PrefixSpan:
    """Mine gap-allowing, single-item sequential patterns.

    The constructor and public attributes match ``prefixspan.PrefixSpan``.
    """

    defaultkey = staticmethod(lambda patt, matches: len(matches))
    _projection_chunk_min_sequences = 32_768

    def __init__(self, db):
        self._db = db
        self.minlen, self.maxlen = 1, 1000
        self._results: list[Any] = []
        self._fast_calls = 0
        self._topk_fast_calls = 0
        self._last_chunked_projections = 0

    def _mine_fast(self, minsup: int, maxlen: int) -> list[tuple[int, Pattern]]:
        db = self._db
        item_to_code: dict[Any, int] = {}
        code_to_item: list[Any] = []
        offsets = np.empty(len(db) + 1, dtype=np.int64)
        offsets[0] = 0
        flat_values: list[int] = []
        longest = 0
        for seq, row in enumerate(db):
            longest = max(longest, len(row))
            for item in row:
                code = item_to_code.get(item)
                if code is None:
                    code = len(code_to_item)
                    item_to_code[item] = code
                    code_to_item.append(item)
                flat_values.append(code)
            offsets[seq + 1] = len(flat_values)

        depth_limit = min(maxlen, longest)
        if depth_limit <= 0 or not code_to_item or not db:
            return []

        flat = np.asarray(flat_values, dtype=np.int64)
        nseq = len(db)
        nitems = len(code_to_item)
        positions = np.empty((depth_limit + 1) * nseq, dtype=np.int64)
        counts = np.empty(depth_limit * nitems, dtype=np.int64)
        order = np.empty(depth_limit * nitems, dtype=np.int64)
        seen = np.empty(nitems, dtype=np.int64)
        pattern = np.empty(depth_limit, dtype=np.int64)
        stats = np.empty(2, dtype=np.int64)
        dummy = np.empty(1, dtype=np.int64)
        required = (
            (flat, int(offsets[-1]), "database"),
            (offsets, nseq + 1, "offset"),
            (positions, (depth_limit + 1) * nseq, "position"),
            (counts, depth_limit * nitems, "count"),
            (order, depth_limit * nitems, "order"),
            (seen, nitems, "seen"),
            (pattern, depth_limit, "pattern"),
            (stats, 2, "statistics"),
        )
        for array, size, name in required:
            _require_capacity(array, size, name)

        args = (
            flat,
            offsets,
            checked_i64(nseq, "sequence count"),
            checked_i64(nitems, "item count"),
            checked_i64(minsup, "minimum support"),
            1,
            checked_i64(depth_limit, "maximum pattern length"),
            positions,
            counts,
            order,
            seen,
            pattern,
            stats,
        )
        # ctypes keeps every ndarray alive for the duration of the native call.
        # Its ndpointer declarations reject non-int64, unaligned, strided, and
        # non-1D buffers before an address crosses the ABI.
        lib().mps_mine(*args, dummy, dummy, dummy, 0)
        pattern_count, item_count = map(int, stats)
        if pattern_count < 0 or item_count < 0:
            raise RuntimeError("native miner returned invalid output sizes")

        supports = np.empty(max(1, pattern_count), dtype=np.int64)
        result_offsets = np.empty(max(1, pattern_count + 1), dtype=np.int64)
        result_items = np.empty(max(1, item_count), dtype=np.int64)
        _require_capacity(supports, pattern_count, "support output")
        _require_capacity(result_offsets, pattern_count + 1, "offset output")
        _require_capacity(result_items, item_count, "item output")
        lib().mps_mine(
            *args,
            supports,
            result_offsets,
            result_items,
            1,
        )
        if tuple(map(int, stats)) != (pattern_count, item_count):
            raise RuntimeError("native miner output changed between sizing and emit passes")
        if (
            int(result_offsets[0]) != 0
            or int(result_offsets[pattern_count]) != item_count
            or np.any(result_offsets[:pattern_count] > result_offsets[1 : pattern_count + 1])
            or np.any(result_items[:item_count] < 0)
            or np.any(result_items[:item_count] >= nitems)
        ):
            raise RuntimeError("native miner returned invalid result offsets or item codes")
        self._fast_calls += 1
        return [
            (
                int(supports[i]),
                [
                    code_to_item[int(code)]
                    for code in result_items[result_offsets[i] : result_offsets[i + 1]]
                ],
            )
            for i in range(pattern_count)
        ]

    def _topk_fast(self, k: int, maxlen: int) -> list[tuple[int, Pattern]] | None:
        db = self._db
        item_to_code: dict[Any, int] = {}
        code_to_item: list[Any] = []
        offsets = np.empty(len(db) + 1, dtype=np.int64)
        offsets[0] = 0
        flat_values: list[int] = []
        longest = 0
        for seq, row in enumerate(db):
            longest = max(longest, len(row))
            for item in row:
                code = item_to_code.get(item)
                if code is None:
                    code = len(code_to_item)
                    item_to_code[item] = code
                    code_to_item.append(item)
                flat_values.append(code)
            offsets[seq + 1] = len(flat_values)

        depth_limit = min(maxlen, longest)
        if depth_limit <= 0 or not code_to_item or not db:
            return []

        try:
            sorted_codes = sorted(
                range(len(code_to_item)), key=code_to_item.__getitem__
            )
            for left_code, right_code in zip(sorted_codes, sorted_codes[1:]):
                left = code_to_item[left_code]
                right = code_to_item[right_code]
                if left != right and not left < right:
                    return None
        except (TypeError, ValueError):
            return None

        flat = np.asarray(flat_values, dtype=np.int64)
        nseq = len(db)
        nitems = len(code_to_item)
        positions = np.empty((depth_limit + 1) * nseq, dtype=np.int64)
        counts = np.empty(depth_limit * nitems, dtype=np.int64)
        order = np.empty(depth_limit * nitems, dtype=np.int64)
        seen = np.empty(nitems, dtype=np.int64)
        pattern = np.empty(depth_limit, dtype=np.int64)
        item_ranks = np.empty(nitems, dtype=np.int64)
        for rank, code in enumerate(sorted_codes):
            item_ranks[code] = rank
        stats = np.empty(2, dtype=np.int64)
        supports = np.empty(k, dtype=np.int64)
        lengths = np.empty(k, dtype=np.int64)
        result_items = np.empty(k * depth_limit, dtype=np.int64)
        required = (
            (flat, int(offsets[-1]), "database"),
            (offsets, nseq + 1, "offset"),
            (positions, (depth_limit + 1) * nseq, "position"),
            (counts, depth_limit * nitems, "count"),
            (order, depth_limit * nitems, "order"),
            (seen, nitems, "seen"),
            (pattern, depth_limit, "pattern"),
            (item_ranks, nitems, "item rank"),
            (stats, 2, "statistics"),
            (supports, k, "support output"),
            (lengths, k, "length output"),
            (result_items, k * depth_limit, "item output"),
        )
        for array, size, name in required:
            _require_capacity(array, size, name)

        lib().mps_topk(
            flat,
            offsets,
            checked_i64(nseq, "sequence count"),
            checked_i64(nitems, "item count"),
            checked_i64(k, "k"),
            checked_i64(self.minlen, "minimum pattern length"),
            checked_i64(depth_limit, "maximum pattern length"),
            positions,
            counts,
            order,
            seen,
            pattern,
            item_ranks,
            stats,
            supports,
            lengths,
            result_items,
            checked_i64(
                self._projection_chunk_min_sequences, "projection chunk threshold"
            ),
        )
        self._topk_fast_calls += 1
        self._last_chunked_projections = int(stats[1])
        result_count = int(stats[0])
        if not 0 <= result_count <= k or np.any(lengths[:result_count] < 0) or np.any(
            lengths[:result_count] > depth_limit
        ):
            raise RuntimeError("native top-k miner returned invalid result sizes")
        used_codes = [
            result_items[slot * depth_limit : slot * depth_limit + int(lengths[slot])]
            for slot in range(result_count)
        ]
        if any(
            np.any(codes < 0) or np.any(codes >= nitems) for codes in used_codes
        ):
            raise RuntimeError("native top-k miner returned an invalid item code")
        results = []
        for slot in range(result_count):
            length = int(lengths[slot])
            base = slot * depth_limit
            results.append(
                (
                    int(supports[slot]),
                    [
                        code_to_item[int(code)]
                        for code in result_items[base : base + length]
                    ],
                )
            )
        return sorted(results, key=lambda entry: (-entry[0], entry[1]))

    def frequent(
        self,
        minsup: int,
        closed: bool = False,
        generator: bool = False,
        key: Callable | None = None,
        bound: Callable | None = None,
        filter: Callable | None = None,
        callback: Callable | None = None,
    ):
        if key is not None or closed or generator:
            return self._frequent_python(
                minsup, closed, generator, key, bound, filter, callback
            )

        internal_maxlen = self.maxlen
        if (
            self.minlen < 1
            or internal_maxlen < 1
            or self.minlen > internal_maxlen
        ):
            return self._frequent_python(
                minsup, closed, generator, key, bound, filter, callback
            )
        mined = self._mine_fast(minsup, internal_maxlen)
        self._results.clear()
        for sup, patt in mined:
            if not (self.minlen <= len(patt) <= self.maxlen):
                continue
            matches = None
            if filter is not None:
                matches = _matches(self._db, patt)
                if not filter(patt, matches):
                    continue
            if closed:
                if matches is None:
                    matches = _matches(self._db, patt)
                if not _is_closed(self._db, patt, matches):
                    continue
            if callback is not None:
                callback(patt, matches if matches is not None else _matches(self._db, patt))
            else:
                self._results.append((sup, patt))
        return None if callback else self._results

    def _frequent_python(
        self, minsup, closed, generator, key, bound, filter, callback
    ):
        if key is None:
            key = bound = PrefixSpan.defaultkey
        self._results.clear()
        occursstack: list[dict[Any, Matches]] = []

        def rec(patt: Pattern, matches: Matches):
            if len(patt) >= self.minlen:
                sup = key(patt, matches)
                if sup >= minsup and (filter is None or filter(patt, matches)):
                    valid = (
                        (not closed or _is_closed(self._db, patt, matches))
                        and (
                            not generator
                            or _is_generator(self._db, patt, matches, occursstack)
                        )
                    )
                    if valid:
                        if callback:
                            callback(patt, matches)
                        else:
                            self._results.append((sup, patt))
                if len(patt) == self.maxlen:
                    return
            occurs = _nextentries(self._db, matches)
            if generator:
                occursstack.append(occurs)
            for item, newmatches in occurs.items():
                newpatt = patt + [item]
                if bound(newpatt, newmatches) < minsup or (
                    closed and _can_closed_prune(self._db, newpatt, newmatches)
                ) or (
                    generator and _can_generator_prune(newpatt, occursstack)
                ):
                    continue
                rec(newpatt, newmatches)
            if generator:
                occursstack.pop()

        rec([], [(i, -1) for i in range(len(self._db))])
        return None if callback else self._results

    def topk(
        self,
        k: int,
        closed: bool = False,
        generator: bool = False,
        key: Callable | None = None,
        bound: Callable | None = None,
        filter: Callable | None = None,
        callback: Callable | None = None,
    ):
        if (
            key is None
            and not closed
            and not generator
            and filter is None
            and callback is None
            and k > 0
            and self.minlen >= 1
            and self.maxlen >= self.minlen
        ):
            fast_results = self._topk_fast(k, self.maxlen)
            if fast_results is not None:
                return fast_results

        if key is None:
            key = bound = PrefixSpan.defaultkey
        self._results.clear()
        occursstack: list[dict[Any, Matches]] = []
        def canpass(score):
            return len(self._results) == k and score <= self._results[0][0]

        def rec(patt: Pattern, matches: Matches):
            if len(patt) >= self.minlen:
                score = key(patt, matches)
                if not canpass(score) and (filter is None or filter(patt, matches)):
                    valid = (
                        (not closed or _is_closed(self._db, patt, matches))
                        and (
                            not generator
                            or _is_generator(self._db, patt, matches, occursstack)
                        )
                    )
                    if valid:
                        entry = (score, patt, matches)
                        if len(self._results) < k:
                            heappush(self._results, entry)
                        else:
                            heappushpop(self._results, entry)
                if len(patt) == self.maxlen:
                    return

            occurs = _nextentries(self._db, matches)
            if generator:
                occursstack.append(occurs)
            ranked = sorted(
                occurs.items(),
                key=lambda pair: key(patt + [pair[0]], pair[1]),
                reverse=True,
            )
            for item, newmatches in ranked:
                newpatt = patt + [item]
                if canpass(bound(newpatt, newmatches)):
                    break
                if (
                    closed and _can_closed_prune(self._db, newpatt, newmatches)
                ) or (
                    generator and _can_generator_prune(newpatt, occursstack)
                ):
                    continue
                rec(newpatt, newmatches)
            if generator:
                occursstack.pop()

        rec([], [(i, -1) for i in range(len(self._db))])
        results = sorted(self._results, key=lambda entry: (-entry[0], entry[1]))
        if callback:
            for _, patt, matches in results:
                callback(patt, matches)
            return None
        return [(score, patt) for score, patt, _ in results]
