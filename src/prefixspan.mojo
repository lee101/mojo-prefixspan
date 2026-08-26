"""Prefix-projected sequential pattern mining over caller-owned buffers."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of as simdwidthof

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PROJECT_CHUNK_SIZE = 4096


def fill_i64(ptr: IPtr, size: Int, value: Int64):
    var i = 0
    var vector_value = SIMD[DType.int64, W](value)
    while i + W <= size:
        ptr.store[alignment=1](i, vector_value)
        i += W
    while i < size:
        ptr[i] = value
        i += 1


def copy_i64(source: IPtr, destination: IPtr, size: Int):
    var i = 0
    while i + W <= size:
        destination.store[alignment=1](
            i,
            source.load[width=W, alignment=1](i),
        )
        i += W
    while i < size:
        destination[i] = source[i]
        i += 1


def mine_rec(
    db: IPtr,
    offsets: IPtr,
    nseq: Int,
    nitems: Int,
    minsup: Int,
    minlen: Int,
    maxlen: Int,
    depth: Int,
    support: Int,
    positions: IPtr,
    counts: IPtr,
    order: IPtr,
    seen: IPtr,
    pattern: IPtr,
    stats: IPtr,
    supports: IPtr,
    pattern_offsets: IPtr,
    pattern_items: IPtr,
    emit: Bool,
):
    if depth >= minlen:
        if emit:
            var result_index = Int(stats[0])
            var item_index = Int(stats[1])
            supports[result_index] = Int64(support)
            pattern_offsets[result_index] = Int64(item_index)
            for i in range(depth):
                pattern_items[item_index + i] = pattern[i]
            pattern_offsets[result_index + 1] = Int64(item_index + depth)
        stats[0] += 1
        stats[1] += Int64(depth)

    if depth >= maxlen:
        return

    var count_row = depth * nitems
    var order_row = depth * nitems
    for item in range(nitems):
        counts[count_row + item] = 0
        seen[item] = -1

    var order_size = 0
    var pos_row = depth * nseq
    for seq in range(nseq):
        var endpos = Int(positions[pos_row + seq])
        if endpos < -1:
            continue
        var start = Int(offsets[seq]) + endpos + 1
        var stop = Int(offsets[seq + 1])
        for at in range(start, stop):
            var item = Int(db[at])
            if Int(seen[item]) != seq:
                seen[item] = Int64(seq)
                if counts[count_row + item] == 0:
                    order[order_row + order_size] = Int64(item)
                    order_size += 1
                counts[count_row + item] += 1

    for oi in range(order_size):
        var item = Int(order[order_row + oi])
        var next_support = Int(counts[count_row + item])
        if next_support < minsup:
            continue

        var next_row = (depth + 1) * nseq
        for seq in range(nseq):
            positions[next_row + seq] = -2
            var endpos = Int(positions[pos_row + seq])
            if endpos < -1:
                continue
            var start = Int(offsets[seq]) + endpos + 1
            var stop = Int(offsets[seq + 1])
            for at in range(start, stop):
                if Int(db[at]) == item:
                    positions[next_row + seq] = Int64(at - Int(offsets[seq]))
                    break

        pattern[depth] = Int64(item)
        mine_rec(
            db,
            offsets,
            nseq,
            nitems,
            minsup,
            minlen,
            maxlen,
            depth + 1,
            next_support,
            positions,
            counts,
            order,
            seen,
            pattern,
            stats,
            supports,
            pattern_offsets,
            pattern_items,
            emit,
        )


def slot_less_than_slot(
    left: Int,
    right: Int,
    maxlen: Int,
    item_ranks: IPtr,
    result_lengths: IPtr,
    result_items: IPtr,
) -> Bool:
    var left_depth = Int(result_lengths[left])
    var right_depth = Int(result_lengths[right])
    var common = min(left_depth, right_depth)
    var left_base = left * maxlen
    var right_base = right * maxlen
    for i in range(common):
        var left_rank = item_ranks[Int(result_items[left_base + i])]
        var right_rank = item_ranks[Int(result_items[right_base + i])]
        if left_rank < right_rank:
            return True
        if left_rank > right_rank:
            return False
    return left_depth < right_depth


def topk_min_slot(
    count: Int,
    maxlen: Int,
    item_ranks: IPtr,
    result_supports: IPtr,
    result_lengths: IPtr,
    result_items: IPtr,
) -> Int:
    var minimum = 0
    for slot in range(1, count):
        if result_supports[slot] < result_supports[minimum] or (
            result_supports[slot] == result_supports[minimum]
            and slot_less_than_slot(
                slot,
                minimum,
                maxlen,
                item_ranks,
                result_lengths,
                result_items,
            )
        ):
            minimum = slot
    return minimum


def project_range(
    db: IPtr,
    offsets: IPtr,
    positions: IPtr,
    pos_row: Int,
    next_row: Int,
    item: Int,
    seq_begin: Int,
    seq_end: Int,
):
    for seq in range(seq_begin, seq_end):
        var endpos = Int(positions[pos_row + seq])
        if endpos < -1:
            continue
        var seq_start = Int(offsets[seq])
        var start = seq_start + endpos + 1
        var stop = Int(offsets[seq + 1])
        for at in range(start, stop):
            if Int(db[at]) == item:
                positions[next_row + seq] = Int64(at - seq_start)
                break


def topk_rec(
    db: IPtr,
    offsets: IPtr,
    nseq: Int,
    nitems: Int,
    k: Int,
    minlen: Int,
    maxlen: Int,
    depth: Int,
    support: Int,
    positions: IPtr,
    counts: IPtr,
    order: IPtr,
    seen: IPtr,
    pattern: IPtr,
    item_ranks: IPtr,
    stats: IPtr,
    result_supports: IPtr,
    result_lengths: IPtr,
    result_items: IPtr,
    parallel_threshold: Int,
):
    if depth >= minlen:
        var result_count = Int(stats[0])
        var accept = result_count < k
        var destination = result_count
        if not accept:
            destination = topk_min_slot(
                result_count,
                maxlen,
                item_ranks,
                result_supports,
                result_lengths,
                result_items,
            )
            accept = support > Int(result_supports[destination])
        if accept:
            if result_count < k:
                stats[0] += 1
            result_supports[destination] = Int64(support)
            result_lengths[destination] = Int64(depth)
            copy_i64(
                pattern,
                result_items + destination * maxlen,
                depth,
            )

    if depth >= maxlen:
        return

    var count_row = depth * nitems
    var order_row = depth * nitems
    fill_i64(counts + count_row, nitems, 0)
    fill_i64(seen, nitems, -1)

    var order_size = 0
    var pos_row = depth * nseq
    for seq in range(nseq):
        var endpos = Int(positions[pos_row + seq])
        if endpos < -1:
            continue
        var start = Int(offsets[seq]) + endpos + 1
        var stop = Int(offsets[seq + 1])
        for at in range(start, stop):
            var item = Int(db[at])
            if Int(seen[item]) != seq:
                seen[item] = Int64(seq)
                if counts[count_row + item] == 0:
                    order[order_row + order_size] = Int64(item)
                    order_size += 1
                counts[count_row + item] += 1

    for i in range(1, order_size):
        var item = order[order_row + i]
        var item_support = counts[count_row + Int(item)]
        var at = i
        while at > 0:
            var previous = order[order_row + at - 1]
            if counts[count_row + Int(previous)] >= item_support:
                break
            order[order_row + at] = previous
            at -= 1
        order[order_row + at] = item

    for oi in range(order_size):
        var item = Int(order[order_row + oi])
        var next_support = Int(counts[count_row + item])
        var result_count = Int(stats[0])
        if result_count == k:
            var minimum = topk_min_slot(
                result_count,
                maxlen,
                item_ranks,
                result_supports,
                result_lengths,
                result_items,
            )
            if next_support <= Int(result_supports[minimum]):
                break

        var next_row = (depth + 1) * nseq
        fill_i64(positions + next_row, nseq, -2)
        if nseq >= parallel_threshold:
            var chunks = (nseq + PROJECT_CHUNK_SIZE - 1) // PROJECT_CHUNK_SIZE

            @__copy_capture(
                db,
                offsets,
                positions,
                nseq,
                pos_row,
                next_row,
                item,
            )
            @__parameter
            def project_chunk(chunk: Int):
                var seq_begin = chunk * PROJECT_CHUNK_SIZE
                var seq_end = min(seq_begin + PROJECT_CHUNK_SIZE, nseq)
                project_range(
                    db,
                    offsets,
                    positions,
                    pos_row,
                    next_row,
                    item,
                    seq_begin,
                    seq_end,
                )

            parallelize[project_chunk](chunks, min(chunks, 8))
            stats[1] += 1
        else:
            project_range(
                db,
                offsets,
                positions,
                pos_row,
                next_row,
                item,
                0,
                nseq,
            )

        pattern[depth] = Int64(item)
        topk_rec(
            db,
            offsets,
            nseq,
            nitems,
            k,
            minlen,
            maxlen,
            depth + 1,
            next_support,
            positions,
            counts,
            order,
            seen,
            pattern,
            item_ranks,
            stats,
            result_supports,
            result_lengths,
            result_items,
            parallel_threshold,
        )


@export("mps_mine")
def mps_mine(
    db_addr: Int,
    offsets_addr: Int,
    nseq: Int,
    nitems: Int,
    minsup: Int,
    minlen: Int,
    maxlen: Int,
    positions_addr: Int,
    counts_addr: Int,
    order_addr: Int,
    seen_addr: Int,
    pattern_addr: Int,
    stats_addr: Int,
    supports_addr: Int,
    pattern_offsets_addr: Int,
    pattern_items_addr: Int,
    emit_flag: Int,
) abi("C"):
    initialize_runtime()
    var db = IPtr(unsafe_from_address=db_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var positions = IPtr(unsafe_from_address=positions_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var seen = IPtr(unsafe_from_address=seen_addr)
    var pattern = IPtr(unsafe_from_address=pattern_addr)
    var stats = IPtr(unsafe_from_address=stats_addr)
    var supports = IPtr(unsafe_from_address=supports_addr)
    var pattern_offsets = IPtr(unsafe_from_address=pattern_offsets_addr)
    var pattern_items = IPtr(unsafe_from_address=pattern_items_addr)

    stats[0] = 0
    stats[1] = 0
    for seq in range(nseq):
        positions[seq] = -1
    if emit_flag != 0:
        pattern_offsets[0] = 0

    mine_rec(
        db,
        offsets,
        nseq,
        nitems,
        minsup,
        minlen,
        maxlen,
        0,
        nseq,
        positions,
        counts,
        order,
        seen,
        pattern,
        stats,
        supports,
        pattern_offsets,
        pattern_items,
        emit_flag != 0,
    )


@export("mps_topk")
def mps_topk(
    db_addr: Int,
    offsets_addr: Int,
    nseq: Int,
    nitems: Int,
    k: Int,
    minlen: Int,
    maxlen: Int,
    positions_addr: Int,
    counts_addr: Int,
    order_addr: Int,
    seen_addr: Int,
    pattern_addr: Int,
    item_ranks_addr: Int,
    stats_addr: Int,
    result_supports_addr: Int,
    result_lengths_addr: Int,
    result_items_addr: Int,
    parallel_threshold: Int,
) abi("C"):
    initialize_runtime()
    var db = IPtr(unsafe_from_address=db_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var positions = IPtr(unsafe_from_address=positions_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var seen = IPtr(unsafe_from_address=seen_addr)
    var pattern = IPtr(unsafe_from_address=pattern_addr)
    var item_ranks = IPtr(unsafe_from_address=item_ranks_addr)
    var stats = IPtr(unsafe_from_address=stats_addr)
    var result_supports = IPtr(unsafe_from_address=result_supports_addr)
    var result_lengths = IPtr(unsafe_from_address=result_lengths_addr)
    var result_items = IPtr(unsafe_from_address=result_items_addr)

    stats[0] = 0
    stats[1] = 0
    fill_i64(positions, nseq, -1)
    topk_rec(
        db,
        offsets,
        nseq,
        nitems,
        k,
        minlen,
        maxlen,
        0,
        nseq,
        positions,
        counts,
        order,
        seen,
        pattern,
        item_ranks,
        stats,
        result_supports,
        result_lengths,
        result_items,
        parallel_threshold,
    )
