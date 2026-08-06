import jax
import jax.numpy as jnp
import numpy as np

from pmpp.distributed import routing as hm

CAPACITY = 12
SHARE_CAPACITY = 4
KEY_FILL = np.iinfo(np.int32).max


def _stream(keys, size, source):
    count = len(keys)
    padded_keys = np.full((size, ), KEY_FILL, dtype=np.int32)
    padded_keys[:count] = keys
    valid = np.arange(size) < count
    base = source * 100 + np.arange(size)
    pmid = np.stack((base, base + 1, base + 2), axis=-1).astype(np.int32)
    disp = np.stack((base + 0.1, base + 0.2, base + 0.3), axis=-1).astype(np.float32)
    vel = disp + 10
    acc = disp + 20
    return tuple(map(jnp.asarray, (padded_keys, pmid, disp, vel, acc, valid)))


def _take_stream(stream, source_slots, size=SHARE_CAPACITY):
    valid_count = len(source_slots)
    slots = np.full((size, ), -1, dtype=np.int32)
    slots[:valid_count] = source_slots
    valid = np.arange(size) < valid_count
    gather_slots = np.where(valid, slots, 0)
    values = []
    for value in stream[:-1]:
        gathered = np.asarray(value)[gather_slots]
        mask = valid.reshape(valid.shape + (1, ) * (gathered.ndim - 1))
        fill = KEY_FILL if value is stream[0] else 0
        values.append(np.where(mask, gathered, fill).astype(value.dtype))
    return (*map(jnp.asarray, values), jnp.asarray(valid)), jnp.asarray(slots)


def _compact_stay(stream, outgoing_slots):
    valid_count = int(np.asarray(stream[-1]).sum())
    outgoing = set(outgoing_slots)
    stay_slots = [slot for slot in range(valid_count) if slot not in outgoing]
    compact, _ = _take_stream(stream, stay_slots, size=stream[0].shape[0])
    return compact, np.asarray(stay_slots, dtype=np.int32)


def _inputs():
    original = _stream([1, 2, 4, 6, 8, 10, 12, 14], CAPACITY, 1)
    outgoing_left, left_slots = _take_stream(original, [1, 6])
    outgoing_right, right_slots = _take_stream(original, [3])
    incoming_b = _stream([0, 5, 11], SHARE_CAPACITY, 2)
    incoming_c = _stream([3, 9, 13], SHARE_CAPACITY, 3)
    stay, expected_stay_slots = _compact_stay(original, [1, 3, 6])
    return (
        original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, stay,
        expected_stay_slots,
    )


def _assert_merge_equal(reference, candidate, valid_field=-1):
    reference = jax.device_get(reference)
    candidate = jax.device_get(candidate)
    valid = np.asarray(reference[valid_field], dtype=bool)
    np.testing.assert_array_equal(candidate[valid_field], reference[valid_field])
    for expected, actual in zip(reference[:valid_field], candidate[:valid_field]):
        np.testing.assert_array_equal(np.asarray(actual)[valid], np.asarray(expected)[valid])


def _reference_merge(*streams, capacity, provenance=False):
    """Test-local stable merge oracle for the active sparse routing helpers."""
    keys = jnp.concatenate(tuple(jnp.where(stream[-1], stream[0], jnp.int32(KEY_FILL)) for stream in streams))
    order = jnp.argsort(keys, stable=True)[:capacity]
    valid = jnp.arange(capacity) < sum(jnp.sum(stream[-1]) for stream in streams)
    mask = valid.reshape((capacity, ) + (1, ) * (streams[0][1].ndim - 1))
    fields = []
    for field_index in range(1, len(streams[0]) - 1):
        values = jnp.concatenate(tuple(stream[field_index] for stream in streams))
        fields.append(jnp.where(mask, values[order], jnp.zeros_like(values[order])))
    merged_keys = jnp.where(valid, keys[order], jnp.int32(KEY_FILL))
    merged = (merged_keys, *fields, valid)
    if not provenance:
        return merged
    source_tag = jnp.concatenate(
        tuple(jnp.full((stream[0].shape[0], ), tag, dtype=jnp.int32) for tag, stream in enumerate(streams))
    )
    source_index = jnp.concatenate(tuple(jnp.arange(stream[0].shape[0], dtype=jnp.int32) for stream in streams))
    return (
        *merged, jnp.where(valid, source_tag[order],
                           jnp.int32(3)), jnp.where(valid, source_index[order], jnp.int32(-1)),
    )


def test_sparse_three_stream_route_matches_compact_then_merge_exactly():
    (
        original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, stay,
        expected_stay_slots,
    ) = _inputs()
    error = "overflow {x} {y}"
    reference = _reference_merge(stay, incoming_b, incoming_c, capacity=CAPACITY)
    candidate, stay_pos, stay_valid = hm._sparse_route_merge_three(
        original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, CAPACITY,
        jnp.int32(KEY_FILL), error,
    )
    _assert_merge_equal(reference, candidate)
    np.testing.assert_array_equal(np.asarray(stay_pos)[np.asarray(stay_valid)], expected_stay_slots)

    reference_prov = _reference_merge(stay, incoming_b, incoming_c, capacity=CAPACITY, provenance=True)
    candidate_prov, _, _ = hm._sparse_route_merge_three(
        original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, CAPACITY,
        jnp.int32(KEY_FILL), error, provenance=True,
    )
    _assert_merge_equal(reference_prov, candidate_prov, valid_field=5)
    np.testing.assert_array_equal(candidate_prov[6], reference_prov[6])
    np.testing.assert_array_equal(candidate_prov[7], reference_prov[7])


def test_sparse_two_stream_and_no_acc_routes_match_exactly():
    original, outgoing_left, left_slots, _, _, incoming_b, _, _, _ = _inputs()
    stay, expected_stay_slots = _compact_stay(original, [1, 6])
    error = "overflow {x} {y}"
    reference = _reference_merge(stay, incoming_b, capacity=CAPACITY)
    candidate, stay_pos, stay_valid = hm._sparse_route_merge_two(
        original, outgoing_left, left_slots, incoming_b, CAPACITY, jnp.int32(KEY_FILL), error,
    )
    _assert_merge_equal(reference, candidate)
    np.testing.assert_array_equal(np.asarray(stay_pos)[np.asarray(stay_valid)], expected_stay_slots)

    original_no_acc = (*original[:4], original[-1])
    outgoing_no_acc = (*outgoing_left[:4], outgoing_left[-1])
    incoming_no_acc = (*incoming_b[:4], incoming_b[-1])
    stay_no_acc = (*stay[:4], stay[-1])
    reference_no_acc = _reference_merge(stay_no_acc, incoming_no_acc, capacity=CAPACITY)
    candidate_no_acc, _, _ = hm._sparse_route_merge_two(
        original_no_acc, outgoing_no_acc, left_slots, incoming_no_acc, CAPACITY, jnp.int32(KEY_FILL), error,
    )
    _assert_merge_equal(reference_no_acc, candidate_no_acc)


def test_sparse_route_payload_gradient_matches_compact_then_merge():
    (
        original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, stay,
        expected_stay_slots,
    ) = _inputs()
    weights = jnp.arange(CAPACITY * 3, dtype=jnp.float32).reshape(CAPACITY, 3) / 19
    expected_slots = jnp.asarray(expected_stay_slots)
    stay_count = expected_slots.shape[0]

    def reference_loss(disp_a, disp_b, disp_c):
        compact_disp = jnp.zeros_like(disp_a).at[:stay_count].set(disp_a[expected_slots])
        stay_dynamic = (stay[0], stay[1], compact_disp, stay[3], stay[4], stay[5])
        incoming_b_dynamic = (incoming_b[0], incoming_b[1], disp_b, incoming_b[3], incoming_b[4], incoming_b[5])
        incoming_c_dynamic = (incoming_c[0], incoming_c[1], disp_c, incoming_c[3], incoming_c[4], incoming_c[5])
        output = _reference_merge(stay_dynamic, incoming_b_dynamic, incoming_c_dynamic, capacity=CAPACITY)
        return jnp.sum(output[2] * weights * output[5][:, None])

    def sparse_loss(disp_a, disp_b, disp_c):
        original_dynamic = (original[0], original[1], disp_a, original[3], original[4], original[5])
        incoming_b_dynamic = (incoming_b[0], incoming_b[1], disp_b, incoming_b[3], incoming_b[4], incoming_b[5])
        incoming_c_dynamic = (incoming_c[0], incoming_c[1], disp_c, incoming_c[3], incoming_c[4], incoming_c[5])
        output, _, _ = hm._sparse_route_merge_three(
            original_dynamic, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b_dynamic,
            incoming_c_dynamic, CAPACITY, jnp.int32(KEY_FILL), "overflow {x} {y}",
        )
        return jnp.sum(output[2] * weights * output[5][:, None])

    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(original[2], incoming_b[2], incoming_c[2])
    actual = jax.grad(sparse_loss, argnums=(0, 1, 2))(original[2], incoming_b[2], incoming_c[2])
    for expected_grad, actual_grad in zip(expected, actual):
        np.testing.assert_array_equal(actual_grad, expected_grad)


def test_sparse_route_randomized_exactness():
    capacity = 24
    share_capacity = 6
    universe = np.arange(80, dtype=np.int32)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(universe, size=20, replace=False)
        original_keys = np.sort(chosen[:15])
        incoming_b_keys = np.sort(chosen[15:18])
        incoming_c_keys = np.sort(chosen[18:20])
        original = _stream(original_keys, capacity, 10 + seed)
        outgoing_slots = np.sort(rng.choice(15, size=5, replace=False))
        left_source_slots = outgoing_slots[::2].tolist()
        right_source_slots = outgoing_slots[1::2].tolist()
        outgoing_left, left_slots = _take_stream(original, left_source_slots, size=share_capacity)
        outgoing_right, right_slots = _take_stream(original, right_source_slots, size=share_capacity)
        incoming_b = _stream(incoming_b_keys, share_capacity, 30 + seed)
        incoming_c = _stream(incoming_c_keys, share_capacity, 50 + seed)
        stay, _ = _compact_stay(original, outgoing_slots.tolist())
        reference = _reference_merge(stay, incoming_b, incoming_c, capacity=capacity)
        candidate, _, _ = hm._sparse_route_merge_three(
            original, outgoing_left, left_slots, outgoing_right, right_slots, incoming_b, incoming_c, capacity,
            jnp.int32(KEY_FILL), "overflow {x} {y}",
        )
        _assert_merge_equal(reference, candidate)
