from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from pmpp.analysis import plotting
from pmpp.core.utils import get_a_schedule, measure_execution_time, pytree_dataclass, wraparound_slice
from pmpp.extras.camels import io as camels_io
from pmpp.numerics.fft import fftfreq, fftfwd, fftinv


def test_fft_helpers_match_jax_reference_and_physical_normalization():
    field = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)
    for norm in (None, "backward", "ortho", "forward"):
        actual = fftfwd(field, norm=norm)
        expected = jnp.fft.rfftn(field, norm=norm)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fftinv(actual, shape=field.shape, norm=norm), field, rtol=1e-13, atol=1e-13)

    spacing = 0.25
    transformed = fftfwd(field, norm=spacing)
    np.testing.assert_allclose(transformed, spacing**3 * jnp.fft.rfftn(field), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(fftinv(transformed, shape=field.shape, norm=spacing), field, rtol=1e-13, atol=1e-13)

    axes = (1, 2)
    transformed_axes = fftfwd(field, shape=(3, 4), axes=axes, norm=spacing)
    expected_axes = spacing**2 * jnp.fft.rfftn(field, s=(3, 4), axes=axes)
    np.testing.assert_allclose(transformed_axes, expected_axes, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        fftinv(transformed_axes, shape=(3, 4), axes=axes, norm=spacing), field, rtol=1e-13, atol=1e-13,
    )

    with pytest.raises(ValueError, match="must be real"):
        fftfwd(field.astype(jnp.complex128))
    with pytest.raises(ValueError, match="Hermitian complex"):
        fftinv(field)


def test_fftfreq_matches_angular_numpy_convention_for_sparse_and_dense_grids():
    shape = (4, 6, 8)
    sparse = fftfreq(shape, spacing=None, dtype=jnp.float32, sparse=True)
    assert [component.shape for component in sparse] == [(4, 1, 1), (1, 6, 1), (1, 1, 5)]
    np.testing.assert_allclose(np.asarray(sparse[0]).ravel(), np.fft.fftfreq(4), rtol=0, atol=0)

    dense = fftfreq(shape, spacing=0.5, dtype=jnp.float64, sparse=False)
    assert all(component.shape == (4, 6, 5) for component in dense)
    np.testing.assert_allclose(dense[2][0, 0], np.fft.rfftfreq(8) * (2 * np.pi / 0.5), rtol=1e-13, atol=1e-13)


def test_pytree_dataclass_static_dynamic_contract_and_transform_behavior():

    @partial(pytree_dataclass, aux_fields="label", frozen=True)
    class State:
        value: jax.Array
        label: str

    state = State(jnp.asarray([1.0, 2.0]), "science")
    leaves, treedef = jax.tree.flatten(state)
    assert len(leaves) == 1
    np.testing.assert_array_equal(leaves[0], state.value)
    restored = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_array_equal(restored.value, state.value)
    assert restored.label == state.label
    assert list(state.named_children()) == [("value", state.value)]
    assert list(state.named_aux_data()) == [("label", "science")]
    assert list(state.children()) == [state.value]
    assert list(state.aux_data()) == ["science"]
    assert state.replace(label="control").label == "control"
    assert "science" in str(state)
    np.testing.assert_array_equal(jax.jit(lambda item: item.value * 3)(state), np.array([3.0, 6.0]))

    @partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
    class StaticState:
        size: int
        label: str

    static = StaticState(4, "fixed")
    static_leaves, static_tree = jax.tree.flatten(static)
    assert static_leaves == []
    assert jax.tree.unflatten(static_tree, static_leaves) is static

    @partial(pytree_dataclass, aux_fields=("dynamic", ), aux_invert=True)
    class InvertedState:
        dynamic: jax.Array
        static: int

    inverted = InvertedState(jnp.asarray(2.0), 3)
    assert list(inverted.named_children()) == [("dynamic", inverted.dynamic)]
    assert list(inverted.named_aux_data()) == [("static", 3)]

    @dataclass
    class AlreadyDataclass:
        value: int

    with pytest.raises(TypeError, match="already be a dataclass"):
        pytree_dataclass(AlreadyDataclass)


def test_schedule_includes_targets_and_respects_maximum_scale_factor_step():
    conf = SimpleNamespace(
        a_nbody_maxstep=0.075, a_nbody=jnp.asarray([0.05, 0.10, 0.20, 0.40, 0.70, 1.0], dtype=jnp.float64),
    )
    targets = jnp.asarray([3.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
    schedule = np.asarray(get_a_schedule(targets, conf))
    assert np.all(np.diff(schedule) >= 0)
    assert np.max(np.diff(schedule)) <= conf.a_nbody_maxstep + 1e-12
    for target_z in np.unique(np.asarray(targets)):
        assert np.any(np.isclose(schedule, 1 / (1 + target_z), rtol=0, atol=1e-8))


def test_wraparound_slice_is_periodic_axis_aware_and_zero_padded():
    array = jnp.arange(12).reshape(3, 4)
    wrapped_rows = wraparound_slice(array, 2, 5, 5, axis=0)
    expected_rows = np.vstack(
        (np.asarray(array[2]), np.asarray(array[0]), np.asarray(array[1]), np.zeros(4), np.zeros(4))
    )
    np.testing.assert_array_equal(wrapped_rows, expected_rows)

    wrapped_columns = wraparound_slice(array, 3, 6, 4, axis=1)
    expected_columns = np.column_stack(
        (np.asarray(array[:, 3]), np.asarray(array[:, 0]), np.asarray(array[:, 1]), np.zeros(3))
    )
    np.testing.assert_array_equal(wrapped_columns, expected_columns)


def test_measure_execution_time_executes_the_requested_blocking_trials():
    calls = []

    def operation():
        calls.append(len(calls))
        return jnp.asarray(len(calls), dtype=jnp.int32)

    mean, std = measure_execution_time(operation, repetitions=2, number=3)
    assert len(calls) == 6
    assert mean >= 0
    assert std >= 0


class _PlotParticles:

    def __init__(self, positions, unused, conf):
        self._positions = positions
        self.unused_index = unused
        self.conf = conf

    def pos(self, dtype, wrap):
        assert dtype == jnp.float32
        assert wrap is True
        return self._positions


def test_plotting_reports_exact_particle_counts_for_single_and_forced_multi_device(monkeypatch):
    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    conf = SimpleNamespace(box_size=(8.0, 8.0, 8.0), mesh_shape=(8, 8, 8), num_devices=2)
    positions = jnp.asarray([[0.5, 0, 0], [1.5, 0, 0], [4.5, 0, 0], [7.5, 0, 0]], dtype=jnp.float32)
    particles = _PlotParticles(positions, jnp.asarray([False, True, False, False]), conf)

    plotting.plot_particle_distribution_on_gpus(particles)
    axis = plotting.plt.gca()
    assert axis.get_title() == "Particle X-Axis Distribution by GPU (n=4)"
    np.testing.assert_array_equal([patch.get_height() for patch in axis.patches], [1, 1, 0, 0, 1, 0, 0, 1])
    plotting.plt.close("all")

    plotting.plot_particle_distribution_on_gpus(particles, force_mGPU=True)
    axis = plotting.plt.gca()
    assert axis.get_title() == "Particle X-Axis Distribution by GPU (n=3)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["0", "1"]
    plotting.plt.close("all")

    no_device = _PlotParticles(np.asarray(positions), np.asarray([False] * 4), conf)
    with pytest.raises(ValueError, match="No devices detected"):
        plotting.plot_particle_distribution_on_gpus(no_device)


def test_plot_helpers_apply_masks_wrapping_titles_and_callback(monkeypatch):
    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    pos = jnp.asarray([[0.5, 0, 0], [8.5, 1, 1], [0, 0, 0], [3.5, 1, 1]], dtype=jnp.float32)
    mask = jnp.asarray([True, True, False, False])
    plotting.plot_particle_bins(pos, 8, title=1, mask=mask)
    axis = plotting.plt.gca()
    assert axis.get_title() == "Particles in halo"
    assert sum(patch.get_height() for patch in axis.patches) == 2
    plotting.plt.close("all")

    plotting.plot_particle_bins(pos, 8, title="custom")
    assert plotting.plt.gca().get_title() == "custom"
    plotting.plt.close("all")

    plotting.plot_pos_distribution(pos, SimpleNamespace(mesh_shape=(8, 8, 8)))
    assert plotting.plt.gca().get_title() == "Particle X-Axis Distribution(n=3)"
    plotting.plt.close("all")

    captured = {}

    def fake_callback(callback, result_shape, *args):
        captured["result_shape"] = result_shape
        callback(*args)

    monkeypatch.setattr(plotting, "io_callback", fake_callback)
    plotting.plot_particle_bins_callback(pos, mask, 8)
    assert captured["result_shape"] == ()
    assert plotting.plt.gca().get_title() == "All particles"
    plotting.plt.close("all")

    assert plotting.resolve_title(10) == "Particles that need to be shared to the right slice"
    assert plotting.resolve_title(999) == "Unknown Title"


def _camels_arrays():
    grid = np.moveaxis(np.indices((2, 2, 2), dtype=np.float32), 0, -1).reshape(-1, 3) * 12.5
    ids = np.arange(10, 18, dtype=np.int64)
    ic_pos = np.mod(grid + np.array([0.25, -0.5, 0.75], np.float32), 25.0)
    ic_vel = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    final_pos = np.mod(ic_pos + np.array([0.5, 0.25, -0.75], np.float32), 25.0)
    final_vel = ic_vel + 2
    return ids, ic_pos, ic_vel, final_pos, final_vel


def _write_camels_npz_pair(base, *, matching_ids=True):
    ids, ic_pos, ic_vel, final_pos, final_vel = _camels_arrays()
    ic_order = np.asarray([5, 0, 7, 2, 1, 6, 4, 3])
    final_order = np.asarray([2, 6, 1, 7, 0, 3, 5, 4])
    final_ids = ids.copy()
    if not matching_ids:
        final_ids[-1] = 99
    np.savez(
        base / "ics.npz", pos=ic_pos[ic_order], vel=ic_vel[ic_order], ids=ids[ic_order], BoxSize=25.0, Omega_m=0.31,
        Omega_l=0.69, redshift=127.0,
    )
    np.savez(
        base / "snapshot_090.npz", pos=final_pos[final_order], vel=final_vel[final_order], ids=final_ids[final_order],
        BoxSize=25.0, Omega_m=0.31, Omega_l=0.69, redshift=0.0,
    )
    params = base / "ICs" / "2LPT.param"
    params.parent.mkdir()
    params.write_text(
        "\nOmega 0.31\nOmegaLambda 0.69\nOmegaBaryon -1\nHubbleParam 0.68\n"
        "Sigma8 0.83\nPrimordialIndex 0.97\nRedshift 127\nmalformed\nignored nope\n", encoding="utf-8",
    )
    return ids, ic_pos, ic_vel, final_pos, final_vel, ic_order


def test_camels_npz_pair_loader_aligns_ids_metadata_and_periodic_fields(tmp_path):
    ids, ic_pos, ic_vel, final_pos, final_vel, ic_order = _write_camels_npz_pair(tmp_path)
    pair = camels_io.load_camels_pair(tmp_path)

    np.testing.assert_array_equal(pair.ids, ids[ic_order] - ids.min())
    np.testing.assert_array_equal(pair.ic_pos, ic_pos[ic_order])
    np.testing.assert_array_equal(pair.ic_vel, ic_vel[ic_order])
    np.testing.assert_array_equal(pair.final_pos, final_pos[ic_order])
    np.testing.assert_array_equal(pair.final_vel, final_vel[ic_order])
    assert pair.metadata.grid_size == 2
    assert pair.metadata.omega_b == camels_io.PLANCK18_OMEGA_B
    assert pair.metadata.a_start == pytest.approx(1 / 128)
    assert pair.metadata.sigma8 == pytest.approx(0.83)

    wrapped = camels_io.periodic_wrap(np.asarray([-0.25, 25.25]), 25.0)
    np.testing.assert_allclose(wrapped, [24.75, 0.25])
    delta = camels_io.periodic_delta(np.asarray([24.75, 0.25]), np.asarray([0.25, 24.75]), 25.0)
    np.testing.assert_allclose(delta, [-0.5, 0.5])


def test_camels_loaders_reject_missing_mismatched_and_non_cubic_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot_090"):
        camels_io.load_camels_pair(tmp_path)

    _write_camels_npz_pair(tmp_path, matching_ids=False)
    with pytest.raises(ValueError, match="particle IDs do not match"):
        camels_io.load_camels_pair(tmp_path)

    assert camels_io._infer_grid_size(27) == 3
    with pytest.raises(ValueError, match="cubic particle grid"):
        camels_io._infer_grid_size(9)


def test_raw_camels_hdf5_loader_sorts_ids_converts_units_and_uses_defaults(tmp_path):
    ics_dir = tmp_path / "ICs"
    ics_dir.mkdir()
    raw_ids = np.asarray([4, 1, 3, 2], dtype=np.int64)
    raw_pos = np.arange(12, dtype=np.float32).reshape(4, 3) * 1000
    raw_vel = np.arange(12, dtype=np.float32).reshape(4, 3) * 100
    for shard, selection in enumerate((slice(0, 2), slice(2, 4))):
        with h5py.File(ics_dir / f"ics.{shard}.hdf5", "w") as handle:
            group = handle.create_group("PartType1")
            group.create_dataset("Coordinates", data=raw_pos[selection])
            group.create_dataset("Velocities", data=raw_vel[selection])
            group.create_dataset("ParticleIDs", data=raw_ids[selection])
    (ics_dir / "2LPT.param").write_text("Box 8000\nOmega 0.32\nOmegaLambda 0.68\n", encoding="utf-8")

    loaded = camels_io._load_camels_ics_hdf5(ics_dir)
    order = np.argsort(raw_ids - 1)
    np.testing.assert_array_equal(loaded["ids"], raw_ids[order] - 1)
    np.testing.assert_array_equal(loaded["pos"], raw_pos[order] / 1000)
    np.testing.assert_allclose(loaded["vel"], raw_vel[order] / 100 / 128, rtol=1e-7, atol=0)
    assert loaded["BoxSize"] == 8.0
    assert loaded["redshift"] == 127.0

    with pytest.raises(FileNotFoundError, match="No CAMELS IC files"):
        camels_io._load_camels_ics_hdf5(tmp_path / "empty")


def _regular_pair(*, ids=None, perturb_positions=False):
    n = 4
    box = 8.0
    anchors = np.moveaxis(np.indices((n, n, n), dtype=np.float32), 0, -1) * (box / n)
    ic_pos = np.mod(anchors + np.array([0.25, -0.25, 0.5], np.float32), box).reshape(-1, 3)
    if perturb_positions:
        ic_pos[:] = 0.123
    final_pos = np.mod(ic_pos + np.array([0.5, 0.25, -0.75], np.float32), box)
    vel = np.broadcast_to(np.array([1.0, -2.0, 3.0], np.float32), ic_pos.shape).copy()
    metadata = camels_io.CamelsMetadata(box, 0.3, 0.7, 0.05, 0.7, 0.8, 0.96, 1 / 128, 0, n)
    if ids is None:
        ids = np.arange(n**3, dtype=np.int64)
    return camels_io.CamelsParticlePair(ic_pos, vel, final_pos, vel * 2, np.asarray(ids), metadata)


def test_camels_coarsening_preserves_periodic_mean_displacements_and_validates_layout():
    pair = _regular_pair()
    assert camels_io.coarsen_camels_pair(pair, 1) is pair
    coarse = camels_io.coarsen_camels_pair(pair, 2)
    assert coarse.metadata.grid_size == 2
    assert coarse.ids.shape == (8, )
    coarse_anchor = np.moveaxis(np.indices((2, 2, 2), dtype=np.float32), 0, -1).reshape(-1, 3) * 4
    np.testing.assert_allclose(
        camels_io.periodic_delta(coarse.ic_pos, coarse_anchor, 8), np.broadcast_to([0.25, -0.25, 0.5], (8, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        camels_io.periodic_delta(coarse.final_pos, coarse_anchor, 8), np.broadcast_to([0.75, 0, -0.25], (8, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(coarse.ic_vel, np.broadcast_to([1, -2, 3], (8, 3)), atol=0)
    np.testing.assert_allclose(coarse.final_vel, np.broadcast_to([2, -4, 6], (8, 3)), atol=0)

    with pytest.raises(ValueError, match="factor must be"):
        camels_io.coarsen_camels_pair(pair, 0)
    with pytest.raises(ValueError, match="not divisible"):
        camels_io.coarsen_camels_pair(pair, 3)

    fallback = _regular_pair(ids=np.full(64, 99, dtype=np.int64))
    fallback_coarse = camels_io.coarsen_camels_pair(fallback, 2)
    np.testing.assert_allclose(fallback_coarse.ic_vel, coarse.ic_vel)

    invalid = _regular_pair(ids=np.full(64, 99, dtype=np.int64), perturb_positions=True)
    with pytest.raises(ValueError, match="do not map cleanly"):
        camels_io.coarsen_camels_pair(invalid, 2)


def test_camels_velocity_conversions_are_dtype_stable_and_unit_exact():
    velocity = np.asarray([[100.0, -200.0, 50.0]], dtype=np.float64)
    converted = camels_io.gadget_velocity_to_pmpp(velocity, redshift=1)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, [[0.5, -1.0, 0.25]], rtol=0, atol=0)

    canonical = camels_io.velocity_kms_to_canonical(velocity, SimpleNamespace(V=2_000_000.0), extra_scale=3)
    np.testing.assert_allclose(canonical, velocity.astype(np.float32) / 2000 * 3, rtol=0, atol=0)
