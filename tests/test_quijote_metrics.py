import numpy as np
import pytest
import jax.numpy as jnp

from pmpp.core import Configuration
from pmpp.extras.quijote import (
    fixed_grid_spectra, gather_authoritative_particles, interlaced_cic_density, periodic_position_metrics,
    pmid_to_lagrangian_idx, summarize_acceptance,
)
from pmpp.nbody import Particles


def _canonical_grid(grid_size):
    return np.moveaxis(np.indices((grid_size, ) * 3, dtype=np.int32), 0, -1).reshape(-1, 3)


def _particle_state_with_padding_and_halo(mesh_ratio):
    grid_size = 4
    conf = Configuration(
        ptcl_spacing=1.0, ptcl_grid_shape=(grid_size, ) * 3, mesh_shape=mesh_ratio, float_dtype=jnp.float32,
    )
    q = _canonical_grid(grid_size)
    pmid = q * mesh_ratio
    offset = np.array([0.15, -0.2, 0.3], dtype=np.float32)
    disp = np.broadcast_to(offset, q.shape).copy()
    velocity = (q + np.array([10, 20, 30])).astype(np.float32)

    order = np.random.default_rng(100 + mesh_ratio).permutation(q.shape[0])
    # Append one halo copy and one padding slot.  Neither may contribute to
    # the authoritative dense state.
    pmid_slots = np.concatenate((pmid[order], pmid[[7]], np.zeros((1, 3), int)))
    disp_slots = np.concatenate((disp[order], disp[[7]], np.zeros((1, 3), np.float32)))
    velocity_slots = np.concatenate((velocity[order], velocity[[7]], np.zeros((1, 3), np.float32)))
    unused = np.zeros(pmid_slots.shape[0], dtype=bool)
    unused[-1] = True
    halo = np.zeros(pmid_slots.shape[0], dtype=bool)
    halo[-2] = True
    particles = Particles(conf, pmid_slots, disp_slots, vel=velocity_slots, unused_index=unused, halo_mask=halo, )
    expected_position = np.mod(q.astype(np.float32) + offset, grid_size)
    return conf, particles, q, expected_position, velocity


@pytest.mark.parametrize("mesh_ratio", [1, 2])
def test_lagrangian_identity_and_authoritative_gather_are_bijective(mesh_ratio):
    conf, particles, q, expected_position, expected_velocity = (_particle_state_with_padding_and_halo(mesh_ratio))

    direct_indices = pmid_to_lagrangian_idx(q * mesh_ratio, conf)
    np.testing.assert_array_equal(np.asarray(direct_indices), np.arange(q.shape[0]))

    dense = gather_authoritative_particles(particles, conf)
    np.testing.assert_array_equal(np.asarray(dense.counts), 1)
    np.testing.assert_allclose(np.asarray(dense.position), expected_position, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(dense.velocity), expected_velocity)


@pytest.mark.parametrize(("replacement", "message"), [((0, 1), "missing=.*duplicates="),
                                                      ((13, 22), "missing=.*duplicates="), ],
                         )
def test_authoritative_gather_rejects_duplicate_and_missing_identity(replacement, message):
    conf, particles, _, _, _ = _particle_state_with_padding_and_halo(2)
    pmid = np.asarray(particles.pmid).copy()
    # Replace one authoritative anchor with another authoritative anchor.  The
    # resulting state has one duplicate and one missing Lagrangian key.
    pmid[replacement[0]] = pmid[replacement[1]]
    broken = particles.replace(pmid=pmid)

    with pytest.raises(ValueError, match=message):
        gather_authoritative_particles(broken, conf)


def test_periodic_position_metrics_use_minimum_image_and_particle_cells():
    box_size = 10.0
    cell_size = 2.0
    target = np.array([[9.9, 1.0, 1.0], [0.1, 2.0, 2.0], [5.0, 9.8, 3.0], [4.0, 4.0, 0.2], ], dtype=np.float32, )
    predicted = np.array([[0.1, 1.0, 1.0], [9.8, 2.0, 2.0], [5.0, 0.2, 3.0], [4.0, 4.0, 9.7], ], dtype=np.float32, )
    expected_distances = np.array([0.2, 0.3, 0.4, 0.5])

    metrics = periodic_position_metrics(predicted, target, box_size, particle_cell_size=cell_size)

    np.testing.assert_allclose(np.asarray(metrics.distances), expected_distances, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(metrics.rmse_cells),
        np.sqrt(np.mean(expected_distances**2)) / cell_size, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics.p99_cells), np.quantile(expected_distances / cell_size, 0.99), atol=1e-6,
    )


def test_identical_positions_have_unit_fixed_grid_cross_correlation():
    rng = np.random.default_rng(44)
    box_size = 25.0
    positions = rng.uniform(0, box_size, size=(400, 3)).astype(np.float32)

    density = interlaced_cic_density(positions, box_size, analysis_grid=8)
    spectra = fixed_grid_spectra(positions, positions, box_size, analysis_grid=8, )

    np.testing.assert_allclose(np.asarray(density).mean(), 1.0, atol=2e-6)
    valid = np.asarray(spectra.nmodes) > 0
    np.testing.assert_allclose(np.asarray(spectra.r)[valid], 1.0, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(spectra.pk_reference), np.asarray(spectra.pk_candidate), atol=1e-5, rtol=2e-7,
    )

    positions_metrics = periodic_position_metrics(positions, positions, box_size, particle_cell_size=box_size / 8)
    summary = summarize_acceptance(spectra, positions_metrics, k_max=np.pi * 8 / box_size, )
    assert bool(np.asarray(summary.passed))
    np.testing.assert_allclose(np.asarray(summary.min_r), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(summary.rmse_cells), 0.0)
    np.testing.assert_allclose(np.asarray(summary.p99_cells), 0.0)
