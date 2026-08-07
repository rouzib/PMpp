import numpy as np
import pytest
import jax.numpy as jnp

from pmpp.core import Configuration
from pmpp.extras.quijote import (
    build_quijote_canonicalization, canonicalize_quijote_arrays, fixed_grid_spectra, gather_authoritative_particles,
    periodic_position_metrics, pmid_to_lagrangian_idx,
)
from pmpp.extras.quijote.metrics import authoritative_particle_mask, validate_lagrangian_bijection
from pmpp.nbody import Particles


def _canonical_inputs(n=2):
    q = np.moveaxis(np.indices((n, n, n), dtype=np.float32), 0, -1).reshape(-1, 3)
    ids = np.arange(100, 100 + n**3, dtype=np.int64)[::-1]
    q = q[::-1]
    velocity = np.zeros_like(q)
    return ids, q, velocity


def _mapping():
    ids, position, velocity = _canonical_inputs()
    return build_quijote_canonicalization(
        ids, position, velocity, box_size=2, a_start=0.01, expansion_rate=10, grid_size=2,
    )


def test_quijote_mapping_preserves_dtype_and_rejects_identity_corruption():
    mapping = _mapping()
    assert mapping.particle_count == 8
    assert mapping.canonical_q_coordinates().dtype.kind == "u"
    assert mapping.canonical_q_coordinates(np.float32).dtype == np.float32

    ids, position, _ = _canonical_inputs()
    canonical, = mapping.apply(ids, position)
    np.testing.assert_array_equal(canonical, np.moveaxis(np.indices((2, 2, 2)), 0, -1).reshape(-1, 3))
    with pytest.raises(TypeError, match="QuijoteCanonicalization"):
        canonicalize_quijote_arrays(object(), ids, position)
    with pytest.raises(ValueError, match="at least one"):
        canonicalize_quijote_arrays(mapping, ids)
    with pytest.raises(ValueError, match="first dimension"):
        canonicalize_quijote_arrays(mapping, ids, np.zeros((7, 3)))
    bad_ids = ids.copy()
    bad_ids[0] = 999
    with pytest.raises(ValueError, match="ID set does not match"):
        mapping.permutation_for(bad_ids)


@pytest.mark.parametrize(("ids", "message"), [(np.zeros((2, 2), dtype=np.int64), "one-dimensional"),
                                              (np.asarray([1.0, 2.0]), "integer dtype"),
                                              (np.asarray([True, False]), "integer dtype"),
                                              (np.asarray([], dtype=np.int64), "must not be empty"),
                                              (np.asarray([1, 1], dtype=np.int64), "unique"), ],
                         )
def test_quijote_id_validation_rejects_ambiguous_particle_identity(ids, message):
    with pytest.raises(ValueError, match=message):
        build_quijote_canonicalization(
            ids, np.zeros((len(ids), 3)), np.zeros((len(ids), 3)), box_size=1, a_start=0.1, expansion_rate=1
        )


def test_quijote_geometry_validation_rejects_nonphysical_or_nonbijective_initial_conditions():
    ids, position, velocity = _canonical_inputs()
    with pytest.raises(ValueError, match="exact integer cube"):
        build_quijote_canonicalization(
            ids[:-1], position[:-1], velocity[:-1], box_size=2, a_start=0.1, expansion_rate=1
        )
    for grid in (True, 0, 2.5, "bad"):
        with pytest.raises(ValueError, match="grid_size"):
            build_quijote_canonicalization(
                ids, position, velocity, box_size=2, a_start=0.1, expansion_rate=1, grid_size=grid
            )
    with pytest.raises(ValueError, match="requires 27 particles"):
        build_quijote_canonicalization(ids, position, velocity, box_size=2, a_start=0.1, expansion_rate=1, grid_size=3)

    with pytest.raises(ValueError, match="ic_pos must have shape"):
        build_quijote_canonicalization(ids, position[:, :2], velocity, box_size=2, a_start=0.1, expansion_rate=1)
    with pytest.raises(ValueError, match="real numeric"):
        build_quijote_canonicalization(
            ids, position.astype(np.complex64), velocity, box_size=2, a_start=0.1, expansion_rate=1
        )
    nonfinite = position.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_quijote_canonicalization(ids, nonfinite, velocity, box_size=2, a_start=0.1, expansion_rate=1)
    for name, kwargs in (("box_size", dict(box_size=-1, a_start=0.1, expansion_rate=1)),
                         ("a_start", dict(box_size=2, a_start="bad", expansion_rate=1)),
                         ("expansion_rate", dict(box_size=2, a_start=0.1, expansion_rate=np.inf)),
                         ):
        with pytest.raises(ValueError, match=name):
            build_quijote_canonicalization(ids, position, velocity, **kwargs)
    for tolerance in (-0.1, 0.5, np.nan, "bad"):
        with pytest.raises(ValueError, match="q_tolerance"):
            build_quijote_canonicalization(
                ids, position, velocity, box_size=2, a_start=0.1, expansion_rate=1, q_tolerance=tolerance
            )

    off_grid = position.copy()
    off_grid[0, 0] += 0.3
    with pytest.raises(ValueError, match="exceeds q_tolerance"):
        build_quijote_canonicalization(
            ids, off_grid, velocity, box_size=2, a_start=0.1, expansion_rate=1, q_tolerance=0.25
        )
    duplicate_cell = position.copy()
    duplicate_cell[0] = duplicate_cell[1]
    with pytest.raises(ValueError, match="not a bijection"):
        build_quijote_canonicalization(ids, duplicate_cell, velocity, box_size=2, a_start=0.1, expansion_rate=1)


def test_lagrangian_index_and_authoritative_mask_shape_errors_fail_closed():
    conf = Configuration(1.0, (2, 2, 2), mesh_shape=1, float_dtype=jnp.float32)
    with pytest.raises(ValueError, match="final dimension"):
        pmid_to_lagrangian_idx(jnp.zeros((3, 2), dtype=jnp.int32), conf)
    with pytest.raises(ValueError, match="signed integer"):
        pmid_to_lagrangian_idx(jnp.zeros((3, 3), dtype=jnp.int32), conf, dtype=jnp.uint32)
    with pytest.raises(ValueError, match="unused_index must match"):
        pmid_to_lagrangian_idx(jnp.zeros((3, 3), dtype=jnp.int32), conf, unused_index=jnp.zeros(2, dtype=jnp.bool_))

    ptcl = Particles.gen_grid(conf)
    with pytest.raises(ValueError, match="unused_index must match"):
        authoritative_particle_mask(ptcl.replace(unused_index=jnp.zeros(7, dtype=jnp.bool_)))
    with pytest.raises(ValueError, match="halo_mask must match"):
        authoritative_particle_mask(ptcl.replace(halo_mask=jnp.zeros(7, dtype=jnp.bool_)))


@pytest.mark.parametrize("particle_count", [True, 0, 2.5, "bad"])
def test_bijection_validation_rejects_invalid_particle_counts(particle_count):
    with pytest.raises(ValueError, match="positive integer"):
        validate_lagrangian_bijection(np.asarray([0, 1]), np.asarray([True, True]), particle_count)


def test_bijection_validation_rejects_shape_dtype_range_missing_and_duplicates():
    with pytest.raises(ValueError, match="same shape"):
        validate_lagrangian_bijection(np.asarray([0, 1]), np.asarray([True]), 2)
    with pytest.raises(ValueError, match="integer dtype"):
        validate_lagrangian_bijection(np.asarray([0.0, 1.0]), np.asarray([True, True]), 2)
    with pytest.raises(ValueError, match="out of range"):
        validate_lagrangian_bijection(np.asarray([0, 3]), np.asarray([True, True]), 2)
    with pytest.raises(ValueError, match="not a bijection"):
        validate_lagrangian_bijection(np.asarray([0, 0]), np.asarray([True, True]), 2)


def test_quijote_analysis_validation_and_odd_grid_spectrum_contracts():
    positions = np.asarray([[0, 0, 0], [1, 1, 1], [0.5, 1.5, 0.25]], dtype=np.float32)
    for grid in (True, 1, 3.5, "bad"):
        with pytest.raises(ValueError, match="analysis_grid"):
            fixed_grid_spectra(positions, positions, 2, analysis_grid=grid)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        fixed_grid_spectra(positions[:, :2], positions[:, :2], 2, analysis_grid=3)
    with pytest.raises(ValueError, match="at least one"):
        fixed_grid_spectra(np.empty((0, 3)), np.empty((0, 3)), 2, analysis_grid=3)
    with pytest.raises(ValueError, match="same shape"):
        fixed_grid_spectra(positions, positions[:2], 2, analysis_grid=3)
    with pytest.raises(ValueError, match="length-3"):
        fixed_grid_spectra(positions, positions, [2, 2], analysis_grid=3)

    integer_positions = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    spectra = fixed_grid_spectra(integer_positions, integer_positions, [2, 2, 2], analysis_grid=3, deconvolve_cic=False)
    valid = np.asarray(spectra.nmodes) > 0
    np.testing.assert_allclose(np.asarray(spectra.r)[valid], 1, rtol=0, atol=2e-6)

    with pytest.raises(ValueError, match="same shape"):
        periodic_position_metrics(positions, positions[:2], 2, 1)
    with pytest.raises(ValueError, match="must be a scalar"):
        periodic_position_metrics(positions, positions, 2, [1, 1, 1])


def test_gather_without_optional_fields_preserves_canonical_counts_in_compiled_mode():
    conf = Configuration(1.0, (2, 2, 2), mesh_shape=1, float_dtype=jnp.float32)
    ptcl = Particles.gen_grid(conf)
    gathered = gather_authoritative_particles(ptcl, validate=False)
    assert gathered.velocity is None
    assert gathered.acceleration is None
    np.testing.assert_array_equal(np.asarray(gathered.counts), 1)
