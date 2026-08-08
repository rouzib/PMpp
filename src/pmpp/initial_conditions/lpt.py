from functools import partial

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from ..cosmology.models import E2
from ..numerics.fft import fftinv, fftfwd, fftfreq
from ..nbody.gravity import laplace_transposed, neg_grad
from ..cosmology.growth import growth
from ..nbody.particles import Particles
from ..nbody.integrator import _halo_move_vjp
from ..core.utils import AXIS_NAME


def _strain(kvec, i, j, pot, conf):
    """LPT strain component sourced by scalar potential only.

     The Nyquist planes are not zeroed when i == j.

    .. _Notes on FFT-based differentiation:
        https://math.mit.edu/~stevenj/fft-deriv.pdf

    """
    k_i, k_j = kvec[i], kvec[j]

    nyquist = jnp.pi / conf.ptcl_spacing
    eps = nyquist * jnp.finfo(conf.float_dtype).eps

    if i != j:
        k_i = jnp.where(jnp.abs(jnp.abs(k_i) - nyquist) <= eps, 0, k_i)
        k_j = jnp.where(jnp.abs(jnp.abs(k_j) - nyquist) <= eps, 0, k_j)

    strain = -k_i * k_j * pot

    if conf.compute_mesh is None:
        strain = jnp.fft.irfftn(strain)
    else:
        strain = conf.mGPU_irfftn_transposed(strain)
    strain = strain.astype(conf.float_dtype)  # no jnp.complex32

    return strain


def _particle_grid_coordinates(pmid, disp, unused_index, conf):
    """Return per-axis particle-grid coordinates without raveling them."""
    grid_shape = jnp.asarray(conf.ptcl_grid_shape, dtype=jnp.int32)
    coordinates = jnp.rint((pmid.astype(conf.float_dtype) * conf.cell_size + disp) /
                           conf.ptcl_spacing, ).astype(jnp.int32)
    coordinates %= grid_shape
    return jnp.where(unused_index[:, None], jnp.zeros_like(coordinates), coordinates)


def _sample_lpt_field(grad, pmid, disp, unused_index, conf):
    """Sample an LPT field with shard-local indices when particles are owned-only."""
    valid = ~unused_index
    use_local_sampling = (conf.compute_mesh is not None and conf.num_devices > 1 and conf.multigpu_mode == "mesh_halo")
    if not use_local_sampling:
        coordinates = _particle_grid_coordinates(pmid, disp, unused_index, conf)
        sampled = grad[tuple(coordinates[:, axis] for axis in range(conf.dim))]
        return jnp.where(valid, sampled, 0)

    grad_spec = P(AXIS_NAME, *([None] * (conf.dim - 1)))

    @partial(
        shard_map, mesh=conf.compute_mesh, in_specs=(grad_spec, P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=P(AXIS_NAME), check_vma=False,
    )
    def sample_local(local_grad, local_pmid, local_disp, local_unused):
        coordinates = _particle_grid_coordinates(local_pmid, local_disp, local_unused, conf)
        local_x_size = conf.ptcl_grid_shape[0] // conf.num_devices
        local_start = jax.lax.axis_index(AXIS_NAME) * local_x_size
        local_x = (coordinates[:, 0] - local_start) % conf.ptcl_grid_shape[0]
        local_valid = (~local_unused) & (local_x < local_x_size)
        local_x = jnp.where(local_valid, local_x, 0)
        local_coordinates = (local_x, ) + tuple(coordinates[:, axis] for axis in range(1, conf.dim))
        sampled = local_grad[local_coordinates]
        return jnp.where(local_valid, sampled, 0)

    return sample_local(grad, pmid, disp, unused_index)


def _streaming_strain(term, kvec, pot, conf):
    """Evaluate one selected 2LPT strain while keeping the inverse FFT scalar."""
    nyquist = jnp.pi / conf.ptcl_spacing
    eps = nyquist * jnp.finfo(conf.float_dtype).eps

    def strain_spectrum(i, j, value):
        k_i, k_j = kvec[i], kvec[j]
        if i != j:
            k_i = jnp.where(jnp.abs(jnp.abs(k_i) - nyquist) <= eps, 0, k_i)
            k_j = jnp.where(jnp.abs(jnp.abs(k_j) - nyquist) <= eps, 0, k_j)
        return -k_i * k_j * value

    branches = tuple(partial(strain_spectrum, i, j) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))
    strain = jax.lax.switch(term, branches, pot)
    if conf.compute_mesh is None:
        strain = jnp.fft.irfftn(strain)
    else:
        strain = conf.mGPU_irfftn_transposed(strain)
    return strain.astype(conf.float_dtype)


def _L_streaming_2lpt(kvec, pot, conf):
    """Build the 2LPT quadratic source with a loop-carried scalar accumulator."""
    if conf.dim != 3:
        raise ValueError("streaming 2LPT requires three spatial dimensions")

    if conf.compute_mesh is None:
        source = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)
    else:
        source = jnp.zeros(
            conf.ptcl_grid_shape, dtype=conf.float_dtype,
            device=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None)),
        )

    left_terms = jnp.asarray((0, 0, 1), dtype=jnp.int32)
    right_terms = jnp.asarray((1, 2, 2), dtype=jnp.int32)
    cross_terms = jnp.asarray((3, 4, 5), dtype=jnp.int32)

    def accumulate(pair, value):
        strain_left = _streaming_strain(left_terms[pair], kvec, pot, conf)
        strain_right = _streaming_strain(right_terms[pair], kvec, pot, conf)
        value = value + strain_left * strain_right
        strain_cross = _streaming_strain(cross_terms[pair], kvec, pot, conf)
        return value - strain_cross * strain_cross

    return jax.lax.fori_loop(0, 3, accumulate, source)


def _low_memory_particle_grid(conf):
    """Generate the canonical owned-only grid from a single fused row iota."""
    num_devices = conf.num_devices or 1
    local_x_size = conf.ptcl_grid_shape[0] // num_devices
    plane_size = conf.ptcl_grid_shape[1] * conf.ptcl_grid_shape[2]
    local_count = local_x_size * plane_size
    capacity = conf.ptcl_num if conf.max_ptcl_per_slice is None else conf.max_ptcl_per_slice

    def build_local(axis):
        rows = jnp.arange(capacity, dtype=jnp.int32)
        valid = rows < local_count
        rows = jnp.where(valid, rows, 0)
        local_x, yz = jnp.divmod(rows, plane_size)
        y, z = jnp.divmod(yz, conf.ptcl_grid_shape[2])
        x = local_x + axis.astype(jnp.int32) * local_x_size
        pmid = jnp.stack((x, y, z), axis=1).astype(conf.pmid_dtype)
        disp = jnp.zeros((capacity, conf.dim), dtype=conf.float_dtype)
        unused_index = ~valid
        halo_mask = jnp.zeros_like(unused_index)
        return pmid, disp, unused_index, halo_mask

    if conf.compute_mesh is None:
        pmid, disp, unused_index, halo_mask = build_local(jnp.int32(0))
    else:

        @partial(
            shard_map, mesh=conf.compute_mesh, in_specs=P(),
            out_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME), P(AXIS_NAME)), check_vma=False,
        )
        def build_all(_):
            return build_local(jax.lax.axis_index(AXIS_NAME))

        pmid, disp, unused_index, halo_mask = build_all(jnp.int32(0))

    return Particles(
        conf, pmid, disp, vel=jnp.zeros_like(disp), acc=None, unused_index=unused_index, halo_mask=halo_mask,
    )


def _sample_canonical_lpt_field(grad, unused_index, conf):
    """Sample a canonical local field by row order, without coordinate arrays."""

    def sample_local(local_grad, local_unused):
        flat = local_grad.reshape(-1)
        padding = local_unused.shape[0] - flat.shape[0]
        sampled = jnp.pad(flat, (0, padding))
        return jnp.where(local_unused, 0, sampled)

    if conf.compute_mesh is None:
        return sample_local(grad, unused_index)

    @partial(
        shard_map, mesh=conf.compute_mesh, in_specs=(P(AXIS_NAME, None, None), P(AXIS_NAME)), out_specs=P(AXIS_NAME),
        check_vma=False,
    )
    def sample_all(local_grad, local_unused):
        return sample_local(local_grad, local_unused)

    return sample_all(grad, unused_index)


@partial(jax.jit, static_argnames=('conf', ), donate_argnums=(0, ))
def _low_memory_first_order_potential(modes, cosmo, conf):
    """Build the first-order LPT potential."""
    modes = modes / conf.ptcl_cell_vol
    if conf.compute_mesh is not None:
        modes = jax.lax.with_sharding_constraint(modes, NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)), )
    return laplace_transposed(conf.kvec_spacing, modes, conf, cosmo)


@partial(jax.jit, static_argnames=('conf', ), donate_argnums=(0, ))
def _low_memory_second_order_potential(pot_1, cosmo, conf):
    """Consume the first-order potential to build the streaming 2LPT potential."""
    src_2 = _L_streaming_2lpt(conf.kvec_spacing, pot_1, conf)
    if conf.compute_mesh is None:
        src_2 = jnp.fft.rfftn(src_2)
    else:
        src_2 = conf.mGPU_rfftn_transposed(src_2)
    return laplace_transposed(conf.kvec_spacing, src_2, conf, cosmo)


def _streaming_lpt_gradient(axis, pot, conf):
    """Transform one selected LPT gradient component to real space."""
    branches = tuple(partial(neg_grad, k, spacing=conf.ptcl_spacing) for k in conf.kvec_spacing)
    grad = jax.lax.switch(axis, branches, pot)
    if conf.compute_mesh is None:
        grad = jnp.fft.irfftn(grad)
    else:
        grad = conf.mGPU_irfftn_transposed(grad)
    return grad.astype(conf.float_dtype)


@partial(jax.jit, static_argnames=('conf', ), donate_argnums=(0, 1))
def _accumulate_low_memory_lpt_order(disp, vel, unused_index, pot, D, a2HDp, conf):
    """Accumulate one LPT order one scalar inverse FFT at a time."""

    def accumulate(axis, state):
        disp_value, vel_value = state
        grad = _streaming_lpt_gradient(axis, pot, conf)
        grad = _sample_canonical_lpt_field(grad, unused_index, conf)
        disp_value = disp_value.at[:, axis].add(D * grad)
        vel_value = vel_value.at[:, axis].add(a2HDp * grad)
        return disp_value, vel_value

    return jax.lax.fori_loop(0, conf.dim, accumulate, (disp, vel))


def _L(kvec, pot_m, pot_n, conf):
    """Quadratic LPT source built from products of strain tensors.

    For second-order LPT this evaluates the invariant
    ``sum_i<j phi_ii psi_jj - phi_ij psi_ji``. When ``pot_n`` is ``None`` the
    same potential is used for both factors, which is the common 2LPT case.
    """
    m_eq_n = pot_n is None
    if m_eq_n:
        pot_n = pot_m

    if conf.compute_mesh is None:
        L = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)
    else:
        L = jnp.zeros(
            conf.ptcl_grid_shape, dtype=conf.float_dtype,
            device=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None)),
        )

    if conf.lpt_cache_strains:
        # Cache diagonal strains to avoid redundant irfftn calls (conf.lpt_cache_strains=True).
        # Saves dim-1 irfftn calls per _L at the cost of keeping dim extra strain arrays
        # (each of shape ptcl_grid_shape) alive simultaneously. Set lpt_cache_strains=False
        # to recompute instead, trading compute for GPU memory.
        diag_m = [_strain(kvec, i, i, pot_m, conf) for i in range(conf.dim)]
        diag_n = diag_m if m_eq_n else [_strain(kvec, i, i, pot_n, conf) for i in range(conf.dim)]

        for i in range(conf.dim):
            for j in range(conf.dim - 1, i, -1):
                L += diag_m[i] * diag_n[j]

            if not m_eq_n:
                for j in range(i - 1, -1, -1):
                    L += diag_m[i] * diag_n[j]
    else:
        for i in range(conf.dim):
            strain_m = _strain(kvec, i, i, pot_m, conf)

            for j in range(conf.dim - 1, i, -1):
                strain_n = _strain(kvec, j, j, pot_n, conf)
                L += strain_m * strain_n

            if not m_eq_n:
                for j in range(i - 1, -1, -1):
                    strain_n = _strain(kvec, j, j, pot_n, conf)
                    L += strain_m * strain_n

    if not m_eq_n:
        L *= 0.5

    # Assuming strain sourced by scalar potential only, symmetric about ``i`` and ``j``,
    # for lpt_order <=3, i.e., m, n <= 2
    for i in range(conf.dim - 1):
        for j in range(i + 1, conf.dim):
            strain_m = _strain(kvec, i, j, pot_m, conf)

            strain_n = strain_m
            if not m_eq_n:
                strain_n = _strain(kvec, j, i, pot_n, conf)

            L -= strain_m * strain_n

    return L


@partial(jax.custom_vjp, nondiff_argnums=(5, ))
def _attach_lpt_halo_move_vjp(disp_before, vel_before, disp_after, vel_after, ptcl_before, conf):
    """Attach the halo-move pullback to LPT outputs.

    LPT builds displacement/velocity on the canonical particle grid and then
    calls the same halo-movement machinery as the N-body drift. This wrapper
    makes the backward pass use that halo-move VJP while returning only the
    post-move floating arrays in the primal result.
    """
    return disp_after, vel_after


def _attach_lpt_halo_move_vjp_fwd(disp_before, vel_before, disp_after, vel_after, ptcl_before, conf):
    """Forward rule for the LPT halo-move custom VJP."""
    return (disp_after, vel_after), (disp_before, vel_before, ptcl_before)


def _attach_lpt_halo_move_vjp_bwd(conf, res, cotangents):
    """Backward rule that routes LPT cotangents through halo movement."""
    disp_before, vel_before, ptcl_before = res
    disp_cot, vel_cot = cotangents
    scratch_acc = disp_before[:, :0]

    disp_before_cot, vel_before_cot, _ = _halo_move_vjp(
        ptcl_before, disp_before, vel_before, scratch_acc, disp_cot, vel_cot, scratch_acc, conf,
    )

    return (disp_before_cot, vel_before_cot, jnp.zeros_like(disp_cot), jnp.zeros_like(vel_cot), None, )


_attach_lpt_halo_move_vjp.defvjp(_attach_lpt_halo_move_vjp_fwd, _attach_lpt_halo_move_vjp_bwd, )


@partial(jax.jit, static_argnames=('conf', ))
@partial(jax.checkpoint, static_argnums=(2, ))
def lpt(modes, cosmo, conf):
    """Lagrangian perturbation theory at ``conf.lpt_order``.

    Parameters
    ----------
    modes : jax.Array
        Linear matter overdensity modes on the particle grid, usually in Fourier
        space as returned by :func:`pmpp.initial_conditions.linear_modes`.
    cosmo : Cosmology
        Cosmology with precomputed growth tables.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Particle state initialized at ``conf.a_start`` with displacements and
        canonical velocities from LPT.

    Raises
    ------
    ValueError
        If ``conf.dim`` or ``conf.lpt_order`` is not supported.
    NotImplementedError
        If ``conf.lpt_order`` is 3. Configuration accepts that value, but the
        third-order source is not implemented.

    Notes
    -----
    After building the LPT displacement and velocity fields on the canonical
    grid, this function routes them through the same multi-GPU ownership logic
    used by the N-body drift so the output particle layout already matches the
    active runtime mode.

    """
    if conf.dim not in (1, 2, 3):
        raise ValueError(f'dim={conf.dim} not supported')
    if conf.lpt_order not in (0, 1, 2, 3):
        raise ValueError(f'lpt_order={conf.lpt_order} not supported')

    modes /= conf.ptcl_cell_vol  # remove volume factor first for convenience

    kvec = conf.kvec_spacing
    if conf.compute_mesh is not None:
        modes = jax.lax.with_sharding_constraint(modes, NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)), )

    pot = []

    if conf.lpt_order > 0:
        src_1 = modes

        pot_1 = laplace_transposed(kvec, src_1, conf, cosmo)
        pot.append(pot_1)

    if conf.lpt_order > 1:
        src_2 = _L(kvec, pot_1, None, conf)

        if conf.compute_mesh is None:
            src_2 = jnp.fft.rfftn(src_2)
        else:
            src_2 = conf.mGPU_rfftn_transposed(src_2)

        pot_2 = laplace_transposed(kvec, src_2, conf, cosmo)
        pot.append(pot_2)

    if conf.lpt_order > 2:
        raise NotImplementedError('TODO')

    a = conf.a_start
    ptcl = Particles.gen_grid(conf, vel=True)
    disp = ptcl.disp
    vel = ptcl.vel

    for order in range(1, 1 + conf.lpt_order):
        D = growth(a, cosmo, conf, order=order)
        dD_dlna = growth(a, cosmo, conf, order=order, deriv=1)
        a2HDp = a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna
        D = D.astype(conf.float_dtype)
        a2HDp = a2HDp.astype(conf.float_dtype)

        for i, k in enumerate(kvec):
            grad = neg_grad(k, pot[order - 1], conf.ptcl_spacing)

            if conf.compute_mesh is None:
                grad = jnp.fft.irfftn(grad)
            else:
                grad = conf.mGPU_irfftn_transposed(grad)
            grad = grad.astype(conf.float_dtype)  # no jnp.complex32

            grad = _sample_lpt_field(grad, ptcl.pmid, ptcl.disp, ptcl.unused_index, conf)

            disp = disp.at[:, i].add(D * grad)
            vel = vel.at[:, i].add(a2HDp * grad)

    disp_before_halo = disp
    vel_before_halo = vel
    if conf.mGPU_halo_moving is None:
        return ptcl.replace(disp=disp, vel=vel, acc=None)

    scratch_acc = disp[:, :0]
    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid, ptcl.disp, disp, vel, scratch_acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    ptcl_after = ptcl.replace(
        pmid=pmid, disp=disp, vel=vel, acc=None, halo_mask=halo_mask, unused_index=unused_indexes,
    )
    disp, vel = _attach_lpt_halo_move_vjp(
        disp_before_halo, vel_before_halo, ptcl_after.disp, ptcl_after.vel, ptcl, conf,
    )

    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=None, halo_mask=halo_mask, unused_index=unused_indexes)


def _validate_low_memory_lpt(conf):
    """Validate the deliberately narrow forward-only LPT profile."""
    if conf.dim != 3:
        raise ValueError("low-memory LPT requires three spatial dimensions")
    if conf.lpt_order != 2:
        raise ValueError("low-memory LPT requires lpt_order=2")
    if tuple(conf.mesh_shape) != tuple(conf.ptcl_grid_shape):
        raise ValueError("low-memory LPT requires mesh_shape equal to ptcl_grid_shape")
    if jnp.dtype(conf.float_dtype) != jnp.dtype(jnp.float32):
        raise ValueError("low-memory LPT requires float32")
    if jnp.dtype(conf.pmid_dtype) != jnp.dtype(jnp.int16):
        raise ValueError("low-memory LPT requires int16 particle coordinates")
    if conf.compute_mesh is not None and conf.multigpu_mode != "mesh_halo":
        raise ValueError("low-memory LPT requires mesh_halo mode")

    num_devices = conf.num_devices or 1
    if conf.ptcl_grid_shape[0] % num_devices:
        raise ValueError("particle-grid x size must be divisible by the device count")
    local_count = conf.ptcl_num // num_devices
    if local_count > jnp.iinfo(jnp.int32).max:
        raise ValueError("low-memory LPT requires fewer than 2^31 particles per device")
    capacity = conf.ptcl_num if conf.max_ptcl_per_slice is None else conf.max_ptcl_per_slice
    if capacity <= 0 or capacity > jnp.iinfo(jnp.int32).max:
        raise ValueError("low-memory LPT requires 0 < max_ptcl_per_slice < 2^31")
    if capacity < local_count:
        raise ValueError("max_ptcl_per_slice is smaller than the canonical local particle count")


def lpt_low_memory_with_telemetry(modes, cosmo, conf):
    """Run forward-only 2LPT and return migration diagnostics.

    This entrypoint is intentionally staged rather than wrapped in one outer
    ``jit``. It is reserved for the low-memory forward runner and rejects JAX
    transformations so the ordinary :func:`lpt` differentiation contract stays
    unchanged. The spectral ``modes`` buffer is donated and must not be reused
    by the caller after this function starts.
    """
    _validate_low_memory_lpt(conf)
    leaves = jax.tree_util.tree_leaves((modes, cosmo))
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        raise NotImplementedError("lpt_low_memory is forward-only and cannot be traced or differentiated")

    pot_1 = _low_memory_first_order_potential(modes, cosmo, conf)
    ptcl = _low_memory_particle_grid(conf)
    disp, vel = ptcl.disp, ptcl.vel
    a = conf.a_start

    D = growth(a, cosmo, conf, order=1).astype(conf.float_dtype)
    dD_dlna = growth(a, cosmo, conf, order=1, deriv=1)
    a2HDp = (a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna).astype(conf.float_dtype)
    disp, vel = _accumulate_low_memory_lpt_order(disp, vel, ptcl.unused_index, pot_1, D, a2HDp, conf)
    pot_2 = _low_memory_second_order_potential(pot_1, cosmo, conf)
    del pot_1

    D = growth(a, cosmo, conf, order=2).astype(conf.float_dtype)
    dD_dlna = growth(a, cosmo, conf, order=2, deriv=1)
    a2HDp = (a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna).astype(conf.float_dtype)
    disp, vel = _accumulate_low_memory_lpt_order(disp, vel, ptcl.unused_index, pot_2, D, a2HDp, conf)
    del pot_2

    fused_mover = getattr(conf, "mGPU_halo_moving_low_memory", None)
    if fused_mover is not None:
        pmid, disp, vel, halo_mask, unused_index, has_failed, max_moved, invalid_count = fused_mover(
            ptcl.pmid, disp, vel, jnp.asarray(0, dtype=conf.float_dtype), ptcl.unused_index,
        )
        if bool(jax.device_get(jnp.any(has_failed))):
            raise RuntimeError("low-memory LPT fused particle routing exceeded a static capacity")
        return (
            ptcl.replace(pmid=pmid, disp=disp, vel=vel, halo_mask=halo_mask, unused_index=unused_index,
                         ), max_moved, invalid_count,
        )

    mover = conf.mGPU_halo_moving_no_acc
    if mover is None:
        if conf.compute_mesh is not None and conf.num_devices > 1:
            raise RuntimeError("low-memory LPT requires the mesh-halo no-acceleration mover")
        return ptcl.replace(disp=disp, vel=vel), jnp.int32(0), jnp.int32(0)

    pmid, disp, vel, halo_mask, unused_index, has_failed, max_moved = mover(
        ptcl.pmid, disp, disp, vel, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    if bool(jax.device_get(jnp.any(has_failed))):
        raise RuntimeError("low-memory LPT particle routing exceeded a static capacity")
    return (
        ptcl.replace(pmid=pmid, disp=disp, vel=vel, halo_mask=halo_mask, unused_index=unused_index,
                     ), max_moved, jnp.int32(0),
    )


def lpt_low_memory(modes, cosmo, conf):
    """Run the forward-only 2LPT path with bounded field liveness.

    The spectral ``modes`` input is donated and must not be reused.  Use
    :func:`lpt_low_memory_with_telemetry` when migration high-water evidence
    is required by a production runner.
    """
    particles, _, _ = lpt_low_memory_with_telemetry(modes, cosmo, conf)
    return particles
