import numpy as np

from src.camels_io import CamelsMetadata, CamelsParticlePair, coarsen_camels_pair


def test_coarsen_camels_pair_preserves_periodic_shift():
    box_size = 4.0
    grid_size = 4
    coarse_grid = grid_size // 2
    shift = np.array([2.2, 0.3, -0.4], dtype=np.float32)

    grid = np.indices((grid_size, grid_size, grid_size), dtype=np.float32)
    anchors = np.moveaxis(grid, 0, -1).reshape((-1, 3))
    ic_pos = anchors.copy()
    ic_vel = anchors * 0.1
    final_pos = np.mod(anchors + shift, box_size)
    final_vel = ic_vel + 1.0

    pair = CamelsParticlePair(
        ic_pos=ic_pos,
        ic_vel=ic_vel,
        final_pos=final_pos,
        final_vel=final_vel,
        ids=np.arange(grid_size ** 3, dtype=np.int64),
        metadata=CamelsMetadata(
            box_size=box_size,
            omega_m=0.3,
            omega_l=0.7,
            omega_b=0.0,
            h=0.67,
            sigma8=0.8,
            n_s=1.0,
            a_start=1 / 128,
            redshift=0.0,
            grid_size=grid_size,
        ),
    )

    coarse = coarsen_camels_pair(pair, factor=2)
    coarse_grid_idx = np.indices((coarse_grid, coarse_grid, coarse_grid), dtype=np.float32)
    coarse_anchor = np.moveaxis(coarse_grid_idx, 0, -1).reshape((-1, 3)) * 2.0
    expected_final_pos = np.mod(coarse_anchor + shift, box_size)
    expected_final_vel = np.full_like(expected_final_pos, 1.0) + (coarse_anchor + 0.5) * 0.1

    np.testing.assert_allclose(coarse.ic_pos, coarse_anchor, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(coarse.final_pos, expected_final_pos, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(coarse.final_vel, expected_final_vel, atol=1e-6, rtol=1e-6)
