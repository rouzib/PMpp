import pytest

pytestmark = pytest.mark.gpu2


@pytest.mark.skip(reason="requires two visible CUDA devices")
def test_fused_halo_matches_two_gpu_force():
    pass
