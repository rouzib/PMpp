import pytest

pytestmark = pytest.mark.gpu2


@pytest.mark.skip(reason="requires two visible CUDA devices; logical shard semantics are covered on CPU")
def test_mesh_halo_cic_boundary():
    pass
