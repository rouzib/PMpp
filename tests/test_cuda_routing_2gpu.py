import pytest

pytestmark = pytest.mark.gpu2


@pytest.mark.skip(reason="requires two visible CUDA devices; covered by the benchmark worker on GPU hosts")
def test_two_gpu_cuda_bidir_route():
    pass
