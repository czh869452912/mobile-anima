import numpy as np

from anima_host.numerical_verify import _compare_arrays, VerificationResult


def test_compare_identical_arrays_passes():
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    result = _compare_arrays(a, b)

    assert isinstance(result, VerificationResult)
    assert result.max_abs_diff == 0.0
    assert result.mean_abs_diff == 0.0
    assert result.mse == 0.0
    assert result.cosine_similarity > 0.9999
    assert result.passed is True


def test_compare_different_arrays_fails():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    result = _compare_arrays(a, b, rtol=1e-5, atol=1e-5)

    assert result.max_abs_diff == 1.0
    assert result.passed is False


def test_compare_arrays_computes_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = _compare_arrays(a, b)

    assert result.cosine_similarity == 0.0
    assert result.passed is False
