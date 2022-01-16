"""Unit tests for distance functions."""

from histopathTDA.homology import wassertein  # type: ignore


def test_wasserstein():
    """Test wasserstein distance function."""
    dgm1 = [[0.1, 0.8], [0.2, 0.9]]
    dgm2 = [[0.2, 0.5], [0.3, 0.7]]
    dist = wassertein(dgm1, dgm2)
    assert dist == 0.5
