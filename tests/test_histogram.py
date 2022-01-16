"""Provides some tests of the histogram functions."""

from histopathTDA.utils import load_example_image
from histopathTDA.homology import color_hist, avg_hist


def test_histograms():
    """Test histogram functions."""
    test_image = load_example_image()
    hist = color_hist(test_image)
    imgs = [test_image]
    average_hist = avg_hist(imgs)
    assert len(hist[0]) == 256
    assert len(hist[1]) == 257
    assert len(average_hist[0]) == 256
    assert len(average_hist[1]) == 257
