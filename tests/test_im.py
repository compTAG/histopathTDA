
"""Provides some tests of the im class.

Includes test of plot_masked_image
"""

from histopathTDA.utils import load_example_image
from histopathTDA.image import Im


def test_constructor():
    """Test template attribute defaults."""
    test_image = load_example_image()
    assert isinstance(test_image, Im)
