"""Provides some tests of the utils module.

Includes test of load_example_image
"""

from histopathTDA.utils import load_example_image, Im


def test_load_example_image():
    """Test template attribute defaults."""
    test_image = load_example_image()
    assert isinstance(test_image, Im)
