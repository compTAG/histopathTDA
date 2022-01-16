"""Provides some tests of the landscape class.

Includes test of constructor.
"""

from numpy import ndarray
from histopathTDA.homology import Diagram, Landscape
from histopathTDA.utils import load_example_image


def test_landscape_attributes():
    """Test landscape attribute defaults."""
    test_image = load_example_image()
    test_diagram = Diagram.from_im(test_image)
    test_landscape = Landscape(test_diagram)
    assert isinstance(test_landscape.landscape, ndarray)
    assert isinstance(test_landscape, Landscape)


# TODO: write test for save method
