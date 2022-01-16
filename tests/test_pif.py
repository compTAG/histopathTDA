"""Provides some tests of the PIF class.

Includes test of constructor.
"""

from histopathTDA.homology import PIF
from histopathTDA.homology import Diagram
from histopathTDA.utils import load_example_image


def test_diagram_attributes():
    """Test diagram attribute defaults."""
    test_image = load_example_image()
    test_diagram = Diagram.from_im(test_image)
    test_pif = PIF(test_diagram)
    assert isinstance(test_pif, PIF)
