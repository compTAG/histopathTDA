"""Provides some tests of the diagram class.

Includes test of constructor.
"""


from os import listdir, remove
from numpy import ndarray
from histopathTDA.homology import Diagram
from histopathTDA.utils import load_example_image


def test_diagram_attributes():
    """Test diagram attribute defaults."""
    test_image = load_example_image()
    test_diagram = Diagram.from_im(test_image)
    assert isinstance(test_diagram.diagram, ndarray)


def test_diagram_save():
    """Test diagram attribute defaults."""
    test_image = load_example_image()
    test_diagram = Diagram.from_im(test_image)
    test_diagram.save("test_diagram.csv")
    dirfiles = set(listdir())
    assert "test_diagram.csv" in dirfiles
    remove("test_diagram.csv")
