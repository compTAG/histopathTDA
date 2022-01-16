"""Utilities for the histopathTDA package.

Includes:
=========

  - A function to load an example image

"""

import pkg_resources  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA.image import Im


def load_example_image():
    """Return an example Gleason 3 grade histopathology image."""
    stream = pkg_resources.resource_stream(__name__, '_assets/example_images/g3.png')
    img = Image.open(stream)
    ex_im = Im(img)
    return ex_im


def row_major_idx(row, col, n_row):
    """Get vector index val for i,j the element"""
    return row * n_row + col


def row_major_idx_inverse(row_idx, n_row):
    """Get row and column indices given vector index"""
    return [row_idx // n_row, row_idx % n_row]
