'''
This module contains core operators and utilities for image morphology.
'''

from histopathTDA.image import Im
import numpy as np


def binary_thresh(image: Im, thresh0: float, thresh1: float = None):
    """Sets all pixels below thresh0 to 0 and above to
 256. Alternatively, if second threshold (thresh1) is given, sets all
 pixels between to 256 and outside to 0.  """
    if thresh0 > thresh1:
        tmp_thresh0 = thresh0
        tmp_thresh1 = thresh1
        thresh0 = tmp_thresh1
        thresh1 = tmp_thresh0
    img = image.as_grayscale()
    if thresh1 is None:
        np_image = img.to_np()
        np_image[np_image < thresh0] = 0
        np_image[np_image >= thresh0] = 255
        thresh_im = Im.from_np(np_image)
        thresh_im = thresh_im.as_binary()
        return thresh_im
    else:
        np_image = img.to_np()
        np_image[np_image < thresh0] = 0
        np_image[np_image > thresh1] = 0
        np_image[np.logical_and(np_image >= thresh0, np_image <= thresh1)] = 255
        thresh_im = Im.from_np(np_image)
        thresh_im = thresh_im.as_binary()
        return thresh_im


# TODO: adapt this fun to have two thresholds
def binary_thresh_inv(image: Im, thresh: float):
    """Sets all pixels above thresh to 0 and below to 256"""
    np_image = image.to_np()
    np_image[np_image > thresh] = 0
    np_image[np_image <= thresh] = 256
    thresh_im = Im.from_np(np_image)
    thresh_im = thresh_im.as_binary()
    return thresh_im


def to_zero_thresh(image: Im, thresh: float):
    """Sets all pixels below thresh to 0"""
    np_image = image.to_np()
    np_image[np_image < thresh] = 0
    thresh_im = Im.from_np(np_image)
    thresh_im = thresh_im.as_binary()
    return thresh_im


def to_zero_thresh_inv(image: Im, thresh: float):
    """Sets all pixels above thresh to 0"""
    np_image = image.to_np()
    np_image[np_image >= thresh] = 0
    thresh_im = Im.from_np(np_image)
    thresh_im = thresh_im.as_binary()
    return thresh_im
