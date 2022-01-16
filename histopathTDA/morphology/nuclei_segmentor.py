'''
This module contains for segementing nuclei.  One is based on image
morphology techniques while another is neural network based.
'''

import os
import numpy as np
import cv2 as cv  # type: ignore
from histocartography.preprocessing import NucleiExtractor  # type: ignore
from histopathTDA.image import Im


def segment_nuclei_v1(image: Im, kernel_size=6):
    """function to segment nuclei given an Im class image"""

    np_image = image.to_np()
    imr = cv.cvtColor(np.stack((np_image[:, :, 0],) * 3, -1), cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(np.stack((np_image[:, :, 1],) * 3, -1), cv.COLOR_BGR2GRAY)
    imb = cv.cvtColor(np.stack((np_image[:, :, 2],) * 3, -1), cv.COLOR_BGR2GRAY)
    retr, threshr = cv.threshold(imr, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    retg, threshg = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    retb, threshb = cv.threshold(imb, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = (threshr == 0) | (threshb == 0) | (threshg == 0)
    img = (~thresh).astype(np.uint8)*255
    im = cv.cvtColor(np.stack((img,) * 3, -1), cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv.erode(im, kernel, iterations=1)
    return erosion


def compute_connected_components(mask, connectivity=4):

    """function takes in a binary mask of an Im class image and returns the centroids of
     the connected components of that image"""

    _, _, centroids = cv.connectedComponentsWithStats(mask, connectivity)
    return centroids

def segment_nuclei_v2(image: Im, use_gpu=True):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    np_image = image.to_np()
    nuclei_extractor = NucleiExtractor()
    mask, centroids = nuclei_extractor.process(np_image)
    mask = (~(mask == 0)).astype(np.uint8)
    return mask, centroids
