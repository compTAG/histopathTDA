"""
The roi_generator module:

This module processes a wholeslide image into rois.
"""


from typing import List
import numpy as np
from cv2 import fillPoly  # type: ignore
from histopathTDA.image import Im


class ROIGenerator():

    """a class to generate rois"""

    def __init__(self, image: Im, polygon, roi_size):
        self.image = image.to_np()
        self.polygon = polygon
        self.mask = fillPoly(np.zeros(self.image.shape), self.polygon, 255)
        self.roi_size = roi_size
        self.cur = 0
        wsi_dims = self.image.shape
        n_cell_x = wsi_dims[1] // self.roi_size
        n_cell_y = wsi_dims[0] // self.roi_size
        self.top_lefts = []  # type: ignore
        for i in range(n_cell_x):
            for j in range(n_cell_y):
                top_left = [(i + 1) * self.roi_size, (j + 1) * self.roi_size]
                self.top_lefts = self.top_lefts + [top_left]

        self.is_contained = []  # type: ignore
        for i in self.top_lefts:
            roi = self._extract_roi(self.mask, i)
            self.is_contained = self.is_contained + [np.all(roi == 255)]

        self.top_lefts = [self.top_lefts[i] for i in range(len(self.top_lefts)) if self.is_contained[i] == True]
        self.length = len(self.top_lefts)

    @classmethod
    def from_filepath(cls, filepath, polygon, roi_size):
        """Construct ROI generator from filepath """
        img_grey = Im.from_filepath(filepath).as_grayscale()
        return cls(img_grey, polygon, roi_size)

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._extract_roi(self.image, self.top_lefts[idx])

    def next(self):
        """get the next element of the generator"""
        self.cur = self.cur + 1
        return self._extract_roi(self.image, self.top_lefts[self.cur])

    def _extract_roi(self, image, top_left):
        """get the roi given a top-left index"""
        return image[top_left[1]:(top_left[1] + self.roi_size), top_left[0]:(top_left[0] + self.roi_size)]
