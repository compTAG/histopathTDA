"""
The image module:

This module contains a thin wrapper around the Pillow image class.
"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image  # type: ignore


class Im:
    """
    A class to represent an image.
    """

    def __init__(self, image):
        """
        Constructs the image class.

        Parameters
        ----------
            image: a PIL image
        """
        self.image = image

    @classmethod
    def from_pil(cls, pilimg):
        """Create Im from pillow image"""
        return cls(pilimg)

    @classmethod
    def from_np(cls, nparray):
        """Create Im from np array"""
        img = Image.fromarray(nparray)
        return cls(img)

    @classmethod
    def from_filepath(cls, filepath):
        """Create Im from opening image at filepath"""
        img = Image.open(filepath)
        return cls(img)

    def to_pil(self):
        """Get underlying PIL image"""
        return self.image

    def to_np(self):
        """Get nd-array of underlying image"""
        return np.array(self.image)

    def plot(self):
        """
        A simple plotting function.
        """
        plt.imshow(self.image)

    def get_dims(self):
        """Get dimensions of image"""
        return self.image.size

    def is_grayscale(self):
        """Grayscale predicate"""
        return self.image.mode == "L"

    def save(self, filename):
        """
        Converts image to grayscale.
        """
        self.image.save(filename)

    def as_grayscale(self):
        """
        Converts image to grayscale.
        """
        return Im(self.image.convert("L"))

    def plot_masked_image(self, mask, scaling_factor=0.6, color="yellow", plot=True):
        """
        Creates masked image and plots the image optionally

        Parameters
        ----------
            mask: an Im class image

            scaling factor: numeric between 0 and 1 that controls opacity of mask

            color: color of mask

            plot: whether or not to plot the masked image
        """
        scalar = int(scaling_factor * 255)
        binmask = mask.to_np()
        masked_im = Image.fromarray(np.array((~binmask).astype(np.uint8) * scalar))
        im2 = Image.new("RGB", self.image.size, color)
        masked_img = Image.composite(im2, self.image, masked_im)
        if plot:
            plt.imshow(masked_img)
            plt.show()
            return masked_img
        return masked_img
