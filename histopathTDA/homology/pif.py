"""
The PIF module:

This module contains a class for generating and working with persistence
intensity functions. Currently, this is just a thin wrapper around the giotto-tda
implementation
"""


import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import PersistenceImages.persistence_images as pimg  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA import Im


class PIF:
    """
    A class to represent a persistence intensity function.

    Attributes
    ----------
        pif: nd-array
            The PIF evaluated on a grid
    """

    def __init__(self, diagram, bandwidth=0.2, nbins=256):
        """
        Constructs all the necessary attributes for the PIF object.

        Parameters
        ----------
        diagram: Diagram
            The Diagram class object to create the PIF from

        bandwidth: numeric
            the bandwidth associated to the KDE

        nbins: int
            the number of evaluation points across the KDE in one dimension
        """
        self.diagram = diagram
        self.bandwidth = bandwidth
        self.nbins = nbins
        d0_idx = self.diagram.diagram[:, 2] == 0
        dgm0 = self.diagram.diagram[d0_idx]
        dgm1 = self.diagram.diagram[~d0_idx]
        pers_imager = pimg.PersistenceImager(birth_range=(0, 256),
                                             pers_range=(0, 256),
                                             pixel_size=256 * (1 / nbins))

        pers_imager.weight = pimg.weighting_fxns.persistence
        pers_imager.weight_params = {'n': 0}
        pers_imager.kernel_params = {'sigma': np.array([[bandwidth * 256, 0],
                                                        [0, bandwidth * 256]])}

        # Calculates Persistence Images
        pif0 = pers_imager.transform(dgm0, skew=False)
        pif1 = pers_imager.transform(dgm1, skew=False)

        # Extract lower diagonal from persistence images
        pif0[np.tril_indices(pif0.shape[0])] = 0
        pif1[np.tril_indices(pif1.shape[0])] = 0

        pif0 *= (1 / (nbins * nbins)) * np.sum(pif0)
        pif1 *= (1 / (nbins * nbins)) * np.sum(pif1)

        self.pif0 = np.transpose(np.flip(pif0, axis=1), axes=(1, 0))
        self.pif1 = np.transpose(np.flip(pif1, axis=1), axes=(1, 0))

    def save_as_img(self, filepath, scaling_factor=1, dimension=0):
        """
        Saves the file as a Pillow image

        Parameters
        ----------
            filepath: the filepath to write the object to
            dimension: which pif dimension to save
        """
        if dimension == 0:
            pif_img = Im.from_np(self.pif0*scaling_factor).as_grayscale()
        else:
            pif_img = Im.from_np(self.pif1*scaling_factor).as_grayscale()
        pif_img.save(filepath)


    def save_as_csv(self, filepath, scaling_factor=1, dimension=0):
        """
        Saves the file as a Pillow image

        Parameters
        ----------
            filepath: the filepath to write the object to
            dimension: which pif dimension to save
        """
        if dimension == 0:
            pif = self.pif0 * scaling_factor
        else:
            pif = self.pif1 * scaling_factor
        np.savetxt(filepath, pif, delimiter=",")


    def plot(self, dimension=0, plot=True):
        """
        Plot PIF function

        Parameters
        ----------
            dimension: which pif dimension to plot
            plot: whether or not to plot the pif
        """
        if dimension == 0:
            pif = self.pif0
        else:
            pif = self.pif1

        if plot:
            plt.imshow(pif)
        else:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(pif)
            return fig, ax
