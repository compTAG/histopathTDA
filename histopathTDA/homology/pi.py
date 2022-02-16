"""
The PI module:

This module contains a class for generating and working with persistence
images.
"""


import pickle
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import PersistenceImages.persistence_images as pimg  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA.image import Im
from histopathTDA.homology import PIF


class PI:
    """
    A class to represent a persistence intensity function.

    Attributes
    ----------
        pif: nd-array
            The PIF evaluated on a grid
    """

    def __init__(self,
                 pi: list,
                 pi_dims: list,
                 pif: PIF,
                 resolutions: list,
                 bounds: list,
                 meshes: list):
        """
        Constructs all the necessary attributes for the PIF object.

        Parameters
        ----------
        pif: PIF
            The persistence intensity function for generating persistence images

        resolution: tuple
            The resolution to evaluate the persistence images
            on. Elements of tuple are numeric and in order homology dimension
        """
        self.pi = pi
        self.pif = pif
        self.resolutions = resolutions
        if bounds is None:
            self.bounds = pif.bounds
        else:
            self.bounds = bounds

        self.meshes = meshes
        self.pi_dims = pif.pif_dims

    @classmethod
    def from_pif(cls, pif: PIF, resolutions, bounds):
        _pif = pif
        _resolutions = resolutions
        if bounds is None:
            _bounds = _pif.bounds
        else:
            _bounds = bounds

        _meshes = []
        _pi_dims = pif.pif_dims
        for i in range(len(pif)):
            # Handle 1 and  2d pif cases, add for each dim
            if _pi_dims[i] == 1:
                tmpx = np.linspace(_bounds[i][0], _bounds[i][1], _resolutions[i]).reshape(_resolutions[i], 1)
                _meshes += [tmpx]
            else:
                tmpx = np.linspace(_bounds[i][0][0], _bounds[i][0][1], _resolutions[i]).reshape(_resolutions[i], 1)
                tmpy = np.linspace(_bounds[i][1][0], _bounds[i][1][1], _resolutions[i]).reshape(_resolutions[i], 1)
                _meshes += [np.array(np.meshgrid(tmpx, tmpy, indexing='ij')).T.reshape(-1, 2)]

        # Evaluate PIF over each dim to compute PI
        _pi = []
        for i in range(len(pif)):
            if _pi_dims[i] == 1:
                tmppi = np.exp(pif[i].score_samples(_meshes[i]))
                _pi += [tmppi]
            else:
                tmppi = np.exp(pif[i].score_samples(_meshes[i])).reshape(_resolutions[i], _resolutions[i])
                # tmppi = np.flip(tmppi, axis=0)
                tmppi[np.triu_indices(tmppi.shape[0])] = 0  # Zero out lower diag (TODO: add transformed check)
                _pi += [tmppi]
        return cls(_pi, _pi_dims, _pif, _resolutions, _bounds, _meshes)

    @classmethod
    def from_filepath(cls, filepath: str):
        with open(filepath, 'rb') as f:
            pi = pickle.load(f)
        _pi = pi.pi
        _pi_dims = pi.pi_dims
        _pif = pi.pif
        _resolutions = pi.resolutions
        _bounds = pi.bounds
        _meshes = pi.meshes
        return cls(_pi, _pi_dims, _pif, _resolutions, _bounds, _meshes)

    def __getitem__(self, index):
        return self.pi[index]


#     def save_as_img(self, filepath, scaling_factor=1, dimension=0):
#         """
#         Saves the file as a Pillow image
#
#         Parameters
#         ----------
#             filepath: the filepath to write the object to
#             dimension: which pif dimension to save
#         """
#         if dimension == 0:
#             pif_img = Im.from_np(self.pif0*scaling_factor).as_grayscale()
#         else:
#             pif_img = Im.from_np(self.pif1*scaling_factor).as_grayscale()
#         pif_img.save(filepath)
#
#     def save_as_csv(self, filepath, scaling_factor=1, dimension=0):
#         """
#         Saves the file as a Pillow image
#
#         Parameters
#         ----------
#             filepath: the filepath to write the object to
#             dimension: which pif dimension to save
#         """
#         if dimension == 0:
#             pif = self.pif0 * scaling_factor
#         else:
#             pif = self.pif1 * scaling_factor
#         np.savetxt(filepath, pif, delimiter=",")
#
#     def save_features_as_csv(self, filepath_prefix, scaling_factor=1, dimension=0):
#         """
#         Saves each pif as a csv with
#
#         Parameters
#         ----------
#             filepath_prefix: the filepath prefix (appended with pif0.csv or pif1.csv)
#             dimension: which pif dimension to save
#         """
#         if dimension == 0:
#             pif = self.pif0 * scaling_factor
#         else:
#             pif = self.pif1 * scaling_factor
#         np.savetxt(filepath, pif, delimiter=",")

    def plot_1d_pi(self, dim, plot=True):
        if plot:
            plt.plot(self.meshes[dim], self.pi[dim])
        else:
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.meshes[dim], self.pi[dim])
            return fig, ax

    def plot_2d_pi(self, dim, plot=True):
        if plot:
            plt.imshow(np.flip(self.pi[dim], axis=0))
        else:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.flip(self.pi[dim], axis=0))
            return fig, ax

    def plot(self, dimensions=(0), plot=True):
        """
        Plot PIs

        Parameters
        ----------
            dimensions: tuple or int
                which pif dimensions to plot
            plot: whether or not to plot the pif (TODO: add better documentation for this)
        """
        if isinstance(dimensions, int):
            nplots = 1
        else:
            nplots = len(dimensions)
        if nplots > 1:
            fig, ax = plt.subplots(1, nplots)
            for i in range(nplots):
                plt.sca(ax[i])
                if (self.pi_dims[i] == 1):
                    self.plot_1d_pi(i, plot=True)
                else:
                    self.plot_2d_pi(i, plot=True)
        else:
            fig, ax = plt.subplots(1, 1)
            if (self.pi_dims[dimensions] == 1):
                self.plot_1d_pi(dimensions, plot=True)
            else:
                self.plot_2d_pi(dimensions, plot=True)

    def save_as_csv(self, fp, dim, vectorized=True):
        """
        Save PI

        Parameters
        ----------
            fp: string
                filepath to file
            dim: int
                index specifying which dimensions to save
        TODO: currently handles only one dim, need to decide on how to save multiple formats... maybe pickle?
        TODO: currently handles full pif in 2d case - add extraction of lower diagonal (vectorized)
        """
        if vectorized and (self.pi_dims[dim] > 1):
            np.savetxt(fp, self.pi[dim][np.tril_indices(self.resolutions[dim])], delimiter=",")
        else:
            np.savetxt(fp, self.pi[dim], delimiter=",")

    def save_pkl(self, filepath):
        """
        Saves the file as a pkl format.

        Remember to save with .pkl extension for load methods

        Parameters
        ----------
            filepath: the filepath to write the pickle file to.
        """
        with open(filepath, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


    # def save_as_img(self, filepath, scaling_factor=1, dimension=0):
    #     """
    #     Saves the file as a Pillow image

    #     Parameters
    #     ----------
    #         filepath: the filepath to write the object to
    #         dimension: which pif dimension to save
    #     """
    #     if dimension == 0:
    #         pif_img = Im.from_np(self.pif0*scaling_factor).as_grayscale()
    #     else:
    #         pif_img = Im.from_np(self.pif1*scaling_factor).as_grayscale()
    #     pif_img.save(filepath)

    # def save_as_csv(self, filepath, scaling_factor=1, dimension=0):
    #     """
    #     Saves the file as a Pillow image

    #     Parameters
    #     ----------
    #         filepath: the filepath to write the object to
    #         dimension: which pif dimension to save
    #     """
    #     if dimension == 0:
    #         pif = self.pif0 * scaling_factor
    #     else:
    #         pif = self.pif1 * scaling_factor
    #     np.savetxt(filepath, pif, delimiter=",")

    # def save_features_as_csv(self, filepath_prefix, scaling_factor=1, dimension=0):
    #     """
    #     Saves each pif as a csv with

    #     Parameters
    #     ----------
    #         filepath_prefix: the filepath prefix (appended with pif0.csv or pif1.csv)
    #         dimension: which pif dimension to save
    #     """
    #     if dimension == 0:
    #         pif = self.pif0 * scaling_factor
    #     else:
    #         pif = self.pif1 * scaling_factor
    #     np.savetxt(filepath, pif, delimiter=",")
