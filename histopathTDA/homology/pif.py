"""
The PIF module:

This module contains a class for generating and working with persistence
intensity functions. Currently, this is just a thin wrapper around the giotto-tda
implementation
"""


import pickle
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import PersistenceImages.persistence_images as pimg  # type: ignore
from sklearn.neighbors import KernelDensity  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA.image import Im
from histopathTDA.homology import Diagram


class PIF:
    """
    A class to represent a persistence intensity function.

    Attributes
    ----------
        pif: nd-array
            The PIF evaluated on a grid
    """

    def __init__(self, diagram: Diagram,
                 dims,
                 bounds,
                 pif_dims,
                 filtration,
                 bandwidths,
                 pif):
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
        self.dims = diagram.dims
        self.num_dims = len(diagram)
        self.bounds = diagram.bounds
        self.pif_dims = []
        for i in range(self.num_dims):
            # Handle 1 and  2d pif cases, add for each dim
            if isinstance(self.bounds[i][0], int):
                self.pif_dims += [1]
            else:
                self.pif_dims += [2]
        self.filtration = diagram.filtration
        self.bandwidths = bandwidths
        if self.bandwidths is None:
            self.banwidths = [.2] * len(self.diagram) # Set default to 0.2... (TODO: set more sensible defaults)
        self.pif = []
        for i in range(len(self.diagram)):
            if self.pif_dims[i] == 1:  # 1d case
                tmpdata = self.diagram[i][:, 1].reshape(len(self.diagram[i][:, 1]), 1)
                self.pif += [KernelDensity(kernel="gaussian", bandwidth=self.bandwidths[i]).fit(tmpdata)]
            else:  # 2d case
                self.pif += [KernelDensity(kernel="gaussian", bandwidth=self.bandwidths[i]).fit(self.diagram[i])]


    @classmethod
    def from_diagram(cls, diagram: Diagram, bandwidths):
        dgm = diagram
        dims = dgm.dims
        num_dims = len(dgm)  # TODO: Handle in __init__?
        bounds = dgm.bounds
        pif_dims = []
        for i in range(num_dims):
            # Handle 1 and  2d pif cases, add for each dim
            if isinstance(bounds[i][0], int):
                pif_dims += [1]
            else:
                pif_dims += [2]
        filtration = dgm.filtration
        bandwidths = bandwidths
        if bandwidths is None:
            bandwidths = [.2] * len(dgm) # Set default to 0.2... (TODO: set more sensible defaults)
        pif = []
        for i in range(len(dgm)):
            if pif_dims[i] == 1:  # 1d case
                tmpdata = dgm[i][:, 1].reshape(len(dgm[i][:, 1]), 1)
                pif += [KernelDensity(kernel="gaussian", bandwidth=bandwidths[i]).fit(tmpdata)]
            else:  # 2d case
                pif += [KernelDensity(kernel="gaussian", bandwidth=bandwidths[i]).fit(dgm[i])]
        return cls(dgm, dims, bounds, pif_dims, filtration, bandwidths, pif)

    @classmethod
    def from_filepath(cls, fp: str):
        with open(fp, 'rb') as f:
            tpif = pickle.load(f)
        diagram = tpif.diagram
        dims = tpif.dims
        bounds = tpif.bounds
        pif_dims = tpif.pif_dims
        filtration = tpif.filtration
        bandwidths = tpif.bandwidths
        pif = tpif.pif
        return cls(diagram, dims, bounds, pif_dims, filtration, bandwidths, pif)

    def __len__(self):
        return self.num_dims

    def __getitem__(self, index):
        return self.pif[index]


    def plot(self, dimension=0, bounds = None, nevals = 50, plot=True):
        """
        Plot PIF function

        Parameters
        ----------
            dimension: which pif dimension to plot
            plot: whether or not to plot the pif
        """
        if bounds is None:
            bds = self.bounds[dimension]
        else:
            bds = bounds
        if self.pif_dims[dimension] == 1:
            mesh = np.linspace(bds[0], bds[1], nevals).reshape(nevals, 1)
            evals = np.exp(self.pif[dimension].score_samples(mesh))

            if plot:
                plt.plot(mesh, evals)
            else:
                fig, ax = plt.subplots(1, 1)
                ax.plot(mesh, evals)
                return fig, ax
                return evals
        else:
            tmpx = np.linspace(bds[0][0], bds[0][1], nevals).reshape(nevals, 1)
            tmpy = np.linspace(bds[1][0], bds[1][1], nevals).reshape(nevals, 1)
            mesh = np.array(np.meshgrid(tmpx, tmpy, indexing='ij')).T.reshape(-1, 2)
            evals = np.exp(self.pif[dimension].score_samples(mesh)).reshape((nevals, nevals))
            evals[np.triu_indices(nevals)] = 0
            if plot:
                plt.imshow(np.flip(evals, axis=0))
            else:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(np.flip(evals, axis=0))
                return fig, ax


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
