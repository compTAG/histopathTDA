"""
The landscape module:

This module contains a class for generating and working with persistence
diagrams. Currently, this is a thin wrapper around the implementation given by
giotto-tda.
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from gtda.diagrams import PersistenceLandscape  # type: ignore
from histopathTDA.homology import Diagram


class Landscape:
    """
    A class to represent a persistence landscape.

    Attributes
    ----------
        landscape: nd-array
            The landscape function values across a grid
    """

    def __init__(self, diagram: Diagram, n_layers=1, n_bins=100, n_jobs=None):
        """
        Constructs all the necessary attributes for the persistence landscape object.

        Parameters
        ----------
            diagram: A Diagram class object
        """
        pers_land = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins, n_jobs=n_jobs)
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.landscape = pers_land.fit_transform(np.expand_dims(diagram.diagram, axis=0))

    def save(self, filepath):
        """
        Saves the file as a csv format.

        Parameters
        ----------
            filepath: the filepath to write the csv file to.
        """
        np.savetxt(filepath, self.landscape, delimiter=",")

    def plot(self, axis=None, dimension=0, main="Landscape", color="m", **kwargs):
        """
        Plots the persistence landscape as a matplotlib plot. Returns
        the plot so that one can incorporate into subplots and edit
        the plot.

        Parameters
        ----------

        """
        axis = axis or plt.gca()
        axis.set_title(main)
        if dimension == 0:
            lscape = axis.plot(list(range(self.n_bins)),
                               self.landscape[0, 0, :],
                               color=color, **kwargs)
            for i in range(self.n_layers):
                lscape = axis.plot(list(range(self.n_bins)),
                                   self.landscape[0, i, :],
                                   color=color, **kwargs)
            return lscape
        lscape = axis.plot(list(range(self.n_bins)),
                           self.landscape[0, 1, :],
                           color=color, **kwargs)
        for i in range(self.n_layers):
            lscape = axis.plot(list(range(self.n_bins)),
                               self.landscape[0, self.n_layers + i, :],
                               color=color, **kwargs)
        return lscape
