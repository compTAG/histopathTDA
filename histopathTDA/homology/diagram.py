"""
The diagram module
===================

This module contains a class for generating and working with persistence
diagrams. Currently, this is specialized for working with black and white image
data, but will be expanded for working with rips filtrations. Long term goals
are to pass in a filtration and use this as a thin wrapper to Dionysus (or
Ghudhi).
"""


import os
import pickle
from gtda.homology import CubicalPersistence  # type: ignore
from gtda.homology import VietorisRipsPersistence #type: ignore
from gtda.images import HeightFiltration
from gtda.images import DilationFiltration
from gtda.images import DensityFiltration
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA.image import Im


class Diagram:
    """
    A class to represent a persistence diagram.

    Attributes
    ----------
        image: pillow image
            A black and white image to construct a cubical complex filtration on.
        diagram: nd-array
            The data (birth,death,dimension) associated to the persistence diagram
    """

    def __init__(self, diagram, filtration, bounds, dimensions):
        """
        Constructs all the necessary attributes for the persistence diagram object.

        Parameters
        ----------
            dgm_input: Either a Im image or a pointset
        """
        self.diagram = diagram
        self.filtration = filtration
        self.bounds = bounds
        self.dims = dimensions
        self.ndims = len(self.dims)
        tdgm = []
        for i in range(self.ndims):
            dim_idx = self.diagram[:, 2] == i
            tdgm += [self.diagram[dim_idx, 0:2]]
        self.diagram = tdgm

    @classmethod
    def from_pointset(cls, pointcloud: np.ndarray, bounds: list,
                      filtration: str = "rips", dims: tuple = (0, 1)):
        if filtration == "rips":
            VR = VietorisRipsPersistence(homology_dimensions=dims)
            diagram = VR.fit_transform(np.expand_dims(pointcloud, 0))[0]
            filtration = "rips"  # TODO: implement other filtrations for points
            return cls(diagram=diagram, filtration=filtration,
                       bounds=bounds, dimensions=dims)
        else:
            print("Error: only rips filtration implemented for pointsets")

    @classmethod
    def from_im(cls, im: Im, filtration: str = "cubical",
                bounds: list = [[0, 256], [0, 256]], dims: tuple = (0, 1)):

        # TODO: figure out where bounds should get used in filtrations?
        # TODO: add check for len(bounds) == len(dims)
        if filtration == "cubical":
            # Image must be single channel for cubical filtration
            # TODO: assess if this statement is true in general
            if not im.is_grayscale():
                img_grey = im.as_grayscale().to_np()
            else:
                img_grey = im.to_np()
            # TODO: assess if there are better ways to reshape img
            img_shape = img_grey.shape
            img_grey = img_grey.reshape((1, img_shape[0], img_shape[1]))
            cubical = CubicalPersistence(homology_dimensions=dims)
            diagram = cubical.fit_transform(img_grey)[0]
            return cls(diagram=diagram, filtration=filtration,
                       bounds=bounds, dimensions=dims)
        elif filtration == "height":
            if not im.is_binary():
                return "Error: Image is not binary"  # TODO: error handling here?
            else:
                # TODO: consider implementing direction into UI
                img_binary = im.to_np()
                img_shape = img_binary.shape
                img_binary = img_binary.reshape((1, img_shape[0], img_shape[1]))
                h_filtration = HeightFiltration().fit_transform(img_binary)
                persistence = CubicalPersistence(homology_dimensions=dims, n_jobs=-1)
                diagram = persistence.fit_transform(h_filtration)[0]
                return cls(diagram=diagram, filtration=filtration,
                           bounds=bounds, dimensions=dims)
        elif filtration == "dilation":
            if not im.is_binary():
                return "Error: Image is not binary"  # TODO: add error handling here?
            else:
                img_binary = im.to_np()
                img_shape = img_binary.shape
                img_binary = img_binary.reshape((1, img_shape[0], img_shape[1]))
                h_filtration = DilationFiltration().fit_transform(img_binary)
                persistence = CubicalPersistence(homology_dimensions=dims, n_jobs=-1)
                diagram = persistence.fit_transform(h_filtration)[0]
                return cls(diagram=diagram, filtration=filtration,
                           bounds=bounds, dimensions=dims)
        else:  # TODO: conduct actual error handling here?
            return "Error: filtration must be one of cubical, height, or dilation"

    @classmethod
    def from_filepath(cls, filepath):
        # TODO: add functionality to load from pickle file
        extension = os.path.splitext(filepath)[1]
        if (extension == ".csv"):
            pointcloud = np.loadtxt(filepath, delimeter=",")
            dims = pointcloud.shape[1]
            VR = VietorisRipsPersistence(homology_dimensions=list(range(dims)))
            diagram = VR.fit_transform(np.expand_dims(pointcloud, 0))[0]
            return cls(diagram)
        if (extension == ".pkl"):
            with open(filepath, 'rb') as inp:
                diagram = pickle.load(inp)
            return diagram
        else:
            im = Im(Image.open(filepath))
            img_grey = im.as_grayscale().to_np()
            img_shape = img_grey.shape
            cubical = CubicalPersistence(homology_dimensions=(0, 1))
            diagram = cubical.fit_transform(img_grey.reshape((1,
                                                              img_shape[0],
                                                              img_shape[1])))[0]
            return cls(diagram)  # TODO: fix this...  doesn't have all needed args.

    def __len__(self):
        """Get the number of diagram dimensions"""
        return len(self.diagram)

    def __getitem__(self, index):
        return self.diagram[index]

    def save_pkl(self, filepath):
        """
        Saves the file as a pkl format.

        Parameters
        ----------
            filepath: the filepath to write the pickle file to.
        """
        with open(filepath, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def save(self, filepath):
        """
        Saves the file as a csv format.

        Parameters
        ----------
            filepath: the filepath to write the csv file to.
        """
        # NOTE: this method seems problematic as it doesn't load parms
        # TODO: fix this to detect rest of parms? Even possible?
        np.savetxt(filepath, self.diagram, delimiter=",")

    def plot(self, ax=None, dimension=0, main="Diagram", color="m",
             alpha=0.2, s=10, **kwargs):
        """
        Plots the persistence diagram as a matplotlib plot. Returns
        the plot so that one can incorporate into subplots and edit
        the plot.

        Parameters
        ----------

            ax: ax
                Axis object to write to.

            dimension: int
                The dimension of the diagram to plot.

            main: str
                The main title for the plot.

            color: str
                The color for plotting characters.

            alpha: num
                The alpha transparency (b/t 0 and 1).

            s: num
                The size of the plotting character.

            **kwargs: dictionary
                The other parameters to pass to scatter and plot from matplotlib.
        """
        data = self.diagram[dimension][:, 0:2]
        ax = ax or plt.gca()
        ax.set_title(main)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        scatter = ax.scatter(data[:, 0], data[:, 1], color=color,
                             alpha=alpha, s=s, **kwargs)
        return scatter

#    def bottleneck(self, diagram):
#        """
#        Computes the bottleneck distance between self and diagram.
#        """
#        bottleneck(self.diagram, diagram)
