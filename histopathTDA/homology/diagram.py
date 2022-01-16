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
from gtda.homology import CubicalPersistence  # type: ignore
from gtda.homology import VietorisRipsPersistence #type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from histopathTDA.image import Im
from histopathTDA import Im


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

    def __init__(self, diagram):
        """
        Constructs all the necessary attributes for the persistence diagram object.

        Parameters
        ----------
            dgm_input: Either a Im image or a pointset
        """
        # TODO: consider filling in with other data that may be needed later
        self.diagram = diagram

    @classmethod
    def from_pointset(cls, pointcloud: np.ndarray):
        dims = pointcloud.shape[1]
        VR = VietorisRipsPersistence(homology_dimensions=list(range(dims)))  # Parameter explained in the text
        diagram = VR.fit_transform(np.expand_dims(pointcloud, 0))[0]
        return cls(diagram)

    @classmethod
    def from_im(cls, im: Im):
        img_grey = im.as_grayscale().to_np()
        img_shape = img_grey.shape
        cubical = CubicalPersistence(homology_dimensions=(0, 1))
        diagram = cubical.fit_transform(img_grey.reshape((1,
                                                          img_shape[0],
                                                          img_shape[1])))[0]
        return cls(diagram)

    @classmethod
    def from_filepath(cls, filepath):
        extension = os.path.splitext(filepath)[1]
        if (extension == ".csv"):
            pointcloud = np.loadtxt(filepath, delimeter=",")
            dims = pointcloud.shape[1]
            VR = VietorisRipsPersistence(homology_dimensions=list(range(dims)))  # Parameter explained in the text
            diagram = VR.fit_transform(np.expand_dims(pointcloud, 0))[0]
            return cls(diagram)
        else:
            im = Im(Image.open(filepath))
            img_grey = im.as_grayscale().to_np()
            img_shape = img_grey.shape
            cubical = CubicalPersistence(homology_dimensions=(0, 1))
            diagram = cubical.fit_transform(img_grey.reshape((1,
                                                            img_shape[0],
                                                            img_shape[1])))[0]
            return cls(diagram)

    def save(self, filepath):
        """
        Saves the file as a csv format.

        Parameters
        ----------
            filepath: the filepath to write the csv file to.
        """
        np.savetxt(filepath, self.diagram, delimiter=",")

    def plot(self, ax=None, dimension=0, main="Diagram", color="m", alpha=0.2, s=10, **kwargs):
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
        data = self.diagram[self.diagram[:, 2] == dimension][:, 0:2]
        ax = ax or plt.gca()
        ax.set_title(main)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        scatter = ax.scatter(data[:, 0], data[:, 1], color=color, alpha=alpha, s=s, **kwargs)
        return scatter

#    def bottleneck(self, diagram):
#        """
#        Computes the bottleneck distance between self and diagram.
#        """
#        bottleneck(self.diagram, diagram)
