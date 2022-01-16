"""
This module contains functions that produce histograms for images
"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from histopathTDA.image import Im


def color_hist(img: Im, bins=256, grayscale=True, plot=True, scale=10000):
    """
    Create a color hisogram for an image

    Parameters
    _________
    img: Im class input to generate histogram
    bins: number of bins for the histogram
    grayscale: whether or not to grayscale the image or produce a RGB histogram
    plot: whether to plot the histogram or save it
    scale: the maximum y-axis value to scale the plots
    Returns
    _______
    n, b, pl: the three returned values from the histogram function in matplotlib
    """
    if grayscale:
        img = img.as_grayscale()
        size = 1
    else:
        size = 3
    np_img = img.to_np()
    np_img = np_img.reshape(-1, size)
    n, b, pl = plt.hist(np_img, histtype="step", bins=bins)
    plt.ylim([0, scale])
    if plot:
        plt.show()
    else:
        plt.savefig("hist.png")
    return n, b, pl


def avg_hist(imgs, bins=256, grayscale=True, plot=True, scale=10000):
    """
    Create an average histogram for the images provided
    Parameters
    __________
    imgs: list containing the images to generate a histogram
    bins: number of bins for the histogram
    grayscale: whether or not to grayscale the image or produce a RGB histogram
    plot: whether to plot the histogram or save it
    scale: the maximum y-axis value to scale the plots
    Returns
    _______
    n, b, pl: the three returned values from the histogram function in matplotlib
    """
    if grayscale:
        size = 1
    else:
        size = 3

    num = len(imgs)
    img = np.array([])
    for im in imgs:
        if grayscale:
            im = im.as_grayscale()
        im = im.to_np()
        im = im.reshape(-1, size)
        img = np.append(img, im)
    img = img.reshape(-1, size)
    counts, bins2 = np.histogram(img, bins=bins)
    n, b, pl = plt.hist(bins2[:-1], bins2, weights=(counts / num), histtype="step")
    plt.ylim([0, scale])
    if plot:
        plt.show()
    else:
        plt.savefig("avg_hist.png")
    return n, b, pl
