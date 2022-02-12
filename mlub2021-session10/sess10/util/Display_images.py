import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import itertools


def display_image(image, title=None, cmap=None):
    """
    Display an image
    :param image: the image to display
    :type image: ndarray
    :param title: the title to display
    :type title: str
    :param cmap: the color map used to display the image
    :type cmap: str
    :return: None
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.show()


def display_image_prop(image, title=None, cmap=None):
    """
    Display an image with its properties
    :param image: the image to display
    :type image: ndarray
    :param title: the title to display
    :type title: str
    :param cmap: the color map used to display the image
    :type cmap: str
    :return: None
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.show()

    print('Range of values: min: ' + str(np.amin(image)) + ' - max: ' + str(np.amax(image)))
    print('Image type: ' + str(image.dtype))
    print('Image shape: ' + str(image.shape))


def display_multiple_images(images, titles=None, figure_size=(20, 20), axes_off=True, cmap_option='gray',
                            vmin_option=None, vmax_option=None, title_size=16, wspace=0.1, hspace=0.1):
    """
    Display multiple images
    :param images: the images to display
    :type images: array of ndarray
    :param titles: the title to display for each image
    :type titles: array of strings
    :param figure_size: the displayed figure size
    :type figure_size: tuple of int
    :param axes_off: show the axis:
                        true - show the axis
                        false - otherwise
    :type axes_off: boolean
    :param cmap_option: the color map used to display the image
    :type cmap_option: array of string
    :param vmin_option: min value of the data range the colormap covers
    :type vmin_option: array of floats
    :param vmax_option: max value of the data range the colormap covers
    :type vmax_option: array of floats
    :param title_size: the title size
    :type title_size: int
    :return: None
    """
    fig = plt.figure(figsize=figure_size)
    c = 0
    for i in range(len(images)):
        for j in range(len(images[i])):
            c += 1
            plt.subplot(len(images), len(images[0]), c)
            if axes_off:
                plt.axis("off")
            if titles:
                plt.title(titles[i][j], size=title_size)
            if cmap_option == 'gray':
                plt.imshow(images[i][j], cmap=cmap_option, vmin=vmin_option, vmax=vmax_option)
            else:
                plt.imshow(images[i][j], cmap=cmap_option[i][j], vmin=vmin_option, vmax=vmax_option)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()