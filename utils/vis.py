import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid


def visualize_data(dset, num_imgs=100, nrow=10, title=None, return_grid=False):
    """
    Visualize a grid of data

    Args:
        dset: Input dataset
        num_imgs: Number of images in grid
        nrow: Number of rows
        title (optional): Figure's title
        return_grid: Return just the grid of images (as a torch.tensor)

    Returns:

    """

    idxs = np.random.choice(len(dset), size=(num_imgs,), replace=False)
    imgs = torch.tensor([])
    for idx in idxs:
        img = dset[idx][0].unsqueeze(0)
        imgs = torch.cat((imgs, img), dim=0)
    grid_img = make_grid(imgs, nrow=nrow)
    if return_grid:
        return grid_img
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    # TODO: optionally save fig in a specified directory
    plt.show()





