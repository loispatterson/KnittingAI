import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors

def plot_knit_grid(mask, class_colors=None, class_labels=None, title=None, show_grid=True):
    """
    Display a segmentation mask as a colored knitting-style grid.

    Parameters
    ----------
    mask : np.ndarray
        2D array of class indices (e.g., 0 = background, 1 = knit, 2 = purl)
    class_colors : list of tuple
        List of RGB tuples (0–1) for each class.
        Example: [(1,1,1), (0.2,0.4,1), (1,0.4,0.4)]
    class_labels : list of str
        Optional labels for legend. Example: ["Background", "Knit", "Purl"]
    title : str
        Optional plot title.
    show_grid : bool
        If True, draw a grid between "stitches".
    """
    if class_colors is None:
        class_colors = [(1, 1, 1), (0.2, 0.4, 1), (1, 0.4, 0.4), (.4, 0.8, 0.5)]  # default 3-color map

    cmap = ListedColormap(class_colors)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(5, 5))
    #plt.imshow(mask, cmap=cmap, interpolation="nearest")
    plt.imshow(mask, cmap=cmap, norm=norm, origin='lower')

    if show_grid:
        h, w = mask.shape
        plt.grid(which="major", color="black", linewidth=0.5)
        plt.xticks(np.arange(-.5, w, 1), [])
        plt.yticks(np.arange(-.5, h, 1), [])
        plt.gca().set_xticks(np.arange(-.5, w, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, h, 1), minor=True)
        plt.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    if class_labels:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in class_colors]
        plt.legend(handles, class_labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def garter_to_knit_purl(mask, start_with="knit"):
    """
    Converts a mask of 'garter' (1s) into alternating knit/purl rows.
    start_with: 'knit' or 'purl'
    Returns a 2D mask where 0 = knit, 1 = purl.
    """
    knit_val, purl_val = 0, 1
    rows, cols = mask.shape
    new_mask = np.zeros_like(mask)

    start = knit_val if start_with == "knit" else purl_val

    for i in range(rows):
        if (i % 2 == 0):
            new_mask[i, :] = start
        else:
            new_mask[i, :] = 1 - start  # alternate
    return new_mask
