import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    1: "#808080",
    3: "#aa805a",
    6: "#d57f34",
    9: "tab:orange",
}

def color_gradient(n):
    """ n equally spaced colors from COLORS[1] to COLORS[9], both included, as hex strings.

    Linear in RGB, which is how COLORS[3] and COLORS[6] sit between the two ends, so
    color_gradient(4) reproduces COLORS.
    """
    start = np.array(mcolors.to_rgb(COLORS[1]))
    end = np.array(mcolors.to_rgb(COLORS[9]))
    return [mcolors.to_hex(start + t * (end - start)) for t in np.linspace(0, 1, n)]

def figsize(fac, doublecol=False):
    # single column width and height.
    w = 3.5
    h = 3.5
    if doublecol:
        fac *= 2.0
    return (w * fac, h * fac)