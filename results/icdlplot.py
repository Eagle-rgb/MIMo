import matplotlib.pyplot as plt

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

def figsize(fac, doublecol=False):
    # single column width and height.
    w = 3.5
    h = 3.5
    if doublecol:
        fac *= 2.0
    return (w * fac, h * fac)