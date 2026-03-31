import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix", 
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 6,
})

PLT_COLORS = ["#99ff99",
    "#ff9999",
    "#9999ff",
    "#9f9f9f",
    "#0FEFEB",
    "#DCEB12"]

def figsize(fac, doublecol=False):
    # single column width and height.
    w = 3.5
    h = 3.5
    if doublecol:
        fac *= 2.0
    return (w * fac, h * fac)