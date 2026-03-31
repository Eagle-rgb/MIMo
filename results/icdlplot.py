import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix", 
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

PLT_COLORS = ["#99ff99",
    "#ff9999",
    "#9999ff",
    "#9f9f9f",
    "#0FEFEB",
    "#DCEB12"]

def figsize(fac):
    w = 3.5
    h = 3.5
    return (w * fac, h * fac)