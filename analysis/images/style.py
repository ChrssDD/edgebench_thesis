# analysis/images/style.py
from matplotlib import pyplot as plt

OKABE_ITO = ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]

def apply_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "axes.linewidth": 1.0,
        "grid.linestyle": "--",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.8",
    })
