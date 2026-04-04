import matplotlib.pyplot as plt
import icdlplot
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent
output_dir = script_dir.parent / "icdl26" / "indpatterns"
output_dir.mkdir(parents=True, exist_ok=True)

def to_perc(arr):
    n = np.sum(arr)
    return arr / n * 100.0

if __name__ == '__main__':
    a_f = np.array([52.0, 5.8, 8.8, 0.7, 0.8, 4.6])
    a_f /= np.sum(a_f) / 100.0

    pat = to_perc(np.array([499, 81, 0, 4, 227, 46]))
    categories = ("A", "B", "C", "D", "E", "F")
    x = np.arange(len(categories))
    np.save(output_dir / f'y_af.npy', a_f)
    np.save(output_dir / f'y_pat.npy', pat)
    width = .8

    plt.figure(figsize=(2,2))
    plt.bar(x, a_f, color=icdlplot.PLT_COLORS[2], edgecolor='black', capsize=5)
    plt.xticks(x, categories)
    plt.ylabel("Count [%]")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(2,2))
    categories = ("TS", "IA", "IAb", "IL", "ILb", "NS")
    plt.bar(x, pat, color=icdlplot.PLT_COLORS[2], edgecolor='black', capsize=5)
    plt.xticks(x, categories)
    plt.ylabel("Count [%]")
    plt.tight_layout()
    plt.show()
