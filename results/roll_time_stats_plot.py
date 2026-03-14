""" This file is used to create a barchart over ages 1, 3, 6, 9 with roll over
durations. """
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    categories = ['9', '6', '3', '1']

    min_duration_ms = [460, 490, 550, 680]

    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(2, 2))

    rects = ax.bar(x, min_duration_ms, edgecolor='black', color='grey', capsize=5)
    # --- Styling ---
    ax.set_ylabel('Min. Duration [ms]')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Age [months]")
    # ax.legend(prop={'size': 8}, frameon=True)

    # Layout optimieren und anzeigen
    plt.savefig('min_duration_age_comparison.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')