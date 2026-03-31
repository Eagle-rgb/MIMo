""" This file is used to plot the maximum displacement velocities as a bar chart 
for younger / older infants just like in Kobayashi 16. For this, manually insert
the displacement velocities for the younger infants as 'means_grey' and for the
older infants as 'means_black'.
"""

import matplotlib.pyplot as plt
import numpy as np
import icdlplot

if __name__ == '__main__':
    # --- Daten vorbereiten ---
    categories = ['TR', 'CA', 'CL']

    means_age1 = [2.46, 2.25, 2.34]
    means_age3 = [3.54, 3.36, 4.24]
    means_age6 = [5.76, 5.99, 4.67]
    means_age9 = [5.67, 6.00, 6.30]

    std_err_age1 = [1.22, 1.19, 1.51]
    std_err_age3 = [1.69, 1.81, 2.04]
    std_err_age6 = [1.36, 1.67, 1.81]
    std_err_age9 = [1.36, 1.49, 1.41]

    # Positionen der Balken auf der x-Achse
    x = np.arange(len(categories))
    width = 0.2  # Breite der einzelnen Balken

    # --- Plot erstellen ---
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    rects1 = ax.bar(x - 1.5*width, means_age1, width, yerr=std_err_age1, 
                    label='1', edgecolor='black', color=icdlplot.PLT_COLORS[0], capsize=5)

    rects2 = ax.bar(x - 0.5*width, means_age3, width, yerr=std_err_age3, 
                    label='3', edgecolor='black', color=icdlplot.PLT_COLORS[1], capsize=5)

    rects3 = ax.bar(x + 0.5*width, means_age6, width, yerr=std_err_age6, 
                    label='6', edgecolor='black', color=icdlplot.PLT_COLORS[2], capsize=5)

    rects4 = ax.bar(x + 1.5*width, means_age9, width, yerr=std_err_age9, 
                    label='9', edgecolor='black', color=icdlplot.PLT_COLORS[3], capsize=5)

    # --- Styling ---
    ax.set_ylabel('Normalized velocity')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(-0.2, 8.0)
    ax.legend(frameon=True)

    # Layout optimieren und anzeigen
    plt.savefig('kobayashi_velocity_barchart.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')