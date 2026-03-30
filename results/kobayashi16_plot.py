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

    # Mittelwerte (Means)
    means_grey = [5.73, 6.03, 4.87]
    means_black = [5.59, 5.92, 6.29]

    # Standardfehler (Std Error)
    std_err_grey = [1.39, 1.68, 1.59]
    std_err_black = [1.34, 1.45, 1.36]

    # Positionen der Balken auf der x-Achse
    x = np.arange(len(categories))
    width = 0.35  # Breite der einzelnen Balken

    # --- Plot erstellen ---
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # Linke Balken (Grau)
    rects1 = ax.bar(x - width/2, means_grey, width, yerr=std_err_grey, 
                    label='Younger', edgecolor='black', color='#99ff99', capsize=5)

    # Rechte Balken (Schwarz)
    rects2 = ax.bar(x + width/2, means_black, width, yerr=std_err_black, 
                    label='Older', edgecolor='black', color='#ff9999', capsize=5)

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