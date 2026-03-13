""" This file is used to plot the maximum displacement velocities as a bar chart 
for younger / older infants just like in Kobayashi 16. For this, manually insert
the displacement velocities for the younger infants as 'means_grey' and for the
older infants as 'means_black'.
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # --- Daten vorbereiten ---
    categories = ['TR', 'CA', 'CL']

    # Mittelwerte (Means)
    means_grey = [0.1562, 0.134, 0.0619]
    means_black = [0.5333, 0.3908, 0.5976]

    # Standardfehler (Std Error)
    std_err_grey = [0.0931, 0.1082, 0.1306]
    std_err_black = [0.1702, 0.2275, 0.2009]

    # Positionen der Balken auf der x-Achse
    x = np.arange(len(categories))
    width = 0.35  # Breite der einzelnen Balken

    # --- Plot erstellen ---
    fig, ax = plt.subplots(figsize=(2, 2))

    # Linke Balken (Grau)
    rects1 = ax.bar(x - width/2, means_grey, width, yerr=std_err_grey, 
                    label='Younger', edgecolor='black', color='grey', capsize=5)

    # Rechte Balken (Schwarz)
    rects2 = ax.bar(x + width/2, means_black, width, yerr=std_err_black, 
                    label='Older', edgecolor='black', color='black', capsize=5)

    # --- Styling ---
    ax.set_ylabel('Normalized velocity')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(-0.2, 1.5)
    ax.legend(prop={'size': 8}, frameon=True)

    # Layout optimieren und anzeigen
    plt.savefig('kobayashi_velocity_barchart.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')