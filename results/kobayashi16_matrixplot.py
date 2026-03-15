import matplotlib.pyplot as plt
import numpy as np

# Definition der Symbole und Farben
# —: Stationary, //: Leading, ●: Synchronous, ○: Following
SYMBOLS = {
    'stationary': {'marker': '_', 'color': 'black', 'fill': False},
    'leading': {'marker': 'o', 'color': 'black', 'hatch': '////', 'fill': False},
    'synchronous': {'marker': 'o', 'color': 'gray', 'fill': True},
    'following': {'marker': 'o', 'color': 'white', 'edgecolor': 'black', 'fill': True}
}

def draw_matrix(ax, data, title):
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-len(data) - 0.5, 0.5)
    
    # Spaltenbeschriftungen
    cols = ['IA', 'IL', 'CA', 'CL', 'Occ. (%)']
    for i, col in enumerate(cols):
        ax.text(i, 0, col, ha='center', va='center', fontweight='bold')

    # Daten zeichnen
    for row_idx, row in enumerate(data):
        y = -(row_idx + 1)
        # Die ersten 4 Spalten sind die Gliedmaßen-Muster
        for col_idx in range(4):
            pattern = row[col_idx]
            style = SYMBOLS.get(pattern)
            
            if pattern == 'stationary':
                ax.text(col_idx, y, '—', ha='center', va='center', fontsize=15)
            else:
                circle = plt.Circle((col_idx, y), 0.3, facecolor=style.get('color', 'white'), 
                                    edgecolor='black', hatch=style.get('hatch', None))
                ax.add_patch(circle)
        
        # Die letzte Spalte ist der Prozentwert
        ax.text(4, y, f"{row[4]:.1f}", ha='center', va='center')

    ax.axis('off')

# Beispiel-Daten für "Two stationary limbs" (Auszug aus deinem Bild)
younger_data = [
    ['stationary', 'stationary', 'synchronous', 'following', 23.3],
    ['stationary', 'stationary', 'synchronous', 'synchronous', 17.4],
    ['stationary', 'stationary', 'leading', 'following', 8.1]
]

older_data = [
    ['stationary', 'stationary', 'synchronous', 'following', 15.0],
    ['stationary', 'stationary', 'synchronous', 'synchronous', 14.2],
    ['stationary', 'stationary', 'leading', 'synchronous', 1.6]
]

# Plot erstellen
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

draw_matrix(ax1, younger_data, "Younger group")
draw_matrix(ax2, older_data, "Older group")

plt.tight_layout()
plt.show()