import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from weasyprint import HTML
import base64
from io import BytesIO
import argparse
from kobayashi16 import SUPPORT_PATTERNS, MOVEMENT_PATTERNS

def pdf_report(df):
    SUPPORT_COLORS = {
        "Two Stationary": "#4CAF50",                     # Grün
        "Stationary IA": "#2196F3",                     # Blau
        "Stationary IA - IL moving backward": "#00BCD4", # Cyan
        "Stationary IL": "#9C27B0",                     # Lila
        "Stationary IL - IA moving backward": "#E91E63", # Pink
        "No stationary": "#FF5722"                       # Orange
    }

    MOVEMENT_COLORS = {
        'A': '#c8e6c9', 'B': '#bbdefb', 'C': '#fff9c4', 
        'D': '#ffe0b2', 'E': '#d1c4e9', 'F': '#ffccbc', 'R': '#f5f5f5'
    }

    # --- 1. Hilfsfunktion für Diagramme ---
    def get_pie_base64(series, title, color_dict):
        counts = series.value_counts()
        if counts.empty: return "" 
        
        # Sicherstellen, dass die Farben immer zum Label passen
        # Wir nehmen nur die Farben aus dem Dictionary, die auch in den Daten vorkommen
        colors = [color_dict.get(label, "#cccccc") for label in counts.index]
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        counts.plot.pie(
            autopct='%1.1f%%', 
            ax=ax, 
            colors=colors, 
            startangle=140,
            wedgeprops={'edgecolor': 'white'}
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- 2. Daten für die Charts vorbereiten ---
    # (Stellen Sie sicher, dass df['Support_Pattern'] etc. existieren)
    support_base64 = get_pie_base64(df['Support_Pattern'], "Verteilung: Support Patterns", SUPPORT_COLORS)
    movement_base64 = get_pie_base64(df['Movement_Pattern'], "Verteilung: Movement Patterns", MOVEMENT_COLORS)

    # Matrix Daten
    # matrix_df = df.pivot(index='Run', columns='Episode', values='Movement_Pattern')
    # --- 1. Sicherstellen, dass nur die ersten 20 Episoden-Namen genommen werden ---
    # Wir holen uns alle einzigartigen Episoden-Namen in ihrer vorkommenden Reihenfolge
    all_episodes = df['Episode'].unique()
    top_20_episodes = all_episodes[:20]

    # Den DataFrame auf diese 20 Episoden filtern
    df_filtered = df[df['Episode'].isin(top_20_episodes)]

    # Jetzt erst die Matrix bauen
    matrix_df = df_filtered.pivot(index='Run', columns='Episode', values='Movement_Pattern')

    # Farbschema
    color_map = MOVEMENT_COLORS

    # --- 3. HTML mit striktem Layout ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 20mm;
            }}
            body {{
                font-family: DejaVu Sans, Arial, sans-serif;
                color: #333;
            }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            h2 {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 40px; }}
            
            /* Diagramm-Container: Explizit untereinander */
            .chart-box {{
                width: 100%;
                text-align: center;
                margin-bottom: 50px;
                display: block;
            }}
            .chart-img {{
                width: 400px; /* Feste Breite in Pixeln für stabilere PDF-Ausgabe */
                max-width: 100%;
            }}

            /* Matrix-Tabelle */
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed; /* Erzwingt Einhaltung der Seitenbreite */
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #999;
                padding: 4px 2px;
                text-align: center;
                font-size: 7pt; /* Sehr klein für viele Spalten */
                overflow: hidden;
            }}
            th {{ background-color: #eee; }}
            
            .legend {{ margin: 20px 0; font-size: 9pt; text-align: center; }}
            .legend-item {{ display: inline-block; margin-right: 15px; }}
            .box {{ display: inline-block; width: 10px; height: 10px; border: 1px solid #333; }}
        </style>
    </head>
    <body>
        <h1>Analysebericht: Lokomotions-Muster</h1>
        
        <h2>Globale Statistiken</h2>
        
        <div class="chart-box">
            <img class="chart-img" src="data:image/png;base64,{support_base64}">
        </div>
        
        <div class="chart-box">
            <img class="chart-img" src="data:image/png;base64,{movement_base64}">
        </div>

        <div style="page-break-before: always;"></div>

        <h2>Bewegungsmuster-Matrix (Details)</h2>
        
        <div class="legend">
            <strong>Legende:</strong><br>
            {' '.join([f'<span class="legend-item"><span class="box" style="background-color:{c}"></span> {p}</span>' for p, c in color_map.items()])}
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 60px;">Run</th>
                    {''.join([f'<th>{e}</th>' for e in matrix_df.columns])}
                </tr>
            </thead>
            <tbody>
    """

    for run, row in matrix_df.iterrows():
        html_content += f"<tr><td><strong>{run}</strong></td>"
        for val in row:
            bg = color_map.get(val, "#ffffff")
            html_content += f'<td style="background-color: {bg}">{val}</td>'
        html_content += "</tr>"

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # PDF erstellen
    HTML(string=html_content).write_pdf('finaler_bericht.pdf')

def export_npy(df: pd.DataFrame):
    support_patterns = np.zeros(6)
    movement_patterns = np.zeros(6)

    cnt_support = df['Support_Pattern'].value_counts()
    cnt_movement = df['Movement_Pattern'].value_counts()

    for i in range(len(SUPPORT_PATTERNS)):
        support_patterns[i] = cnt_support.get(SUPPORT_PATTERNS[i], 0)

    # Exlude the remainder (R) category in movement patterns, because we want
    # to normalize the A--F to 100%.
    for i in range(len(MOVEMENT_PATTERNS)-1):
        movement_patterns[i] = cnt_movement.get(MOVEMENT_PATTERNS[i], 0)

    total_cnt = len(df)
    support_patterns /= total_cnt
    movement_patterns /= total_cnt

    # Normalize movement counts
    sum_movement = sum(movement_patterns)
    movement_patterns /= sum_movement

    # In percentage
    support_patterns *= 100.0
    movement_patterns *= 100.0

    print("Exporting .npy files 'support.npy' and 'movement.npy'.")

    np.save('support.npy', support_patterns)
    np.save('movement.npy', movement_patterns)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_csv', required=True, type=str,
                        help="Load the full kobayashi report csv.")
    parser.add_argument('--pdf_report', required=False, action='store_true',
                        help="Generate full pdf report.")
    parser.add_argument('--npy', required=False, action='store_true',
                        help="Export support and movement patterns to .npy files " \
                        "support.npy and movement.npy. " \
                        "Movement patterns are normalized to sum up to 100%.")
    args = parser.parse_args()

    # CSV Export
    df = pd.read_csv(args.load_csv)

    if args.pdf_report:
        pdf_report(df)

    if args.npy:
        export_npy(df)

    