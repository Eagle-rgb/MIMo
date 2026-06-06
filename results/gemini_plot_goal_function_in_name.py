import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse
from datetime import datetime
from utils import DATE_FORMAT, valid_date

# --- Konfiguration ---
BASE_DIR = "."  # Das Hauptverzeichnis, in dem Ihre Ordner liegen
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate"]
OUTPUT_CSV = "rl_comparison_data.csv"

# --- 1. Daten laden und zusammenführen ---

def load_tensorboard_runs(base_dir, date, suffix, tags):
    """
    Durchsucht rekursiv das Basisverzeichnis, lädt die angegebenen TensorBoard-Tags
    aus den PPO_0-Ordnern und fasst die Daten in einer einzigen DataFrame zusammen.
    """
    all_data_list = []
    
    # Muster für die Ordnerstruktur:
    # <date>_<haltung>_<belohnung>_<modelsuffix>_run_<nummer>/PPO_0/...
    # Dabei ist <date> als YY-mm-dd (also %Y-%m-%d) formatiert.
    date_str = date.strftime(DATE_FORMAT)

    # Regex-Muster zum Extrahieren von Haltung und Belohnung aus dem **Elternordner**
    # Der Ordnername muss exakt dem Muster entsprechen, bevor PPO_0 kommt.
    re_str = date_str + r'_([a-z]+)_([a-z]+)_' + suffix + r'_run_(\d+)'
    pattern = re.compile(re_str)

    print(f"Suche nach TensorBoard Logs in {base_dir}...")
    
    # Rekursives Durchsuchen, um alle 'PPO_0'-Ordner zu finden
    for root, dirs, files in os.walk(base_dir):
        # Wir sind im Ordner 'PPO_0'
        if os.path.basename(root) != 'PPO_0':
            continue

        # Der Elternordner von PPO_0 ist der eigentliche Run-Ordner
        run_folder_name = os.path.basename(os.path.dirname(root))
        
        # Überprüfe, ob der Run-Pfad dem gewünschten Muster entspricht
        match = pattern.search(run_folder_name)
        
        if not match:
            continue

        haltung, belohnung, run_num = match.groups()
        
        print(f"Lade Run: Date={date_str}, Haltung={haltung}, Belohnung={belohnung}, Suffix={suffix}, Run={run_num} aus {root}")

        try:
            # SummaryReader erwartet den Pfad zum Ordner (hier: PPO_0),
            # der die 'events.out.tfevents'-Datei enthält.
            reader = SummaryReader(root)
            
            # Direktes Laden aller skalaren Daten in eine DataFrame
            df_scalars = reader.scalars
            
            # Filtern auf die gewünschten Tags
            tag_data = df_scalars[df_scalars['tag'].isin(tags)].copy()
            
            if not tag_data.empty:
                # Hinzufügen der Experiment-Metadaten
                tag_data['Belohnung'] = belohnung
                tag_data['Haltung'] = haltung
                tag_data['Run'] = int(run_num)
                
                # Umbenennen der Spalten für Konsistenz
                tag_data.rename(columns={'step': 'Step', 'value': 'Value', 'tag': 'Tag'}, inplace=True)
                
                # Hinzufügen zur Gesamtliste
                all_data_list.append(tag_data[['Belohnung', 'Haltung', 'Run', 'Tag', 'Step', 'Value']])
                
        except Exception as e:
            print(f"Fehler beim Laden von {root}: {e}")

    # Zusammenführen aller individuellen DataFrames am Ende
    if all_data_list:
        df = pd.concat(all_data_list, ignore_index=True)
        return df
    else:
        return pd.DataFrame()

# --- 2. Datenaggregieren und Plotten ---

def create_and_save_individual_plots(df, plot_dir, date, suffix):
    """
    Aggregiert die Daten und speichert vier individuelle Plots (Tag x Haltung).
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # Berechne den Mittelwert und die Standardabweichung einmal für alle Daten
    aggregated = df.groupby(['Haltung', 'Belohnung', 'Tag', 'Step'])['Value'].agg(['mean', 'std']).reset_index()
    
    haltungen = ['prone', 'supine']
    date_str = date.strftime(DATE_FORMAT)
    
    # Übersetzungen für Titel und Achsenbeschriftungen
    title_map = {
        "rollout/ep_rew_mean": "Episode Reward Mean",
        "rollout/success_rate": "Success Rate"
    }
    y_label_map = {
        "rollout/ep_rew_mean": "Episode Reward Mean",
        "rollout/success_rate": "Success Rate"
    }

    plot_count = 0
    
    for tag in TAGS_TO_LOAD:
        tag_data = aggregated[aggregated['Tag'] == tag]
        
        for haltung in haltungen:
            # 1. Filtern der aggregierten Daten für die aktuelle Haltung und
            # check, ob mindestens ein Eintrag mit der Haltung existiert.
            haltung_group = tag_data[tag_data['Haltung'] == haltung]
            if (len(haltung_group) == 0): continue

            # 2. Plot-Setup starten
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            
            # 3. Iteriere über die Belohnungsfunktionen
            for belohnung, group in haltung_group.groupby('Belohnung'):
                # Plot des Mittelwerts
                ax.plot(group['Step'], group['mean'], label=label, linewidth=2)
                
                # Plot der Standardabweichung als Fehlerband
                ax.fill_between(
                    group['Step'], 
                    group['mean'] - group['std'], 
                    group['mean'] + group['std'], 
                    alpha=0.15 
                )
            
            # 4. Achsen- und Titelkonfiguration
            plot_title = f'{title_map.get(tag)}: {haltung.capitalize()}'
            ax.set_title(plot_title, fontsize=14)
            ax.set_xlabel("Step (Training)", fontsize=12)
            ax.set_ylabel(y_label_map.get(tag, "Wert"), fontsize=12)
            if tag == 'rollout/success_rate':
                ax.set_ylim(0.0, 1.0)
            ax.legend(title='Goal Function', loc='best')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # 5. Speichern des Plots
            filename = f"{date_str}_{suffix}_{haltung}_{tag.replace('/', '_')}.png"
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path)
            plt.close() # Schließt die Figur, um Speicher freizugeben
            
            print(f"   Plot gespeichert: {save_path}")
            plot_count += 1
            
    print(f"\n✅ {plot_count} Plots wurden erfolgreich im Ordner '{plot_dir}' gespeichert.")

if __name__ == "__main__":
    # 0. Argumente laden. Zeit und Modelsuffix.
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, type=valid_date, help="Date of the runs.")
    parser.add_argument('--suffix', required=True, type=str, help="Model name suffix.")
    parser.add_argument('--save_df', required=False, action='store_true', help="If set, saved pandas Dataframe csv.")
    args = parser.parse_args()
    date = args.date
    suffix = args.suffix
    save_df = args.save_df

    # 1. Daten laden
    base_dir = os.path.abspath(BASE_DIR)
    full_df = load_tensorboard_runs(base_dir, date, suffix, TAGS_TO_LOAD)

    if full_df.empty:
        print("Es wurden keine TensorBoard-Daten gefunden. Bitte überprüfen Sie den BASE_DIR und die Ordnerstruktur.")
    else:
        # 2. DataFrame speichern (optional, aber empfohlen)
        if save_df:
            full_df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✅ Alle Daten wurden erfolgreich in '{OUTPUT_CSV}' gespeichert.")

        # 3. Plots erstellen
        print("\nErstelle Plots...")
        
        # Plot für die durchschnittliche Episodenbelohnung und Erfolgsrate
        create_and_save_individual_plots(full_df, '.', date, suffix)