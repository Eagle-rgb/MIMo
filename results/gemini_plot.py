import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse
from datetime import datetime
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict

# --- Konfiguration ---
BASE_DIR = "."  # Das Hauptverzeichnis, in dem Ihre Ordner liegen
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate"]
OUTPUT_CSV = "rl_comparison_data.csv"
DATE_FORMAT = r'%y-%m-%d'

# --- 2. Datenaggregieren und Plotten ---
def create_and_save_individual_plots(df, plot_dir, date, suffixes, ind_runs):
    """
    Aggregiert die Daten und speichert vier individuelle Plots (Tag x Haltung).

    04.01.2026: Der Name der gespeicherten Datei ist:
    yy-mm-dd_<suffix>_<prone/supine>_rollout_<ep_rew_mean/success_rate>.png

    Ich habe in dieser Version den <suffix> hinzugefügt. Dieser steht dort nur, wenn 'suffixes' als
    Liste genau einen Eintrag (genau einen Suffix) hat.

    Arguments:
        - df (pd.DataFrame): Pandas Dataframe holding run data.
        - plot_dir: The directory to save the plot in.
        - date: Date filter.
        - suffixed: Suffixes filter.
        - ind_runs: Plot individual runs instead of mean and std.
    """
    os.makedirs(plot_dir, exist_ok=True)

    haltungen = ['prone', 'supine']
    
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
        for haltung in haltungen:
            # 1. Plot-Setup starten
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            # 2. Daten nach Haltung und Tag filtern.
            sub_df = df[(df['Tag'] == tag) & (df['Haltung'] == haltung)]
            if len(sub_df) == 0:
                continue # Es gibt keinen Eintrag für diese Haltung/Tag. Überspringe.

            num_runs = len(sub_df['Run'].unique())

            # 3. Iteriere über alle Suffixe.
            for suffix, groupby in sub_df.groupby(['Suffix']):
                # Die einzelnen runs dieses suffixes für den tag und die Haltung.
                run_data = interpolate_runs_to_dict(groupby, 500)

                if not ind_runs:
                    # Plot des Mittelwerts
                    ax.plot(run_data['steps'], run_data['mean'], label=suffix, linewidth=2)
                
                if not ind_runs:
                    # Plot der Standardabweichung als Fehlerband
                    ax.fill_between(
                        run_data['steps'], 
                        run_data['mean'] - run_data['std'], 
                        run_data['mean'] + run_data['std'], 
                        alpha=0.15 
                    )

                if ind_runs:
                    # 16.01.26 Also plot each individual run lightly in the background.
                    for key in run_data['runs'].keys():
                        values = run_data['runs'][key]
                        ax.plot(run_data['steps'], values, label=str(key), linewidth=2, alpha=1)
            
            # 4. Achsen- und Titelkonfiguration
            haltung_opposite = haltungen[0] if haltung == haltungen[1] else haltungen[1]
            roll_over_title = f'{haltung.capitalize()} to {haltung_opposite.capitalize()}'
            plot_title = f'{title_map.get(tag)}: {roll_over_title}, {num_runs} runs.'
            ax.set_title(plot_title, fontsize=14)
            ax.set_xlabel("Step (Training)", fontsize=12)
            ax.set_ylabel(y_label_map.get(tag, "Wert"), fontsize=12)
            #if tag == 'rollout/success_rate':
            #    ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()

            # 5. Speichern des Plots
            # If there is exactly one suffix specified, we include that in
            # the output file name.
            include_suffix_in_name = len(suffixes) == 1
            if date and include_suffix_in_name:
                date_str = date.strftime(DATE_FORMAT)
                filename = f"{date_str}_{suffixes[0]}_{haltung}_{tag.replace('/', '_')}.png"
            elif date and not include_suffix_in_name:
                date_str = date.strftime(DATE_FORMAT)
                filename = f"{date_str}_{haltung}_{tag.replace('/', '_')}.png"
            elif not date and include_suffix_in_name:
                filename = f"{suffixes[0]}_{haltung}_{tag.replace('/', '_')}.png"
            else:
                filename = f"{haltung}_{tag.replace('/', '_')}.png"

            save_path = os.path.join(plot_dir, filename)

            plt.savefig(save_path)
            plt.close() # Schließt die Figur, um Speicher freizugeben
            
            print(f"   Plot gespeichert: {save_path}")
            plot_count += 1
            
    print(f"\n✅ {plot_count} Plots wurden erfolgreich im Ordner '{plot_dir}' gespeichert.")

# --- Hauptausführung ---
def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {s!r}")

if __name__ == "__main__":
    # 0. Argumente laden. Zeit und Modelsuffix.
    # '--date' and '--suffix' are optional parameters used to filter which training
    # data to include in the plot. '--date' may be specified and is specified followed
    # by a date in 'yy-mm-dd' format. If a date is specified, this date is used in
    # the output .png file name.
    # '--suffix' is specified by following it up with
    # a list of suffixes that should be allowed. If exactly one suffix is specified, then
    # this suffix is included in the output .png file name.
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=False, type=valid_date, help="Date of the runs.")
    parser.add_argument('--save_df', required=False, action='store_true', help="If set, saved pandas Dataframe csv.")
    parser.add_argument('--suffix', '--names-list', nargs='+', required=False, default=[], help="Model name suffixes to use. Leave empty if allow all.")
    parser.add_argument('--ind_runs', action='store_true', help="Plot individual runs. This leaves out std and mean.")
    args = parser.parse_args()
    date = args.date
    suffixes = args.suffix
    save_df = args.save_df
    ind_runs = args.ind_runs

    # 1. Daten laden
    base_dir = os.path.abspath(BASE_DIR)
    full_df = load_tensorboard_runs(base_dir, TAGS_TO_LOAD, date.strftime(DATE_FORMAT), suffixes)

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
        create_and_save_individual_plots(full_df, '.', date, suffixes, ind_runs)