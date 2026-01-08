import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import argparse
from datetime import datetime

# --- Konfiguration ---
BASE_DIR = "."  # Das Hauptverzeichnis, in dem Ihre Ordner liegen
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate"]
OUTPUT_CSV = "rl_comparison_data.csv"
DATE_FORMAT = r'%y-%m-%d'
ALGORITHM_FOLDERS = ['PPO_0', 'SAC_0', 'TD3_0', 'DDPG_0', 'A2C_0']

# --- 1. Daten laden und zusammenführen ---
def load_tensorboard_runs(base_dir, date, suffixes, tags):
    """
    Durchsucht rekursiv das Basisverzeichnis, lädt die angegebenen TensorBoard-Tags
    aus den <ALG>_0-Ordnern und fasst die Daten in einer einzigen DataFrame zusammen.
    <ALG> ist dabei einer von 'PPO', 'SAC', 'TD3', 'DDPG', 'A2C'.
    """
    all_data_list = []
    
    # Muster für die Ordnerstruktur:
    # <date>_<haltung>_<modelsuffix>_run_<nummer>/<ALG>_0/...
    # Dabei ist <date> als YY-mm-dd (also %Y-%m-%d) formatiert.

    # If date is specified, then exactly this date string must be included
    # in the model folder name.
    if date:
        date_str = date.strftime(DATE_FORMAT)
        date_regex = date_str
    else:
        # Else we allow any valid date string.
        date_regex = r'([0-9][0-9]-[0-9][0-9]-[0-9][0-9])' 

    # Regex-Muster zum Extrahieren von Haltung, suffix und run_num aus dem **Elternordner**
    # Der Ordnername muss exakt dem Muster entsprechen, bevor <ALG>_0 kommt.
    re_str = date_regex + r'_([a-z]+)_([a-z0-9_]+)_run_(\d+)'
    pattern = re.compile(re_str)

    print(f"Suche nach TensorBoard Logs in {base_dir}...")
    
    # Rekursives Durchsuchen, um alle '<ALG>_0'-Ordner zu finden
    for root, dirs, files in os.walk(base_dir):
        # The name of the folder we are currently in. We want this to be in the format 'PPO_0'
        # or any other algorithm prefix from the algorithm list above.
        this_folder_name = os.path.basename(root)

        if this_folder_name not in ALGORITHM_FOLDERS:
            continue

        # Der Elternordner von PPO_0 ist der eigentliche Run-Ordner
        run_folder_name = os.path.basename(os.path.dirname(root))
        
        # Überprüfe, ob der Run-Pfad dem gewünschten Muster entspricht
        match = pattern.search(run_folder_name)
        
        if not match:
            continue

        if date:
            haltung, suffix, run_num = match.groups()
        else:
            date_str, haltung, suffix, run_num = match.groups()

        # Überprüfe, ob suffix erlaubt ist, also als Konsolenargument spezifiziert ist. Falls keine suffixe angegeben sind,
        # erlaube alle.
        if len(suffixes) != 0 and suffix not in suffixes:
            continue  # Suffixe angegeben, aber dieser suffix passt leider nicht.
        
        print(f"Lade Run: Date={date_str}, Haltung={haltung}, Suffix={suffix}, Run={run_num} aus {root}")

        try:
            # SummaryReader erwartet den Pfad zum Ordner (hier: z.B. PPO_0),
            # der die 'events.out.tfevents'-Datei enthält.
            reader = SummaryReader(root)
            
            # Direktes Laden aller skalaren Daten in eine DataFrame
            df_scalars = reader.scalars
            
            # Filtern auf die gewünschten Tags
            tag_data = df_scalars[df_scalars['tag'].isin(tags)].copy()
            
            if not tag_data.empty:
                # Hinzufügen der Experiment-Metadaten
                tag_data['Haltung'] = haltung
                tag_data['Run'] = int(run_num)
                tag_data['Suffix'] = suffix
                tag_data['Date'] = date_str
                
                # Umbenennen der Spalten für Konsistenz
                tag_data.rename(columns={'step': 'Step', 'value': 'Value', 'tag': 'Tag'}, inplace=True)
                
                # Hinzufügen zur Gesamtliste
                all_data_list.append(tag_data[['Date', 'Haltung', 'Run', 'Suffix', 'Tag', 'Step', 'Value']])
                
        except Exception as e:
            print(f"Fehler beim Laden von {root}: {e}")

    # Zusammenführen aller individuellen DataFrames am Ende
    if all_data_list:
        df = pd.concat(all_data_list, ignore_index=True)
        return df
    else:
        return pd.DataFrame()

def interpolate_runs(group_df, n_points):
    """ 07.01.2026: DDPG and SAC have steps in their tensorboard logs that do not follow a "pattern", i.e.
    we can not just do a group_by on the step and then get a list of the values for each run. We must
    manipulate the pandas dataframe samples such that the steps match how we would have them in PPO or
    A2C, i.e. each 500 steps in A2C or 2048 in PPO there is one entry in the tensorboard log.
    
    Args:
      group_df (panda.DataFrame): DataFrame consisting of arbitrary number of runs.
      n_points: Number of sample points on the x-axis.
    
    Returns:
      resampled_values [List]: List of resampled values: One entry per point on the common x-axis.
    """
    if group_df.empty: return None

    # Step 1: Build a common x axis to sample values from.
    min_step = group_df['Step'].min()
    max_step = group_df['Step'].max()
    common_steps = np.linspace(min_step, max_step, n_points)

    # Step 2: Resample values such that 'Step' lies on the 'common_steps' axis. After this, 'resampled_values'
    # contains an entry for each run, with each run containing the list of resampled values for this run.
    resampled_values = []

    # Iterate over all runs.
    for run in group_df['Run'].unique():
        run_data = group_df[group_df['Run'] == run].sort_values('Step')
        if len(run_data) < 2: continue

        # Interpolate the values of this run.
        interp_vals = np.interp(common_steps, run_data['Step'], run_data['Value'])
        resampled_values.append(interp_vals)

    if not resampled_values: return None

    # Step 3: Average and std over resamples values.
    resampled_values = np.array(resampled_values)
    mean_vals = np.mean(resampled_values, axis=0)
    std_vals = np.std(resampled_values, axis=0)

    return pd.DataFrame({
        'Step': common_steps,
        'mean': mean_vals,
        'std': std_vals
    })


# --- 2. Datenaggregieren und Plotten ---

def create_and_save_individual_plots(df, plot_dir, date, suffixes):
    """
    Aggregiert die Daten und speichert vier individuelle Plots (Tag x Haltung).

    04.01.2026: Der Name der gespeicherten Datei ist:
    yy-mm-dd_<suffix>_<prone/supine>_rollout_<ep_rew_mean/success_rate>.png

    Ich habe in dieser Version den <suffix> hinzugefügt. Dieser steht dort nur, wenn 'suffixes' als
    Liste genau einen Eintrag (genau einen Suffix) hat.
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

            # 3. Iteriere über alle Suffixe.
            for suffix, groupby in sub_df.groupby(['Suffix']):
                # Die einzelnen runs dieses suffixes für den tag und die Haltung.
                run_data = interpolate_runs(groupby, 100)

                # Plot des Mittelwerts
                ax.plot(run_data['Step'], run_data['mean'], label=suffix, linewidth=2)
                
                # Plot der Standardabweichung als Fehlerband
                ax.fill_between(
                    run_data['Step'], 
                    run_data['mean'] - run_data['std'], 
                    run_data['mean'] + run_data['std'], 
                    alpha=0.15 
                )
            
            # 4. Achsen- und Titelkonfiguration
            plot_title = f'{title_map.get(tag)}: {haltung.capitalize()}'
            ax.set_title(plot_title, fontsize=14)
            ax.set_xlabel("Step (Training)", fontsize=12)
            ax.set_ylabel(y_label_map.get(tag, "Wert"), fontsize=12)
            if tag == 'rollout/success_rate':
                ax.set_ylim(0.0, 1.0)
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
    args = parser.parse_args()
    date = args.date
    suffixes = args.suffix
    save_df = args.save_df

    # 1. Daten laden
    base_dir = os.path.abspath(BASE_DIR)
    full_df = load_tensorboard_runs(base_dir, date, suffixes, TAGS_TO_LOAD)

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
        create_and_save_individual_plots(full_df, '.', date, suffixes)