import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict

# --- Konfiguration ---
BASE_DIR = "."
TAGS_TO_LOAD = ["rollout/ep_rew_mean",
                "rollout/success_rate",
                "rollout/ep_end_hip_deg_mean",
                "rollout/ep_end_chest_deg_mean",
                "rollout/side_lying_success_rate"]
N_POINTS = 500  # Auflösung der X-Achse
DATE_FORMAT = r'%y-%m-%d'

def create_dual_comparison_plots_single_model(df, plot_dir, date, suffix):
    """ Creates a dual comparison plot for a single model.
    
    Arguments:
        - df: Individual tensorboard run data of just thist model.
        - plot_dir: The output directory.
        - date: Date of the model.
        - suffix: Suffix of the model.
    """
    for tag in TAGS_TO_LOAD:
        for haltung in df['Haltung'].unique():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

            sub_df = df[(df['Tag'] == tag) & (df['Haltung'] == haltung)]
            if sub_df.empty: continue

            num_runs = len(sub_df['Run'].unique())

            for m_suffix, groupby in sub_df.groupby(['Suffix']):
                # Die einzelnen runs dieses suffixes für den tag und die Haltung.
                run_data = interpolate_runs_to_dict(groupby, N_POINTS, min_)

                # Plot des Mittelwerts
                ax2.plot(run_data['steps'], run_data['mean'], label=m_suffix, linewidth=2)
            
                # Plot der Standardabweichung als Fehlerband
                ax2.fill_between(
                    run_data['steps'], 
                    run_data['mean'] - run_data['std'], 
                    run_data['mean'] + run_data['std'], 
                    alpha=0.15 
                )

                # 16.01.26 Also plot each individual run lightly in the background.
                for key in run_data['runs'].keys():
                    values = run_data['runs'][key]
                    ax1.plot(run_data['steps'], values, label=str(key), linewidth=2)

            # --- Layout & Titel ---
            # This gets 'success rate' out of 'rollout/success_rate'.
            tag_name = tag.split('/')[-1].replace('_', ' ').title()

            haltung_opposite = 'prone' if haltung == 'supine' else 'supine'
            direction_roll_over = f'{haltung.capitalize()} to {haltung_opposite.capitalize()}'
            
            # Gemeinsamer Haupttitel
            fig.suptitle(f"{tag_name} - {direction_roll_over}, {num_runs} runs", fontsize=18, fontweight='bold', y=0.98)
            
            # Metadaten / Hyperparameter Text unter dem Titel
            folder = sub_df['Folder'].unique()[0]
            hyperparams = load_model_hyperparams(folder)
            if hyperparams:
                fig.text(0.5, 0.91, hyperparams, ha='center', fontsize=11, style='italic', color='dimgray', wrap=True)

            # Achsenbeschriftungen
            ax1.set_title("Individual Training Runs", fontsize=14, pad=10)
            ax2.set_title("Aggregated (Mean ± Std)", fontsize=14, pad=10)

            ax1.legend(loc='best', fontsize=10)
            
            for ax in [ax1, ax2]:
                ax.set_xlabel("Steps", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)
                #if tag == 'rollout/success_rate':
                #    ax.set_ylim(0.0, 1.0)
            
            ax1.set_ylabel(tag_name, fontsize=12)

            # Speichern
            plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Platz oben für Titel lassen

            # 5. Speichern des Plots
            # If there is exactly one suffix specified, we include that in
            # the output file name.
            date_str = date.strftime(DATE_FORMAT)
            filename = f"{date_str}_{suffix}_{haltung}_{tag.replace('/', '_')}.png"
            save_path = os.path.join(plot_dir, "png", filename)
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"Erfolg: {filename} gespeichert.")

# # --- 3. Dual-Plotting Funktion ---
# def create_dual_comparison_plots(df, plot_dir):
#     os.makedirs(plot_dir, exist_ok=True)

#     suffixes = df['Suffix'].unique()
#     dates = df['Date'].unique()
    
#     for tag in TAGS_TO_LOAD:
#         for haltung in df['Haltung'].unique():
#             # Erstelle Figur mit 2 Subplots. ax1 is the subplot that contains
#             # all the individual runs and ax2 is the subplot with the std and mean.
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            
#             sub_df = df[(df['Tag'] == tag) & (df['Haltung'] == haltung)]
#             if sub_df.empty: continue

#             # Farben festlegen, damit beide Plots dieselben Farben nutzen
#             colors = plt.cm.tab10(np.linspace(0, 1, len(sub_df['Belohnung'].unique())))

#             for i, belohnung in enumerate(sorted(sub_df['Belohnung'].unique())):
#                 res = interpolate_runs_to_dict(sub_df[sub_df['Belohnung'] == belohnung], N_POINTS)
#                 if res is None: continue
                
#                 color = colors[i]
#                 label_name = belohnung.capitalize()

#                 # --- LINKER PLOT: Individual Runs ---
#                 for run_id, values in res['runs'].items():
#                     ax1.plot(res['steps'], values, color=color, alpha=0.3, linewidth=1)
#                 # Dummy-Linie für Legende links
#                 ax1.plot([], [], color=color, label=f"{label_name} Runs")

#                 # --- RECHTER PLOT: Mean + Std ---
#                 ax2.plot(res['steps'], res['mean'], color=color, linewidth=3, label=f"{label_name} Mean")
#                 ax2.fill_between(res['steps'], res['mean'] - res['std'], res['mean'] + res['std'], 
#                                  color=color, alpha=0.15)

#             # --- Layout & Titel ---
#             tag_name = tag.split('/')[-1].replace('_', ' ').title()
            
#             # Gemeinsamer Haupttitel
#             fig.suptitle(f"{tag_name} Comparison - {haltung.capitalize()}", fontsize=18, fontweight='bold', y=0.98)
            
#             # Metadaten / Hyperparameter Text unter dem Titel
#             fig.text(0.5, 0.91, HYPERPARAMS_TEXT, ha='center', fontsize=11, style='italic', color='dimgray')

#             # Achsenbeschriftungen
#             ax1.set_title("Individual Training Runs", fontsize=14, pad=10)
#             ax2.set_title("Aggregated (Mean ± Std)", fontsize=14, pad=10)
            
#             for ax in [ax1, ax2]:
#                 ax.set_xlabel("Steps", fontsize=12)
#                 ax.grid(True, linestyle='--', alpha=0.5)
#                 ax.legend(loc='best', fontsize=10)
            
#             ax1.set_ylabel(tag_name, fontsize=12)

#             # Speichern
#             plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Platz oben für Titel lassen
#             filename = f"dual_{tag.replace('/', '_')}_{haltung}.png"
#             plt.savefig(os.path.join(plot_dir, filename), dpi=200)
#             plt.close()
#             print(f"Erfolg: {filename} gespeichert.")

def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {s!r}")

# --- Start ---
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
    parser.add_argument('--date', required=True, type=valid_date, help="Date of the runs")
    # parser.add_argument('--save_df', required=False, action='store_true', help="If set, saved pandas Dataframe csv.")
    parser.add_argument('--suffix', required=True, help="Model name suffix")
    args = parser.parse_args()
    date = args.date
    suffix = args.suffix
    # save_df = args.save_df

    base_dir = os.path.abspath(BASE_DIR)
    data = load_tensorboard_runs(base_dir, TAGS_TO_LOAD, date.strftime(DATE_FORMAT), [suffix])

    if data.empty:
        print("Es wurden keine TensorBoard-Daten gefunden. Bitte überprüfen Sie den BASE_DIR und die Ordnerstruktur.")
    else:
        # 3. Plots erstellen
        print("\nErstelle Plots...")
        
        # Plot für die durchschnittliche Episodenbelohnung und Erfolgsrate
        create_dual_comparison_plots_single_model(data, '.', date, suffix)