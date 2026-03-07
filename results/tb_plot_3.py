import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict

# --- Konfiguration ---
BASE_DIR = "."
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate", "rollout/side_lying_success_rate"]
N_POINTS = 500  # Auflösung der X-Achse
DATE_FORMAT = r'%y-%m-%d'

def plot_suffix_run_data(axIndi, axAggre, groupby, label):
    """ Plots run data for one suffix into individual plot and into
    aggregated plot.
    
    Parameters:
        axIndi: The individual axes object from pyplot
        axAggre: The aggregated axes object from pyplot
        groupby: rundata, i.e. dataframe grouped by suffix
        label: The label for this suffix of the model
    """
    # Die einzelnen runs dieses suffixes für den tag und die Haltung.
    run_data = interpolate_runs_to_dict(groupby, N_POINTS)

    # Plot des Mittelwerts
    axAggre.plot(run_data['steps'], run_data['mean'], label=label, linewidth=2)

    # Plot der Standardabweichung als Fehlerband
    axAggre.fill_between(
        run_data['steps'], 
        run_data['mean'] - run_data['std'], 
        run_data['mean'] + run_data['std'], 
        alpha=0.15 
    )

    # 16.01.26 Also plot each individual run lightly in the background.
    for key in run_data['runs'].keys():
        values = run_data['runs'][key]
        axIndi.plot(run_data['steps'], values, label=str(key), linewidth=2)

def create_tri_comparison_plots_dual_model(df, plot_dir, suffix_1, suffix_2, display1, display2, tag):
    """ Creates a dual comparison plot for a single model.
    
    Arguments:
        - df: Individual tensorboard run data of two models you want to compare.
        - plot_dir: The output directory.
        - date: Date of the model.
        - suffix1: Suffix of the model 1.
        - suffix2: Suffix of the model 2.
        - display1: Title and legend label for model 1.
        - display2: Title and legend label for model 2.
        - tag: Tag to load. Specify without 'rollout/' prefix.
    """
    rollout_tag = "rollout/" + tag
    df_1 = df[(df['Tag'] == rollout_tag) & (df['Suffix'] == suffix_1)]
    df_2 = df[(df['Tag'] == rollout_tag) & (df['Suffix'] == suffix_2)]

    assert not df_1.empty and not df_2.empty
    haltung_1 = df_1['Haltung'].unique()[0]
    haltung_2 = df_2['Haltung'].unique()[0]

    assert haltung_1 == haltung_2

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8), sharey=True)

    num_runs_1 = len(df_1['Run'].unique())
    num_runs_2 = len(df_2['Run'].unique())

    plot_suffix_run_data(ax1, ax3, df_1, display1)
    plot_suffix_run_data(ax2, ax3, df_2, display2)

    tag_name = tag.replace('_', ' ').title()

    haltung_opposite = 'prone' if haltung_1 == 'supine' else 'supine'
    direction_roll_over = f'{haltung_1.capitalize()} to {haltung_opposite.capitalize()}'
        
    # Gemeinsamer Haupttitel
    fig.suptitle(f"{tag_name} - {direction_roll_over}", fontsize=18, fontweight='bold', y=0.98)

    # Metadata ToDo. We have two models so we would really need to include both metadata. It would probably
    # be best to have common metadata on the top and for ax1, ax2 the individual - differing metadata.
        # Metadaten / Hyperparameter Text unter dem Titel
        #folder = sub_df['Folder'].unique()[0]
        #hyperparams = load_model_hyperparams(folder)
        #if hyperparams:
        #    fig.text(0.5, 0.91, hyperparams, ha='center', fontsize=11, style='italic', color='dimgray')

    # Achsenbeschriftungen
    ax1.set_title(f"{num_runs_1} Individual Runs {display1}", fontsize=14, pad=10)
    ax2.set_title(f"{num_runs_2} Individual Runs {display2}", fontsize=14, pad=10)
    ax3.set_title("Aggregated (Mean ± Std)", fontsize=14, pad=10)

    # We do not need run 1, 2, 3, 4, ... legend.
    #ax1.legend(loc='best', fontsize=10)
    #ax2.legend(loc='best', fontsize=10)
    ax3.legend(loc='best', fontsize=10)
        
    for ax in [ax1, ax2, ax3]:
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
    filename = f"tricomp_{suffix1}_{haltung_1}_{tag.replace('/', '_')}.png"
    save_path = os.path.join(plot_dir, "png", filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Erfolg: {filename} gespeichert.")

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
    parser.add_argument('--date1', required=True, type=valid_date, help="Date of the runs 1")
    parser.add_argument('--date2', required=True, type=valid_date, help="Date of the runs 2")
    parser.add_argument('--suffix1', required=True, help="Model name suffix 1")
    parser.add_argument('--suffix2', required=True, help="Model name suffix 2")
    parser.add_argument('--display1', required=False, help="Display for model 1")
    parser.add_argument('--display2', required=False, help="Display for model 2")
    parser.add_argument('--tag', required=True, help="Tag to load. Supply tags without the rollout/ prefix.")
    parser.add_argument('--pos', required=False, type=str, help="Fixed starting position. Either 'prone' or 'supine'")
    args = parser.parse_args()
    date1 = args.date1
    date2 = args.date2
    suffix1 = args.suffix1
    suffix2 = args.suffix2
    display1 = args.display1
    display2 = args.display2
    tag = args.tag
    pos = args.pos
    # save_df = args.save_df

    base_dir = os.path.abspath(BASE_DIR)
    data = load_tensorboard_runs(base_dir, ["rollout/" + tag],
                                 [date1.strftime(DATE_FORMAT), date2.strftime(DATE_FORMAT)],
                                 [suffix1, suffix2], pos)

    if data.empty:
        print("Es wurden keine TensorBoard-Daten gefunden. Bitte überprüfen Sie den BASE_DIR und die Ordnerstruktur.")
    else:
        # 3. Plots erstellen
        print("\nErstelle Plots...")
        
        # Plot für die durchschnittliche Episodenbelohnung und Erfolgsrate
        create_tri_comparison_plots_dual_model(data, '.', suffix1, suffix2, display1, display2, tag)