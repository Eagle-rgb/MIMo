""" This file was created for only one purpose: Make one plot with three curves in it:
1. Success Rate full rollover
2. Success Rate Side Lying during training of full roll over
3. Success Rate Side Lying in explicit side lying model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict
from utils import DATE_FORMAT, valid_date

# --- Konfiguration ---
BASE_DIR = "."
N_POINTS = 500  # Auflösung der X-Achse

def create_side_lying_comparison_plot(plot_dir, date_full, date_side, suffix_full, suffix_side):
    """ Creates a dual comparison plot for a single model.
    
    Arguments:
        - df: Individual tensorboard run data of two models you want to compare.
        - plot_dir: The output directory.
        - date_full: Date of the fully trained model.
        - date_side: Date of the side lying trained model.
        - suffix_full: Suffix of the fully trained model.
        - suffix_side: Suffix of the side lying trained model.
    """
    tag_side_lying = "rollout/side_lying_success_rate"
    tag_full = "rollout/success_rate"

    df = load_tensorboard_runs(base_dir=os.path.abspath(BASE_DIR), tags=[tag_side_lying, tag_full],
        date_filter=[date_side, date_full], suffix_filter=[suffix_full, suffix_side])
    
    assert not df.empty
    
    df_full_tag_side = df[(df['Suffix'] == suffix_full) & (df['Tag'] == tag_side_lying)]
    df_full_tag_full = df[(df['Suffix'] == suffix_full) & (df['Tag'] == tag_full)]
    df_side_tag_side = df[(df['Suffix'] == suffix_side) & (df['Tag'] == tag_full)]

    df_list = [df_full_tag_side, df_full_tag_full, df_side_tag_side]
    label_list = ["Side Lying Implicit", "Full", "Side Lying Explicit"]

    plt.figure(figsize=(5,8))

    for i in range(len(df_list)):
        df = df_list[i]
        run_data = interpolate_runs_to_dict(df, N_POINTS)

        plt.plot(run_data['steps'], run_data['mean'], label=label_list[i], linewidth=2)
        plt.fill_between(
            run_data['steps'],
            run_data['mean'] - run_data['std'],
            run_data['mean'] + run_data['std'],
            alpha=0.15
        )

    num_runs = min([len(df_full_tag_side), len(df_full_tag_full), len(df_side_tag_side)])
        
    # Gemeinsamer Haupttitel
    plt.suptitle(f"Supine to Prone Side Lying Comparison, {num_runs} runs", fontsize=18, fontweight='bold', y=0.98)

    # Metadata ToDo. We have two models so we would really need to include both metadata. It would probably
    # be best to have common metadata on the top and for ax1, ax2 the individual - differing metadata.
        # Metadaten / Hyperparameter Text unter dem Titel
        #folder = sub_df['Folder'].unique()[0]
        #hyperparams = load_model_hyperparams(folder)
        #if hyperparams:
        #    fig.text(0.5, 0.91, hyperparams, ha='center', fontsize=11, style='italic', color='dimgray')

    # Achsenbeschriftungen
    plt.set_title("Aggregated (Mean ± Std)", fontsize=14, pad=10)

    # We do not need run 1, 2, 3, 4, ... legend.
    #ax1.legend(loc='best', fontsize=10)
    #ax2.legend(loc='best', fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.xlabel("Steps", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylabel('Success Rate')

    # Speichern
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Platz oben für Titel lassen

    # 5. Speichern des Plots
    # If there is exactly one suffix specified, we include that in
    # the output file name.
    filename = f"side_lying_comparison.png"
    save_path = os.path.join(plot_dir, "png", filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Erfolg: {filename} gespeichert.")

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
    parser.add_argument('--date_side', required=True, type=valid_date, help="Date of the side lying model.")
    parser.add_argument('--date_full', required=True, type=valid_date, help="Date of the full trained model.")
    parser.add_argument('--suffix_side', required=True, help="Side lying model name.")
    parser.add_argument('--suffix_full', required=True, help="Full trained model name.")

    args = parser.parse_args()
    date_side = args.date_side
    date_full = args.date_full
    suffix_side = args.suffix_side
    suffix_full = args.suffix_full

    create_side_lying_comparison_plot(plot_dir='.', date_full=date_full, date_side=date_side, suffix_full=suffix_full, suffix_side=suffix_side)