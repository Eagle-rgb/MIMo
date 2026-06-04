import pandas as pd
import matplotlib.pyplot as plt
import icdlplot
from signal_utils import resample_df_to_60hz, smooth_x_butterworth
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent
output_dir = script_dir.parent / "icdl26" / "actuation"
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    df = pd.read_csv('kobayashidataact_ind_model_2_age9.csv')
    df = df[df["Time"] >= 50]

    pattern_letters = ["A", "B", "C", "D", "E", "F"]
    episodes = [3, 16, 6, 137, 24, 2]
    act_keys = ["Act_Torso", "Act_Right_Leg", "Act_Left_Leg", "Act_Right_Arm", "Act_Left_Arm"]
    labels = ["TR", "CL", "IL", "CA", "IA"]

    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(3,6), sharey=True)

    for i in range(len(episodes)):
        episode = episodes[i]
        df_ep = df[df['Episode']==episode][['Time', 'Act_Torso', 'Act_Right_Leg', 'Act_Left_Leg', 'Act_Right_Arm', 'Act_Left_Arm']]
        df_ep = df_ep.set_index('Time')
        df_ep = resample_df_to_60hz(df_ep, original_fs=100, target_fs=60)
        df_ep = df_ep.apply(smooth_x_butterworth)
        x = df_ep.index.values

        ax = ax1 if i <= 2 else ax2
        ax = ax[i] if i <= 2 else ax[i-3]
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)
        y_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(y_ticks)
        ax.set_axisbelow(True)
        file_path = output_dir / f"actuation_x_{pattern_letters[i]}.npy"
        np.save(file_path, x)

        for j in range(len(act_keys)):
            act_key = act_keys[j]
            act_val = df_ep[act_key].values
            np.save(output_dir / f'actuation_y_{pattern_letters[i]}_{labels[j]}.npy', act_val)
            ax.plot(x, act_val, label=labels[j], color=icdlplot.PLT_COLORS[j])
            ax.set_title(f"Pattern {pattern_letters[i]}")

        if i != 0 and i != 3:
            # ax.yaxis.set_tick_params(labelleft=False)
            pass
        else:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(y_ticks)
            #ax.set_yticklabels([f"{val:1.f}" for val in y_ticks])

    ax2[2].set_xlabel("Time [ms]")

    handles, labels = ax1[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        bbox_to_anchor=(0.5,0.02),
        frameon=False,
        handlelength=1,
        handleheight=1,
        borderaxespad=0,
        columnspacing=0.5,
        fontsize=10
    )
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)
    plt.show()
