import matplotlib.pyplot as plt
import numpy as np
import icdlplot
from pathlib import Path

script_dir = Path(__file__).parent
output_dir = script_dir.parent / "icdl26" / "patterns"
output_dir.mkdir(parents=True, exist_ok=True)

def plot_bar_util(data_list, categories, save_file, show_ylabel=True, show_legend=True):
    labels = ["Original", "Adaptive", "Kobayashi 16", "Kobayashi 21", "Siegel 24"]
    colors = ["#99ff99",
            "#ff9999",
            "#9999ff",
            "#9f9f9f",
            "#DCEB12"]
    for i in reversed(range(len(data_list))):
        if data_list[i] is None:
            del data_list[i]
            del labels[i]
            del colors[i]

    offset = 0.1
    width = (1.0-2*offset) / len(data_list)
    fig, ax = plt.subplots(figsize=(3.8, 2.0))

    x = np.arange(len(categories))

    for i in range(len(data_list)):
        data = data_list[i]
        if data is None:
            continue
        label = labels[i]
        pos = x - (len(data_list)-1) * width / 2.0 + width * i
        ax.bar(pos, data, width, label=label,
            edgecolor='black', color=colors[i], capsize=5)

    if show_ylabel:
        ax.set_ylabel("Number of rolls [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if show_legend:
        ax.legend(frameon=True)
    plt.savefig(f'{save_file}.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')

def abcdef_plot(params_ori, params_adap, age, save_file):
    """ Creates and saves a bar plot of the A, B, C, D, E, F
    categories. Expects 'params_ori' and  'params_adap' to
    be a tuple with 6 entries. The first entry is the percentage
    of patterns 'A', the second entry for pattern 'B' and so on,
    until pattern 'F'.
    Also requires age parameter so that we can
    retrieve the original values from kobayashi for that
    age group (age9 = older, age6 = younger). """
    kobayashi_16_age9 = np.array([
        23.3,
        17.4,
        11.6,
        5.8,
        1.2,
        1.2
    ])
    kobayashi_16_age9 /= np.sum(kobayashi_16_age9) / 100.0
    kobayashi_16_age6 = np.array([
        14.2,
        15.0,
        23.6,
        7.9,
        5.5,
        3.9
    ])
    kobayashi_16_age6 /= np.sum(kobayashi_16_age6) / 100.0
    kobayashi_21_age9 = np.array([8.5, 15.5, 14.0, 13.2, 6.2, 6.2])
    kobayashi_21_age9 /= np.sum(kobayashi_21_age9) / 100.0
    kobayashi_21_age6 = None
    siegel_age9 = None
    siegel_age6 = np.array([6.9, 9.7, 26.4, 20.8, 9.7, 26.4])
    categories = ("A", "B", "C", "D", "E", "F")

    if age == 9:
        data_kobayashi16 = kobayashi_16_age9
        data_kobayashi21 = kobayashi_21_age9
        data_siegel = siegel_age9
    elif age == 6:
        data_kobayashi16 = kobayashi_16_age6
        data_kobayashi21 = kobayashi_21_age6
        data_siegel = siegel_age6
    else:
        raise ValueError

    data_list = [params_ori, params_adap, data_kobayashi16, data_kobayashi21, data_siegel]
    plot_bar_util(data_list, categories, save_file)

def stationary_plot(params_ori, params_adap, age, save_file):
    """ Creates and saves a bar plot of the TS, IA, ...
    categories. Expects 'params_ori' and  'params_adap' to
    be a tuple with 6 entries. The first entry is the percentage
    of pattern 'TS', the second entry for pattern 'IA' and so on,
    until pattern 'NS'.
    Also requires age parameter so that we can
    retrieve the original values from kobayashi for that
    age group (age9 = older, age6 = younger). """
    kobayashi_16_age9 = (34.60, 40.90, 10.20, 8.70, 0, 5.5)
    kobayashi_16_age6 = (62.80, 32.60, 0, 1.2, 0, 3.5)
    kobayashi_21_age9 = (30.2, 38.0, 3.1, 13.2, 0.0, 15.5)
    kobayashi_21_age6 = None
    categories = ("TS", "IA", "IAb", "IL", "ILb", "NS")

    if age == 9:
        data_kobayashi16 = kobayashi_16_age9
        data_kobayashi21 = kobayashi_21_age9
    elif age == 6:
        data_kobayashi16 = kobayashi_16_age6
        data_kobayashi21 = kobayashi_21_age6
    else:
        raise ValueError

    data_list = [params_ori, params_adap, data_kobayashi16, data_kobayashi21, None]
    plot_bar_util(data_list, categories, save_file)

def to_perc(arr):
    n = np.sum(arr)
    return arr / n * 100.0

if __name__ == '__main__':
    # fig, ax = plt.subplots(figsize=(5,5))
    # stationary / non stationary plot. We have them here as absolute values, so we convert
    # them in percentage.
    params_ori_age9 = to_perc(np.array([49, 59, 0, 3, 103, 118]))
    params_adap_age9 = to_perc(np.array([168, 56, 0, 8, 58, 42]))
    params_ori_age6 = to_perc(np.array([0, 117, 0, 0, 0, 6]))
    params_adap_age6 = to_perc(np.array([26, 92, 0, 1, 0, 4]))
    stationary_plot(params_ori_age9, params_adap_age9, age=9, save_file="stationary_plot_age9")
    stationary_plot(params_ori_age6, params_adap_age6, age=6, save_file="stationary_plot_age6")

    np.save(output_dir / "pat_ori_age9.npy", params_ori_age9)
    np.save(output_dir / "pat_ori_age6.npy", params_ori_age6)
    np.save(output_dir / "pat_adap_age9.npy", params_adap_age9)
    np.save(output_dir / "pat_adap_age6.npy", params_adap_age6)

    # abcdef plot
    params_ori_age9 = np.array([14.2, 0.6, 31.3, 0.0, 0.0, 34.3])
    params_ori_age9 /= np.sum(params_ori_age9) / 100.0
    params_adap_age9 = np.array([12.3, 28.6, 6.3, 4.8, 7.8, 1.8])
    params_adap_age9 /= np.sum(params_adap_age9) / 100.0
    params_ori_age6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 4.9])
    params_ori_age6 /= np.sum(params_ori_age6) / 100.0
    params_adap_age6 = np.array([13.0, 8.1, 0.8, 0.0, 6.5, 3.3])
    params_adap_age6 /= np.sum(params_adap_age6) / 100.0
    abcdef_plot(params_ori_age9, params_adap_age9, age=9, save_file="abcdef_plot_age9")
    abcdef_plot(params_ori_age6, params_adap_age6, age=6, save_file="abcdef_plot_age6")

    np.save(output_dir / "af_ori_age9.npy", params_ori_age9)
    np.save(output_dir / "af_ori_age6.npy", params_ori_age6)
    np.save(output_dir / "af_adap_age9.npy", params_adap_age9)
    np.save(output_dir / "af_adap_age6.npy", params_adap_age6)

    kobayashi_16_age9 = np.array([
        23.3,
        17.4,
        11.6,
        5.8,
        1.2,
        1.2
    ])
    kobayashi_16_age9 /= np.sum(kobayashi_16_age9) / 100.0
    kobayashi_16_age6 = np.array([
        14.2,
        15.0,
        23.6,
        7.9,
        5.5,
        3.9
    ])
    kobayashi_16_age6 /= np.sum(kobayashi_16_age6) / 100.0
    kobayashi_21_age9 = np.array([8.5, 15.5, 14.0, 13.2, 6.2, 6.2])
    kobayashi_21_age9 /= np.sum(kobayashi_21_age9) / 100.0
    kobayashi_21_age6 = None
    siegel_age9 = None
    siegel_age6 = np.array([6.9, 9.7, 26.4, 20.8, 9.7, 26.4])

    np.save(output_dir / "af_16_age9.npy", kobayashi_16_age9)
    np.save(output_dir / "af_16_age6.npy", kobayashi_16_age6)
    np.save(output_dir / "af_21_age9.npy", kobayashi_21_age9)
    np.save(output_dir / "af_siegel_age6.npy", siegel_age6)

    kobayashi_16_age9 = np.array([34.60, 40.90, 10.20, 8.70, 0, 5.5])
    kobayashi_16_age6 = np.array([62.80, 32.60, 0, 1.2, 0, 3.5])
    kobayashi_21_age9 = np.array([30.2, 38.0, 3.1, 13.2, 0.0, 15.5])
    kobayashi_21_age6 = None

    np.save(output_dir / "pat_16_age9.npy", kobayashi_16_age9)
    np.save(output_dir / "pat_16_age6.npy", kobayashi_16_age6)
    np.save(output_dir / "pat_21_age9.npy", kobayashi_21_age9)
