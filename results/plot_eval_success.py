"""Success plot: how many of a batch's seeds learned to roll, per posture.

The figure of 'results/icdl26_francisco_plots/supine_success.ipynb' -- one bar per group, its
height the percentage of models whose success rate clears 0.75 -- but driven by the JSON that
'mimoEnv/eval_rollover.py --group --json=...' writes instead of by the CSVs of
'results/success_after_training.py', and drawing prone and supine side by side.

    MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
        --group=models/roll_over/26-03-09/supine/26-03-09_supine_age1 --json=eval/supine_age1.json
    ... once per group and posture ...

    python results/plot_eval_success.py \
        --json 1=eval/supine_age1.json --json 1=eval/prone_age1.json \
        --json 3=eval/supine_age3.json --json 3=eval/prone_age3.json \
        --out=age_success.pdf

A sweep whose files already sit in one directory needs no per-file argument -- --label_regex
turns the batch names into the bar labels, and numeric labels are sorted numerically:

    python results/plot_eval_success.py --dir results/horizon/supine \
        --label_regex 'ep(\\d+)' --xlabel 'Training episode length [steps]' --annotate \
        --out=results/horizon/horizon_supine_success.pdf

The posture of each bar is read out of the JSON ('starting_position' per run), so the same label
appears once per posture and the two panels line up. Success is recomputed from each run's roll
rate rather than read off the 'successful' flag, so --threshold can be changed after the
evaluation without re-running it.

Old '<date>_<haltung>_<suffix>_test_success_rate.csv' files are accepted too, which is what lets
the ICDL figures be redrawn from the data they were originally made from. Those carry no posture,
so pass --posture with them.
"""
import argparse
import glob
import json
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# The house style of results/icdl26_francisco_plots/*.ipynb. Kept identical so a figure produced
# here can sit next to one produced there in the same document.
STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 3.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# The age palette of the ICDL notebooks, used when every label is one of these ages.
AGE_COLORS = {
    '1': "#808080",
    '3': "#aa805a",
    '6': "#d57f34",
    '9': "tab:orange",
}
# Otherwise interpolate the same grey -> orange ramp across however many bars there are.
RAMP = ("#808080", "#f08a24")

POSTURES = ('supine', 'prone')
DEFAULT_THRESHOLD = 0.75


def parse_source(spec):
    """Split a '--json LABEL=PATH[,PATH...]' value. The label may be omitted."""
    label, sep, paths = spec.partition('=')
    if not sep:
        label, paths = None, spec
    return label, [p for p in paths.split(',') if p]


def read_source(path):
    """Return (rate, posture, steps) for every model in one eval file.

    Handles the three shapes that exist:

    * a --group payload: one entry per run, posture from the run's own row;
    * a single --model payload: one model, the row without a policy_goal (a --policy_goal_sweep
      payload has no single roll rate and is rejected);
    * an old '..._test_success_rate.csv': 'Run,Success_Rate', no posture.
    """
    if path.lower().endswith('.csv'):
        import csv
        with open(path) as fh:
            return [(float(row['Success_Rate']), None, None) for row in csv.DictReader(fh)]

    with open(path) as fh:
        payload = json.load(fh)
    rows = payload.get('rows') or []
    if not rows:
        raise ValueError(f"{path}: no rows.")
    if 'summary' in payload:                      # --group
        return [(float(row['rolled']), row.get('starting_position'), row.get('episode_steps'))
                for row in rows]
    honest = [row for row in rows if row.get('policy_goal') is None]
    if len(honest) != 1:
        raise ValueError(f"{path}: this is a --policy_goal_sweep payload ({len(rows)} rows), "
                         f"which has no single success rate. Plot it with "
                         f"results/plot_her_goal_response.py.")
    return [(float(honest[0]['rolled']), payload.get('starting_position'),
             payload.get('episode_steps'))]


def source_name(path):
    """The name --label_regex matches against: the 'group' a --group payload records, else the
    file name. The group is the save path of the batch, so it carries the sweep parameter
    ('..._sac_her_ep250') even when the file was renamed."""
    if not path.lower().endswith('.csv'):
        try:
            with open(path) as fh:
                group = json.load(fh).get('group')
            if group:
                return os.path.basename(group.rstrip('/'))
        except (OSError, ValueError):
            pass
    return os.path.splitext(os.path.basename(path))[0]


def label_for(path, regex):
    """The bar label for one file: the regex's first capture group, else the file's own name."""
    name = source_name(path)
    if regex is None:
        return os.path.splitext(os.path.basename(path))[0]
    match = re.search(regex, name)
    if not match:
        raise ValueError(f"--label_regex {regex!r} does not match {name!r} (from {path}).")
    return match.group(1) if match.groups() else match.group(0)


def expand_sources(args):
    """The (label, [path, ...]) list, from --json arguments and --dir directories."""
    sources = [parse_source(spec) for spec in args.json]
    for directory in args.dir:
        found = sorted(glob.glob(os.path.join(directory, '*.json')))
        if not found:
            raise ValueError(f"No .json files in {directory}.")
        sources.extend((None, [path]) for path in found)
    resolved = []
    for label, paths in sources:
        resolved.append((label if label is not None else label_for(paths[0], args.label_regex),
                         paths))
    return resolved


def sort_labels(labels, how):
    """Bar order. 'auto' sorts numerically when every label is a number -- which is what a sweep
    over episode lengths or ages wants -- and otherwise keeps the order the files came in."""
    if how == 'given':
        return labels
    if how == 'name':
        return sorted(labels)
    try:
        numeric = sorted(labels, key=float)
    except ValueError:
        if how == 'numeric':
            raise ValueError(f"--sort=numeric, but these labels are not numbers: {labels}")
        return labels
    return numeric


def collect(sources, forced_posture):
    """Group every model's roll rate by (posture, label), keeping the order of the arguments."""
    data = OrderedDict()          # posture -> label -> [rate, ...]
    labels = []                   # label order, shared by both panels
    horizons = set()              # evaluation horizons seen, to warn when they differ
    for label, paths in sources:
        if label not in labels:
            labels.append(label)
        for path in paths:
            for rate, posture, steps in read_source(path):
                posture = forced_posture or posture
                if posture is None:
                    raise ValueError(f"{path}: no posture recorded (old CSVs do not carry one). "
                                     f"Pass --posture.")
                data.setdefault(posture, OrderedDict()).setdefault(label, []).append(rate)
                if steps is not None:
                    horizons.add(int(steps))
    return data, labels, horizons


def bar_colors(labels):
    if all(label in AGE_COLORS for label in labels):
        return [AGE_COLORS[label] for label in labels]
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list('mimo_ramp', RAMP)
    if len(labels) == 1:
        return [ramp(1.0)]
    return [ramp(i / (len(labels) - 1)) for i in range(len(labels))]


def percentage(rates, threshold):
    """Share of models that clear the threshold, in percent. Strictly greater, as in the notebook."""
    rates = np.asarray(rates, dtype=float)
    return 100.0 * float((rates > threshold).sum()) / rates.size


def bar_height(rates, args):
    """(height, error) for one bar, in percent.

    'successful' is the notebook's metric: how many seeds cleared the threshold. It is a count
    over seeds, so with six runs it can only take seven values -- 'roll_rate' shows the mean roll
    rate with its spread across seeds instead, which separates a batch that just misses the
    threshold from one that does not roll at all.
    """
    rates = np.asarray(rates, dtype=float)
    if args.metric == 'roll_rate':
        return 100.0 * float(rates.mean()), 100.0 * float(rates.std())
    return percentage(rates, args.threshold), None


def draw(data, labels, args):
    postures = [p for p in POSTURES if p in data] + [p for p in data if p not in POSTURES]
    colors = bar_colors(labels)
    width = args.panel_width or (10 * 0.24 * max(len(labels), 2) / 4)
    fig, axes = plt.subplots(1, len(postures), figsize=(width * len(postures), args.height),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, posture in zip(axes, postures):
        for i, label in enumerate(labels):
            rates = data[posture].get(label)
            if not rates:
                continue                        # this group was not evaluated in this posture
            height, error = bar_height(rates, args)
            ax.bar(x=i, height=height, width=0.8, color=colors[i], zorder=0)
            if error is not None:
                ax.errorbar(i, height, yerr=error, fmt='none', ecolor='#555555',
                            capsize=4, linewidth=1.2, zorder=2)
            if args.annotate:
                successful = int((np.asarray(rates) > args.threshold).sum())
                ax.text(i, height + (error or 0) + 2, f"{successful}/{len(rates)}",
                        ha='center', va='bottom', fontsize=8, color='#333333')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(args.xlabel)
        ax.set_ylim(0, 112 if args.annotate else 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(postures) > 1:
            ax.set_title(posture.capitalize(),
                         bbox=dict(facecolor='#f0f0f0', edgecolor='none',
                                   boxstyle='round,pad=0.3'),
                         fontsize=12, color='#333333')
    axes[0].set_ylabel(args.ylabel)
    if args.title:
        fig.suptitle(args.title)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='append', default=[], metavar='LABEL=PATH',
                        help="One evaluation file, optionally labelled ('9=eval/supine_age9.json'). "
                             "Repeat for every bar. Several comma-separated paths under one label "
                             "are pooled into a single bar. The label defaults to the file name; "
                             "the posture comes from the file. Old "
                             "'..._test_success_rate.csv' files are accepted with --posture.")
    parser.add_argument('--dir', action='append', default=[], metavar='PATH',
                        help="Every '*.json' directly inside this directory becomes a bar. "
                             "Repeatable, and combinable with --json. Use --label_regex to turn "
                             "the file names into short labels.")
    parser.add_argument('--label_regex', default=None, metavar='RE',
                        help="Take each bar's label from this regex's first capture group, "
                             "matched against the batch's save-path name -- e.g. 'ep(\\d+)' turns "
                             "'26-08-19_supine_sac_her_ep100' into '100'.")
    parser.add_argument('--sort', default='auto', choices=['auto', 'given', 'numeric', 'name'],
                        help="Bar order. 'auto' (default) sorts numerically when every label is a "
                             "number and otherwise keeps the order the files came in.")
    parser.add_argument('--metric', default='successful', choices=['successful', 'roll_rate'],
                        help="Bar height: 'successful' (default) is the share of models above "
                             "--threshold, the notebook's metric; 'roll_rate' is the mean roll "
                             "rate across models with its standard deviation as an error bar.")
    parser.add_argument('--posture', default=None, choices=list(POSTURES),
                        help="Override the posture recorded in the files. Required for CSV input.")
    parser.add_argument('--threshold', default=DEFAULT_THRESHOLD, type=float,
                        help="A model counts as successful when it rolls in more than this "
                             "fraction of its episodes (default 0.75, the thesis standard).")
    parser.add_argument('--out', default='eval_success.pdf',
                        help="Output file. The extension picks the format (.pdf for the thesis).")
    parser.add_argument('--xlabel', default='Age [months]')
    parser.add_argument('--ylabel', default=None,
                        help="Defaults to the metric's own name.")
    parser.add_argument('--title', default=None)
    parser.add_argument('--annotate', action='store_true',
                        help="Print 'successful/total' above each bar. A bar built from 3 seeds "
                             "and one built from 18 look identical otherwise.")
    parser.add_argument('--panel_width', default=None, type=float,
                        help="Width of one posture panel in inches. Defaults to the ICDL "
                             "notebooks' 2.4 in for four bars, scaled with the bar count.")
    parser.add_argument('--height', default=3.0, type=float)
    args = parser.parse_args()

    if not args.json and not args.dir:
        parser.error("Pass at least one --json or --dir.")
    if args.ylabel is None:
        args.ylabel = ('Mean roll rate [%]' if args.metric == 'roll_rate'
                       else 'Successful models [%]')

    try:
        data, labels, horizons = collect(expand_sources(args), args.posture)
        labels = sort_labels(labels, args.sort)
    except ValueError as error:
        parser.error(str(error))

    print(f"threshold           : rolls in more than {args.threshold * 100:.0f} % of episodes")
    if len(horizons) > 1:
        print(f"WARNING             : the batches were evaluated over different horizons "
              f"({sorted(horizons)} steps), so their bars are not directly comparable. "
              f"Re-run eval_rollover.py --group with a common --episode_steps.")
    elif horizons:
        print(f"evaluation horizon  : {horizons.pop()} steps for every batch")
    for posture, groups in data.items():
        print(f"\n{posture}")
        for label in labels:
            rates = groups.get(label)
            if not rates:
                print(f"  {label:<12}  (not evaluated)")
                continue
            successful = int((np.asarray(rates) > args.threshold).sum())
            print(f"  {label:<12}  {successful:>3} / {len(rates):<3} models "
                  f"({percentage(rates, args.threshold):5.1f} %)   "
                  f"mean roll rate {np.mean(rates) * 100:5.1f} %")

    matplotlib.rcParams.update(STYLE)
    fig = draw(data, labels, args)
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\nwritten: {args.out}")


if __name__ == '__main__':
    main()
