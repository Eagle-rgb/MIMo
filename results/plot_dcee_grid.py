"""DCEE grid: one policy scored on every embodiment.

The figure of 'results/cee/dcee_success.ipynb' -- a square per (actuation age, body age), coloured
by how many of the experiment's seeds clear the success line -- but driven by the JSON that
'mimoEnv/eval_rollover.py --group --embodiment_grid --json=...' writes, instead of by sixteen
'..._test_success_rate.csv' files whose names had to be looked up in a hand-maintained table.

    MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \\
        --group=models/roll_over/26-03-09/supine/26-03-09_supine_age1 \\
        --embodiment_grid --episodes=40 --json=eval/dcee_age1.json

    python results/plot_dcee_grid.py --json eval/dcee_age1.json --out dcee_age1.pdf

Several payloads become the notebook's four-panel figure, one panel per source embodiment, in the
order given:

    python results/plot_dcee_grid.py --json eval/dcee_age{1,3,6,9}.json \\
        --title 'Age of learned embodiment (supine to prone)' --out dcee_supine.pdf

Success is recomputed from each run's roll rate rather than read off the 'successful' flag, so
--threshold can be changed after the evaluation without re-running it.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# The house style, applied on import. Layered over by --rcparams so that mimolab can hand this
# script the rcParams from its Settings dialog and get the same figure it shows on screen.
try:
    from results import icdlplot                          # noqa: F401  (sets rcParams)
    COLORS = icdlplot.COLORS
except ImportError:                                       # run from inside results/
    import icdlplot                                       # noqa: F401
    COLORS = icdlplot.COLORS

DEFAULT_THRESHOLD = 0.75
# The notebook's ramp: the 1-month grey up to the 9-month orange, i.e. the same two inks the age
# figures use, so a DCEE panel sits beside them without introducing a third palette.
RAMP = (COLORS[1], COLORS[9])
CELL = 0.8                                                # square size, on a 1.0 grid pitch


def read_payload(path):
    """(cells, ages, source, posture) from one --embodiment_grid payload."""
    with open(path) as fh:
        payload = json.load(fh)
    grid = payload.get('embodiment_grid')
    if not grid:
        raise ValueError(
            f"{path}: not an --embodiment_grid payload. Produce one with "
            f"'eval_rollover.py --group=... --embodiment_grid --json={os.path.basename(path)}'.")
    cells = {}
    for cell in payload.get('cells', []):
        rows = [row for row in cell.get('rows', []) if row.get('rolled') is not None]
        if rows:
            cells[(cell['age_physio'], cell['age_morph'])] = [float(r['rolled']) for r in rows]
    if not cells:
        raise ValueError(f"{path}: the grid has no evaluated cell.")
    source = grid.get('source') or {}
    postures = {row.get('starting_position') for row in payload.get('rows', [])}
    posture = postures.pop() if len(postures) == 1 else None
    return cells, list(grid['ages']), (source.get('physio'), source.get('morph')), posture


def default_title(source):
    """The panel title: the source embodiment, as the notebook writes it when both ages agree."""
    physio, morph = source
    if physio is None or morph is None:
        return ''
    if physio == morph:
        return f"{format(physio, 'g')} {'Month' if physio == 1 else 'Months'}"
    return f"act {format(physio, 'g')} / body {format(morph, 'g')}"


def cell_value(rates, metric, threshold):
    """The number the colour encodes, in [0, 1]."""
    if metric == 'roll_rate':
        return float(np.mean(rates))
    return float(np.mean([rate > threshold for rate in rates]))


def draw_panel(ax, cells, ages, source, cmap, metric, threshold, label_axes, title):
    for i, age_physio in enumerate(ages):
        for j, age_morph in enumerate(ages):
            rates = cells.get((age_physio, age_morph))
            if rates is None:
                # An empty cell is left empty rather than painted at 0 -- "not evaluated" and
                # "no seed rolled" are different claims and the ramp cannot say both.
                ax.add_patch(plt.Rectangle((j, i), CELL, CELL, facecolor='none',
                                           edgecolor=COLORS[1], linewidth=0.6, linestyle=':'))
                continue
            ax.add_patch(plt.Rectangle((j, i), CELL, CELL,
                                       color=cmap(cell_value(rates, metric, threshold))))
    # The embodiment the policy was trained on: every other cell is read against it, and on a
    # single panel nothing else says where on the grid the run came from.
    if source in [(p, m) for p in ages for m in ages]:
        i, j = ages.index(source[0]), ages.index(source[1])
        ax.add_patch(plt.Rectangle((j, i), CELL, CELL, facecolor='none',
                                   edgecolor='#333333', linewidth=1.2, zorder=3))

    ax.set_xlim(-0.2, len(ages) + 0.2)
    ax.set_ylim(-0.2, len(ages) + 0.2)
    ax.set_xticks(np.arange(len(ages)) + CELL / 2)
    ax.set_yticks(np.arange(len(ages)) + CELL / 2)
    ax.set_xticklabels([format(age, 'g') for age in ages])
    ax.set_yticklabels([format(age, 'g') for age in ages])
    ax.set_aspect('equal')
    if label_axes:
        ax.set_xlabel("Body Age [months]")
        ax.set_ylabel("Actuation Age [months]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, bbox=dict(facecolor='#f0f0f0', edgecolor='none',
                                      boxstyle='round,pad=0.3'),
                     color='#333333')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='append', required=True, metavar='PATH',
                        help="An --embodiment_grid payload. Repeat for one panel per source "
                             "embodiment, in the order given.")
    parser.add_argument('--out', required=True, help="Output file; the suffix picks the format.")
    parser.add_argument('--metric', default='successful', choices=['successful', 'roll_rate'],
                        help="'successful' (default) colours by the share of seeds above "
                             "--threshold; 'roll_rate' by the mean roll rate over seeds.")
    parser.add_argument('--threshold', default=DEFAULT_THRESHOLD, type=float,
                        help="Roll rate above which a seed counts as successful (default 0.75).")
    parser.add_argument('--width', default=None, type=float,
                        help="Figure width in inches. Default 2.8 for one panel (half a text "
                             "block), 5.6 for more than one.")
    parser.add_argument('--height', default=None, type=float, help="Figure height in inches.")
    parser.add_argument('--title', default=None, help="Figure title above the panels.")
    parser.add_argument('--panel_titles', default=None,
                        help="Comma-separated titles, one per panel. 'auto' uses each payload's "
                             "source embodiment, which is the default for a multi-panel figure "
                             "-- there the title is the only thing telling the panels apart. A "
                             "single panel gets no title unless one is asked for: it would "
                             "repeat what the caption already says.")
    parser.add_argument('--no_colorbar', action='store_true')
    parser.add_argument('--cbar_fraction', default=None, type=float,
                        help="Width of the colorbar as a fraction of the axes (matplotlib's "
                             "'fraction'). Default 0.05 for one panel, 0.02 for several -- the "
                             "notebook's 0.02 is sized for four panels across a text block and "
                             "is a hairline on a single one.")
    parser.add_argument('--cbar_label', default=None,
                        help="Colorbar label. Default follows --metric.")
    parser.add_argument('--rcparams', default=None, metavar='JSON',
                        help="matplotlib rcParams as a JSON object, applied over the house "
                             "style. This is how mimolab hands the script the settings it was "
                             "configured with, so the exported figure matches the app.")
    args = parser.parse_args()

    if args.rcparams:
        matplotlib.rcParams.update(json.loads(args.rcparams))
    # icdlplot.py sets savefig.bbox='tight', which crops the page to whatever the content needs
    # and would return some width other than the one asked for. A figure that goes into the
    # document at a fixed \includegraphics width has to come out at that width.
    matplotlib.rcParams['savefig.bbox'] = None

    panels = [read_payload(path) for path in args.json]
    if args.panel_titles == 'auto' or (args.panel_titles is None and len(panels) > 1):
        titles = [default_title(src) for _c, _a, src, _p in panels]
    elif args.panel_titles:
        titles = args.panel_titles.split(',')
    else:
        titles = [''] * len(panels)

    width = args.width if args.width is not None else (2.8 if len(panels) == 1 else 5.6)
    height = args.height if args.height is not None else (2.8 if len(panels) == 1 else 3.0)
    cmap = mcolors.LinearSegmentedColormap.from_list('dcee', list(RAMP))

    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True,
                             constrained_layout=True, squeeze=False)
    axes = list(axes[0])
    for index, ((cells, ages, source, _posture), ax) in enumerate(zip(panels, axes)):
        draw_panel(ax, cells, ages, source, cmap, args.metric, args.threshold,
                   label_axes=(index == 0), title=titles[index] if index < len(titles) else '')

    if not args.no_colorbar:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
        mappable.set_array([])
        label = args.cbar_label or ("Mean Roll Rate [%]" if args.metric == 'roll_rate'
                                   else "Success Rate [%]")
        fraction = (args.cbar_fraction if args.cbar_fraction is not None
                    else (0.05 if len(panels) == 1 else 0.02))
        cbar = fig.colorbar(mappable, ax=axes, orientation='vertical', fraction=fraction,
                            location='right', pad=0.01)
        # The notebook pulls the label onto the bar with labelpad=-20; at a single-panel width
        # that lands it on top of the 0%/100% ticks, so it sits just clear of them instead.
        cbar.set_label(label, rotation=90, labelpad=2)
        cbar.outline.set_visible(False)
        cbar.set_ticks([0, 100])
        cbar.set_ticklabels(['0%', '100%'])

    if args.title:
        fig.suptitle(args.title)

    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({width} x {height} in, {len(panels)} panel(s))")


if __name__ == '__main__':
    main()
