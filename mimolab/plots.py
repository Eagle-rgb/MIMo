"""Server-side chart rendering.

Charts are PNGs from matplotlib rather than a JS charting library: the same code that draws a
curve here is the code in results/, so a curve in the app and a curve in the thesis cannot drift.

Colour follows the dataviz reference palette, in fixed slot order, never cycled. These are line
charts, so the adjacent pairlist applies and all eight slots are usable; the three light-mode
slots that sit below 3:1 on the surface are covered by the relief rule -- every series is direct-
labelled and a legend is always present, so identity is never carried by colour alone.
"""

import contextlib
import contextvars
import io
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import db
from .config import SETTINGS

# Categorical slots 1-8, in the fixed order the palette defines.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# Sequential blue, 100 -> 700, for magnitude (the age grid).
SEQ_LIGHT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

THEMES = {
    "light": {"ink": "#131920", "muted": "#5C6472", "faint": "#9AA3AE",
              "grid": "#DFE3E8", "surface": "#FFFFFF", "series": SERIES_LIGHT},
    "dark": {"ink": "#E2E7EC", "muted": "#A3AEBA", "faint": "#6E7883",
             "grid": "#2C363F", "surface": "#161D24", "series": SERIES_DARK},
}

HEADLINE_TAG = "rollout/ep_rho_max_mean"

# Paper style for exported figures, mirroring results/icdlplot.py so a figure exported here and a
# figure produced by the analysis scripts sit on the same page without looking like two documents.
# fonttype 42 embeds TrueType rather than matplotlib's default Type 3, which several thesis and
# journal templates reject outright.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# results/icdlplot.py: single column is 3.5 in wide. A chart on screen is much wider than that,
# so the export is not the screen figure re-encoded -- it is re-laid out at the target width.
COLUMN_WIDTHS = {"single": (3.5, 2.6), "double": (7.0, 3.9)}

# Render options for the request being served. A ContextVar rather than module state because
# matplotlib's rcParams are global and FastAPI serves sync handlers on a threadpool: two
# concurrent exports would otherwise fight over the style.
_OPTS = contextvars.ContextVar("mimolab_render_opts", default=None)


def _opts():
    return _OPTS.get() or {"fmt": "png", "paper": False, "size": None}


@contextlib.contextmanager
def render_as(fmt="png", column=None):
    """Render everything inside this block as 'fmt', optionally at a fixed column width."""
    paper = fmt == "pdf"
    size = COLUMN_WIDTHS.get(column) if column else None
    token = _OPTS.set({"fmt": fmt, "paper": paper, "size": size})
    try:
        if paper:
            with plt.rc_context(PAPER_RC):
                yield
        else:
            yield
    finally:
        _OPTS.reset(token)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "chart").lower()).strip("_")


# Model names in this fork run long ("sac_her_sparse_goal_lohi_curriculum_nodone_entropy92_
# lrlinearsched"). Untruncated they drag the figure to three times its width, because
# bbox_inches="tight" expands to fit the direct labels.
LABEL_MAX = 26


def shorten(name, limit=LABEL_MAX):
    if name is None:
        return "?"
    if len(name) <= limit:
        return name
    return name[:limit - 1] + "\u2026"


def elide_middle(name, limit=LABEL_MAX):
    """Shorten from the middle, keeping both ends.

    Names here differ at the tail as often as at the head ('..._curriculum' vs
    '..._curriculum_nodone' vs '..._nodone_entropy92_lrlinearsched'), so left-anchored truncation
    collapses distinct experiments to the same label.
    """
    if name is None:
        return "?"
    if len(name) <= limit:
        return name
    head = max(4, (limit - 1) * 4 // 10)
    tail = limit - 1 - head
    return name[:head] + "\u2026" + name[-tail:]


def distinguish(names):
    """Label runs by what differs between them, not by what they share.

    Model names in this fork are built by accretion, so a selection often shares a long prefix
    ('sac_her_sparse_goal_lohi_curriculum...'). Truncating those from the left yields three
    identical labels. Strip the common leading and trailing underscore-tokens instead, and label
    with the part that actually distinguishes the experiments.
    """
    names = list(names)
    if len(names) < 2:
        return {n: shorten(n) for n in names}

    split = [(n or "").split("_") for n in names]
    head = 0
    while all(len(t) > head + 1 for t in split) and len({t[head] for t in split}) == 1:
        head += 1
    tail = 0
    while (all(len(t) > head + tail + 1 for t in split)
           and len({t[len(t) - 1 - tail] for t in split}) == 1):
        tail += 1

    out = {}
    for name, tokens in zip(names, split):
        core = "_".join(tokens[head:len(tokens) - tail]) or name
        out[name] = elide_middle(core)
    return out

# Tags whose meaning is easy to misread. Surfaced as a caption on the chart itself, because the
# mistake this prevents (reading success_rate as a roll rate) already happened.
TAG_NOTES = {
    "rollout/success_rate":
        "scored against the SAMPLED goal, which can be as low as 0.25 -- not a roll rate",
    "rollout/ep_rho_max_mean":
        "episode maximum rotation; the quantity comparable to eval_rollover.py",
    "rollout/goal_high_effective":
        "goal curriculum, removed 25.08.2026 -- only runs trained before then have this curve",
    "rollout/side_lying_success_rate":
        "final step of the episode, not the maximum",
}


def _figure(theme, width=9.0, height=4.6):
    t = THEMES[theme]
    size = _opts()["size"]
    if size:
        # Keep the caller's aspect ratio, but honour the requested column width.
        width, height = size[0], size[0] * (height / width)
    # Paper figures are laid out constrained so everything fits *inside* the requested size.
    # With bbox_inches="tight" the page grows to whatever the legend and labels need, and a
    # figure asked for at 3.5 in comes out at 7.65 in -- useless for a column.
    fig, ax = plt.subplots(figsize=(width, height), dpi=110,
                           layout="constrained" if _opts()["paper"] else None)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=3, width=0.8)
    ax.grid(True, color=t["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax, t


def _emit(fig):
    opts = _opts()
    buf = io.BytesIO()
    if opts["fmt"] == "pdf":
        # Opaque white, not transparent: a transparent PDF dropped into a document picks up
        # whatever is behind it, and the axis labels are near-black.
        fig.savefig(buf, format="pdf", dpi=300, facecolor="white", edgecolor="none")
    else:
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return buf.getvalue()


def _placeholder(theme, message):
    fig, ax, t = _figure(theme, height=2.2)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=t["muted"], fontsize=10)
    return _emit(fig)


def _smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _common_grid(series, points=200):
    """Resample runs onto one step grid so seeds with different logging cadence can be averaged."""
    lo = max(s[0][0] for s in series)
    hi = min(s[0][-1] for s in series)
    if hi <= lo:
        return None, None
    grid = np.linspace(lo, hi, points)
    stacked = np.vstack([np.interp(grid, np.asarray(steps), np.asarray(vals))
                         for steps, vals in series])
    return grid, stacked


def curve(run_ids, tag, theme="light", aggregate=False, smooth=1, title=None):
    """Training curves for a set of runs.

    aggregate=False draws one line per run; aggregate=True collapses each experiment group
    (same date/posture/model name) to a mean with a min-max band across its seeds.
    """
    series = []
    for run_id in run_ids:
        steps, vals = db.series(run_id, tag)
        if steps:
            series.append((run_id, steps, vals))

    if not series:
        return _placeholder(theme, f"no '{tag}' logged for this selection")

    fig, ax, t = _figure(theme)
    colors = t["series"]

    if aggregate:
        groups = {}
        for run_id, steps, vals in series:
            row = db.one("SELECT date, posture, model_name FROM runs WHERE run_id=?", (run_id,))
            key = (row["date"], row["posture"], row["model_name"]) if row else (None, None, run_id)
            groups.setdefault(key, []).append((steps, vals))
        items = list(groups.items())
        names = distinguish([k[2] for k in groups])
        # A model_name can repeat across dates ('ep100' on 26-08-21 and 26-08-22) and across
        # postures ('ep100_18' prone and supine on the same day). Labelling on the name alone
        # paints two different experiments identically, which is worse than a long label -- so
        # append whichever key components actually vary inside each colliding set, and nothing
        # more. A group key is (date, posture, model_name).
        seen = {}
        for key in groups:
            seen.setdefault(names.get(key[2], key[2]), []).append(key)
        labels = {}
        for base, keys in seen.items():
            if len(keys) == 1:
                labels[keys[0]] = base
                continue
            vary_posture = len({k[1] for k in keys}) > 1
            vary_date = len({k[0] for k in keys}) > 1
            for key in keys:
                extra = []
                if vary_posture:
                    extra.append(key[1])
                if vary_date:
                    extra.append(key[0] or "?")
                labels[key] = " ".join([base] + extra) if extra else base
        # A 9th series is never a generated hue: fold the tail rather than cycling the palette.
        folded = items[len(colors):]
        items = items[:len(colors)]
        for i, (key, members) in enumerate(items):
            grid, stacked = _common_grid(members)
            if grid is None:
                continue
            mean = _smooth(stacked.mean(axis=0), smooth)
            # 'ep100_18 prone' already says prone; only add the posture when it is not there.
            base_label = labels[key]
            detail = f"n={len(members)}" if key[1] in base_label else f"{key[1]}, n={len(members)}"
            label = f"{base_label} ({detail})"
            ax.fill_between(grid, stacked.min(axis=0), stacked.max(axis=0),
                            color=colors[i], alpha=0.15, linewidth=0)
            ax.plot(grid, mean, color=colors[i], linewidth=2.0, label=label,
                    solid_capstyle="round")
            if len(items) <= 4 and not _opts()["paper"]:
                ax.annotate(labels[key], xy=(grid[-1], mean[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            color=colors[i], fontsize=8, va="center", fontweight="semibold")
        if folded:
            n = sum(len(m) for _, m in folded)
            ax.plot([], [], color=t["faint"], linewidth=2.0,
                    label=f"+{len(folded)} more groups ({n} runs) not shown")
    else:
        shown = series[:len(colors)]
        meta = {run_id: db.one(
            "SELECT model_name, seed_idx, posture FROM runs WHERE run_id=?", (run_id,))
            for run_id, _, _ in shown}
        names = distinguish([(meta[r]["model_name"] if meta[r] else r) for r, _, _ in shown])
        for i, (run_id, steps, vals) in enumerate(shown):
            row = meta[run_id]
            label = run_id
            if row:
                seed = "" if row["seed_idx"] is None else f" #{row['seed_idx']}"
                label = f"{names.get(row['model_name'], shorten(row['model_name']))}{seed}"
            ax.plot(np.asarray(steps), _smooth(np.asarray(vals), smooth),
                    color=colors[i], linewidth=1.8, label=label, solid_capstyle="round")
        if len(series) > len(shown):
            ax.plot([], [], color=t["faint"], linewidth=1.8,
                    label=f"+{len(series) - len(shown)} more runs not shown")

    paper = _opts()["paper"]
    ax.set_xlabel("environment steps", color=t["muted"], fontsize=None if paper else 9)
    ax.set_ylabel(tag.split("/")[-1], color=t["muted"], fontsize=None if paper else 9)
    ax.margins(x=0.02)
    # The italic caption is on-screen guidance about how to read the metric. A figure in a
    # document gets its caption from the document, and at 3.5 in the line runs off the page.
    note = None if paper else TAG_NOTES.get(tag)
    ax.set_title(title or tag, color=t["ink"], fontsize=None if paper else 11.5, loc="left",
                 pad=26 if note else 12, fontweight="semibold")
    if note:
        ax.text(0, 1.015, note, transform=ax.transAxes, color=t["muted"], fontsize=8.2,
                va="bottom", style="italic")

    if tag == HEADLINE_TAG:
        ax.axhline(0.95, color=t["faint"], linewidth=1.0, linestyle=(0, (4, 3)))
        # Right-aligned: the upper left is where the earliest data and the busiest labels sit.
        ax.text(0.998, 0.95, "roll threshold 0.95 ", transform=ax.get_yaxis_transform(),
                color=t["muted"], fontsize=8, va="bottom", ha="right")

    # Legend below the axes, never over the data.
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        if paper:
            # "outside" is what makes constrained layout *reserve* room for the legend. With the
            # axes-anchored version it lands on top of the x-axis label at column width.
            # A single column stacks the entries; at double width they fit side by side.
            wide = (_opts()["size"] or (7.0,))[0] >= 6.0
            legend = fig.legend(handles, legend_labels, loc="outside lower center", frameon=False,
                                ncols=min(3, len(handles)) if wide else 1,
                                handlelength=1.4, columnspacing=1.0)
        else:
            legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.2,
                               frameon=False, ncols=min(4, len(handles)), handlelength=1.6,
                               columnspacing=1.6, borderaxespad=0)
        for text in legend.get_texts():
            text.set_color(t["muted"])
    return _emit(fig)


def goal_response(rows, theme="light", title=None):
    """Policy response to a lied-to desired_goal -- the output of --policy_goal_sweep.

    'rows' is [(fed_goal, rolled, side, rho_mean), ...].
    """
    if not rows:
        return _placeholder(theme, "no sweep results yet")

    rows = sorted(rows)
    x = np.array([r[0] for r in rows])
    fig, ax, t = _figure(theme, height=4.2)
    colors = t["series"]

    ax.plot(x, [r[3] for r in rows], color=colors[0], linewidth=2.2, marker="o",
            markersize=5, label="rho_max (mean)", solid_capstyle="round")
    ax.plot(x, [r[1] for r in rows], color=colors[1], linewidth=2.0, marker="s",
            markersize=4.5, label="roll rate", solid_capstyle="round")
    ax.plot(x, [r[2] for r in rows], color=colors[2], linewidth=2.0, marker="^",
            markersize=4.5, label="side-lying rate", solid_capstyle="round")

    ax.axvline(0.95, color=t["faint"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.95, 1.005, " real goal", transform=ax.get_xaxis_transform(),
            color=t["muted"], fontsize=8, va="bottom")

    ax.set_xlabel("desired_goal fed to the policy", color=t["muted"], fontsize=9)
    ax.set_ylabel("score", color=t["muted"], fontsize=9)
    ax.set_title(title or "Goal response", color=t["ink"], fontsize=11.5, loc="left",
                 pad=14, fontweight="semibold")
    ax.text(0, 1.02, "the environment keeps the real goal, so rho_max stays honest",
            transform=ax.transAxes, color=t["muted"], fontsize=8.2, va="bottom", style="italic")
    legend = ax.legend(loc="best", fontsize=8.2, frameon=False)
    for text in legend.get_texts():
        text.set_color(t["muted"])
    return _emit(fig)


def age_grid(grid, theme="light", ages=(1, 3, 6, 9), title=None):
    """morph x physio as a heatmap. Sequential single hue: this is magnitude, not identity."""
    from matplotlib.colors import LinearSegmentedColormap

    ages = list(ages)
    matrix = np.full((len(ages), len(ages)), np.nan)
    counts = np.zeros((len(ages), len(ages)), dtype=int)
    for i, morph in enumerate(ages):
        for j, physio in enumerate(ages):
            cell = grid.get(morph, {}).get(physio)
            if cell and cell.get("rho") is not None:
                matrix[i, j] = cell["rho"]
                counts[i, j] = cell["n"]

    if np.all(np.isnan(matrix)):
        return _placeholder(theme, "no runs with both ages recorded in this selection")

    fig, ax, t = _figure(theme, width=5.6, height=4.8)
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_LIGHT)
    cmap.set_bad(t["grid"])
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(ages)):
        for j in range(len(ages)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color=t["muted"], fontsize=9)
                continue
            # Ink flips on the dark half of the ramp so the label stays legible on its cell.
            ink = "#FFFFFF" if matrix[i, j] > 0.55 else "#0d366b"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=ink,
                    fontsize=10, fontweight="semibold")
            ax.text(j, i + 0.28, f"n={counts[i, j]}", ha="center", va="center", color=ink,
                    fontsize=7.5, alpha=0.85)

    ax.set_xticks(range(len(ages)), [f"{a} mo" for a in ages])
    ax.set_yticks(range(len(ages)), [f"{a} mo" for a in ages])
    ax.set_xlabel("physiological age (actuation)", color=t["muted"], fontsize=9)
    ax.set_ylabel("morphological age (body)", color=t["muted"], fontsize=9)
    ax.set_title(title or "Best rho_max by embodiment", color=t["ink"], fontsize=11.5,
                 loc="left", pad=14, fontweight="semibold")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=t["muted"], labelsize=8)
    return _emit(fig)


def horizon_warning(run_ids):
    """Runs at different episode_steps are scored over lengths they never saw. Say so."""
    rows = db.query(
        f"SELECT DISTINCT episode_steps FROM runs WHERE run_id IN ({','.join('?' * len(run_ids))})",
        run_ids) if run_ids else []
    horizons = sorted({(r["episode_steps"] or 500) for r in rows})
    if len(horizons) > 1:
        return ("This selection mixes episode horizons (" +
                ", ".join(str(h) for h in horizons) +
                " steps). Curves are not directly comparable across them.")
    return None
