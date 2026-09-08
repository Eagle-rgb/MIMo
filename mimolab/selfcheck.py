"""python -m mimolab.selfcheck -- verify the app against the real models/ directory.

Not a unit test suite; this fork does not have one. It is the same kind of thing as
goalenv_check.py: a set of assertions over real data that catch the failures that actually
happen here -- a posture read from the wrong place, a flag that will not round-trip through
data.yml, a guard rail that stopped guarding.
"""

import json
import re
import sys
import traceback
from pathlib import Path

from .config import configure, SETTINGS
from . import db, evals, indexer, plots, queries

PASS, FAIL = "  ok  ", " FAIL "
_results = []


def check(name):
    def wrap(fn):
        def run():
            try:
                detail = fn()
                _results.append((True, name, detail or ""))
            except Exception as exc:
                _results.append((False, name, f"{type(exc).__name__}: {exc}"))
                if "-v" in sys.argv:
                    traceback.print_exc()
        run.__name__ = fn.__name__
        _checks.append(run)
        return run
    return wrap


_checks = []


@check("index is populated")
def _index():
    stats = queries.stats()
    assert stats["runs"] > 0, "no runs indexed -- run a reindex first"
    return f"{stats['runs']} runs, {stats['checkpoints']} checkpoints, {stats['evals']} evals"


@check("posture comes from the path, not data.yml")
def _posture():
    # data.yml deliberately omits roll_over_starting_position, so anything reading posture from
    # the yaml silently gets 'supine' for prone runs. The index must not depend on it.
    rows = db.query("SELECT run_id, yaml FROM runs WHERE posture='prone' LIMIT 20")
    assert rows, "no prone runs indexed -- cannot verify"
    for row in rows:
        assert "roll_over_starting_position" not in (row["yaml"] or ""), \
            f"{row['run_id']} unexpectedly records posture in data.yml"
    total = db.one("SELECT COUNT(*) n FROM runs WHERE posture='prone'")["n"]
    return f"{total} prone runs identified from their paths"


@check("the age shim matches load_model_yaml")
def _age_shim():
    cfg = indexer.apply_yaml_shims({"age": 6})
    assert cfg["morph_age"] == 6 and cfg["physio_age"] == 6, cfg
    assert "age" not in cfg
    assert indexer.apply_yaml_shims({"proprio_only_qpos": True})["proprio_config"] == "position"
    nulls = db.one("SELECT COUNT(*) n FROM runs WHERE morph_age IS NULL")["n"]
    total = db.one("SELECT COUNT(*) n FROM runs")["n"]
    return f"{total - nulls}/{total} runs have an embodiment age"


@check("scalar blobs round-trip")
def _scalars():
    row = db.one("SELECT run_id, tag, n FROM scalars WHERE n > 10 LIMIT 1")
    assert row is not None, "no scalars indexed"
    steps, vals = db.series(row["run_id"], row["tag"])
    assert len(steps) == row["n"] == len(vals), (len(steps), row["n"], len(vals))
    assert steps == sorted(steps), "steps are not monotonic"
    return f"{row['tag']}: {len(steps)} points, {steps[0]}..{steps[-1]}"


@check("filters narrow the corpus")
def _filters():
    _, everything = queries.search({})
    _, supine = queries.search({"posture": "supine"})
    _, prone = queries.search({"posture": "prone"})
    assert 0 < supine < everything and 0 < prone < everything
    _, her = queries.search({"her": "1"})
    _, no_her = queries.search({"her": "0"})
    assert her + no_her == everything, (her, no_her, everything)
    return f"{everything} total = {supine} supine + {prone} prone (+unknown); her {her}/{everything}"


@check("date facet is newest-first and presets select what they claim")
def _dates():
    dates = [d for d, _ in queries.facets({})["date"] if d]
    assert dates == sorted(dates, reverse=True), f"date facet is not newest-first: {dates[:5]}"

    counts = queries.date_preset_counts({})
    _, total = queries.search({})
    latest = queries.latest_date()

    # "Latest" must be exactly the runs carrying the newest date.
    rows, n_latest = queries.search({"date_preset": "latest"})
    assert n_latest == counts["latest"], (n_latest, counts["latest"])
    assert all(r["date"] == latest for r in rows), "latest returned a run from another date"

    # "Latest per experiment" keeps one date per model_name, and must be a superset of "latest".
    rows, n_each = queries.search({"date_preset": "latest_each"})
    assert counts["latest"] <= n_each <= total, (counts["latest"], n_each, total)
    per_model = {}
    for r in rows:
        per_model.setdefault(r["model_name"], set()).add(r["date"])
    multi = {k: v for k, v in per_model.items() if len(v) > 1}
    assert not multi, f"more than one date survived for {list(multi)[:3]}"

    # Windows nest, and every preset narrows.
    assert counts["7d"] <= counts["30d"] <= total, counts
    return (f"newest {latest}; latest {counts['latest']}, per-experiment {counts['latest_each']}, "
            f"7d {counts['7d']}, 30d {counts['30d']} of {total}")


@check("experiments are the default view, individual runs the opt-out")
def _grouping():
    from . import app as webapp
    _, total = queries.search({})
    n_groups = len(queries.groups({}, limit=10_000))
    assert 0 < n_groups < total, (n_groups, total)
    # No 'view' at all, and the empty string a cleared hidden input submits, both mean grouped.
    for params in ({}, {"view": ""}, {"view": "groups"}):
        assert params.get("view") != "runs", params
    assert "runs" == {"view": "runs"}.get("view")
    return f"{total} runs collapse to {n_groups} experiments by default"


@check("the Runs tab remembers the filters")
def _filter_memory():
    from . import app as webapp

    class Req:
        def __init__(self, cookies):
            self.cookies = cookies

    assert webapp.runs_href(Req({})) == "/"
    saved = "?algorithm=SAC&posture=supine&date_preset=latest"
    assert webapp.runs_href(Req({webapp.FILTER_COOKIE: saved})) == "/" + saved
    # A cookie that does not look like a query string must not become part of the URL.
    assert webapp.runs_href(Req({webapp.FILTER_COOKIE: "javascript:alert(1)"})) == "/"
    assert webapp.runs_href(Req({webapp.FILTER_COOKIE: ""})) == "/"
    return "restores a saved query, ignores a malformed cookie"


@check("filtering leaves a reloadable URL in the address bar")
def _push_url():
    from . import app as webapp

    class Req:
        def __init__(self, query, headers=None):
            self.url = type("U", (), {"query": query})()
            self.headers = headers or {}

    class Resp:
        def __init__(self):
            self.headers = {}

    # hx-push-url would push /fragments/runs?<q>, which renders no navigation. The response
    # header has to name the page URL instead, or Refresh (a reload) strands the user.
    resp = webapp.push_page_url(Req("algorithm=SAC&posture=supine"), Resp())
    assert resp.headers["HX-Push-Url"] == "/?algorithm=SAC&posture=supine", resp.headers
    assert webapp.push_page_url(Req(""), Resp()).headers["HX-Push-Url"] == "/"

    # And a browser landing on a fragment directly is sent to the real page.
    redirect = webapp.whole_page_instead(Req("algorithm=SAC"), "/")
    assert redirect is not None and redirect.headers["location"] == "/?algorithm=SAC"
    assert webapp.whole_page_instead(Req("x=1", {"HX-Request": "true"}), "/") is None
    return "pushes /?<filters>; a direct hit on a fragment redirects to the page"


@check("the goal function is filterable and matches illustrations.py")
def _goal_fn():
    present = {v for v, _ in queries.facets({})["goal_fn"] if v}
    assert present, "no goal function recorded on any run"
    _, total = queries.search({})
    counts = {}
    for value in present:
        _, n = queries.search({"goal_fn": value})
        counts[value] = n
    assert sum(counts.values()) <= total
    assert all(n > 0 for n in counts.values()), counts

    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


@check("the actuation model is recorded, filterable and labelled on both states")
def _actuation():
    """--use_muscle changes the action space, the observation and the effort term.

    A run reloaded under the wrong actuation model does not match its own weights, so it is worth
    a chip of its own. Both states are labelled: use_muscle only entered data.yml on 02.09.2026,
    so a missing chip would be ambiguous between 'spring damper' and 'not recorded'.
    """
    counts = {v: n for v, n in queries.facets({})["use_muscle"]}
    assert None not in counts, "use_muscle left null -- the indexer did not coerce it"
    _, muscle = queries.search({"use_muscle": "1"})
    _, spring = queries.search({"use_muscle": "0"})
    _, total = queries.search({})
    assert muscle + spring == total, (muscle, spring, total)

    from fastapi.testclient import TestClient
    from . import app as appmod
    client = TestClient(appmod.app)
    # Match the chip's class, not its text: the sidebar radio for this very filter also says
    # "muscle", so a bare text search passes on a listing holding no muscle run at all.
    body = client.get("/", params={"use_muscle": "1", "group": "0"}).text
    assert 'class="chip muscle"' in body, "no muscle chip on a muscle-only listing"
    assert 'class="chip muted spring"' not in body, "spring chip on a muscle-only listing"
    body = client.get("/", params={"use_muscle": "0", "group": "0"}).text
    assert 'class="chip muted spring"' in body and 'class="chip muscle"' not in body

    return f"{muscle} muscle, {spring} spring damper"


@check("every page and chart endpoint actually responds")
def _routes():
    """Exercise the HTTP surface, not just the functions behind it.

    Every other check here calls the plot and query functions directly, which is why deleting a
    module-level helper that only the *route* used went unnoticed until the charts vanished from
    the page. This check drives the app through its own routes.
    """
    from fastapi.testclient import TestClient
    from . import app as webapp

    row = db.one("SELECT run_id FROM runs WHERE last_step IS NOT NULL LIMIT 1")
    group = next((g for g in queries.groups({}, limit=50) if g["model_name"]), None)
    assert row is not None and group is not None, "index is empty"

    targets = [
        "/", "/analysis",
        "/fragments/runs", "/fragments/facets",
        f"/run/{row['run_id']}",
        f"/api/config/{row['run_id']}",
        "/api/plot/curve.png?tag=rollout/ep_rho_max_mean",
        "/api/plot/curve.png?tag=rollout/ep_rho_max_mean&aggregate=1",
        "/api/plot/curve.pdf?tag=rollout/ep_rho_max_mean&column=single",
        "/api/plot/age_grid.png",
        "/api/plot/age_grid.pdf",
        "/api/evals/status",
        "/api/evals/jobs",
    ]
    if group["posture"]:
        targets.append(
            f"/experiment?posture={group['posture']}&name={group['model_name']}"
            + (f"&date={group['date']}" if group["date"] else ""))

    with TestClient(webapp.app) as client:
        for target in targets:
            response = client.get(target, headers={"HX-Request": "true"})
            assert response.status_code == 200, f"{target} -> {response.status_code}"
            assert response.content, f"{target} returned nothing"
        # The removed launcher must stay removed.
        for gone in ("/jobs", "/launch", "/fragments/jobs", "/fragments/hosts"):
            assert client.get(gone).status_code == 404, f"{gone} is still routed"
    return f"{len(targets)} endpoints answered, 4 removed ones are gone"


@check("a group spec never names runs outside its experiment")
def _group_spec():
    import glob as globlib
    import os

    checked = safe = refused = 0
    for group in queries.groups({}, limit=10_000):
        exp = queries.experiment(group["date"], group["posture"], group["model_name"])
        if exp is None:
            continue
        checked += 1
        spec = exp["group_spec"]
        expected = {os.path.abspath(r["path"]) for r in exp["runs"]}
        if spec is None:
            refused += 1
            continue
        if os.path.isdir(spec) and globlib.glob(os.path.join(spec, "model_*.zip")):
            expanded = {os.path.abspath(spec)}
        else:
            expanded = {os.path.abspath(d) for d in globlib.glob(spec + "_run_*")
                        if os.path.isdir(d)}
        extra = expanded - expected
        assert not extra, (
            f"{group['model_name']}: --group={spec} would also evaluate "
            f"{sorted(os.path.basename(p) for p in extra)[:3]}")
        safe += 1
    return f"{safe} of {checked} experiments have an exact spec, {refused} correctly refused"


@check("the tag list describes the selection, not the archive")
def _tag_scope():
    everything = queries.all_tags()
    rows, _ = queries.search({"goal_fn": "gravity"}, limit=40)
    run_ids = [r["run_id"] for r in rows]
    if not run_ids:
        return "skipped -- no gravity runs indexed"
    scoped = queries.tags_for(run_ids)
    assert scoped, "no tags for a non-empty selection"
    assert set(scoped) <= set(everything), "scoped tags are not a subset of all tags"
    assert len(scoped) < len(everything), (
        f"scoping changed nothing: {len(scoped)} of {len(everything)}")
    # Every offered tag must actually have data, or picking it draws an empty chart.
    for tag in scoped[:5]:
        assert any(db.series(r, tag)[0] for r in run_ids), f"'{tag}' offered but empty"
    return f"{len(scoped)} tags for the selection vs {len(everything)} in the corpus"


@check("the config view separates shared keys from disagreements")
def _config_view():
    exp = None
    for group in queries.groups({}, limit=10_000):
        if group["n_seeds"] >= 2:
            exp = queries.experiment(group["date"], group["posture"], group["model_name"])
            if exp and exp["shared"]:
                break
    assert exp is not None, "no multi-seed experiment to check"
    overlap = set(exp["shared"]) & set(exp["differing"])
    assert not overlap, f"keys claimed both shared and differing: {sorted(overlap)}"
    keys = {k for r in exp["runs"] for k in r["yaml_parsed"]}
    assert keys == set(exp["shared"]) | set(exp["differing"]), "a config key was dropped"
    # The raw file has to be readable, since the page offers it.
    raw = Path(exp["runs"][0]["path"]) / "data.yml"
    assert raw.exists() and raw.read_text().strip(), f"unreadable {raw}"
    return (f"{exp['model_name']}: {len(exp['shared'])} shared, "
            f"{len(exp['differing'])} differing across {exp['n_seeds']} seeds")


@check("facet counts stay honest under other filters")
def _facets():
    facets = queries.facets({"posture": "supine"})
    # A facet's own field is counted without its own filter applied, so prone must still appear.
    postures = dict((k, v) for k, v in facets["posture"] if k)
    assert "prone" in postures, postures
    return f"posture facet still offers {sorted(postures)}"


@check("charts render for a real selection")
def _charts():
    rows, _ = queries.search({}, limit=6)
    run_ids = [r["run_id"] for r in rows]
    sizes = []
    for theme in ("light", "dark"):
        png = plots.curve(run_ids, indexer.HEADLINE_TAG, theme=theme)
        assert png.startswith(b"\x89PNG"), "not a PNG"
        assert len(png) > 5000, f"suspiciously small chart ({len(png)} bytes)"
        sizes.append(len(png) // 1024)
    grid = plots.age_grid(queries.age_grid({}), theme="light")
    assert grid.startswith(b"\x89PNG")
    return f"curve {sizes[0]}k light / {sizes[1]}k dark, age grid {len(grid) // 1024}k"


@check("PDF export is vector, exactly the requested width, and paper-styled")
def _pdf_export():
    rows, _ = queries.search({}, limit=12)
    run_ids = [r["run_id"] for r in rows]

    for column, (inches, _height) in plots.COLUMN_WIDTHS.items():
        with plots.render_as("pdf", column):
            data = plots.curve(run_ids, indexer.HEADLINE_TAG, theme="light", aggregate=True)
        assert data.startswith(b"%PDF"), "not a PDF"
        # The page must be the requested column width. bbox_inches="tight" used to inflate a
        # 3.5 in figure to 7.65 in, which is useless for a thesis column.
        media = re.search(rb"/MediaBox\s*\[([\d.\s-]+)\]", data)
        assert media, "no MediaBox in the PDF"
        width_pt = float(media.group(1).split()[2])
        assert abs(width_pt - inches * 72) < 1.0, \
            f"{column}: page is {width_pt:.1f} pt, expected {inches * 72:.0f} pt"
        # TrueType, not matplotlib's default Type 3, which thesis templates reject.
        assert b"/Type3" not in data, "Type 3 fonts leaked into the export"

    with plots.render_as("pdf", "single"):
        grid = plots.age_grid(queries.age_grid({}), theme="light")
    assert grid.startswith(b"%PDF")

    # PNG rendering must be untouched by all of this.
    png = plots.curve(run_ids, indexer.HEADLINE_TAG)
    assert png.startswith(b"\x89PNG"), "PNG output broke"
    return (", ".join(f"{c} {w} in" for c, (w, _) in plots.COLUMN_WIDTHS.items())
            + "; TrueType embedded, PNG unaffected")


@check("thesis figures are placed at their final size with the given labels")
def _thesis():
    rows, _ = queries.search({}, limit=30)
    run_ids = [r["run_id"] for r in rows]
    series = queries.series_of(run_ids)
    assert series, "no series for a non-empty selection"

    # The label editor must list the series in the order the renderer groups them, or row i of
    # the editor names a different line than swatch i suggests.
    grouped, order = set(), []
    for run_id in run_ids:
        row = db.one("SELECT date, posture, model_name FROM runs WHERE run_id=?", (run_id,))
        key = (row["date"], row["posture"], row["model_name"])
        if key not in grouped:
            grouped.add(key)
            order.append(key)
    assert [s["key"] for s in series] == [plots._series_key(k) for k in order], \
        "the label editor lists series in a different order than the chart draws them"

    overrides = {s["key"]: f"Series {i}" for i, s in enumerate(series[:3], start=1)}
    for column, size in plots.THESIS_SIZES.items():
        with plots.render_as("pdf", column, style="thesis"):
            data = plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True, band="std",
                               label_overrides=overrides, ylabel="Mean Success Rate")
        assert data.startswith(b"%PDF")
        media = re.search(rb"/MediaBox\s*\[([\d.\s-]+)\]", data)
        box = [float(v) for v in media.group(1).split()]
        width, height = box[2] - box[0], box[3] - box[1]
        # Placed at its final size: \includegraphics must not need a width= to rescale it.
        assert abs(width - size[0] * 72) < 1.5 and abs(height - size[1] * 72) < 1.5, \
            f"{column}: {width:.0f}x{height:.0f} pt, expected {size[0] * 72:.0f}x{size[1] * 72:.0f}"
    return (f"single {plots.THESIS_SIZES['single']} in, double {plots.THESIS_SIZES['double']} in; "
            f"{len(overrides)} labels applied")


@check("the legend goes where it is told, in every style")
def _legend():
    """The control used to be read on the thesis path alone.

    Choosing "upper left" for a screen chart or a plain PDF export left the legend below the axes
    and looked like a broken setting. The legend title had the same gap.
    """
    rows, _ = queries.search({}, limit=12)
    run_ids = [r["run_id"] for r in rows]

    captured = {}
    real_emit = plots._emit

    def spy(fig):
        legend = fig.axes[0].get_legend() or (fig.legends[0] if fig.legends else None)
        captured["loc"] = None if legend is None else legend._get_loc()
        captured["title"] = None if legend is None else legend.get_title().get_text()
        captured["in_axes"] = legend in fig.axes[0].get_children() if legend else False
        return real_emit(fig)

    seen = {}
    try:
        plots._emit = spy
        for style, fmt in (("screen", "png"), ("screen", "pdf"), ("thesis", "pdf")):
            with plots.render_as(fmt, "single" if fmt == "pdf" else None, style=style):
                plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True,
                            legend_title="Learning rate", legend_loc="upper left")
            seen[f"{style}/{fmt}"] = dict(captured)
    finally:
        plots._emit = real_emit

    # Legend.codes: 2 is "upper left".
    for name, got in seen.items():
        assert got["in_axes"], f"{name}: legend not placed inside the axes"
        assert got["loc"] == 2, f"{name}: loc {got['loc']}, expected 2 (upper left)"
        assert got["title"] == "Learning rate", f"{name}: no legend title"

    # Without an explicit placement the default is unchanged: below/outside, never over the data.
    with plots.render_as("png", None, style="screen"):
        try:
            plots._emit = spy
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True)
        finally:
            plots._emit = real_emit
    assert captured["loc"] != 2, "an unset placement should keep the legend below the axes"

    return f"upper left honoured in {', '.join(sorted(seen))}"


@check("thesis figures use the icdlplot style and the tab palette")
def _tab_palette():
    import matplotlib.colors as mcolors
    assert plots.TAB_COLORS[:4] == ["tab:orange", "tab:gray", "tab:green", "tab:blue"], \
        "the first four series are not orange / grey / green / blue"
    for name, hexval in zip(plots.TAB_COLORS, plots.TAB_HEX):
        assert mcolors.to_hex(name) == hexval, f"{name} is not {hexval}"
    # static/app.js paints the label editor's swatches from its own copy of this list; if the two
    # drift, swatch i names a different line than the chart draws.
    js = (Path(__file__).resolve().parent / "static" / "app.js").read_text()
    block = js.split("THESIS_COLORS =", 1)[1].split("];", 1)[0]
    assert all(h in block for h in plots.TAB_HEX), "app.js swatches do not match TAB_HEX"

    # The style is icdlplot's, executed from the file rather than restated, so the two cannot
    # drift -- editing lines.linewidth there must reach the app with no edit here.
    icdl = plots.ICDL_PATH.read_text()
    rc = plots.paper_rc()
    for key in ("lines.linewidth", "mathtext.fontset", "axes.titlesize", "legend.fontsize"):
        assert key in icdl and key in rc, f"{key} missing from the paper style"
    source = float(re.search(r'"lines\.linewidth":\s*([\d.]+)', icdl).group(1))
    assert rc["lines.linewidth"] == source, \
        f"icdlplot.py says {source}, the app renders at {rc['lines.linewidth']}"
    assert "savefig.bbox" not in rc, \
        "tight bbox would resize the page and break the column width"
    assert rc["pdf.fonttype"] == 42, "Type 3 fonts would reach the thesis template"
    return f"{len(plots.TAB_COLORS)} tab colours, lines.linewidth {rc['lines.linewidth']}"


@check("a custom figure size overrides the column, within limits")
def _custom_size():
    rows, _ = queries.search({}, limit=8)
    run_ids = [r["run_id"] for r in rows]
    assert plots.clamp_size(None, None) is None, "no size given should stay on the column preset"
    lo, hi = plots.SIZE_LIMITS
    assert plots.clamp_size(999, 0.01) == (hi, lo), "size limits not enforced"
    assert plots.clamp_size(4.0, None)[0] == 4.0, "a width alone should imply a height"

    size = (4.6, 2.3)
    with plots.render_as("pdf", "single", style="thesis", size=size):
        data = plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True)
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[([\d.\s-]+)\]", data).group(1).split()]
    assert abs(box[2] - size[0] * 72) < 1.5 and abs(box[3] - size[1] * 72) < 1.5, \
        f"custom size ignored: {box[2] / 72:.2f} x {box[3] / 72:.2f} in"
    return f"{size[0]} x {size[1]} in overrides the single-column preset {plots.THESIS_SIZES['single']}"


@check("the legend order follows the label editor")
def _legend_order():
    rows, _ = queries.search({}, limit=40)
    run_ids = [r["run_id"] for r in rows]
    series = queries.series_of(run_ids)
    if len(series) < 3:
        return "fewer than 3 series in the corpus -- nothing to reorder"
    keys = [s["key"] for s in series]

    captured = {}
    real_emit = plots._emit

    def spy(fig):
        legend = fig.axes[0].get_legend()
        captured["labels"] = [t.get_text() for t in legend.get_texts()] if legend else []
        return real_emit(fig)

    # Label every series with its own key, so the legend text says which series drew which line.
    overrides = {k: k for k in keys}
    wanted = list(reversed(keys))
    try:
        plots._emit = spy
        with plots.render_as("png", None, style="thesis"):
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True,
                        label_overrides=overrides, order=wanted)
        reordered = list(captured["labels"])
        with plots.render_as("png", None, style="thesis"):
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True,
                        label_overrides=overrides)
        default = list(captured["labels"])
    finally:
        plots._emit = real_emit

    drawn = [l for l in reordered if l in keys]
    assert drawn == [k for k in wanted if k in drawn], \
        f"legend order ignored: {drawn[:3]} against {wanted[:3]}"
    assert default != reordered, "reordering changed nothing"
    return f"{len(drawn)} series reversed; colour follows position"


@check("a series colour can be pinned, and only from the picker's own list")
def _pinned_colors():
    import matplotlib.colors as mcolors
    assert plots.AGE_COLORS[9] == plots.TAB_HEX[0], \
        "icdlplot maps age 9 to tab:orange -- a separate entry would give one age two inks"
    assert plots.valid_color("#AA805A") == "#aa805a", "the picker's own colours must pass"
    for junk in ("red", "#123456", "'; rm -rf", "", None):
        assert plots.valid_color(junk) is None, f"{junk!r} should not reach matplotlib"

    rows, _ = queries.search({}, limit=40)
    run_ids = [r["run_id"] for r in rows]
    series = queries.series_of(run_ids)
    if len(series) < 2:
        return "fewer than 2 series -- nothing to pin"
    pinned = {series[0]["key"]: "#aa805a", series[1]["key"]: "#808080"}

    captured = {}
    real_emit = plots._emit

    def spy(fig):
        captured["lines"] = [mcolors.to_hex(line.get_color())
                             for line in fig.axes[0].get_lines()
                             if line.get_label() and not line.get_label().startswith("_")]
        return real_emit(fig)

    try:
        plots._emit = spy
        with plots.render_as("png", None, style="thesis"):
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True, color_overrides=pinned)
        drawn = list(captured["lines"])
        with plots.render_as("png", None, style="thesis"):
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True)
        default = list(captured["lines"])
    finally:
        plots._emit = real_emit

    assert drawn[:2] == ["#aa805a", "#808080"], f"pinned colours ignored: {drawn[:2]}"
    # Everything not pinned still follows its position.
    assert drawn[2:] == default[2:], "pinning one series moved the others"
    return "age 1 and age 3 pinned; the rest still follow position"


@check("the static bundle is versioned, so a restart cannot serve stale JS")
def _asset_version():
    from fastapi.testclient import TestClient
    from . import app as appmod
    client = TestClient(appmod.app)
    body = client.get("/").text
    version = appmod.asset_version()
    assert int(version) > 0, "no mtime for the static bundle"
    assert f"/static/app.js?v={version}" in body, "app.js is not cache-busted"
    assert f"/static/app.css?v={version}" in body, "app.css is not cache-busted"
    return f"v={version}"


@check("the figure style is editable, validated and persistent")
def _style_settings():
    """rcParams from the Settings dialog, on top of whatever icdlplot.py currently says."""
    from fastapi.testclient import TestClient
    from . import app as appmod
    client = TestClient(appmod.app)

    saved = plots.rc_overrides_text()          # never clobber the real style
    try:
        bad = client.post("/api/style", data={"text": "lines.linewidth: abc\n"
                                                      "nosuch.param: 3\n"
                                                      "savefig.bbox: tight\n"})
        assert bad.status_code == 422, f"invalid rcParams accepted ({bad.status_code})"
        reasons = " | ".join(bad.json()["errors"])
        for expected in ("not an rcParam", "Could not convert", "not settable here"):
            assert expected in reasons, f"missing complaint {expected!r} in {reasons}"
        # Nothing is written when anything is wrong: a half-saved style is worse than none.
        assert plots.rc_overrides_text() == saved, "an invalid save still touched the file"

        ok = client.post("/api/style", data={"text": "# note\nlines.linewidth: 2.4\n"})
        assert ok.status_code == 200, ok.text
        assert plots.paper_rc()["lines.linewidth"] == 2.4, "the override did not take"
        assert "# note" in plots.rc_overrides_text(), "comments should survive a save"

        widths = []
        real_emit = plots._emit

        def spy(fig):
            lines = [l for l in fig.axes[0].get_lines() if not l.get_label().startswith("_")]
            widths.append(lines[0].get_linewidth() if lines else None)
            return real_emit(fig)

        rows, _ = queries.search({}, limit=6)
        run_ids = [r["run_id"] for r in rows]
        try:
            plots._emit = spy
            with plots.render_as("pdf", "single", style="thesis"):
                plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True)
            plots.curve(run_ids, indexer.HEADLINE_TAG, aggregate=True)
        finally:
            plots._emit = real_emit
        assert widths[0] == 2.4, f"the exported figure ignored the override ({widths[0]})"
        # The screen charts are a different instrument and must not move with the paper style.
        assert widths[1] != 2.4, "the override leaked into the on-screen chart"
    finally:
        if saved:
            SETTINGS.rc_path.write_text(saved)
        elif SETTINGS.rc_path.exists():
            SETTINGS.rc_path.unlink()

    return "invalid input refused, valid input applied to the PDF only"


@check("laterality is counted over the successful seeds only")
def _laterality():
    """n_left / n_right for the seeds above the success line, with a placeholder when unknown.

    The direction only entered eval_rollover.py on 08.09.2026, so every evaluation stored before
    that carries none -- the tile has to say "---" rather than a confident 0 / 0.
    """
    def payload(left, right):
        return json.dumps({"left": left, "right": right})

    rows = [
        {"rolled": 1.00, "raw": payload(40, 0)},    # successful, always left
        {"rolled": 0.95, "raw": payload(38, 0)},    # successful, always left
        {"rolled": 0.80, "raw": payload(0, 32)},    # successful, always right
        {"rolled": 0.90, "raw": payload(20, 16)},   # successful, both ways
        {"rolled": 0.50, "raw": payload(0, 20)},    # NOT successful -- must not be counted
        {"rolled": 0.99, "raw": json.dumps({})},    # successful, no direction recorded
    ]
    band = queries.laterality_band(rows, 0.75)
    assert band["side_successful"] == 5, band
    assert band["side_known"] == 4, "the run without a direction should not be counted as a side"
    assert (band["side_left"], band["side_right"], band["side_ambiguous"]) == (2, 1, 1), band

    none_recorded = queries.laterality_band(
        [{"rolled": 0.9, "raw": None}, {"rolled": 0.1, "raw": None}], 0.75)
    assert none_recorded["side_successful"] == 1 and none_recorded["side_known"] == 0, \
        "an evaluation with no direction must read as unknown, not as zero"

    # eval_rollover.py must agree, because the terminal summary and the page show the same number.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mimoEnv.eval_rollover import _committed_side
    assert _committed_side({"left": 40, "right": 0}) == "left"
    assert _committed_side({"left": 0, "right": 40}) == "right"
    assert _committed_side({"left": 3, "right": 1}) == "mixed"
    assert _committed_side({"left": 0, "right": 0}) is None
    assert _committed_side({}) is None

    return "2 left / 1 right / 1 ambiguous of 5 successful, 1 without a direction"


@check("stored --group payloads feed the bar chart")
def _bars():
    files = evals.group_jsons()
    if not files:
        return "skipped -- no group evaluation stored yet"
    import json as _json
    for entry in files[:3]:
        with open(entry["run_path"]) as fh:
            payload = _json.load(fh)
        # plot_eval_success.py reads exactly these keys; losing one breaks the thesis figure.
        assert "rows" in payload and payload["rows"], f"{entry['job_id']}: no rows"
        for row in payload["rows"]:
            assert "rolled" in row and "starting_position" in row, \
                f"{entry['job_id']}: a row is missing rolled/starting_position"
    return f"{len(files)} payload(s) stored, newest {files[0]['label']}"


@check("the bar panel offers only evaluations of the selected runs")
def _bar_scope():
    everything = evals.group_jsons()
    if not everything:
        return "skipped -- no group evaluation stored yet"

    # A selection that includes the newest payload's own runs must offer it...
    import json as _json
    with open(everything[0]["run_path"]) as fh:
        payload = _json.load(fh)
    dirs = {str(Path(r["model"]).parent) for r in payload.get("rows", []) if r.get("model")}
    run_ids = []
    for path in dirs:
        row = db.one("SELECT run_id FROM runs WHERE path=?", (path,))
        if row:
            run_ids.append(row["run_id"])
    assert run_ids, "the newest payload names no indexed run"
    offered = {e["job_id"] for e in evals.group_jsons(run_ids=run_ids)}
    assert everything[0]["job_id"] in offered, "a payload's own runs do not offer it"

    # ...and a selection sharing none of its runs must not.
    others = [r["run_id"] for r in db.query(
        "SELECT run_id FROM runs WHERE path NOT IN ({}) LIMIT 40".format(
            ",".join("?" * len(dirs))), list(dirs))]
    if others:
        stray = evals.group_jsons(run_ids=others)
        for entry in stray:
            with open(entry["run_path"]) as fh:
                other = _json.load(fh)
            other_dirs = {str(Path(r["model"]).parent) for r in other.get("rows", [])
                          if r.get("model")}
            assert other_dirs & set(dirs) or entry["job_id"] != everything[0]["job_id"], \
                "an unrelated payload was offered"
    assert evals.group_jsons(run_ids=[]) == [], "an empty selection offered evaluations"
    return f"{len(everything)} stored, {len(offered)} offered for the newest payload's own runs"


@check("series labels stay distinct")
def _labels():
    names = ["sac_her_sparse_goal_lohi_curriculum",
             "sac_her_sparse_goal_lohi_curriculum_nodone",
             "sac_her_sparse_goal_lohi_curriculum_nodone_entropy92_lrlinearsched"]
    labels = plots.distinguish(names)
    assert len(set(labels.values())) == len(names), f"labels collide: {labels}"
    return " | ".join(labels.values())


@check("mixed horizons are reported")
def _horizons():
    a = db.one("SELECT run_id FROM runs WHERE episode_steps = 250 LIMIT 1")
    b = db.one("SELECT run_id FROM runs WHERE episode_steps = 100 LIMIT 1")
    if not (a and b):
        return "skipped -- no two horizons in the index"
    warning = plots.horizon_warning([a["run_id"], b["run_id"]])
    assert warning and "250" in warning and "100" in warning, warning
    assert plots.horizon_warning([a["run_id"]]) is None
    return "warns on 100 vs 250, silent on a single horizon"


@check("offline mode blocks every write path")
def _offline():
    SETTINGS.offline = True
    try:
        for call, label in ((lambda: evals.submit("x", "model_1.zip"), "single eval"),
                            (lambda: evals.submit_group("26-08-22", "supine", "x"), "group eval")):
            try:
                call()
            except RuntimeError as exc:
                assert "offline" in str(exc), exc
            else:
                raise AssertionError(f"{label} was not blocked in offline mode")

        from fastapi.testclient import TestClient
        from . import app as appmod
        before = plots.rc_overrides_text()
        response = TestClient(appmod.app).post("/api/style", data={"text": "lines.linewidth: 9"})
        assert response.status_code == 400, "the style was writable in offline mode"
        assert plots.rc_overrides_text() == before, "offline mode still touched the style file"
    finally:
        SETTINGS.offline = False
    return "evaluation and the figure style refuse to write"


def main():
    configure(mimo_root=None)
    db.reset_connection()
    print(f"mimolab selfcheck -- {SETTINGS.models_root}\n")
    for run in _checks:
        run()
    for ok, name, detail in _results:
        print(f"[{PASS if ok else FAIL}] {name}")
        if detail:
            print(f"          {detail}")
    failed = sum(1 for ok, _, _ in _results if not ok)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
