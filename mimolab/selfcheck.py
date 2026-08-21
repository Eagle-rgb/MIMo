"""python -m mimolab.selfcheck -- verify the app against the real models/ directory.

Not a unit test suite; this fork does not have one. It is the same kind of thing as
goalenv_check.py: a set of assertions over real data that catch the failures that actually
happen here -- a posture read from the wrong place, a flag that will not round-trip through
data.yml, a guard rail that stopped guarding.
"""

import sys
import traceback

from .config import configure, SETTINGS
from . import db, evals, fleet, indexer, plots, queries

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


@check("facet counts stay honest under other filters")
def _facets():
    facets = queries.facets({"posture": "supine"})
    # A facet's own field is counted without its own filter applied, so prone must still appear.
    postures = dict((k, v) for k, v in facets["posture"] if k)
    assert "prone" in postures, postures
    return f"posture facet still offers {sorted(postures)}"


@check("every experiment-defining flag round-trips through data.yml")
def _yaml_coverage():
    missing = fleet.check_yaml_coverage()
    assert not missing, ("these flags are not stored in yaml_data in illustrations.py, so a model "
                         f"trained with them would reload with different settings: {missing}")
    return f"{sum(1 for f in fleet.TRAIN_FLAGS if f[4])} flags verified against illustrations.py"


@check("guard rails match what illustrations.py enforces")
def _guards():
    # Blocking, because illustrations.py raises on these.
    blocking = [
        ({"algorithm": "PPO", "her": True}, "off-policy"),
        ({"algorithm": "SAC", "pbrs": True, "no_done_active": True}, "discontinuous"),
        ({"algorithm": "SAC", "goal_low": 0.25}, "both"),
        ({"algorithm": "SAC", "goal_curriculum": True}, "goal range"),
    ]
    for values, needle in blocking:
        found = " ".join(fleet.errors(values))
        assert needle in found, f"{values} should be an error for '{needle}', got: {found}"

    # Advisory only: illustrations.py warns or silently auto-corrects, so blocking here would
    # refuse configurations that train fine.
    advisory = [
        ({"algorithm": "SAC", "her": True, "no_done_active": False, "goal_low": 0.2,
          "goal_high": 0.9}, "bootstraps"),
        ({"algorithm": "SAC", "save_every": 1_000_000, "train_for": 1_000_000}, "final"),
        ({"algorithm": "SAC", "her": True, "goal_low": 0.5, "goal_high": 0.5,
          "no_done_active": True}, "variation"),
        ({"algorithm": "SAC", "her": True, "no_done_active": True, "goal_low": 0.2,
          "goal_high": 0.9, "learning_starts": 100, "episode_steps": 500}, "raised"),
    ]
    for values, needle in advisory:
        assert needle in " ".join(fleet.warnings(values)), f"{values} should warn for '{needle}'"
        assert needle not in " ".join(fleet.errors(values)), f"{values} must not block"

    # The exact configuration running across the pool right now: --pbrs is inert under
    # --sparse_reward, so the discontinuous-potential guard does not apply.
    live = {"algorithm": "SAC", "her": True, "sparse_reward": True, "pbrs": True,
            "no_done_active": True, "goal_low": 0.25, "goal_high": 0.95, "episode_steps": 100}
    assert not fleet.errors(live), f"a configuration that actually runs was blocked: {fleet.errors(live)}"

    for name, preset in fleet.PRESETS.items():
        assert not fleet.errors(preset["flags"]), f"preset {name}: {fleet.errors(preset['flags'])}"
    return (f"{len(blocking)} blocked, {len(advisory)} advisory, "
            f"the live pool configuration accepted")


@check("the built command matches the shell scripts")
def _command():
    cmd, _ = fleet.build_command(fleet.PRESETS["sac_her_sparse"]["flags"], "x_run_0")
    for expected in ("--algorithm=SAC", "--her", "--sparse_reward", "--goal_curriculum",
                     "--no_done_active", "--goal_low=0.25", "--goal_high=0.95",
                     "--roll_over_model_path_auto", "--save_model=x_run_0"):
        assert expected in cmd, f"missing {expected} in: {cmd}"
    ppo, _ = fleet.build_command(fleet.PRESETS["ppo_pbrs"]["flags"], "y_run_0")
    assert "--pbrs" in ppo and "--her" not in ppo
    return "both presets reproduce their rbi_autorun script"


@check("numeric flags are not mistaken for switches")
def _numeric_flags():
    # train_freq=1 and gradient_steps=1 hold the value 1, which is also what a checked checkbox
    # submits. Deciding by value rather than by declared kind drops the "=1" and produces a
    # different run than the form shows.
    cmd, _ = fleet.build_command({"train_freq": 1, "gradient_steps": 1, "pbrs": True,
                                  "sparse_reward": False}, "x_run_0")
    assert "--train_freq=1" in cmd, cmd
    assert "--gradient_steps=1" in cmd, cmd
    assert "--pbrs" in cmd and "--pbrs=" not in cmd, cmd
    assert "--sparse_reward" not in cmd, cmd
    return "value-1 integers keep their '=1'; switches stay bare"


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
        for call, label in ((lambda: fleet.launch({}, "x", "adrastos"), "launch"),
                            (lambda: evals.submit("x", "model_1.zip"), "eval")):
            try:
                call()
            except RuntimeError as exc:
                assert "offline" in str(exc), exc
            else:
                raise AssertionError(f"{label} was not blocked in offline mode")
    finally:
        SETTINGS.offline = False
    return "launch and evaluate refuse to run"


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
