"""Filtering and aggregation over the run index."""

import datetime
import json
from collections import defaultdict

from . import db

# Filters exposed in the UI. Each maps to a SQL predicate over 'runs'.
TEXT_FILTERS = ["posture", "algorithm", "reward_shape", "goal_fn", "collection", "date",
                "lr_schedule", "state"]
BOOL_FILTERS = ["her", "sparse_reward", "pbrs", "goal_curriculum", "no_done_active",
                "isr", "side_lying"]
INT_FILTERS = ["morph_age", "physio_age", "episode_steps"]

# Quick date filters. Dates are stored as 'yy-mm-dd', so lexicographic order is chronological
# and a plain string comparison is a correct date comparison -- no parsing needed.
DATE_PRESETS = {
    "latest":     "the most recent date in the index",
    "7d":         "runs from the last 7 days",
    "30d":        "runs from the last 30 days",
    "latest_each": "only each experiment's most recent date",
}


def _cutoff(days):
    return (datetime.date.today() - datetime.timedelta(days=days)).strftime("%y-%m-%d")


SORTABLE = {
    "date": "date", "name": "model_name", "seed": "seed_idx", "algorithm": "algorithm",
    "posture": "posture", "steps": "last_step", "rho": "best_rho", "final_rho": "final_rho",
    "checkpoints": "n_checkpoints", "state": "state", "morph": "morph_age", "physio": "physio_age",
}


def _multi(values):
    """Query params arrive as a list when a filter is multi-valued; normalise and drop blanks."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [v for v in values if v not in ("", None)]


def build_where(params):
    """Translate request params into a WHERE clause. Returns (sql, bind_values)."""
    clauses, binds = [], []

    for field in TEXT_FILTERS:
        values = _multi(params.get(field))
        if values:
            clauses.append(f"{field} IN ({','.join('?' * len(values))})")
            binds.extend(values)

    for field in INT_FILTERS:
        values = _multi(params.get(field))
        if values:
            # 'none' selects runs where the key is absent -- e.g. the 485 runs with no
            # episode_steps recorded, which ran at the 500-step default.
            concrete = [int(v) for v in values if v != "none"]
            parts = []
            if concrete:
                parts.append(f"{field} IN ({','.join('?' * len(concrete))})")
                binds.extend(concrete)
            if "none" in values:
                parts.append(f"{field} IS NULL")
            clauses.append("(" + " OR ".join(parts) + ")")

    for field in BOOL_FILTERS:
        value = params.get(field)
        if isinstance(value, list):
            value = value[0] if value else None
        if value in ("1", "0"):
            clauses.append(f"{field} = ?")
            binds.append(int(value))

    preset = params.get("date_preset")
    if isinstance(preset, list):
        preset = preset[0] if preset else None
    if preset == "latest":
        clauses.append("date = (SELECT MAX(date) FROM runs)")
    elif preset in ("7d", "30d"):
        clauses.append("date >= ?")
        binds.append(_cutoff(7 if preset == "7d" else 30))
    elif preset == "latest_each":
        # Each experiment's newest date, computed over the whole corpus rather than over the
        # current filter -- "the latest date of this model" means latest, not latest-among-what-
        # is-currently-shown, which would shift as other filters change.
        clauses.append("""run_id IN (
            SELECT r.run_id FROM runs r
            JOIN (SELECT model_name, MAX(date) AS newest FROM runs
                  WHERE model_name IS NOT NULL AND date IS NOT NULL
                  GROUP BY model_name) latest
              ON r.model_name = latest.model_name AND r.date = latest.newest)""")

    q = params.get("q")
    if isinstance(q, list):
        q = q[0] if q else None
    if q:
        clauses.append("(run_id LIKE ? OR model_name LIKE ?)")
        binds.extend([f"%{q}%", f"%{q}%"])

    for field, op in (("min_steps", ">="), ("min_rho", ">=")):
        value = params.get(field)
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in ("", None):
            column = "last_step" if field == "min_steps" else "best_rho"
            clauses.append(f"{column} {op} ?")
            binds.append(float(value))

    has_eval = params.get("has_eval")
    if isinstance(has_eval, list):
        has_eval = has_eval[0] if has_eval else None
    if has_eval == "1":
        clauses.append("run_id IN (SELECT run_id FROM evals)")
    elif has_eval == "0":
        clauses.append("run_id NOT IN (SELECT run_id FROM evals)")

    return (" WHERE " + " AND ".join(clauses)) if clauses else "", binds


# The corpus passed 500 runs on 19.08.2026 and the old cap silently truncated the table. Rows are
# cheap (the table has its own scrollport) and the index is local, so the cap exists only to stop a
# runaway query, not to paginate.
DEFAULT_LIMIT = 2000


def search(params, limit=DEFAULT_LIMIT, offset=0):
    where, binds = build_where(params)

    sort = params.get("sort")
    if isinstance(sort, list):
        sort = sort[0] if sort else None
    direction = "ASC" if params.get("dir") == "asc" else "DESC"
    column = SORTABLE.get(sort or "date", "date")
    # Runs that never logged a headline metric sort last either way, rather than clumping at the
    # top of a descending sort as NULLs otherwise do in SQLite.
    order = f"ORDER BY {column} IS NULL, {column} {direction}, model_name, seed_idx"

    rows = db.query(
        f"SELECT * FROM runs {where} {order} LIMIT ? OFFSET ?", binds + [limit, offset])
    total = db.one(f"SELECT COUNT(*) AS n FROM runs {where}", binds)["n"]
    return [dict(r) for r in rows], total


def facets(params):
    """Counts for each filter value under the *other* active filters, so the sidebar stays honest."""
    out = {}
    for field in TEXT_FILTERS + BOOL_FILTERS + INT_FILTERS:
        others = {k: v for k, v in params.items() if k != field}
        where, binds = build_where(others)
        # Dates read as a timeline, so they are ordered newest-first; every other facet is a set
        # of unordered categories, where the useful order is by how much data sits behind each.
        order = "v IS NULL, v DESC" if field == "date" else "n DESC"
        rows = db.query(
            f"SELECT {field} AS v, COUNT(*) AS n FROM runs {where} GROUP BY 1 ORDER BY {order}",
            binds)
        out[field] = [(r["v"], r["n"]) for r in rows]
    return out


def date_preset_counts(params):
    """How many runs each quick date filter would select, under the other active filters."""
    others = {k: v for k, v in params.items() if k != "date_preset"}
    counts = {}
    for key in DATE_PRESETS:
        merged = dict(others, date_preset=key)
        where, binds = build_where(merged)
        counts[key] = db.one(f"SELECT COUNT(*) AS n FROM runs {where}", binds)["n"]
    return counts


def latest_date():
    row = db.one("SELECT MAX(date) AS d FROM runs")
    return row["d"] if row else None


def get_run(run_id):
    row = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    if row is None:
        return None
    run = dict(row)
    run["yaml_parsed"] = json.loads(run["yaml"] or "{}")
    run["checkpoint_list"] = json.loads(run["checkpoints"] or "[]")
    run["tags"] = [r["tag"] for r in db.query(
        "SELECT tag FROM scalars WHERE run_id=? ORDER BY tag", (run_id,))]
    run["evals"] = [dict(r) for r in db.query(
        "SELECT * FROM evals WHERE run_id=? ORDER BY created_at DESC", (run_id,))]
    return run


def siblings(run):
    """The other seeds of the same experiment -- same date, posture and model name."""
    if not run.get("model_name"):
        return []
    rows = db.query(
        "SELECT * FROM runs WHERE model_name=? AND date IS ? AND posture=? ORDER BY seed_idx",
        (run["model_name"], run["date"], run["posture"]))
    return [dict(r) for r in rows]


def groups(params, limit=200):
    """Collapse runs into experiments: one row per (date, posture, model_name) with seed stats."""
    where, binds = build_where(params)
    rows = db.query(f"""
        SELECT date, posture, model_name, algorithm, reward_shape, her,
               morph_age, physio_age, episode_steps, collection,
               COUNT(*)        AS n_seeds,
               AVG(best_rho)   AS rho_mean,
               MAX(best_rho)   AS rho_best,
               MIN(best_rho)   AS rho_worst,
               MAX(last_step)  AS last_step,
               SUM(n_checkpoints) AS n_checkpoints,
               GROUP_CONCAT(run_id, char(10)) AS run_ids
        FROM runs {where}
        GROUP BY date, posture, model_name
        ORDER BY date DESC, model_name
        LIMIT ?""", binds + [limit])
    out = []
    for r in rows:
        g = dict(r)
        g["run_ids"] = (g["run_ids"] or "").split("\n")
        out.append(g)
    return out


def age_grid(params, run_ids=None):
    """The 4x4 morph x physio matrix, averaged over best_rho.

    'run_ids' pins the grid to an explicit selection. The analysis page passes it because its
    query string carries run= parameters that build_where has no predicate for -- without this
    the chart would silently summarise the whole corpus while claiming to show the selection.
    """
    if run_ids:
        where = f" WHERE run_id IN ({','.join('?' * len(run_ids))})"
        binds = list(run_ids)
    else:
        where, binds = build_where(params)
    rows = db.query(f"""
        SELECT morph_age, physio_age, COUNT(*) AS n, AVG(best_rho) AS rho
        FROM runs {where} GROUP BY 1, 2""", binds)
    grid = defaultdict(dict)
    for r in rows:
        if r["morph_age"] is None or r["physio_age"] is None:
            continue
        grid[r["morph_age"]][r["physio_age"]] = {"n": r["n"], "rho": r["rho"]}
    return grid


def age_pairs(params):
    """Run count per (morph, physio) cell, so the matrix control shows where data exists."""
    where, binds = build_where(params)
    rows = db.query(f"""SELECT morph_age AS m, physio_age AS p, COUNT(*) AS n
                        FROM runs {where} GROUP BY 1, 2""", binds)
    return {(r["m"], r["p"]): r["n"] for r in rows if r["m"] is not None and r["p"] is not None}


def stats():
    row = db.one("""SELECT COUNT(*) AS runs, SUM(n_checkpoints) AS ckpts,
                           COUNT(DISTINCT model_name) AS experiments FROM runs""")
    evals = db.one("SELECT COUNT(*) AS n FROM evals")["n"]
    running = db.one("SELECT COUNT(*) AS n FROM runs WHERE state='running'")["n"]
    return {"runs": row["runs"] or 0, "checkpoints": row["ckpts"] or 0,
            "experiments": row["experiments"] or 0, "evals": evals, "running": running}


def all_tags():
    return [r["tag"] for r in db.query(
        "SELECT tag, COUNT(*) c FROM scalars GROUP BY 1 ORDER BY c DESC, tag")]
