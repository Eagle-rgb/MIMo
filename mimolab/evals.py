"""The evaluation queue.

eval_rollover.py is the authority for reportable numbers -- it forces ISR off, pins the goal,
disables done_active, uses deterministic actions and aggregates rho_max >= 0.95 per episode. None
of that is reimplemented here; this module only schedules it and stores what comes back.

Serial by construction: one MIMo env is ~3.6 GB RSS, so a depth-1 queue is a correctness
constraint, not a tunable.
"""

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import db
from .config import SETTINGS, MUJOCO_GL, EVAL_CONCURRENCY

_queue = queue.Queue()
_worker = None
_lock = threading.Lock()
_current = {"task": None, "started": None}


def _run_group_eval(task):
    """Invoke eval_rollover.py --group once, streaming its output to a log.

    Popen rather than run(): a group of 16 seeds takes several minutes and eval_rollover prints
    '[i/n] <run>' as it goes, which is the only progress signal there is. Parsing it as it
    arrives is what lets the page say 'run 7 of 16' instead of just spinning. An embodiment grid
    counts through all its cells in the same '[i/n]', so the same parser covers both -- a DCEE run
    is 16x the work and a bar that restarted at 1 in every cell would be useless.
    """
    job_id = task["job_id"]
    log_path = SETTINGS.log_dir / f"{job_id}.log"
    # Kept, not temporary: results/plot_eval_success.py draws the thesis bar chart straight from
    # this payload, so the file is the handover between evaluation and figure.
    SETTINGS.eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = SETTINGS.eval_dir / f"{job_id}.json"

    argv = [SETTINGS.python, "mimoEnv/eval_rollover.py",
            f"--group={task['group_spec']}",
            f"--episodes={int(task.get('episodes', 40))}",
            f"--checkpoint={task.get('checkpoint', 'last')}",
            f"--json={json_path}"]
    if task.get("embodiment_grid"):
        argv.append(f"--embodiment_grid={task['embodiment_grid']}")
    if task.get("starting_position"):
        argv.append(f"--starting_position={task['starting_position']}")
    if task.get("success_threshold") is not None:
        argv.append(f"--success_threshold={task['success_threshold']}")
    if task.get("csv"):
        argv.append(f"--csv={task['csv']}")

    env = dict(os.environ, MUJOCO_GL=MUJOCO_GL, PYTHONUNBUFFERED="1")
    progress = re.compile(r"^\[(\d+)/(\d+)\]")
    with open(log_path, "w") as log:
        log.write(" ".join(argv) + "\n\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=SETTINGS.mimo_root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            match = progress.match(line)
            if match:
                done, total = match.group(1), match.group(2)
                db.execute("UPDATE jobs SET note=? WHERE job_id=?",
                           (f"run {done} of {total}", job_id))
        proc.wait()

    try:
        with open(json_path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        tail = ""
        try:
            with open(log_path, errors="replace") as fh:
                tail = "".join(fh.readlines()[-25:])
        except OSError:
            pass
        raise RuntimeError(tail or f"eval_rollover exited with {proc.returncode} and wrote no JSON")
    payload["_json_path"] = str(json_path)
    return payload


def _store_group(payload):
    """Fold each run's row into 'evals', so the group result and the per-run pages agree.

    Rows are matched back to the index by the run directory, not by the printed name: the name is
    only a basename and two collections can hold the same one.
    """
    now = time.time()
    stored, missing = 0, []
    for row in payload.get("rows", []):
        run_dir = str(Path(row["model"]).parent)
        found = db.one("SELECT run_id FROM runs WHERE path=?", (run_dir,))
        if found is None:
            missing.append(row.get("run"))
            continue
        db.execute("""INSERT INTO evals (run_id, checkpoint, episodes, goal, policy_goal,
                                         episode_steps, starting_position, rolled, side,
                                         rho_mean, rho_min, rho_max, steps_mean, raw, created_at)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (found["run_id"], row.get("checkpoint"), row.get("episodes"), row.get("goal"),
                    None, row.get("episode_steps"), row.get("starting_position"),
                    row.get("rolled"), row.get("side"), row.get("rho_mean"), row.get("rho_min"),
                    row.get("rho_max"), row.get("steps_mean"), json.dumps(row), now))
        stored += 1
    return stored, missing


def _store_dcee(payload):
    """Fold the source cell of a grid into 'evals', and nothing else.

    A DCEE payload holds sixteen evaluations of the same seeds, one per embodiment. Storing them
    all would make the experiment page average a run against bodies it was never trained on. The
    top-level 'rows' of the payload are the run's own embodiment -- exactly what a plain --group
    run would have produced -- so folding those keeps the two agreeing.
    """
    return _store_group(payload)


def _run_eval(task):
    """Invoke eval_rollover.py once and return the parsed payload."""
    model = task["model"]
    run_dir = Path(model).parent
    posture = task.get("starting_position")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    argv = [SETTINGS.python, "mimoEnv/eval_rollover.py",
            f"--model={model}",
            f"--episodes={int(task.get('episodes', 50))}",
            f"--json={json_path}"]
    if posture:
        # Always explicit. data.yml never records the posture, so eval_rollover's own fallback
        # would read 'supine' for every prone run.
        argv.append(f"--starting_position={posture}")
    if task.get("goal") is not None:
        argv.append(f"--goal={task['goal']}")
    if task.get("policy_goal") is not None:
        argv.append(f"--policy_goal={task['policy_goal']}")
    if task.get("policy_goal_sweep"):
        argv.append(f"--policy_goal_sweep={task['policy_goal_sweep']}")
    if task.get("episode_steps"):
        argv.append(f"--episode_steps={int(task['episode_steps'])}")

    env = dict(os.environ, MUJOCO_GL=MUJOCO_GL)
    proc = subprocess.run(argv, cwd=SETTINGS.mimo_root, env=env,
                          capture_output=True, text=True, timeout=task.get("timeout", 7200))
    try:
        with open(json_path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        payload = None
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    if payload is None:
        raise RuntimeError((proc.stderr or proc.stdout or "eval produced no output")[-2000:])
    payload["stdout"] = proc.stdout[-8000:]
    return payload


def _store(task, payload):
    run_id = task["run_id"]
    checkpoint = os.path.basename(task["model"])
    now = time.time()
    for row in payload.get("rows", []):
        db.execute("""INSERT INTO evals (run_id, checkpoint, episodes, goal, policy_goal,
                                         episode_steps, starting_position, rolled, side,
                                         rho_mean, rho_min, rho_max, steps_mean, raw, created_at)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (run_id, checkpoint, payload.get("episodes"), payload.get("goal"),
                    row.get("policy_goal"), payload.get("episode_steps"),
                    payload.get("starting_position"), row.get("rolled"), row.get("side"),
                    row.get("rho_mean"), row.get("rho_min"), row.get("rho_max"),
                    row.get("steps_mean"), json.dumps(row), now))


def _delete_job(job, payload):
    """Remove one finished evaluation completely: its stored rows, its payload, its log, its job.

    The rows are found by the job's own time window. That is exact because the queue is serial --
    no other evaluation can write between a job being queued and finishing -- and it is narrowed to
    the job's own runs on top. Verified on 11.09.2026 against seven real jobs: every window held
    exactly as many rows as the experiment has seeds.
    """
    run_ids = []
    for row in payload.get("rows", []):
        if not row.get("model"):
            continue
        found = db.one("SELECT run_id FROM runs WHERE path=?", (str(Path(row["model"]).parent),))
        if found:
            run_ids.append(found["run_id"])
    if run_ids and job.get("started_at") and job.get("finished_at"):
        db.execute(f"""DELETE FROM evals WHERE created_at BETWEEN ? AND ?
                       AND run_id IN ({','.join('?' * len(run_ids))})""",
                   [job["started_at"], job["finished_at"]] + run_ids)
    for path in (job.get("run_path"), SETTINGS.log_dir / f"{job['job_id']}.log"):
        try:
            if path and str(path).endswith((".json", ".log")):
                os.unlink(path)
        except OSError:
            pass
    db.execute("DELETE FROM jobs WHERE job_id=?", (job["job_id"],))


def _supersede(job_id, kind, payload):
    """Strictly replace every older evaluation of the same kind, group and checkpoint.

    11.09.2026 Re-running used to keep both, and the directory filled with copies a notebook
    globbing it would count twice. Asked for as a strict overwrite, so the older job goes entirely
    -- file, log, job and its rows in 'evals'.

    Called only once the new job has been stored, so a re-evaluation that fails never costs the
    result it was meant to replace. The checkpoint is part of the key because 'best' and 'last'
    are two different measurements, and the kind is, because a DCEE grid does not replace a plain
    group evaluation of the same seeds or the other way round.
    """
    key = (payload.get("group"), payload.get("checkpoint"))
    removed = []
    for old in db.query("""SELECT * FROM jobs WHERE kind=? AND state='finished' AND job_id != ?
                           AND run_path LIKE '%.json'""", (kind, job_id)):
        old = dict(old)
        try:
            with open(old["run_path"]) as fh:
                old_payload = json.load(fh)
        except (OSError, ValueError):
            continue
        if (old_payload.get("group"), old_payload.get("checkpoint")) != key:
            continue
        _delete_job(old, old_payload)
        removed.append(old["job_id"])
    return removed


def _loop():
    while True:
        task = _queue.get()
        if task is None:
            return
        job_id = task["job_id"]
        with _lock:
            _current["task"] = task
            _current["started"] = time.time()
        db.execute("UPDATE jobs SET state='running' WHERE job_id=?", (job_id,))
        try:
            if task.get("kind") in ("group", "dcee"):
                payload = _run_group_eval(task)
                stored, missing = (_store_dcee(payload) if task["kind"] == "dcee"
                                   else _store_group(payload))
                db.execute("UPDATE jobs SET run_path=? WHERE job_id=?",
                           (payload.get("_json_path") or task["group_spec"], job_id))
                cells = payload.get("cells")
                note = (f"{len(cells)} embodiments, {stored} run(s) stored" if cells
                        else f"{stored} run(s) stored")
                if missing:
                    note += f"; {len(missing)} not in the index"
                # The replacement is complete and stored; only now may the old result go.
                try:
                    removed = _supersede(job_id, task["kind"], payload)
                    if removed:
                        note += f"; replaced {len(removed)} older evaluation(s)"
                except Exception as exc:              # never fail a good job over the cleanup
                    note += f"; could not remove the older evaluation: {exc}"
                db.execute("UPDATE jobs SET note=? WHERE job_id=?", (note, job_id))
            else:
                payload = _run_eval(task)
                _store(task, payload)
            db.execute("UPDATE jobs SET state='finished', finished_at=?, exit_code=0 WHERE job_id=?",
                       (time.time(), job_id))
        except Exception as exc:
            db.execute("""UPDATE jobs SET state='failed', finished_at=?, exit_code=1, note=?
                          WHERE job_id=?""", (time.time(), str(exc)[-2000:], job_id))
        finally:
            with _lock:
                _current["task"] = None
                _current["started"] = None
            _queue.task_done()


def start_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, name="mimolab-eval", daemon=True)
        _worker.start()
    return _worker


def submit(run_id, checkpoint, episodes=50, goal=None, policy_goal=None,
           policy_goal_sweep=None, episode_steps=None, starting_position=None):
    """Queue one evaluation. Returns the job row."""
    if SETTINGS.offline:
        raise RuntimeError("offline mode: evaluation is disabled")

    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    if run is None:
        raise KeyError(run_id)
    model = str(Path(run["path"]) / checkpoint)
    if not Path(model).exists():
        raise FileNotFoundError(model)

    posture = starting_position or (run["posture"] if run["posture"] in ("prone", "supine") else None)
    job_id = f"eval-{time.strftime('%y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    label = f"{run['model_name']} / {checkpoint}"
    if policy_goal_sweep:
        label += f" (sweep {policy_goal_sweep})"

    db.execute("""INSERT INTO jobs (job_id, kind, host, cmd, label, run_path, started_at, state)
                  VALUES (?,?,?,?,?,?,?,?)""",
               (job_id, "eval", "local", f"eval_rollover.py {checkpoint}", label,
                run["path"], time.time(), "queued"))

    task = {"job_id": job_id, "run_id": run_id, "model": model, "episodes": episodes,
            "goal": goal, "policy_goal": policy_goal, "policy_goal_sweep": policy_goal_sweep,
            "episode_steps": episode_steps, "starting_position": posture}
    start_worker()
    _queue.put(task)
    return dict(db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,)))


def submit_group(date, posture, model_name, episodes=40, checkpoint="last",
                 success_threshold=0.75):
    """Queue an eval_rollover.py --group over every seed of one experiment."""
    if SETTINGS.offline:
        raise RuntimeError("offline mode: evaluation is disabled")

    from . import queries
    experiment = queries.experiment(date, posture, model_name)
    if experiment is None:
        raise KeyError(f"{date} / {posture} / {model_name}")
    spec = experiment["group_spec"]
    if not spec:
        raise ValueError("could not derive a --group path for this experiment")

    job_id = f"group-{time.strftime('%y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    label = f"{model_name} ({posture}, {experiment['n_seeds']} seeds)"
    db.execute("""INSERT INTO jobs (job_id, kind, cmd, label, run_path, started_at, state)
                  VALUES (?,?,?,?,?,?,?)""",
               (job_id, "group", f"eval_rollover.py --group {spec}", label, spec,
                time.time(), "queued"))

    task = {"job_id": job_id, "kind": "group", "group_spec": spec, "episodes": episodes,
            "checkpoint": checkpoint, "success_threshold": success_threshold,
            "starting_position": posture if posture in ("prone", "supine") else None,
            "run_ids": [r["run_id"] for r in experiment["runs"]]}
    start_worker()
    _queue.put(task)
    return dict(db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,)))


def submit_dcee(date, posture, model_name, episodes=40, checkpoint="last",
                success_threshold=0.75, ages="1,3,6,9"):
    """Queue a cross-embodiment grid: every seed of one experiment, at every (act, body) age.

    Same job as submit_group with --embodiment_grid added, so it shares the queue, the log and
    the progress parsing. It is 'len(ages)**2' times the work of a plain group run -- one env for
    the whole grid, but every cell is a full pass over every seed.
    """
    if SETTINGS.offline:
        raise RuntimeError("offline mode: evaluation is disabled")

    from . import queries
    experiment = queries.experiment(date, posture, model_name)
    if experiment is None:
        raise KeyError(f"{date} / {posture} / {model_name}")
    spec = experiment["group_spec"]
    if not spec:
        raise ValueError("could not derive a --group path for this experiment")
    grid = ",".join(str(float(a)).rstrip("0").rstrip(".") for a in
                    [float(part) for part in str(ages).replace(" ", "").split(",") if part])
    if not grid:
        raise ValueError("no ages given for the embodiment grid")

    job_id = f"dcee-{time.strftime('%y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    n_cells = len(grid.split(",")) ** 2
    label = f"{model_name} ({posture}, {experiment['n_seeds']} seeds, {n_cells} embodiments)"
    db.execute("""INSERT INTO jobs (job_id, kind, cmd, label, run_path, started_at, state)
                  VALUES (?,?,?,?,?,?,?)""",
               (job_id, "dcee", f"eval_rollover.py --group {spec} --embodiment_grid={grid}",
                label, spec, time.time(), "queued"))

    task = {"job_id": job_id, "kind": "dcee", "group_spec": spec, "episodes": episodes,
            "checkpoint": checkpoint, "success_threshold": success_threshold,
            "embodiment_grid": grid,
            "starting_position": posture if posture in ("prone", "supine") else None,
            "run_ids": [r["run_id"] for r in experiment["runs"]]}
    start_worker()
    _queue.put(task)
    return dict(db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,)))


def dcee_json(date, posture, model_name):
    """The most recent embodiment-grid payload for this experiment, if one was ever produced."""
    from . import queries
    experiment = queries.experiment(date, posture, model_name)
    if experiment is None:
        return None
    spec = experiment["group_spec"]
    if not spec:
        return None
    row = db.one("""SELECT job_id, run_path, finished_at FROM jobs
                    WHERE kind='dcee' AND state='finished' AND run_path LIKE '%.json'
                      AND cmd LIKE ?
                    ORDER BY finished_at DESC LIMIT 1""", (f"%{spec} %",))
    if row and Path(row["run_path"]).exists():
        return dict(row)
    return None


def group_json(date, posture, model_name):
    """The most recent --group payload for this experiment, if one was ever produced."""
    from . import queries
    experiment = queries.experiment(date, posture, model_name)
    if experiment is None:
        return None
    spec = experiment["group_spec"]
    row = db.one("""SELECT run_path FROM jobs WHERE kind='group' AND state='finished'
                      AND (cmd LIKE ? OR run_path LIKE ?)
                    ORDER BY finished_at DESC LIMIT 1""",
                 (f"%{spec}%" if spec else "%", "%.json"))
    if row and row["run_path"] and row["run_path"].endswith(".json") \
            and Path(row["run_path"]).exists():
        return row["run_path"]
    return None


def dcee_payloads(limit=50):
    """Stored embodiment-grid payloads, newest first."""
    rows = db.query("""SELECT job_id, label, run_path, finished_at FROM jobs
                       WHERE kind='dcee' AND state='finished' AND run_path LIKE '%.json'
                       ORDER BY finished_at DESC LIMIT ?""", (limit,))
    return [dict(r) for r in rows if r["run_path"] and Path(r["run_path"]).exists()]


def group_jsons(limit=200, run_ids=None, include_superseded=False):
    """Stored --group payloads, newest first, one per (group, checkpoint).

    Since 11.09.2026 a new evaluation strictly replaces the old one (see _supersede), so there is
    normally nothing to fold away here; this stays as the guard for copies written before then or
    left behind by a cleanup that failed. Before that, every job wrote its own '<job_id>.json' and
    the directory kept the history. That history is not what a figure wants, though -- by 11.09.2026
    seven experiments had been evaluated twice, age9 once before laterality was recorded and once
    after, and sac_her_ep200_tf2 at 0 and then 4 successful seeds from the same model_1.zip
    because the protocol changed in between. Offering both means ticking the stale one by accident,
    or counting every seed twice. So the newest payload for a (group, checkpoint) supersedes the
    older ones here. Newest wins even at a different episode count, which is the same rule
    queries.group_eval_summary applies per run, so the bar panel and the experiment page agree.
    A different checkpoint is a different evaluation and is never folded away.

    'include_superseded' returns the older ones too, flagged 'superseded' -- for resolving a job id
    that is already in a URL, not for listing.

    'run_ids' restricts the list to payloads that actually evaluated runs in that selection --
    the bar panel sits on the Analysis page, where offering every evaluation ever made means the
    chart can be built from experiments the page is not showing. Matching is done on the run
    directories inside each payload rather than on the job's label, because a label is free text
    and two experiments can share one.
    """
    rows = db.query("""SELECT job_id, label, run_path, finished_at FROM jobs
                       WHERE kind='group' AND state='finished' AND run_path LIKE '%.json'
                       ORDER BY finished_at DESC LIMIT ?""", (limit,))
    found = [dict(r) for r in rows if r["run_path"] and Path(r["run_path"]).exists()]

    # Read each payload once: both the supersession and the selection filter need it.
    kept, seen = [], set()
    for entry in found:
        try:
            with open(entry["run_path"]) as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            continue
        key = (payload.get("group"), payload.get("checkpoint"))
        if key in seen:
            if not include_superseded:
                continue
            entry["superseded"] = True
        seen.add(key)
        kept.append((entry, payload))

    if run_ids is None:
        return [entry for entry, _payload in kept]

    wanted = set()
    for run_id in run_ids:
        row = db.one("SELECT path FROM runs WHERE run_id=?", (run_id,))
        if row and row["path"]:
            wanted.add(os.path.abspath(row["path"]))
    if not wanted:
        return []

    matching = []
    for entry, payload in kept:
        dirs = {os.path.abspath(str(Path(row["model"]).parent))
                for row in payload.get("rows", []) if row.get("model")}
        if dirs & wanted:
            entry["n_runs"] = len(dirs)
            entry["n_selected"] = len(dirs & wanted)
            matching.append(entry)
    return matching


def job(job_id):
    row = db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    return dict(row) if row else None


def recent(limit=25):
    return [dict(r) for r in db.query(
        "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,))]


def tail_log(job_id, lines=200):
    path = SETTINGS.log_dir / f"{job_id}.log"
    if not path.exists():
        return ""
    with open(path, errors="replace") as fh:
        return "".join(fh.readlines()[-lines:])


def status():
    with _lock:
        current = dict(_current["task"]) if _current["task"] else None
        started = _current["started"]
    pending = _queue.qsize()
    # ~40 s per 50 episodes, measured. Enough to be worth showing, not enough to promise.
    eta = None
    if current or pending:
        # ~40 s per 50 episodes, measured; a group multiplies that by its seed count.
        seeds = len((current or {}).get("run_ids") or [1])
        per_job = 40.0 * (current or {}).get("episodes", 50) / 50.0 * seeds
        eta = int(per_job * (pending + (1 if current else 0)))
    return {"current": current, "started": started, "pending": pending,
            "concurrency": EVAL_CONCURRENCY, "eta_seconds": eta}


def sweep_rows(run_id, checkpoint):
    """Every fed-goal result for one checkpoint, for the goal-response chart."""
    rows = db.query("""SELECT policy_goal, rolled, side, rho_mean FROM evals
                       WHERE run_id=? AND checkpoint=? AND policy_goal IS NOT NULL
                       ORDER BY policy_goal""", (run_id, checkpoint))
    return [(r["policy_goal"], r["rolled"], r["side"], r["rho_mean"]) for r in rows]


def latest(run_id, checkpoint=None):
    if checkpoint:
        return db.one("""SELECT * FROM evals WHERE run_id=? AND checkpoint=? AND policy_goal IS NULL
                         ORDER BY created_at DESC LIMIT 1""", (run_id, checkpoint))
    return db.one("""SELECT * FROM evals WHERE run_id=? AND policy_goal IS NULL
                     ORDER BY created_at DESC LIMIT 1""", (run_id,))
