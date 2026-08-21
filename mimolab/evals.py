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


def status():
    with _lock:
        current = dict(_current["task"]) if _current["task"] else None
        started = _current["started"]
    pending = _queue.qsize()
    # ~40 s per 50 episodes, measured. Enough to be worth showing, not enough to promise.
    eta = None
    if current or pending:
        per_job = 40.0 * (current or {}).get("episodes", 50) / 50.0
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
