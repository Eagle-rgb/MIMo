"""Walk models/, parse data.yml and TensorBoard event files into the SQLite index.

Cost, measured on this checkout: ~63 s to parse all 800 event files cold (1.56 s per 20). Far too
slow per request, entirely fine as a cached build -- so each event file is keyed on (path, mtime,
size) and re-parsed only when it changed. The steady state is a directory walk plus 800 stat calls.
"""

import json
import re
import time
from pathlib import Path

import yaml

from .config import SETTINGS
from . import db

# Per tag, per run. Enough to draw a curve; small enough that 800 runs stay a few MB.
MAX_POINTS = 500

# A run whose event file has not advanced in this long is reported as stalled rather than running,
# which a PID check alone would never catch.
STALL_SECONDS = 20 * 60

RUN_DIR_RE = re.compile(r"^(\d{2}-\d{2}-\d{2})_(prone|supine)_(.*?)(?:_run_(\d+))?$")

# Columns lifted out of data.yml into their own field. Everything else stays in the yaml blob.
YAML_COLUMNS = [
    "algorithm", "her", "sparse_reward", "pbrs", "pbrs_w", "pen_factor",
    "morph_age", "physio_age", "episode_steps", "goal_low", "goal_high",
    "goal_curriculum", "no_done_active", "isr", "side_lying", "lr",
    "lr_schedule", "num_train", "target_entropy", "buffer_size", "obs_noise",
    "intrinsic_goal", "proprio_config",
]

# Stored under a different name in data.yml than in the index.
YAML_ALIASES = {"goal_fn": "goal_achievement_function"}

HEADLINE_TAG = "rollout/ep_rho_max_mean"


def parse_run_dir(run_path, models_root):
    """Identity of a run from its path.

    Posture comes from the directory name, never from data.yml: all 539 data.yml files in this
    checkout omit 'roll_over_starting_position' -- it is on the deliberate exclusion list in
    illustrations.py. Anything reading posture out of the yaml silently gets 'supine' for prone
    runs, which is exactly the bug in eval_rollover.py's default.
    """
    rel = run_path.relative_to(models_root)
    name = run_path.name
    parts = rel.parts

    date = posture = model_name = None
    seed_idx = None

    m = RUN_DIR_RE.match(name)
    if m:
        date, posture, model_name, seed = m.groups()
        seed_idx = int(seed) if seed is not None else None
    else:
        # The 12 video-render dirs (vid_white_prone, pinkvid_smoke, model) do not follow the
        # convention. Recover what we can rather than dropping them from the index.
        model_name = name
        for token in ("prone", "supine"):
            if name.endswith("_" + token) or f"_{token}_" in name:
                posture = token
                break

    if posture is None:
        for token in ("prone", "supine"):
            if token in parts:
                posture = token
                break
    if date is None:
        for part in parts:
            if re.fullmatch(r"\d{2}-\d{2}-\d{2}", part):
                date = part
                break

    return {
        "run_id": str(rel),
        "path": str(run_path),
        "date": date,
        "posture": posture or "unknown",
        "model_name": model_name,
        "seed_idx": seed_idx,
        "collection": parts[0] if parts else "",
    }


def apply_yaml_shims(cfg):
    """Mirror the back-compat rewrites in mimoEnv.utils.load_model_yaml.

    Not cosmetic: 354 of the 539 stored runs predate the morph_age/physio_age split and record a
    single 'age'. Reading the raw yaml leaves both ages null for them, so an age filter would hide
    two thirds of the corpus. The app must see a run the way --load_model would.
    """
    cfg = dict(cfg)
    if "age" in cfg:
        age = cfg.pop("age")
        cfg.setdefault("morph_age", age)
        cfg.setdefault("physio_age", age)
    if cfg.get("proprio_only_qpos"):
        cfg.setdefault("proprio_config", "position")
    if cfg.get("no_proprio"):
        cfg.setdefault("proprio_config", "")
    return cfg


def reward_shape(cfg):
    if cfg.get("sparse_reward"):
        return "sparse"
    if cfg.get("pbrs"):
        return "pbrs"
    return "distance"


def load_yaml(path):
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def find_event_files(run_path):
    return sorted(run_path.glob("**/events.out.tfevents.*"))


def downsample(points):
    """Uniform stride down to MAX_POINTS, always keeping the first and last sample."""
    if len(points) <= MAX_POINTS:
        return points
    stride = len(points) / float(MAX_POINTS - 1)
    picked = [points[int(i * stride)] for i in range(MAX_POINTS - 1)]
    picked.append(points[-1])
    return picked


def read_scalars(event_paths):
    """Parse scalar tags out of one run's event files.

    Imported lazily: pulling in tensorboard costs ~1 s and the app should start without it when
    the index is already warm.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged = {}
    for path in event_paths:
        try:
            acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:
            # A run killed mid-write leaves a truncated event file. Index what parsed and move on.
            continue
        for tag in acc.Tags().get("scalars", []):
            try:
                events = acc.Scalars(tag)
            except Exception:
                continue
            merged.setdefault(tag, []).extend((e.step, e.value) for e in events)

    for tag, points in merged.items():
        points.sort(key=lambda p: p[0])
        merged[tag] = downsample(points)
    return merged


def derive_state(last_step, num_train, event_mtime, now):
    if last_step is None:
        return "unknown"
    if num_train and last_step >= 0.99 * num_train:
        return "done"
    if event_mtime and now - event_mtime < STALL_SECONDS:
        return "running"
    return "partial"


def index_run(run_path, models_root, conn, force=False, now=None):
    """Index one run directory. Returns (run_id, reparsed_events)."""
    now = now or time.time()
    ident = parse_run_dir(run_path, models_root)
    run_id = ident["run_id"]
    cfg = apply_yaml_shims(load_yaml(run_path / "data.yml"))

    checkpoints = sorted(p.name for p in run_path.glob("model*.zip"))
    event_paths = find_event_files(run_path)

    # Has anything changed since the last index pass?
    cached = {r["path"]: (r["mtime"], r["size"]) for r in
              conn.execute("SELECT path, mtime, size FROM event_files WHERE run_id=?", (run_id,))}
    current = {}
    for path in event_paths:
        try:
            st = path.stat()
        except OSError:
            continue
        current[str(path)] = (st.st_mtime, st.st_size)

    stale = force or current != cached
    event_mtime = max((m for m, _ in current.values()), default=None)

    if stale and event_paths:
        scalars = read_scalars(event_paths)
        conn.execute("DELETE FROM scalars WHERE run_id=?", (run_id,))
        rows = []
        for tag, points in scalars.items():
            if not points:
                continue
            steps_blob, vals_blob = db.pack_series(points)
            rows.append((run_id, tag, len(points), steps_blob, vals_blob))
        conn.executemany(
            "INSERT INTO scalars (run_id, tag, n, steps, vals) VALUES (?,?,?,?,?)", rows)
        conn.execute("DELETE FROM event_files WHERE run_id=?", (run_id,))
        conn.executemany(
            "INSERT OR REPLACE INTO event_files (path, run_id, mtime, size) VALUES (?,?,?,?)",
            [(p, run_id, m, s) for p, (m, s) in current.items()])
    elif not event_paths:
        scalars = {}
    else:
        scalars = None  # unchanged; the stored rows still stand

    if scalars is not None:
        last_step = max((points[-1][0] for points in scalars.values() if points), default=None)
        headline = scalars.get(HEADLINE_TAG) or []
    else:
        last_step = None
        headline = []
        for r in conn.execute("SELECT tag, steps, vals FROM scalars WHERE run_id=?", (run_id,)):
            steps, vals = db.unpack_series(r["steps"], r["vals"])
            if steps:
                last_step = steps[-1] if last_step is None else max(last_step, steps[-1])
            if r["tag"] == HEADLINE_TAG:
                headline = list(zip(steps, vals))

    best_rho = max((v for _, v in headline), default=None)
    final_rho = headline[-1][1] if headline else None

    record = dict(ident)
    record.update({key: cfg.get(key) for key in YAML_COLUMNS})
    record.update({col: cfg.get(key) for col, key in YAML_ALIASES.items()})
    for flag in ("her", "sparse_reward", "pbrs", "goal_curriculum", "no_done_active",
                 "isr", "side_lying"):
        record[flag] = int(bool(record.get(flag)))
    record.update({
        "reward_shape": reward_shape(cfg),
        "yaml": json.dumps(cfg),
        "n_checkpoints": len(checkpoints),
        "checkpoints": json.dumps(checkpoints),
        "last_step": last_step,
        "best_rho": best_rho,
        "final_rho": final_rho,
        "event_mtime": event_mtime,
        "state": derive_state(last_step, cfg.get("num_train"), event_mtime, now),
        "indexed_at": now,
    })

    columns = list(record)
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({','.join(columns)}) "
        f"VALUES ({','.join('?' * len(columns))})",
        [record[c] for c in columns])
    return run_id, stale


def walk_runs(models_root):
    """Every directory holding a data.yml is a run."""
    for data_yml in sorted(Path(models_root).glob("**/data.yml")):
        yield data_yml.parent


def reindex(models_root=None, force=False, progress=None):
    models_root = Path(models_root or SETTINGS.models_root)
    if not models_root.exists():
        return {"runs": 0, "reparsed": 0, "seconds": 0.0, "error": f"no such directory: {models_root}"}

    conn = db.connect()
    started = time.time()
    seen, reparsed = [], 0
    run_dirs = list(walk_runs(models_root))

    for i, run_path in enumerate(run_dirs):
        run_id, stale = index_run(run_path, models_root, conn, force=force)
        seen.append(run_id)
        reparsed += int(bool(stale))
        if progress and i % 25 == 0:
            progress(i + 1, len(run_dirs))
    conn.commit()

    # Drop runs whose directory disappeared, so the table cannot outlive the filesystem.
    known = {r["run_id"] for r in conn.execute("SELECT run_id FROM runs")}
    for gone in known - set(seen):
        conn.execute("DELETE FROM runs WHERE run_id=?", (gone,))
        conn.execute("DELETE FROM scalars WHERE run_id=?", (gone,))
        conn.execute("DELETE FROM event_files WHERE run_id=?", (gone,))
    conn.commit()

    return {"runs": len(seen), "reparsed": reparsed, "removed": len(known - set(seen)),
            "seconds": round(time.time() - started, 1)}
