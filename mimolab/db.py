"""SQLite schema and connection handling.

The run directory is the source of truth; this database is a cache over it and can be deleted at
any time. The one exception is 'jobs', which records processes launched on the cluster -- that
table is authoritative, because a PID exists nowhere else on disk.
"""

import array
import json
import sqlite3
import threading
from pathlib import Path

from .config import SETTINGS

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,   -- path relative to models_root
    path              TEXT NOT NULL,
    date              TEXT,               -- yy-mm-dd
    posture           TEXT,               -- prone | supine | unknown
    model_name        TEXT,               -- the --save_model stem, seed suffix removed
    seed_idx          INTEGER,            -- the _run_<i> suffix
    collection        TEXT,               -- top-level grouping: roll_over | arch | ...
    algorithm         TEXT,
    goal_fn           TEXT,               -- goal_achievement_function
    her               INTEGER,
    sparse_reward     INTEGER,
    pbrs              INTEGER,
    pbrs_w            REAL,
    pen_factor        REAL,
    morph_age         INTEGER,
    physio_age        INTEGER,
    episode_steps     INTEGER,
    goal_low          REAL,
    goal_high         REAL,
    goal_curriculum   INTEGER,
    no_done_active    INTEGER,
    isr               INTEGER,
    side_lying        INTEGER,
    lr                REAL,
    lr_schedule       TEXT,
    num_train         INTEGER,
    target_entropy    REAL,
    buffer_size       INTEGER,
    obs_noise         REAL,
    intrinsic_goal    TEXT,
    proprio_config    TEXT,
    reward_shape      TEXT,               -- sparse | pbrs | distance
    yaml              TEXT,               -- full data.yml as JSON, for anything not columnised
    n_checkpoints     INTEGER,
    checkpoints       TEXT,               -- JSON list of filenames
    last_step         INTEGER,
    best_rho          REAL,               -- max of rollout/ep_rho_max_mean, if logged
    final_rho         REAL,
    event_mtime       REAL,
    state             TEXT,               -- running | done | partial | unknown
    indexed_at        REAL
);
CREATE INDEX IF NOT EXISTS runs_date ON runs(date);
CREATE INDEX IF NOT EXISTS runs_name ON runs(model_name);

-- One row per (run, tag), with the series packed into two blobs. Stored row-per-point this
-- table was 4.3 M rows and 800 MB for 539 runs, because run_id and tag were repeated on every
-- sample; packed it is a few tens of MB and a curve is a single read.
CREATE TABLE IF NOT EXISTS scalars (
    run_id  TEXT NOT NULL,
    tag     TEXT NOT NULL,
    n       INTEGER NOT NULL,
    steps   BLOB NOT NULL,          -- array('i')
    vals    BLOB NOT NULL,          -- array('f')
    PRIMARY KEY (run_id, tag)
) WITHOUT ROWID;

-- Cache key for the incremental indexer: an event file is re-parsed only when its size or mtime
-- changed. A cold build over 800 files takes ~63 s; this makes the steady state stat-only.
CREATE TABLE IF NOT EXISTS event_files (
    path    TEXT PRIMARY KEY,
    run_id  TEXT NOT NULL,
    mtime   REAL NOT NULL,
    size    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS evals (
    eval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    checkpoint    TEXT NOT NULL,
    episodes      INTEGER,
    goal          REAL,
    policy_goal   REAL,
    episode_steps INTEGER,
    starting_position TEXT,
    rolled        REAL,
    side          REAL,
    rho_mean      REAL,
    rho_min       REAL,
    rho_max       REAL,
    steps_mean    REAL,
    raw           TEXT,
    created_at    REAL
);
CREATE INDEX IF NOT EXISTS evals_run ON evals(run_id);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,            -- train | eval
    host        TEXT,
    pid         INTEGER,
    cmd         TEXT,
    label       TEXT,
    run_path    TEXT,                     -- filled in once the run dir is known
    log_path    TEXT,
    started_at  REAL,
    finished_at REAL,
    exit_code   INTEGER,
    state       TEXT,                     -- launching | running | stalled | finished | failed | killed
    note        TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
"""


def connect():
    """One connection per thread. FastAPI's threadpool runs sync handlers on many threads."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        SETTINGS.ensure_dirs()
        conn = sqlite3.connect(SETTINGS.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        _local.conn = conn
    return conn


def reset_connection():
    """Drop the cached handle, e.g. after configure() moved the database."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn = None


def query(sql, params=()):
    return connect().execute(sql, params).fetchall()


def one(sql, params=()):
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql, params=()):
    conn = connect()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def as_dict(row):
    if row is None:
        return None
    d = dict(row)
    if "yaml" in d and d["yaml"]:
        try:
            d["yaml_parsed"] = json.loads(d["yaml"])
        except (ValueError, TypeError):
            d["yaml_parsed"] = {}
    return d


def pack_series(points):
    """(step, value) pairs -> two little-endian blobs."""
    steps = array.array("i", [int(s) for s, _ in points])
    vals = array.array("f", [float(v) for _, v in points])
    if array.array("i", [1]).tobytes()[0] != 1:      # big-endian host
        steps.byteswap()
        vals.byteswap()
    return steps.tobytes(), vals.tobytes()


def unpack_series(steps_blob, vals_blob):
    steps = array.array("i")
    vals = array.array("f")
    steps.frombytes(steps_blob)
    vals.frombytes(vals_blob)
    if array.array("i", [1]).tobytes()[0] != 1:
        steps.byteswap()
        vals.byteswap()
    return list(steps), list(vals)


def series(run_id, tag):
    """The (steps, values) of one run's tag, or ([], []) if it was never logged."""
    row = one("SELECT steps, vals FROM scalars WHERE run_id=? AND tag=?", (run_id, tag))
    if row is None:
        return [], []
    return unpack_series(row["steps"], row["vals"])
