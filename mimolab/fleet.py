"""Launching, monitoring and killing training runs on the RBI host pool.

What this adds over rbi_autorun*.sh:

  * The scripts background the ssh itself, so stdout lives in the launching terminal and dies with
    it, and no PID is ever recorded. That is why the only kill available today is
    'killall python' -- which ends every Python process the user owns on that host. Here the
    remote process is detached with setsid+nohup, its output is redirected into the shared home,
    and the PID comes back on stdout and is stored.
  * Hosts are probed immediately before launch rather than assumed free. One MIMo env is ~3.6 GB
    RSS and four parallel runs were OOM-killed, so the unit of allocation is the whole host.
"""

import json
import os
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import db
from .config import SETTINGS, MUJOCO_GL

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8"]

# Processes that mean a host is occupied by this project.
#
# The bracketed first character is the classic self-exclusion trick: the regex still matches
# "illustrations.py", but the probe's own command line contains the literal "[i]llustrations.py",
# which it does not. Without this, pgrep reports the probing shell itself, every genuinely free
# host reads as "foreign", and allocation never finds anywhere to launch.
BUSY_PATTERN = "[i]llustrations.py|[e]val_rollover.py"


def ssh(host, remote_cmd, timeout=25):
    """Run one command on a host. Returns (rc, stdout, stderr)."""
    target = f"{SETTINGS.ssh_user}@{SETTINGS.fqdn(host)}" if SETTINGS.ssh_user else SETTINGS.fqdn(host)
    argv = ["ssh", *SSH_OPTS, target, remote_cmd]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 255, "", "timed out"
    except OSError as exc:
        return 255, "", str(exc)


def probe_host(host):
    """What is this host doing right now?

    Reports 'mine' only for PIDs the registry knows about. A MIMo process we have no record of is
    reported as 'foreign' rather than assumed ours -- the home is shared, so a shell script or a
    second app instance can have launched it, and claiming the host would OOM both runs.
    """
    rc, out, err = ssh(host, f"pgrep -af '{BUSY_PATTERN}' ; echo '---' ; free -m | awk '/Mem:/{{print $7}}'")
    if rc != 0 and not out:
        return {"host": host, "state": "unreachable", "detail": err or f"rc={rc}", "procs": []}

    proc_text, _, mem_text = out.partition("---")
    procs = []
    for line in proc_text.strip().splitlines():
        pid, _, cmd = line.partition(" ")
        if pid.isdigit():
            procs.append({"pid": int(pid), "cmd": cmd.strip()})

    try:
        free_mb = int(mem_text.strip().splitlines()[0])
    except (ValueError, IndexError):
        free_mb = None

    known = {r["pid"] for r in db.query(
        "SELECT pid FROM jobs WHERE host=? AND state IN ('running','launching','stalled')", (host,))}
    if not procs:
        state = "free"
    elif all(p["pid"] in known for p in procs):
        state = "mine"
    else:
        state = "foreign"

    return {"host": host, "state": state, "free_mb": free_mb, "procs": procs,
            "detail": "" if state != "foreign" else "a MIMo process this app did not launch"}


def probe_all(hosts=None, workers=10):
    hosts = hosts or SETTINGS.hosts()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(probe_host, hosts))


def free_hosts(count, probes=None):
    probes = probes if probes is not None else probe_all()
    return [p["host"] for p in probes if p["state"] == "free"][:count]


# ---------------------------------------------------------------------------------------------
# Building the training command
# ---------------------------------------------------------------------------------------------

# Flags the launch form may set. 'yaml' marks the ones that illustrations.py records in data.yml:
# a flag that defines the experiment but is NOT recorded there will not survive --load_model, so
# the run would later be evaluated under different settings than it was trained with.
# check_yaml_coverage() below asserts this list stays in step with the source.
TRAIN_FLAGS = [
    # (flag, kind, default, group, in_yaml)
    ("train_for",                 "int",    1_000_000, "run",      False),
    ("save_every",                "int",    200_000,   "run",      False),
    ("algorithm",                 "choice", "PPO",     "run",      True),
    ("roll_over_starting_position", "choice", "supine", "run",     False),
    ("morph_age",                 "int",    9,         "embodiment", True),
    ("physio_age",                "int",    9,         "embodiment", True),
    ("episode_steps",             "int",    None,      "run",      True),
    ("lr",                        "float",  3e-4,      "optim",    True),
    ("lr_schedule",               "choice", "constant","optim",    True),
    ("target_entropy",            "float",  None,      "optim",    True),
    ("pen_factor",                "float",  0.02,      "reward",   True),
    ("goal_achievement_function", "choice", "cos",     "reward",   True),
    ("pbrs",                      "bool",   False,     "reward",   True),
    ("pbrs_w",                    "float",  100,       "reward",   True),
    ("sparse_reward",             "bool",   False,     "reward",   True),
    ("nopen",                     "bool",   False,     "reward",   True),
    ("side_lying",                "bool",   False,     "reward",   True),
    ("her",                       "bool",   False,     "her",      True),
    ("n_sampled_goal",            "int",    4,         "her",      True),
    ("goal_selection_strategy",   "choice", "future",  "her",      True),
    ("goal_low",                  "float",  None,      "her",      True),
    ("goal_high",                 "float",  None,      "her",      True),
    ("goal_curriculum",           "bool",   False,     "her",      True),
    ("no_done_active",            "bool",   False,     "her",      True),
    ("buffer_size",               "int",    300_000,   "offpolicy", True),
    ("train_freq",                "int",    1,         "offpolicy", True),
    ("gradient_steps",            "int",    1,         "offpolicy", True),
    ("learning_starts",           "int",    100,       "offpolicy", True),
    ("eval_every",                "int",    0,         "eval",     True),
    ("eval_episodes",             "int",    20,        "eval",     True),
    ("isr",                       "bool",   False,     "misc",     True),
    ("obs_noise",                 "float",  0.0,       "misc",     True),
    ("obs_norm",                  "bool",   False,     "misc",     True),
    ("save_intermediate",         "bool",   False,     "misc",     False),
]

CHOICES = {
    "algorithm": ["PPO", "SAC", "TD3", "DDPG", "A2C"],
    "roll_over_starting_position": ["supine", "prone"],
    "lr_schedule": ["constant", "linear"],
    "goal_achievement_function": ["angle", "cos", "intrinsic", "gravity"],
    "goal_selection_strategy": ["future", "final", "episode"],
}

OFF_POLICY = {"SAC", "TD3", "DDPG"}

PRESETS = {
    "ppo_pbrs": {
        "label": "PPO + PBRS",
        "note": "The shaped-reward configuration from rbi_autorun.sh.",
        "flags": {"algorithm": "PPO", "pbrs": True, "pbrs_w": 100, "pen_factor": 0.02,
                   "goal_achievement_function": "cos", "episode_steps": 250,
                   "train_for": 1_000_000, "save_every": 200_000, "lr": 3e-4,
                   "morph_age": 9, "physio_age": 9},
    },
    "sac_her_sparse": {
        "label": "SAC + HER + sparse",
        "note": "The headline roll-over configuration from rbi_autorun_sac.sh.",
        "flags": {"algorithm": "SAC", "her": True, "sparse_reward": True,
                   "goal_low": 0.25, "goal_high": 0.95, "goal_curriculum": True,
                   "no_done_active": True, "eval_every": 25_000, "eval_episodes": 20,
                   "target_entropy": -92, "lr_schedule": "linear", "pen_factor": 0.02,
                   "goal_achievement_function": "cos", "train_for": 1_000_000,
                   "save_every": 200_000, "lr": 3e-4, "morph_age": 9, "physio_age": 9},
    },
}


def check_yaml_coverage(illustrations_path=None):
    """Guard the data.yml round-trip.

    Any experiment-defining flag the form can set must appear in the yaml_data dict in
    illustrations.py, or --load_model will silently evaluate the model under different settings.
    Returns the flags that are missing, so the UI can refuse to offer them.
    """
    path = illustrations_path or (SETTINGS.mimo_root / "mimoEnv" / "illustrations.py")
    try:
        source = open(path).read()
    except OSError:
        return []
    block = source.split("yaml_data = {", 1)
    if len(block) < 2:
        return []
    block = block[1].split("\n    }", 1)[0]
    # The dict stores some flags under a different key than the CLI name.
    aliases = {"goal_achievement_function": "goal_achievement_function", "nopen": "nopen"}
    missing = []
    for flag, _kind, _default, _group, in_yaml in TRAIN_FLAGS:
        if not in_yaml:
            continue
        key = aliases.get(flag, flag)
        if f"'{key}'" not in block and f'"{key}"' not in block:
            missing.append(flag)
    return missing


def validate(values):
    """Check a configuration against the rules illustrations.py actually enforces.

    Returns a list of (severity, message), severity in {"error", "warning"}.

    The split matters and is not cosmetic. illustrations.py *raises* on four combinations and
    merely *warns* or silently auto-corrects on several others; a UI that blocks on the warnings
    refuses configurations that train perfectly well. In particular
    '--pbrs --sparse_reward --no_done_active' is legal -- the guard is
    'pbrs and not sparse_reward and not done_active', because a sparse reward has no potential to
    be discontinuous -- and that is the configuration currently running across the pool.
    """
    out = []
    algorithm = values.get("algorithm", "PPO")
    off_policy = algorithm in OFF_POLICY
    her = bool(values.get("her"))
    sparse = bool(values.get("sparse_reward"))
    pbrs = bool(values.get("pbrs"))
    no_done = bool(values.get("no_done_active"))
    goal_low, goal_high = values.get("goal_low"), values.get("goal_high")

    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # --- the four that raise in illustrations.py -------------------------------------------
    if her and not off_policy:
        out.append(("error", f"--her needs an off-policy algorithm; {algorithm} is not one of "
                             f"{', '.join(sorted(OFF_POLICY))}."))
    if (goal_low in (None, "")) != (goal_high in (None, "")):
        out.append(("error", "Provide both --goal_low and --goal_high, or neither."))
    if values.get("goal_curriculum") and goal_low in (None, ""):
        out.append(("error", "--goal_curriculum needs a goal range: the curriculum moves the upper "
                             "end of it, so with a fixed goal it has nothing to do."))
    if pbrs and not sparse and not no_done:
        # Only unsound while the potential is live. With --sparse_reward there is no potential.
        pass
    if pbrs and not sparse and no_done:
        out.append(("error", "--pbrs with --no_done_active is unsound: the potential is "
                             "discontinuous at the goal, so leaving the goal region pays about "
                             "-pbrs_w * reward_success (~-50000) and the critic diverges. Drop "
                             "--no_done_active, or use --sparse_reward."))

    # --- advisories: illustrations.py warns or auto-corrects, so these must not block --------
    if her and not no_done:
        out.append(("warning", "--her without --no_done_active: episodes terminate on success, so "
                               "relabelled transitions are not marked terminal and the critic "
                               "bootstraps past the virtual goal."))
    if pbrs and sparse:
        out.append(("warning", "--pbrs alongside --sparse_reward: the sparse reward wins and the "
                               "PBRS shaping is inert. Harmless, but --pbrs_w means nothing here."))

    episode_steps = number(values.get("episode_steps")) or 500
    learning_starts = number(values.get("learning_starts"))
    if her and learning_starts is not None and learning_starts <= episode_steps:
        out.append(("warning", f"--learning_starts will be raised automatically to "
                               f"{int(2 * episode_steps)}; it must exceed the "
                               f"{int(episode_steps)}-step horizon under --her."))

    save_every, train_for = number(values.get("save_every")), number(values.get("train_for"))
    if off_policy and save_every and train_for and save_every >= train_for:
        out.append(("warning", "save_every >= train_for saves only the final checkpoint, and for "
                               "off-policy runs the last one is not reliably the best -- a HER run "
                               "collapsed after 600k steps and its final model scored 2 %. "
                               "Consider --save_every=200000."))
    if her and goal_low not in (None, "") and number(goal_low) == number(goal_high):
        out.append(("warning", "--goal_low equals --goal_high: HER needs goal variation, or the "
                               "policy never learns to condition on the goal."))
    return out


def errors(values):
    return [m for severity, m in validate(values) if severity == "error"]


def warnings(values):
    return [m for severity, m in validate(values) if severity == "warning"]


def build_command(values, save_model, remote_root=None):
    """The illustrations.py invocation, as a single shell-quoted string."""
    root = remote_root or SETTINGS.remote_root
    parts = [SETTINGS.python, "mimoEnv/illustrations.py",
             "--roll_over_model_path_auto",
             f"--save_model={shlex.quote(save_model)}"]

    for flag, kind, default, _group, _in_yaml in TRAIN_FLAGS:
        if flag not in values:
            continue
        value = values[flag]
        if value is None or value == "":
            continue
        if kind == "bool":
            if value:
                parts.append(f"--{flag}")
        else:
            parts.append(f"--{flag}={shlex.quote(str(value))}")
    return " ".join(parts), root


def launch(values, save_model, host, label=None):
    """Start one training run on one host. Returns the job row."""
    if SETTINGS.offline:
        raise RuntimeError("offline mode: launching is disabled")

    blocking = errors(values)
    if blocking:
        raise ValueError("; ".join(blocking))

    job_id = f"{time.strftime('%y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    command, root = build_command(values, save_model)
    log_path = SETTINGS.remote_log_path(job_id)

    # setsid detaches from the ssh session so the run survives the connection closing; nohup
    # covers the SIGHUP; stdin from /dev/null stops it blocking on a read. 'exec' makes the PID
    # echoed back the Python process itself rather than a wrapping shell, so kill hits the run.
    inner = (f"mkdir -p {shlex.quote(root)}/.mimolab/logs && "
             f"conda activate {shlex.quote(SETTINGS.conda_env)} && "
             f"cd {shlex.quote(root)} && "
             f"MUJOCO_GL={MUJOCO_GL} exec {command}")
    remote = (f"setsid nohup bash -lc {shlex.quote(inner)} "
              f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!")

    db.execute("""INSERT INTO jobs (job_id, kind, host, pid, cmd, label, log_path,
                                    started_at, state)
                  VALUES (?,?,?,?,?,?,?,?,?)""",
               (job_id, "train", host, None, command, label or save_model, log_path,
                time.time(), "launching"))

    rc, out, err = ssh(host, remote, timeout=40)
    pid = int(out.strip().splitlines()[-1]) if rc == 0 and out.strip().splitlines() \
        and out.strip().splitlines()[-1].isdigit() else None

    if pid is None:
        db.execute("UPDATE jobs SET state='failed', finished_at=?, note=? WHERE job_id=?",
                   (time.time(), (err or out or f"rc={rc}")[:2000], job_id))
    else:
        db.execute("UPDATE jobs SET pid=?, state='running' WHERE job_id=?", (pid, job_id))

    return dict(db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,)))


def launch_sweep(values, name, n_seeds, hosts=None):
    """One job per seed, named <name>_run_<i> to match the existing convention.

    The naming matters beyond tidiness: results/ scripts locate a batch by --date and --suffix
    built from exactly this pattern, so a sweep launched here stays readable by them.
    """
    probes = probe_all()
    chosen = hosts or free_hosts(n_seeds, probes)
    if len(chosen) < n_seeds:
        raise RuntimeError(
            f"only {len(chosen)} of {n_seeds} hosts are free "
            f"({sum(1 for p in probes if p['state'] == 'foreign')} busy with foreign processes, "
            f"{sum(1 for p in probes if p['state'] == 'unreachable')} unreachable)")

    jobs = []
    for i, host in enumerate(chosen):
        jobs.append(launch(values, f"{name}_run_{i}", host, label=f"{name} #{i}"))
    return jobs


def kill_job(job_id):
    """Terminate one run by PID. Never 'killall python' -- that ends unrelated work."""
    row = db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    if row is None:
        raise KeyError(job_id)
    if row["pid"] is None:
        db.execute("UPDATE jobs SET state='failed', finished_at=? WHERE job_id=?",
                   (time.time(), job_id))
        return {"ok": False, "detail": "no PID was ever recorded for this job"}

    rc, out, err = ssh(row["host"], f"kill -TERM {int(row['pid'])} && echo killed")
    ok = "killed" in out
    db.execute("UPDATE jobs SET state=?, finished_at=?, note=? WHERE job_id=?",
               ("killed" if ok else row["state"], time.time() if ok else None,
                (err or "")[:500] or None, job_id))
    return {"ok": ok, "detail": err or out}


def refresh(job_ids=None):
    """Reconcile stored job state with what is actually running."""
    where = "WHERE state IN ('launching','running','stalled')"
    binds = []
    if job_ids:
        where += f" AND job_id IN ({','.join('?' * len(job_ids))})"
        binds = list(job_ids)
    rows = db.query(f"SELECT * FROM jobs {where}", binds)
    if not rows:
        return []

    by_host = {}
    for row in rows:
        by_host.setdefault(row["host"], []).append(row)

    def check(host):
        rc, out, _ = ssh(host, "pgrep -af '%s' || true" % BUSY_PATTERN)
        alive = set()
        for line in out.splitlines():
            pid, _, _ = line.partition(" ")
            if pid.isdigit():
                alive.add(int(pid))
        return host, alive

    with ThreadPoolExecutor(max_workers=10) as pool:
        alive_by_host = dict(pool.map(check, by_host))

    updated = []
    for host, host_rows in by_host.items():
        alive = alive_by_host.get(host, set())
        for row in host_rows:
            state = "running" if row["pid"] in alive else "finished"
            if state != row["state"]:
                db.execute(
                    "UPDATE jobs SET state=?, finished_at=? WHERE job_id=?",
                    (state, time.time() if state == "finished" else None, row["job_id"]))
            updated.append({"job_id": row["job_id"], "state": state})
    return updated


def tail_log(job_id, lines=200):
    row = db.one("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    if row is None:
        return ""
    # The log lives on the shared home, so it is readable locally when the app runs on the cluster.
    local = SETTINGS.log_dir / f"{job_id}.log"
    if local.exists():
        with open(local, errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    rc, out, err = ssh(row["host"], f"tail -n {int(lines)} {shlex.quote(row['log_path'])}")
    return out if rc == 0 else (err or "log not readable")


def jobs(limit=100, states=None):
    where, binds = "", []
    if states:
        where = f"WHERE state IN ({','.join('?' * len(states))})"
        binds = list(states)
    return [dict(r) for r in db.query(
        f"SELECT * FROM jobs {where} ORDER BY started_at DESC LIMIT ?", binds + [limit])]
