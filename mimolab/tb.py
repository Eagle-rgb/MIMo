"""On-demand TensorBoard for a selection of runs.

Not proxied into the page: TensorBoard has its own routing and embedding it means fighting
--path_prefix for little gain. It is spawned on a reserved port instead and linked out through
the same SSH tunnel, one instance at a time, replaced on the next request.

--logdir_spec, not --logdir: it gives each run its own legend name. Pointed at a shared parent
directory TensorBoard invents names from the path, which for this fork's long run names is
unreadable.
"""

import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import db
from .config import SETTINGS

_lock = threading.Lock()
_state = {"proc": None, "runs": [], "started": None, "port": None}

# A TensorBoard nobody has asked about for this long is reaped.
IDLE_SECONDS = 60 * 60


def _spec(run_ids):
    """name:path pairs, with names short enough for the legend and unique within the selection."""
    parts, used = [], set()
    for run_id in run_ids:
        row = db.one("SELECT path, model_name, seed_idx, posture FROM runs WHERE run_id=?", (run_id,))
        if row is None:
            continue
        seed = "" if row["seed_idx"] is None else f"_{row['seed_idx']}"
        name = f"{(row['model_name'] or 'run')[:28]}{seed}_{row['posture'][:3]}"
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        base, i = name, 2
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        parts.append(f"{name}:{row['path']}")
    return ",".join(parts)


def status():
    with _lock:
        proc = _state["proc"]
        alive = proc is not None and proc.poll() is None
        return {"alive": alive, "runs": list(_state["runs"]), "port": _state["port"],
                "started": _state["started"],
                "url": f"http://localhost:{_state['port']}" if alive else None}


def stop():
    with _lock:
        proc = _state["proc"]
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        _state.update({"proc": None, "runs": [], "started": None, "port": None})
    return {"alive": False}


def launch(run_ids):
    """Replace any running instance with one serving exactly these runs."""
    if SETTINGS.offline:
        raise RuntimeError("offline mode: TensorBoard is disabled")
    spec = _spec(run_ids)
    if not spec:
        raise ValueError("none of those runs are in the index")

    stop()
    port = SETTINGS.tb_port
    argv = ["tensorboard", f"--logdir_spec={spec}", f"--port={port}",
            "--host=127.0.0.1", "--reload_interval=30"]
    proc = subprocess.Popen(argv, cwd=SETTINGS.mimo_root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with _lock:
        _state.update({"proc": proc, "runs": list(run_ids), "started": time.time(), "port": port})

    # TensorBoard takes a moment to bind; a link handed over too early 404s in the browser.
    for _ in range(40):
        if proc.poll() is not None:
            raise RuntimeError("tensorboard exited immediately -- is it on PATH?")
        try:
            import socket
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.25)
    return status()


def reap_if_idle():
    with _lock:
        started = _state["started"]
        alive = _state["proc"] is not None and _state["proc"].poll() is None
    if alive and started and time.time() - started > IDLE_SECONDS:
        stop()
