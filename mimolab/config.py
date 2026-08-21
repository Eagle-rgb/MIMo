"""Paths, host pool and runtime settings.

Everything the app writes lives under MIMO_ROOT/.mimolab so that it sits on the shared RBI home:
job state then survives an app restart and stays greppable from any host without the app running.
"""

import os
from pathlib import Path

# The RBI host pool, in the order used by rbi_autorun*.sh. 'anemoi' is last and is the suggested
# app host -- see APP_HOST -- so it is excluded from allocation by default.
HOST_PREFIXES = [
    "adrastos", "alkmene", "ajax", "anaxo", "achilles", "axylos", "aktor",
    "admeta", "amata", "agylla", "adamas", "arabia", "adonis", "aither", "apate",
    "atropos", "aletheia", "acheloos", "anemoi",
]
HOST_DOMAIN = "rbi.cs.uni-frankfurt.de"

# Held out of the training rotation because the app itself runs here.
APP_HOST = "anemoi"

# One MIMo env costs ~3.6 GB RSS; four parallel runs were OOM-killed. The unit of allocation is
# therefore the whole host, and evaluations run one at a time.
ENV_RSS_GB = 3.6
EVAL_CONCURRENCY = 1

AGES = [1, 3, 6, 9]

# Rendering: 'egl' fails in the mimo conda env regardless of what older notes in this repo claim.
MUJOCO_GL = "osmesa"


class Settings:

    def __init__(self, mimo_root=None, models_root=None, offline=False,
                 ssh_user=None, remote_root=None, conda_env="mimo",
                 tb_port=8771, python="python"):
        self.mimo_root = Path(mimo_root or os.environ.get("MIMO_ROOT") or Path.cwd()).resolve()
        self.models_root = Path(models_root or self.mimo_root / "models").resolve()
        # 'offline' disables everything that spawns a process: launching, killing, evaluating.
        # Used when browsing a scp'd copy of models/ on a machine that is not the cluster.
        self.offline = offline
        self.ssh_user = ssh_user or os.environ.get("MIMO_SSH_USER") or os.environ.get("USER")
        # Path to the MIMo checkout *on the RBI hosts*. Shared home, so one value covers all of them.
        self.remote_root = remote_root or os.environ.get("MIMO_REMOTE_ROOT") or "~/MIMo"
        self.conda_env = conda_env
        self.tb_port = tb_port
        self.python = python

        self.state_dir = self.mimo_root / ".mimolab"
        self.log_dir = self.state_dir / "logs"
        self.plot_dir = self.state_dir / "plots"
        self.db_path = self.state_dir / "index.db"

    def ensure_dirs(self):
        for d in (self.state_dir, self.log_dir, self.plot_dir):
            d.mkdir(parents=True, exist_ok=True)

    def hosts(self, include_app_host=False):
        return [h for h in HOST_PREFIXES if include_app_host or h != APP_HOST]

    def fqdn(self, host):
        return host if "." in host else f"{host}.{HOST_DOMAIN}"

    # Remote log paths must be expressed against the remote root, not the local one -- they are
    # the same directory over NFS, but the app may be reading it under a different mount point.
    def remote_log_path(self, job_id):
        return f"{self.remote_root}/.mimolab/logs/{job_id}.log"


SETTINGS = Settings()


def configure(**kwargs):
    global SETTINGS
    SETTINGS = Settings(**kwargs)
    SETTINGS.ensure_dirs()
    return SETTINGS
