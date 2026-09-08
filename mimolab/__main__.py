"""python -m mimolab -- start the server.

Defaults to the current checkout and 127.0.0.1. Binding to loopback is deliberate: the app can
launch and kill processes on the cluster, so it must not be reachable from the department network.
The SSH tunnel is the authentication.
"""

import argparse
import os
from pathlib import Path

import uvicorn

from .config import configure


def main():
    parser = argparse.ArgumentParser(prog="mimolab")
    parser.add_argument("--mimo-root", default=None, help="MIMo checkout (default: cwd)")
    parser.add_argument("--models-root", default=None, help="override models/ location")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8770, type=int)
    parser.add_argument("--tb-port", default=8771, type=int,
                        help="port for on-demand TensorBoard; tunnel this too")
    parser.add_argument("--ssh-user", default=None, help="RBI username")
    parser.add_argument("--remote-root", default=None,
                        help="path to the MIMo checkout on the RBI hosts (default ~/MIMo)")
    parser.add_argument("--conda-env", default="mimo")
    parser.add_argument("--offline", action="store_true",
                        help="browse only: no launching, killing or evaluating")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    settings = configure(mimo_root=args.mimo_root, models_root=args.models_root,
                         offline=args.offline, ssh_user=args.ssh_user,
                         remote_root=args.remote_root, conda_env=args.conda_env,
                         tb_port=args.tb_port)

    # --reload imports the app in a child process that never runs this function, so the settings
    # have to travel by environment or they revert to the defaults there -- an --offline server
    # that quietly becomes writable is the failure that matters.
    os.environ.update({
        "MIMO_ROOT": str(settings.mimo_root),
        "MIMO_MODELS_ROOT": str(settings.models_root),
        "MIMO_OFFLINE": "1" if settings.offline else "0",
        "MIMO_SSH_USER": settings.ssh_user or "",
        "MIMO_REMOTE_ROOT": settings.remote_root,
        "MIMO_CONDA_ENV": settings.conda_env,
        "MIMO_TB_PORT": str(settings.tb_port),
    })

    # reload_dirs: templates and static are not Python, so uvicorn does not watch them by default
    # and a template edit would not reach the page -- which is most of this app.
    uvicorn.run("mimolab.app:app", host=args.host, port=args.port, reload=args.reload,
                reload_dirs=[str(Path(__file__).resolve().parent)] if args.reload else None,
                reload_includes=["*.py", "*.html", "*.css", "*.js"] if args.reload else None,
                log_level="info")


if __name__ == "__main__":
    main()
