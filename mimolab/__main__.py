"""python -m mimolab -- start the server.

Defaults to the current checkout and 127.0.0.1. Binding to loopback is deliberate: the app can
launch and kill processes on the cluster, so it must not be reachable from the department network.
The SSH tunnel is the authentication.
"""

import argparse

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

    configure(mimo_root=args.mimo_root, models_root=args.models_root, offline=args.offline,
              ssh_user=args.ssh_user, remote_root=args.remote_root, conda_env=args.conda_env,
              tb_port=args.tb_port)

    uvicorn.run("mimolab.app:app", host=args.host, port=args.port, reload=args.reload,
                log_level="info")


if __name__ == "__main__":
    main()
