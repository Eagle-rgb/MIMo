"""MIMo Lab -- browse trained roll-over policies, launch runs on the RBI pool, evaluate and plot.

This package deliberately imports nothing from mimoEnv at request time. Training and evaluation
are shelled out to 'mimoEnv/illustrations.py' and 'mimoEnv/eval_rollover.py' exactly as they are
run by hand, which keeps the ~3.6 GB MuJoCo model out of the web process.
"""

__version__ = "0.1.0"
