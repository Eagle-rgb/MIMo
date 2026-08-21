# mimolab

Browse every trained roll-over policy, launch and watch sweeps on the RBI pool, run evaluations
and plot the results. Server-rendered FastAPI + HTMX; no node, no build step.

## Running it

Because `~/MIMo` is a shared network home, one process on one RBI host sees every run every other
host ever wrote. Run it on a host held out of the training rotation and reach it over SSH:

```bash
# on the app host
conda activate mimo && cd ~/MIMo
python -m mimolab --ssh-user=<rbi-user>

# from the laptop
ssh -L 8770:localhost:8770 -L 8771:localhost:8771 <rbi-user>@anemoi.rbi.cs.uni-frankfurt.de
# then open http://localhost:8770   (8771 serves on-demand TensorBoard)
```

It binds to `127.0.0.1` deliberately: the app can start and kill processes across the pool, so it
must not be reachable from the department network. The tunnel is the authentication.

To browse a local copy of `models/` with every write path disabled:

```bash
python -m mimolab --offline
```

Verify an install against the real `models/` directory:

```bash
python -m mimolab.selfcheck
```

## What it is built on

The run directory *is* the database. `models/roll_over/<date>/<posture>/<run>/` already holds the
hyperparameters (`data.yml`), the training history (the TensorBoard event file) and the artefacts
(`model_*.zip`), so nothing new is recorded at training time and `illustrations.py` is unchanged.

| Module | Job |
|---|---|
| `indexer.py` | Walks `models/`, parses `data.yml` + event files into SQLite |
| `queries.py` | Filtering, faceting, grouping by experiment |
| `plots.py` | matplotlib charts (light and dark), rendered server-side |
| `fleet.py` | Host probing, detached launch over SSH, PID registry, kill |
| `evals.py` | The serial evaluation queue around `eval_rollover.py` |
| `tb.py` | On-demand TensorBoard for a selection |
| `app.py` | Routes |

State lives in `~/MIMo/.mimolab/` — on the shared home on purpose, so job state survives an app
restart and every log stays greppable from any host without the app running.

## Things that are the way they are for a reason

**Posture comes from the path, never from `data.yml`.** All 539 stored runs omit
`roll_over_starting_position` — it is on the deliberate exclusion list in `illustrations.py`.
Anything reading posture from the yaml silently gets `supine` for prone runs. (That was a live bug
in `eval_rollover.py`; it now falls back to the path.)

**The index is a cache keyed on `(path, mtime, size)`.** A cold build over 800 event files takes
~100 s; the steady state is a directory walk plus 800 `stat` calls, about 1.3 s. Scalars are
stored as packed blobs per `(run, tag)` — row-per-point cost 4.3 M rows and 800 MB for the same
data that now fits in 40 MB.

**One run per host, one evaluation at a time.** A MIMo env is ~3.6 GB RSS and four parallel runs
were OOM-killed. The queue depth of 1 is a correctness constraint, not a tunable.

**Launching is detached and kills are by PID.** `rbi_autorun*.sh` backgrounds the `ssh` itself, so
stdout dies with the terminal and no PID is recorded — which is why the only kill available there
is `killall python`, ending every Python process the user owns on that host. Here the remote
process is started under `setsid nohup`, logged into the shared home, and its PID stored.

**The launch form is checked against `yaml_data`.** `fleet.check_yaml_coverage()` reads
`illustrations.py` and reports any experiment-defining flag the form can set that would not
round-trip through `data.yml`. The UI shows it as a blocking notice; the selfcheck asserts it.

**Dates are a timeline, every other facet is a set.** The date facet is ordered newest-first by
value; every other facet is ordered by count, because there the useful question is where the data
is. Four quick presets sit above it -- Latest, Latest per experiment, Last 7 / 30 days. "Latest per
experiment" resolves each `model_name`'s newest date over the whole corpus, not within the current
filter, so it does not shift as you change other filters.

**Experiments are the default view.** 565 runs collapse to 69 experiments, and the seeds of one
experiment are rarely interesting apart from each other, so the table groups by
`(date, posture, model_name)` unless `view=runs` says otherwise. The control shows both states
rather than naming the one you are not in.

**The Runs tab remembers its filters in a cookie**, set by both the page and the htmx fragment
(filters normally change through the fragment alone, so storing it on the page only would remember
a stale query). Analysis and Jobs have their own URLs, so the link has to be built server-side
before the browser gets there; a plain link to `/` clears the cookie, which is what makes "Reset
filters" work.

**The address bar always holds a real page URL.** `hx-push-url="true"` pushes the URL of the
request htmx made -- `/fragments/runs?<filters>`, which renders no navigation -- so a reload
stranded the user on a bare table. The fragment sends an `HX-Push-Url` header naming the page URL
instead, and any plain browser navigation to a `/fragments/*` route is redirected to the page that
renders it properly. That is what makes Refresh, bookmarking and the back button work.

**TensorBoard is spawned, not proxied**, with `--logdir_spec` so each run gets a legible legend
name instead of one invented from a shared parent path.
