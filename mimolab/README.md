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
| `evals.py` | The serial evaluation queue around `eval_rollover.py`, single runs and `--group` |
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

**Experiments are evaluated as a group, and the `--group` path is verified before it is used.**
The experiment page runs `eval_rollover.py --group` over every seed -- the *last* checkpoint of
each, never `model_best.zip`, with identical episode seeds so the runs are paired. The spec is the
run path with the `_run_<i>` tail removed, but that prefix is expanded and compared against the
experiment's own runs before being offered: some experiments were sorted by hand into
subdirectories (`intrinsic_only_vesti/bad/..._run_7` next to `good/..._run_11`), and a path wide
enough to catch those would also evaluate other experiments living in the same parent. 90 of 94
experiments get an exact spec; the other 4 are refused with an explanation rather than quietly
evaluating the wrong set. Results are folded back into `evals` keyed by run directory, so the
group summary and the per-run pages cannot disagree.

**One evaluation at a time.** A MIMo env is ~3.6 GB RSS, so the queue depth of 1 is a correctness
constraint, not a tunable. Subprocesses are launched with `sys.executable`, not `"python"` -- the
server runs inside the `mimo` env but a subprocess inherits `PATH`, where `python` can be the base
interpreter with no numpy, and that only fails when an evaluation is finally launched.

**The tag list describes the selection, not the archive.** The corpus spans a year of changing
callbacks, so the union of all tags offers curriculum and eval metrics that most selections never
recorded -- picking one drew an empty chart. `tags_for()` restricts the dropdown to tags the
selected runs actually logged: 30 instead of 63 for a typical experiment.

**The configuration is shown three ways.** A key/value table, the raw `data.yml` verbatim
(`/api/config/<run_id>`), and -- on an experiment -- split into what all seeds agree on and where
they disagree. Seeds should differ only by the random seed; anything else that varies is either a
deliberate sub-sweep or a mistake, and either way it is the first thing worth seeing.

**Thesis figures are rendered at their final size, with the legend you choose.** The Analysis page
has a `thesis` style that follows `results/training_plot_no_sig.py` and the reference figure in
`results/proprio_ablations/`: `results/icdlplot.py`'s palette, a mean ± std band clipped to [0, 1],
a dashed grid, a framed legend inside the axes, and serif axis names taken from a lookup rather
than the raw tag. Each series gets a free-text legend label, keyed on `date|posture|model_name`, so
renaming one cannot move it onto another line; `series_of()` lists them in the order the renderer
groups them, so the swatch beside an input is the colour of the line it names. The page is emitted
at 3.4 x 3.2 in (one column) or 6.9 x 3.6 in (two), which is why `\includegraphics` needs no
`width=` -- rescaling in LaTeX is what makes the type in one figure disagree with the next.

**The evaluation bar chart is drawn by `results/plot_eval_success.py`, not reimplemented.** A
`--group` run keeps its JSON in `.mimolab/evals/`, and the bar panel shells out to that script with
the stored payloads. It is the script the thesis figures already come from, so a second
implementation would only give the document two subtly different charts. Column width maps to
`--panel_width`, divided by the number of postures the payload contains, because the script lays
one panel out per posture and multiplies the width by their count.

**Charts export as vector PDF, re-laid-out rather than re-encoded.** Every chart has an Export PDF
control; the link is the chart's own URL with `.png` swapped for `.pdf`, so the figure you export
is the figure you are looking at. The PDF is not the screen figure in another container: it is
rendered again in the paper style from `results/icdlplot.py` (STIX serif, 10 pt, `pdf.fonttype 42`
so no Type 3 fonts reach a thesis template), always light, at a real column width -- 3.5 in single
or 7.0 in double. That needs `layout="constrained"` and *no* `bbox_inches="tight"`: tight cropping
grows the page to whatever the legend needs, which turned a 3.5 in request into 7.65 in. In paper
mode the on-screen caption and the right-hand direct labels are dropped, and the legend moves to
`loc="outside lower center"` so constrained layout reserves room instead of letting it land on the
x-axis label.

**TensorBoard is spawned, not proxied**, with `--logdir_spec` so each run gets a legible legend
name instead of one invented from a shared parent path.
