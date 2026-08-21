"""FastAPI routes for MIMo Lab.

Server-rendered HTML with HTMX for the live parts. No bundler: node is not installed on the
cluster and adding a build step to a research tool is a maintenance cost with no payoff.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, Response)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db, evals, fleet, indexer, plots, queries, tb
from .config import AGES, SETTINGS

HERE = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app):
    SETTINGS.ensure_dirs()
    db.connect()
    evals.start_worker()
    yield


app = FastAPI(title="MIMo Lab", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


# ---------------------------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------------------------

def human_steps(value):
    if value is None:
        return "--"
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M".replace(".00M", "M")
    if value >= 1_000:
        return f"{value // 1000}k"
    return str(value)


def human_age(seconds):
    if not seconds:
        return "--"
    delta = time.time() - seconds
    for limit, unit, div in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600)):
        if delta < limit:
            return f"{int(delta / div)}{unit} ago"
    return f"{int(delta / 86400)}d ago"


def progress(run):
    if not run.get("num_train") or run.get("last_step") is None:
        return None
    return max(0.0, min(1.0, run["last_step"] / float(run["num_train"])))


templates.env.globals.update(human_steps=human_steps, human_age=human_age,
                             progress=progress, AGES=AGES, settings=SETTINGS,
                             PRESETS=fleet.PRESETS, TRAIN_FLAGS=fleet.TRAIN_FLAGS,
                             CHOICES=fleet.CHOICES, all_tags=queries.all_tags,
                             DATE_PRESETS=queries.DATE_PRESETS)


# Remembering the Runs filters in a cookie rather than in the page: Analysis and Jobs have URLs
# of their own, so the tab link has to be built before the browser gets there. A cookie is sent
# with every request, so the link is correct on first paint, survives a reload or a new tab, and
# needs no JavaScript. Single-user local tool, so one global slot is enough.
FILTER_COOKIE = "mimolab_runs_query"


def runs_href(request: Request):
    """Where the Runs tab should point: the last filter query this browser used, or plain '/'."""
    saved = request.cookies.get(FILTER_COOKIE, "")
    return "/" + saved if saved.startswith("?") else "/"


def _register_runs_href():
    templates.env.globals["runs_href"] = runs_href


def push_page_url(request: Request, response):
    """Tell htmx to put the *page* URL in the address bar, not this fragment's URL.

    hx-push-url="true" pushes the URL of the request htmx made, which here is
    /fragments/runs?<filters> -- a layout-less HTML fragment. The address bar then holds a URL
    that renders no navigation, so a reload (the Reindex button does one) drops the user on a bare
    table with no way back. The HX-Push-Url response header overrides that with the real page URL,
    which keeps reload, bookmarking and the back button all working.
    """
    query = str(request.url.query or "")
    response.headers["HX-Push-Url"] = ("/?" + query) if query else "/"
    return response


def remember_filters(request: Request, response):
    """Store the current filter query so the Runs tab returns to it.

    Called from both the page and the fragment: filters are usually changed by htmx, which only
    ever hits the fragment, so storing it on the page alone would remember a stale query. An empty
    query clears the cookie, which is what makes 'Reset filters' (a plain link to '/') work.
    """
    query = str(request.url.query or "")
    # 'run' belongs to the Analysis selection, not to the filters, and would bloat the cookie.
    keep = "&".join(p for p in query.split("&") if p and not p.startswith("run="))
    if keep:
        response.set_cookie(FILTER_COOKIE, "?" + keep, max_age=60 * 60 * 24 * 30,
                            samesite="lax", path="/")
    else:
        response.delete_cookie(FILTER_COOKIE, path="/")
    return response


_register_runs_href()


def params_of(request: Request):
    """Multi-valued query params, as the filter builder expects them."""
    out = {}
    for key in request.query_params.keys():
        values = request.query_params.getlist(key)
        out[key] = values if len(values) > 1 else values[0]
    return out


# ---------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def page_runs(request: Request):
    params = params_of(request)
    rows, total = queries.search(params)
    # Grouped by default: 565 runs collapse to 69 experiments, and the seeds of one experiment are
    # rarely interesting apart from each other. 'view=runs' is the explicit opt-out.
    grouped = params.get("view") != "runs"
    response = templates.TemplateResponse(request, "runs.html", {
        "runs": rows, "total": total, "params": params, "grouped": grouped,
        "facets": queries.facets(params), "stats": queries.stats(),
        "age_pairs": queries.age_pairs(params),
        "date_preset_counts": queries.date_preset_counts(params),
        "latest_date": queries.latest_date(),
        "groups": queries.groups(params) if grouped else [],
        "nav": "runs",
    })
    return remember_filters(request, response)


def whole_page_instead(request: Request, page="/"):
    """Redirect a plain browser navigation away from a fragment endpoint.

    Fragments render without the topbar or navigation, so a browser that lands on one -- a stale
    bookmark, a history entry, a middle-click -- gets a page with no way out. htmx sets HX-Request
    on its own calls, so this only ever fires for real navigations.
    """
    if "HX-Request" in request.headers:
        return None
    query = str(request.url.query or "")
    return RedirectResponse((page + "?" + query) if query else page, status_code=302)


@app.get("/fragments/runs", response_class=HTMLResponse)
def fragment_runs(request: Request):
    redirect = whole_page_instead(request, "/")
    if redirect is not None:
        return redirect

    params = params_of(request)
    rows, total = queries.search(params)
    grouped = params.get("view") != "runs"
    template = "_groups_table.html" if grouped else "_runs_table.html"
    response = templates.TemplateResponse(request, template, {
        "runs": rows, "total": total, "params": params, "grouped": grouped,
        "groups": queries.groups(params) if grouped else [],
    })
    return push_page_url(request, remember_filters(request, response))


@app.get("/fragments/facets", response_class=HTMLResponse)
def fragment_facets(request: Request):
    redirect = whole_page_instead(request, "/")
    if redirect is not None:
        return redirect
    params = params_of(request)
    return templates.TemplateResponse(request, "_facets.html", {
"params": params, "facets": queries.facets(params)})


@app.get("/run/{run_id:path}", response_class=HTMLResponse)
def page_run(request: Request, run_id: str):
    run = queries.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run indexed at '{run_id}'")
    return templates.TemplateResponse(request, "run.html", {
"run": run, "siblings": queries.siblings(run),
        "stats": queries.stats(), "nav": "runs",
        "sweep": evals.sweep_rows(run_id, run["checkpoint_list"][-1]) if run["checkpoint_list"] else [],
    })


@app.get("/analysis", response_class=HTMLResponse)
def page_analysis(request: Request):
    params = params_of(request)
    selected = request.query_params.getlist("run")
    if not selected:
        rows, _ = queries.search(params, limit=60)
        selected = [r["run_id"] for r in rows]
    return templates.TemplateResponse(request, "analysis.html", {
"params": params, "selected": selected,
        "warning": plots.horizon_warning(selected), "stats": queries.stats(),
        "tags": queries.all_tags(), "nav": "analysis",
        "runs": [queries.get_run(r) for r in selected[:200]],
    })


@app.get("/jobs", response_class=HTMLResponse)
def page_jobs(request: Request):
    return templates.TemplateResponse(request, "jobs.html", {
"jobs": fleet.jobs(), "stats": queries.stats(),
        "eval_status": evals.status(), "nav": "jobs",
        "yaml_gaps": fleet.check_yaml_coverage(),
    })


@app.get("/fragments/jobs", response_class=HTMLResponse)
def fragment_jobs(request: Request):
    redirect = whole_page_instead(request, "/jobs")
    if redirect is not None:
        return redirect
    fleet.refresh()
    return templates.TemplateResponse(request, "_jobs_table.html", {
"jobs": fleet.jobs(), "eval_status": evals.status()})


@app.get("/fragments/hosts", response_class=HTMLResponse)
def fragment_hosts(request: Request):
    redirect = whole_page_instead(request, "/jobs")
    if redirect is not None:
        return redirect
    probes = [] if SETTINGS.offline else fleet.probe_all()
    return templates.TemplateResponse(request, "_hosts.html", {
"probes": probes})


@app.get("/launch", response_class=HTMLResponse)
def page_launch(request: Request):
    return templates.TemplateResponse(request, "launch.html", {
"stats": queries.stats(), "nav": "jobs",
        "yaml_gaps": fleet.check_yaml_coverage(),
        "preset": request.query_params.get("preset", "sac_her_sparse"),
    })


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------

def _png(data):
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/plot/curve.png")
def plot_curve(request: Request,
               tag: str = Query("rollout/ep_rho_max_mean"),
               theme: str = Query("light"),
               aggregate: int = Query(0),
               smooth: int = Query(1)):
    run_ids = request.query_params.getlist("run")
    if not run_ids:
        params = params_of(request)
        rows, _ = queries.search(params, limit=60)
        run_ids = [r["run_id"] for r in rows]
    return _png(plots.curve(run_ids, tag, theme=theme if theme in ("light", "dark") else "light",
                            aggregate=bool(aggregate), smooth=max(1, min(51, smooth))))


@app.get("/api/plot/age_grid.png")
def plot_age_grid(request: Request, theme: str = Query("light")):
    params = params_of(request)
    run_ids = request.query_params.getlist("run")
    return _png(plots.age_grid(queries.age_grid(params, run_ids=run_ids or None),
                               theme=theme if theme in ("light", "dark") else "light"))


@app.get("/api/plot/goal_response.png")
def plot_goal_response(run: str, checkpoint: str, theme: str = Query("light")):
    rows = evals.sweep_rows(run, checkpoint)
    return _png(plots.goal_response(rows, theme=theme if theme in ("light", "dark") else "light",
                                    title=f"Goal response -- {checkpoint}"))


# ---------------------------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------------------------

@app.post("/api/reindex")
def api_reindex(force: int = Query(0)):
    return JSONResponse(indexer.reindex(force=bool(force)))


@app.post("/api/evals")
def api_eval(run_id: str = Form(...), checkpoint: str = Form(...),
             episodes: int = Form(50), goal: str = Form(""),
             policy_goal_sweep: str = Form(""), episode_steps: str = Form("")):
    try:
        job = evals.submit(
            run_id, checkpoint, episodes=episodes,
            goal=float(goal) if goal else None,
            policy_goal_sweep=policy_goal_sweep or None,
            episode_steps=int(episode_steps) if episode_steps else None)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"job": job, "queue": evals.status()})


@app.get("/api/evals/status")
def api_eval_status():
    return JSONResponse(evals.status())


def _form_values(form):
    """Form fields -> the values dict fleet.build_command expects.

    Kinds come from TRAIN_FLAGS, never from the submitted value: a numeric field that happens to
    hold "1" (train_freq, gradient_steps) is not a boolean, and treating it as one silently drops
    the "=1".
    """
    values = {}
    for flag, kind, _default, _group, _in_yaml in fleet.TRAIN_FLAGS:
        if kind == "bool":
            values[flag] = form.get(flag) in ("1", "on", "true")
        elif flag in form and form[flag] not in ("", None):
            values[flag] = form[flag]
    return values


@app.post("/api/jobs/preview")
async def api_preview(request: Request):
    """Render the exact command the launch would run, plus any guard-rail errors.

    Served by the same build_command the launcher uses, so the preview cannot disagree with what
    actually gets executed -- which is the only thing a preview is for.
    """
    form = dict(await request.form())
    name = (form.get("name") or "").strip() or "<name>"
    values = _form_values(form)
    command, _ = fleet.build_command(values, f"{name}_run_0")
    return JSONResponse({"command": command,
                         "errors": fleet.errors(values),
                         "warnings": fleet.warnings(values)})


@app.post("/api/jobs/launch")
async def api_launch(request: Request):
    form = dict(await request.form())
    name = (form.pop("name", "") or "").strip()
    seeds = int(form.pop("seeds", 1) or 1)
    if not name:
        raise HTTPException(400, "Name the sweep -- it becomes the --save_model stem.")

    values = _form_values(form)
    blocking = fleet.errors(values)
    if blocking:
        raise HTTPException(400, " ".join(blocking))
    try:
        jobs = fleet.launch_sweep(values, name, seeds)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"jobs": jobs})


@app.post("/api/jobs/{job_id}/kill")
def api_kill(job_id: str):
    try:
        return JSONResponse(fleet.kill_job(job_id))
    except KeyError:
        raise HTTPException(404, job_id)


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def api_log(job_id: str, lines: int = 200):
    return fleet.tail_log(job_id, lines=lines)


@app.post("/api/tensorboard")
async def api_tensorboard(request: Request):
    form = await request.form()
    run_ids = form.getlist("run")
    if not run_ids:
        raise HTTPException(400, "Select at least one run.")
    try:
        return JSONResponse(tb.launch(run_ids))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/tensorboard")
def api_tensorboard_status():
    tb.reap_if_idle()
    return JSONResponse(tb.status())


@app.post("/api/tensorboard/stop")
def api_tensorboard_stop():
    return JSONResponse(tb.stop())
