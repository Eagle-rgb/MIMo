/* MIMo Lab client.
 *
 * Deliberately small: HTMX does the fetching, this handles selection state, theme, chart theming
 * and the few POST actions. No framework, no build step.
 */
(function () {
  'use strict';

  var SEL_KEY = 'mimolab.selection';

  /* ---------- theme ---------------------------------------------------------------------- */

  function currentTheme() {
    var stamped = document.documentElement.getAttribute('data-theme');
    if (stamped) return stamped;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  var stored = localStorage.getItem('mimolab.theme');
  if (stored) document.documentElement.setAttribute('data-theme', stored);

  document.addEventListener('click', function (e) {
    var toggle = e.target.closest('[data-theme-toggle]');
    if (!toggle) return;
    var next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('mimolab.theme', next);
    refreshCharts();
  });

  /* ---------- charts --------------------------------------------------------------------- */
  /* Charts are server-rendered PNGs, so the theme has to travel with the request. */

  function chartUrl(base) {
    var sep = base.indexOf('?') === -1 ? '?' : '&';
    return base + sep + 'theme=' + currentTheme();
  }

  /* The export link is the chart's own URL with .png swapped for .pdf, so the two can never
     describe different figures. Column width is a PDF-only concern; the server ignores it on PNG. */
  function exportHref(base) {
    var column = (document.getElementById('column') || {}).value || 'double';
    var url = base.replace('/curve.png', '/curve.pdf')
                  .replace('/age_grid.png', '/age_grid.pdf')
                  .replace('/goal_response.png', '/goal_response.pdf');
    return url + (url.indexOf('?') === -1 ? '?' : '&') + 'column=' + column;
  }

  function syncExport(scope, base) {
    var link = scope && scope.querySelector('[data-export]');
    if (link) link.setAttribute('href', exportHref(base));
  }

  function refreshCharts() {
    document.querySelectorAll('img[data-chart]').forEach(function (img) {
      var base = img.getAttribute('data-chart');
      img.src = chartUrl(base);
      /* the export link sits beside the .chart wrapper, so look one level up */
      var scope = img.closest('.chart') && img.closest('.chart').parentElement;
      syncExport(scope, base);
    });
    var main = document.getElementById('mainchart');
    if (main) drawMain();
  }

  function drawMain() {
    var img = document.getElementById('mainchart');
    if (!img) return;
    var base = img.getAttribute('data-chart-base');
    var tag = (document.getElementById('tagsel') || {}).value || 'rollout/ep_rho_max_mean';
    var agg = (document.getElementById('agg') || {}).checked ? 1 : 0;
    var smooth = (document.getElementById('smooth') || {}).value || 1;
    var query = base + 'tag=' + encodeURIComponent(tag) + '&aggregate=' + agg +
                '&smooth=' + smooth;
    img.src = chartUrl(query);
    var link = document.getElementById('exportmain');
    if (link) link.setAttribute('href', exportHref(query));
  }

  ['tagsel', 'agg', 'smooth', 'column'].forEach(function (id) {
    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === id) {
        drawMain();
        if (e.target.id === 'column') refreshCharts();
      }
    });
  });

  /* ---------- selection ------------------------------------------------------------------ */

  function selection() {
    try { return JSON.parse(sessionStorage.getItem(SEL_KEY) || '[]'); }
    catch (err) { return []; }
  }

  function setSelection(list) {
    var unique = Array.from(new Set(list));
    sessionStorage.setItem(SEL_KEY, JSON.stringify(unique));
    paintSelection();
  }

  function paintSelection() {
    var chosen = selection();
    var lookup = new Set(chosen);
    document.querySelectorAll('input[data-pick]').forEach(function (box) {
      box.checked = lookup.has(box.value);
      var row = box.closest('tr');
      if (row) row.classList.toggle('picked', box.checked);
    });
    document.querySelectorAll('input[data-pick-group]').forEach(function (box) {
      var members = box.value.split(' ').filter(Boolean);
      box.checked = members.length > 0 && members.every(function (m) { return lookup.has(m); });
      var row = box.closest('tr');
      if (row) row.classList.toggle('picked', box.checked);
    });

    var tray = document.getElementById('tray');
    if (!tray) return;
    tray.setAttribute('data-open', chosen.length ? '1' : '0');
    var count = tray.querySelector('[data-tray-count]');
    if (count) count.textContent = chosen.length;
  }

  document.addEventListener('change', function (e) {
    var box = e.target;
    if (box.matches && box.matches('input[data-pick]')) {
      var list = selection();
      list = box.checked ? list.concat([box.value])
                         : list.filter(function (v) { return v !== box.value; });
      setSelection(list);
    } else if (box.matches && box.matches('input[data-pick-group]')) {
      var members = box.value.split(' ').filter(Boolean);
      var current = selection();
      if (box.checked) {
        setSelection(current.concat(members));
      } else {
        var drop = new Set(members);
        setSelection(current.filter(function (v) { return !drop.has(v); }));
      }
    }
  });

  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-select-page]')) {
      var page = Array.from(document.querySelectorAll('input[data-pick]'))
        .map(function (b) { return b.value; });
      setSelection(selection().concat(page));
    }
    if (e.target.closest('[data-tray-clear]')) setSelection([]);

    var compare = e.target.closest('[data-tray-compare]');
    if (compare) {
      e.preventDefault();
      var query = selection().map(function (r) { return 'run=' + encodeURIComponent(r); }).join('&');
      window.location.href = '/analysis?' + query;
    }
  });

  /* ---------- sorting and view toggles --------------------------------------------------- */

  document.addEventListener('click', function (e) {
    var sorter = e.target.closest('a[data-sort]');
    if (sorter) {
      e.preventDefault();
      setHidden('sort', sorter.getAttribute('data-sort'));
      setHidden('dir', sorter.getAttribute('data-dir'));
      submitFilters();
    }
    var view = e.target.closest('[data-view]');
    if (view) {
      /* '' means grouped -- the default -- so only the opt-out carries a value. */
      var wanted = view.getAttribute('data-view');
      setHidden('view', wanted === 'runs' ? 'runs' : '');
      document.querySelectorAll('.segmented button[data-view]').forEach(function (b) {
        b.setAttribute('aria-pressed', b === view ? 'true' : 'false');
      });
      submitFilters();
    }
  });

  function setHidden(name, value) {
    var input = document.querySelector('#filters [name="' + name + '"]');
    if (input) input.value = value;
  }

  function submitFilters() {
    var form = document.getElementById('filters');
    if (form && window.htmx) htmx.trigger(form, 'change');
  }

  /* Date presets are mutually exclusive, and clicking the active one clears it. */
  document.addEventListener('click', function (e) {
    var chip = e.target.closest('.quick button[data-date-preset]');
    if (!chip) return;
    var wasOn = chip.getAttribute('aria-pressed') === 'true';
    document.querySelectorAll('.quick button[data-date-preset]').forEach(function (b) {
      b.setAttribute('aria-pressed', 'false');
    });
    if (!wasOn) chip.setAttribute('aria-pressed', 'true');
    setHidden('date_preset', wasOn ? '' : chip.getAttribute('data-date-preset'));
    submitFilters();
  });

  /* The age matrix writes into the two hidden age inputs; a cell is one (morph, physio) pair. */
  document.addEventListener('click', function (e) {
    var cell = e.target.closest('.agegrid button[data-age]');
    if (!cell) return;
    var pressed = cell.getAttribute('aria-pressed') === 'true';
    document.querySelectorAll('.agegrid button[data-age]').forEach(function (b) {
      b.setAttribute('aria-pressed', 'false');
    });
    var pair = cell.getAttribute('data-age').split(':');
    cell.setAttribute('aria-pressed', pressed ? 'false' : 'true');
    setHidden('morph_age', pressed ? '' : pair[0]);
    setHidden('physio_age', pressed ? '' : pair[1]);
    submitFilters();
  });

  /* ---------- configuration view --------------------------------------------------------- */

  function loadRawConfig() {
    var pre = document.querySelector('[data-cfg-src]');
    if (!pre || pre.dataset.loaded) return;
    pre.dataset.loaded = '1';
    fetch(pre.getAttribute('data-cfg-src'))
      .then(function (r) { return r.text(); })
      .then(function (text) { pre.textContent = text; })
      .catch(function (err) { pre.textContent = 'Could not read data.yml: ' + err; });
  }

  document.addEventListener('click', function (e) {
    var toggle = e.target.closest('[data-cfg]');
    if (!toggle) return;
    var wanted = toggle.getAttribute('data-cfg');
    document.querySelectorAll('[data-cfg]').forEach(function (b) {
      b.setAttribute('aria-pressed', b === toggle ? 'true' : 'false');
    });
    document.querySelectorAll('[data-cfg-panel]').forEach(function (panel) {
      panel.hidden = panel.getAttribute('data-cfg-panel') !== wanted;
    });
    if (wanted === 'raw') loadRawConfig();
  });

  /* ---------- group evaluation ------------------------------------------------------------ */

  var groupPoll = null;

  function watchGroup(jobId, msg) {
    var box = document.getElementById('grouplog');
    var pre = document.querySelector('[data-group-log]');
    if (box) box.hidden = false;
    clearInterval(groupPoll);
    groupPoll = setInterval(function () {
      fetch('/api/evals/' + jobId + '/log?lines=60')
        .then(function (r) { return r.text(); })
        .then(function (text) { if (pre) { pre.textContent = text; pre.scrollTop = pre.scrollHeight; } });
      fetch('/api/evals/jobs').then(function (r) { return r.json(); }).then(function (data) {
        var job = (data.jobs || []).filter(function (j) { return j.job_id === jobId; })[0];
        if (!job) return;
        if (msg) msg.textContent = job.note ? job.state + ' \u2014 ' + job.note : job.state;
        if (job.state === 'finished' || job.state === 'failed') {
          clearInterval(groupPoll);
          if (job.state === 'finished') setTimeout(function () { window.location.reload(); }, 1200);
        }
      });
    }, 3000);
  }

  document.addEventListener('submit', function (e) {
    var form = e.target.closest('[data-group-form]');
    if (!form) return;
    e.preventDefault();
    var msg = form.querySelector('[data-group-msg]');
    var button = form.querySelector('button[type=submit]');
    button.disabled = true;
    msg.textContent = 'Queueing\u2026';
    post('/api/evals/group', new FormData(form)).then(function (data) {
      button.disabled = false;
      msg.textContent = 'Queued. This runs one environment at a time.';
      watchGroup(data.job.job_id, msg);
    }).catch(function (err) {
      button.disabled = false;
      msg.textContent = err.message;
    });
  });

  /* ---------- actions -------------------------------------------------------------------- */

  function post(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
      body: body instanceof FormData ? body : JSON.stringify(body || {})
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) throw new Error(data.detail || ('request failed (' + res.status + ')'));
        return data;
      });
    });
  }

  document.addEventListener('click', function (e) {
    var reindex = e.target.closest('[data-reindex]');
    if (reindex) {
      /* Rescan models/, then reload the page you are already on. The reload is what refreshes the
         facet counts and the header totals, which a fragment swap would leave stale. */
      var label = reindex.textContent;
      reindex.disabled = true;
      reindex.textContent = 'Refreshing…';
      post('/api/reindex').then(function () {
        window.location.reload();
      }).catch(function (err) {
        reindex.disabled = false;
        reindex.textContent = label;
        alert('Refresh failed: ' + (err.message || err));
      });
    }

    var tbSel = e.target.closest('[data-tb-selected]');
    var tbRun = e.target.closest('[data-tb-runs]');
    if (tbSel || tbRun) {
      var runs = tbRun ? tbRun.getAttribute('data-tb-runs').split(' ') : selection();
      if (!runs.length) { alert('Select at least one run first.'); return; }
      var form = new FormData();
      runs.forEach(function (r) { form.append('run', r); });
      var button = tbSel || tbRun;
      button.disabled = true;
      var original = button.textContent;
      button.textContent = 'Starting TensorBoard';
      post('/api/tensorboard', form).then(function (data) {
        button.disabled = false;
        button.textContent = original;
        if (data.url) window.open(data.url, '_blank', 'noopener');
      }).catch(function (err) {
        button.disabled = false;
        button.textContent = original;
        alert(err.message);
      });
    }

    if (e.target.closest('[data-tray-tb]')) {
      var chosen = selection();
      if (!chosen.length) return;
      var fd = new FormData();
      chosen.forEach(function (r) { fd.append('run', r); });
      post('/api/tensorboard', fd).then(function (data) {
        if (data.url) window.open(data.url, '_blank', 'noopener');
      }).catch(function (err) { alert(err.message); });
    }
  });

  /* eval submission */
  document.addEventListener('submit', function (e) {
    var form = e.target.closest('[data-eval-form]');
    if (!form) return;
    e.preventDefault();
    var msg = form.querySelector('[data-eval-msg]');
    var button = form.querySelector('button[type=submit]');
    button.disabled = true;
    msg.textContent = 'Queueing…';
    post('/api/evals', new FormData(form)).then(function (data) {
      button.disabled = false;
      msg.textContent = 'Queued. ' + data.queue.pending + ' waiting' +
        (data.queue.eta_seconds ? ', about ' + Math.round(data.queue.eta_seconds / 60) + ' min.' : '.');
    }).catch(function (err) {
      button.disabled = false;
      msg.textContent = err.message;
    });
  });

  /* ---------- boot ----------------------------------------------------------------------- */

  function boot() {
    paintSelection();
    refreshCharts();
    loadRawConfig();
  }

  document.addEventListener('DOMContentLoaded', boot);
  document.body && document.addEventListener('htmx:afterSwap', function () {
    paintSelection();
    refreshCharts();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshCharts);
})();
