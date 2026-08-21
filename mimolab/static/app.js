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

  function refreshCharts() {
    document.querySelectorAll('img[data-chart]').forEach(function (img) {
      img.src = chartUrl(img.getAttribute('data-chart'));
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
    img.src = chartUrl(base + 'tag=' + encodeURIComponent(tag) + '&aggregate=' + agg +
                       '&smooth=' + smooth);
  }

  ['tagsel', 'agg', 'smooth'].forEach(function (id) {
    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === id) drawMain();
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

    var kill = e.target.closest('[data-kill]');
    if (kill) {
      var jobId = kill.getAttribute('data-kill');
      if (!window.confirm('Send SIGTERM to this run? Training stops at its current step.')) return;
      kill.disabled = true;
      post('/api/jobs/' + jobId + '/kill').then(function () {
        if (window.htmx) htmx.trigger(document.getElementById('jobs'), 'load');
      });
    }

    var log = e.target.closest('[data-log]');
    if (log) {
      var id = log.getAttribute('data-log');
      var row = document.querySelector('[data-logrow="' + id + '"]');
      var body = document.querySelector('[data-logbody="' + id + '"]');
      if (!row) return;
      row.hidden = !row.hidden;
      if (!row.hidden) {
        fetch('/api/jobs/' + id + '/log?lines=300')
          .then(function (r) { return r.text(); })
          .then(function (text) { body.textContent = text || '(log is empty)'; });
      }
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

  /* ---------- launch form ---------------------------------------------------------------- */

  function launchValues() {
    var form = document.getElementById('launchform');
    if (!form) return null;
    return new FormData(form);
  }

  /* The preview is rendered by the server, by the same build_command the launcher calls. Doing
     it here would mean a second implementation of the flag rules that can silently disagree with
     the one that actually runs -- and a preview that lies is worse than no preview. */
  var previewPending = null;

  function renderPreview() {
    var form = document.getElementById('launchform');
    var preview = document.getElementById('preview');
    if (!form || !preview) return;
    clearTimeout(previewPending);
    previewPending = setTimeout(function () {
      post('/api/jobs/preview', new FormData(form)).then(function (data) {
        preview.textContent = data.command.replace(/ --/g, ' \\\n    --');
        var box = document.getElementById('launcherrors');
        if (!box) return;
        box.innerHTML = '';
        /* Errors block the launch; warnings are things illustrations.py only prints or silently
           corrects, so they inform and must never disable the button. */
        [['bad', data.errors || []], ['', data.warnings || []]].forEach(function (pair) {
          pair[1].forEach(function (message) {
            var div = document.createElement('div');
            div.className = 'notice ' + pair[0];
            div.style.marginTop = '.6rem';
            div.textContent = message;
            box.appendChild(div);
          });
        });
        var submit = form.querySelector('button[type=submit]');
        if (submit) submit.disabled = (data.errors || []).length > 0;
      }).catch(function (err) { preview.textContent = String(err.message || err); });
    }, 150);
  }

  document.addEventListener('input', function (e) {
    if (e.target.closest('#launchform')) renderPreview();
  });
  document.addEventListener('change', function (e) {
    if (e.target.closest('#launchform')) renderPreview();
  });

  document.addEventListener('click', function (e) {
    var preset = e.target.closest('[data-preset]');
    if (!preset) return;
    var values = JSON.parse(preset.getAttribute('data-values'));
    var form = document.getElementById('launchform');
    form.querySelectorAll('input[type=checkbox]').forEach(function (b) { b.checked = false; });
    Object.keys(values).forEach(function (key) {
      var field = form.querySelector('[name="' + key + '"]');
      if (!field) return;
      if (field.type === 'checkbox') field.checked = Boolean(values[key]);
      else field.value = values[key];
    });
    renderPreview();
  });

  document.addEventListener('submit', function (e) {
    var form = e.target.closest('#launchform');
    if (!form) return;
    e.preventDefault();
    var msg = document.getElementById('launchmsg');
    var seeds = form.querySelector('[name=seeds]').value;
    if (!window.confirm('Launch ' + seeds + ' run(s) on the RBI pool?')) return;
    msg.textContent = 'Allocating hosts…';
    post('/api/jobs/launch', new FormData(form)).then(function (data) {
      msg.textContent = 'Launched ' + data.jobs.length + ' run(s).';
      setTimeout(function () { window.location.href = '/jobs'; }, 800);
    }).catch(function (err) { msg.textContent = err.message; });
  });

  /* ---------- boot ----------------------------------------------------------------------- */

  function boot() {
    paintSelection();
    refreshCharts();
    renderPreview();
  }

  document.addEventListener('DOMContentLoaded', boot);
  document.body && document.addEventListener('htmx:afterSwap', function () {
    paintSelection();
    refreshCharts();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshCharts);
})();
