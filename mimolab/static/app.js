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

  /* ---------- figure style (rcParams) ----------------------------------------------------- */

  function renderStyle(data) {
    document.getElementById('styletext').value = data.text || '';
    var base = document.getElementById('stylebase');
    base.innerHTML = '';
    Object.keys(data.base).forEach(function (key) {
      var dt = document.createElement('dt'); dt.textContent = key;
      var dd = document.createElement('dd'); dd.textContent = data.base[key];
      base.appendChild(dt); base.appendChild(dd);
    });
    document.getElementById('styleblocked').textContent =
      'Ignored from every source: ' + data.blocked.join(', ') +
      ' \u2014 they would change the page size or the renderer.';
    document.getElementById('stylenote').textContent = data.offline
      ? 'offline mode: read-only'
      : Object.keys(data.base).length + ' rcParams from ' + data.source;
    var box = document.getElementById('styleerr');
    box.hidden = !(data.errors && data.errors.length);
    if (!box.hidden) box.textContent = 'Not saved \u2014 ' + data.errors.join('  |  ');
  }

  function loadStyle() {
    fetch('/api/style').then(function (r) { return r.json(); }).then(renderStyle);
  }

  document.addEventListener('click', function (e) {
    if (!e.target.closest) return;
    if (e.target.closest('[data-settings]')) {
      loadStyle();
      document.getElementById('settings').showModal();
      return;
    }
    if (e.target.closest('[data-style-revert]')) { loadStyle(); return; }
    var save = e.target.closest('[data-style-save]');
    if (!save) return;
    var body = new FormData();
    body.append('text', document.getElementById('styletext').value);
    save.disabled = true;
    fetch('/api/style', { method: 'POST', body: body })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderStyle(data);
        if (!data.errors || !data.errors.length) {
          document.getElementById('stylenote').textContent = 'Saved. Redrawing the figures.';
          refreshCharts();
        }
      })
      .catch(function () {
        var box = document.getElementById('styleerr');
        box.hidden = false;
        box.textContent = 'Could not reach the server.';
      })
      .finally(function () { save.disabled = false; });
  });

  /* ---------- charts --------------------------------------------------------------------- */
  /* Charts are server-rendered PNGs, so the theme has to travel with the request. */

  function chartUrl(base) {
    var sep = base.indexOf('?') === -1 ? '?' : '&';
    return base + sep + 'theme=' + currentTheme();
  }

  /* The export link is the chart's own URL with .png swapped for .pdf, so the two can never
     describe different figures. Column width is a PDF-only concern; the server ignores it on PNG. */
  /* An <a href="#"> that looks like a button is a trap: clicking it scrolls to the top and
     exports nothing, which is indistinguishable from a silent failure. Export links carry their
     state instead -- disabled until there is something to export, and they say so when clicked. */
  function setExportLink(el, href) {
    if (!el) return;
    if (href) {
      el.setAttribute('href', href);
      el.removeAttribute('aria-disabled');
      el.removeAttribute('title');
    } else {
      el.setAttribute('href', '#');
      el.setAttribute('aria-disabled', 'true');
      el.setAttribute('title', 'Nothing to export yet');
    }
  }

  function flashExported(el) {
    if (!el || el.dataset.flashing) return;
    var original = el.textContent;
    el.dataset.flashing = '1';
    el.textContent = 'Saving\u2026';
    setTimeout(function () {
      el.textContent = original;
      delete el.dataset.flashing;
    }, 1400);
  }

  document.addEventListener('click', function (e) {
    var link = e.target.closest('a[download]');
    if (!link) return;
    if (link.getAttribute('aria-disabled') === 'true' || link.getAttribute('href') === '#') {
      /* Without this the browser follows "#" and jumps to the top of the page. */
      e.preventDefault();
      var hint = document.getElementById('barhint');
      if (link.id === 'exportbars' && hint) {
        hint.hidden = false;
        hint.textContent = 'Tick at least one evaluation first, then export.';
      }
      return;
    }
    /* The response is an attachment, so the page does not change -- say that something happened. */
    flashExported(link);
  });

  function exportHref(base) {
    var column = (document.getElementById('column') || {}).value || 'double';
    var url = base.replace('/curve.png', '/curve.pdf')
                  .replace('/age_grid.png', '/age_grid.pdf')
                  .replace('/goal_response.png', '/goal_response.pdf');
    return url + (url.indexOf('?') === -1 ? '?' : '&') + 'column=' + column;
  }

  function syncExport(scope, base) {
    var link = scope && scope.querySelector('[data-export]');
    setExportLink(link, exportHref(base));
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

  /* The two palettes the renderer uses, mirrored so the swatch beside a label input matches the
     line it will name. Thesis mode uses the tab palette of results/icdlplot.py. */
  var SCREEN_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                       '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
  // plots.TAB_HEX -- matplotlib's tab cycle, reordered orange / grey / green / blue.
  var THESIS_COLORS = ['#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4',
                       '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf'];

  /* The picker. The tab cycle carries its matplotlib name; results/icdlplot.py's age ramp is
     appended, and age 9 is a note on tab:orange rather than an eleventh colour, because that is
     exactly what icdlplot maps it to -- two entries would mean two inks for one age. */
  var TAB_NAMES = ['orange', 'gray', 'green', 'blue', 'red',
                   'purple', 'brown', 'pink', 'olive', 'cyan'];
  var AGE_NOTE = { '#ff7f0e': 'age 9' };
  var AGE_COLORS = [['#808080', 'age 1'], ['#aa805a', 'age 3'], ['#d57f34', 'age 6']];

  function pickedColors() {
    var out = {};
    document.querySelectorAll('#labelrows tr[data-color]').forEach(function (tr) {
      out[tr.getAttribute('data-series-row')] = tr.getAttribute('data-color');
    });
    return out;
  }

  function colorParams() {
    var picked = pickedColors();
    return Object.keys(picked).map(function (key) {
      return 'color=' + encodeURIComponent(key + '=' + picked[key]);
    }).join('&');
  }

  function paintSwatches() {
    var thesis = (document.getElementById('style') || {}).value === 'thesis';
    var palette = thesis ? THESIS_COLORS : SCREEN_COLORS;
    /* Colour follows position, so the swatch is read off the row's current index rather than the
       index the server rendered -- otherwise reordering repaints nothing and the swatches lie.
       A pinned colour wins, and says so with a ring. */
    document.querySelectorAll('[data-swatch]').forEach(function (el, i) {
      var row = el.closest('tr');
      var pinned = row && row.getAttribute('data-color');
      el.style.background = pinned || palette[i % palette.length];
      el.setAttribute('data-pinned', pinned ? '1' : '0');
    });
  }

  function closePicker() {
    var open = document.querySelector('.colorpop');
    if (open) open.remove();
  }

  function openPicker(swatch) {
    closePicker();
    var row = swatch.closest('tr');
    var pop = document.createElement('div');
    pop.className = 'colorpop';
    var entries = THESIS_COLORS.map(function (hex, i) {
      var note = AGE_NOTE[hex];
      return [hex, 'tab:' + TAB_NAMES[i] + (note ? ' \u00b7 ' + note : '')];
    }).concat(AGE_COLORS.map(function (pair) {
      return [pair[0], 'icdlplot \u00b7 ' + pair[1]];
    }));
    entries.forEach(function (entry) {
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('data-color-pick', entry[0]);
      b.innerHTML = '<i style="background:' + entry[0] + '"></i><span>' + entry[1] + '</span>';
      pop.appendChild(b);
    });
    var auto = document.createElement('button');
    auto.type = 'button';
    auto.className = 'auto';
    auto.setAttribute('data-color-pick', '');
    auto.textContent = 'Follow legend position';
    pop.appendChild(auto);

    row.querySelector('td').appendChild(pop);
    pop.addEventListener('click', function (e) {
      var choice = e.target.closest('[data-color-pick]');
      if (!choice) return;
      var value = choice.getAttribute('data-color-pick');
      if (value) row.setAttribute('data-color', value);
      else row.removeAttribute('data-color');
      closePicker();
      drawMain();
      swatch.focus();
    });
  }

  document.addEventListener('click', function (e) {
    var swatch = e.target.closest && e.target.closest('.swatch[data-swatch]');
    if (swatch) {
      e.preventDefault();
      if (document.querySelector('.colorpop')) closePicker();
      else openPicker(swatch);
      return;
    }
    if (!e.target.closest || !e.target.closest('.colorpop')) closePicker();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closePicker();
  });

  /* Legend order. The rows of the label editor are the order of the legend, and the server is
     told about it as a list of series keys -- the same keys the labels are keyed on. */
  function orderParams() {
    var rows = document.querySelectorAll('#labelrows tr[data-series-row]');
    if (rows.length < 2) return '';
    return Array.prototype.map.call(rows, function (tr) {
      return 'order=' + encodeURIComponent(tr.getAttribute('data-series-row'));
    }).join('&');
  }

  document.addEventListener('click', function (e) {
    var button = e.target.closest && e.target.closest('[data-move]');
    if (!button) return;
    e.preventDefault();
    var row = button.closest('tr');
    var sibling = button.getAttribute('data-move') === 'up'
      ? row.previousElementSibling : row.nextElementSibling;
    if (!sibling) return;
    if (button.getAttribute('data-move') === 'up') row.parentNode.insertBefore(row, sibling);
    else row.parentNode.insertBefore(sibling, row);
    button.focus();
    drawMain();
  });

  function labelParams() {
    var parts = [];
    document.querySelectorAll('.labelin').forEach(function (input) {
      var text = (input.value || '').trim();
      if (text) {
        parts.push('glabel=' + encodeURIComponent(input.getAttribute('data-series') + '=' + text));
      }
    });
    return parts.join('&');
  }

  function drawMain() {
    var img = document.getElementById('mainchart');
    if (!img) return;
    var base = img.getAttribute('data-chart-base');
    var tag = (document.getElementById('tagsel') || {}).value || 'rollout/ep_rho_max_mean';
    var agg = (document.getElementById('agg') || {}).checked ? 1 : 0;
    var smooth = (document.getElementById('smooth') || {}).value || 1;
    var style = (document.getElementById('style') || {}).value || 'screen';
    var band = (document.getElementById('bandsel') || {}).value || 'std';
    var query = base + 'tag=' + encodeURIComponent(tag) + '&aggregate=' + agg +
                '&smooth=' + smooth + '&style=' + style + '&band=' + band;
    ['ylabelin=ylabel', 'xlabelin=xlabel', 'legendloc=legend_loc',
     'legendtitle=legend_title'].forEach(function (pair) {
      var bits = pair.split('=');
      var el = document.getElementById(bits[0]);
      var value = el && (el.value || '').trim();
      if (value && !(bits[0] === 'legendloc' && value === 'best')) {
        query += '&' + bits[1] + '=' + encodeURIComponent(value);
      }
    });
    ['figw=figw', 'figh=figh'].forEach(function (pair) {
      var bits = pair.split('=');
      var el = document.getElementById(bits[0]);
      var value = el && (el.value || '').trim();
      if (value) query += '&' + bits[1] + '=' + encodeURIComponent(value);
    });
    var labels = labelParams();
    if (labels) query += '&' + labels;
    var order = orderParams();
    if (order) query += '&' + order;
    var colors = colorParams();
    if (colors) query += '&' + colors;
    paintSwatches();
    img.src = chartUrl(query);
    setExportLink(document.getElementById('exportmain'), exportHref(query));
  }

  ['tagsel', 'agg', 'smooth', 'column', 'style', 'bandsel', 'legendloc'].forEach(function (id) {
    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === id) {
        drawMain();
        if (e.target.id === 'column') refreshCharts();
      }
    });
  });

  /* ---------- evaluation bar chart -------------------------------------------------------- */

  function drawBars() {
    var img = document.getElementById('barchart');
    var hint = document.getElementById('barhint');
    if (!img) return;
    var picked = Array.prototype.filter.call(
      document.querySelectorAll('.barpick'), function (b) { return b.checked; });
    if (!picked.length) {
      img.hidden = true;
      setExportLink(document.getElementById('exportbars'), null);
      if (hint) { hint.hidden = false; hint.textContent = 'Tick at least one evaluation to draw the chart.'; }
      return;
    }
    var query = picked.map(function (box) {
      var input = document.querySelector('.barlabel[data-job="' + box.value + '"]');
      var text = input && (input.value || '').trim();
      return 'src=' + encodeURIComponent((text || '') + '=' + box.value);
    }).join('&');
    query += '&metric=' + ((document.getElementById('barmetric') || {}).value || 'successful');
    query += '&column=' + ((document.getElementById('barcolumn') || {}).value || 'single');
    if ((document.getElementById('barannotate') || {}).checked) query += '&annotate=1';
    ['barx=xlabel', 'bary=ylabel'].forEach(function (pair) {
      var bits = pair.split('=');
      var el = document.getElementById(bits[0]);
      var value = el && (el.value || '').trim();
      if (value) query += '&' + bits[1] + '=' + encodeURIComponent(value);
    });
    img.hidden = false;
    if (hint) hint.hidden = true;
    img.src = '/api/plot/eval_bars.png?' + query;
    setExportLink(document.getElementById('exportbars'), '/api/plot/eval_bars.pdf?' + query);
  }

  var barTimer = null;
  document.addEventListener('change', function (e) {
    if (e.target.matches && e.target.matches('.barpick, #barmetric, #barcolumn, #barannotate')) {
      drawBars();
    }
  });
  document.addEventListener('input', function (e) {
    if (!e.target.matches || !e.target.matches('.barlabel, #barx, #bary')) return;
    clearTimeout(barTimer);
    barTimer = setTimeout(drawBars, 350);
  });

  var labelTimer = null;
  document.addEventListener('input', function (e) {
    if (!e.target.matches ||
        !e.target.matches('.labelin, #ylabelin, #xlabelin, #legendtitle, #figw, #figh')) return;
    clearTimeout(labelTimer);
    labelTimer = setTimeout(drawMain, 350);
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

  /* 'kind' is 'group' or 'dcee': same queue, same log, same progress counter -- an embodiment
     grid just counts through 16x as many runs, so one watcher covers both. */
  function watchGroup(jobId, msg, kind) {
    kind = kind || 'group';
    var box = document.getElementById(kind + 'log');
    var pre = document.querySelector('[data-' + kind + '-log]');
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
    var form = e.target.closest('[data-group-form], [data-dcee-form]');
    if (!form) return;
    e.preventDefault();
    var kind = form.hasAttribute('data-dcee-form') ? 'dcee' : 'group';
    var msg = form.querySelector('[data-' + kind + '-msg]');
    var button = form.querySelector('button[type=submit]');
    button.disabled = true;
    msg.textContent = 'Queueing\u2026';
    post('/api/evals/' + kind, new FormData(form)).then(function (data) {
      button.disabled = false;
      msg.textContent = 'Queued. This runs one environment at a time.';
      watchGroup(data.job.job_id, msg, kind);
    }).catch(function (err) {
      button.disabled = false;
      msg.textContent = err.message;
    });
  });

  /* The stored grid is redrawn on the server from the payload on disk, so restyling it is a new
     image URL and never another evaluation. */
  function drawDcee() {
    var img = document.getElementById('dceechart');
    if (!img) return;
    var query = img.getAttribute('data-dcee-base');
    query += '&metric=' + ((document.getElementById('dceemetric') || {}).value || 'successful');
    [['dceepaneltitle', 'panel_title'], ['dceetitle', 'title'], ['dceew', 'width'],
     ['dceeh', 'height'], ['dceethreshold', 'threshold']].forEach(function (pair) {
      var el = document.getElementById(pair[0]);
      var value = el && (el.value || '').trim();
      if (value) query += '&' + pair[1] + '=' + encodeURIComponent(value);
    });
    /* 0 means "no colour bar", which is a real choice and not an empty field -- so the value is
       read before the blank test, not after it. */
    var bar = document.getElementById('dceecbar');
    var barValue = bar && (bar.value || '').trim();
    if (barValue !== '' && barValue !== null && barValue !== undefined) {
      query += parseFloat(barValue) > 0 ? '&cbar_fraction=' + encodeURIComponent(barValue)
                                        : '&cbar=0';
    }
    img.src = query;
    setExportLink(document.getElementById('exportdcee'),
                  query.replace('/dcee.png?', '/dcee.pdf?'));
  }

  document.addEventListener('change', function (e) {
    if (e.target && e.target.id === 'dceemetric') drawDcee();
  });
  var dceeTimer = null;
  document.addEventListener('input', function (e) {
    if (!e.target.matches || !e.target.matches('#dceeopts input')) return;
    clearTimeout(dceeTimer);
    dceeTimer = setTimeout(drawDcee, 350);
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
    paintSwatches();
    refreshCharts();
    drawBars();          // settles the bar panel's export link into its disabled state
    loadRawConfig();
  }

  document.addEventListener('DOMContentLoaded', boot);
  document.body && document.addEventListener('htmx:afterSwap', function () {
    paintSelection();
    refreshCharts();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshCharts);
})();
