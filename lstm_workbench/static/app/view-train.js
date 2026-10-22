/* 训练：门禁指标卡 + 六张 ECharts 图 + 等宽日志。
   在线模式下 POST /train 会一直阻塞到训完，期间 progress.json 每一轮都在重写，
   曲线和门禁卡就按它报的真实 history 逐轮长出来；收尾以 POST 的返回为准。
   一轮都还没跑过时，页面按 settings 里的真实契约显示一张空表，但按钮是可点的。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;

  var refs = {};
  var cursor = 0;        /* 展示到第几轮，0 = 整轮 */
  var training = false;
  var liveLog = [];
  var afterLog = null;   /* 跑完之后继续显示这一轮的日志 */
  var liveRun = null;    /* 正在跑的这一轮：契约是真的，曲线随着 progress.json 长 */
  var liveRecipe = null; /* 训练台回报的这一轮五项参数和设备；没回报之前用已保存的设置 */

  /* 一轮都没跑过时用它顶着：结构参数全部来自真实 settings 与 constants，没有一个是编的。 */
  function draftRun() {
    var d = global.OD.data, c = d.constants;
    var split = d.split || {};
    return {
      sample: false, draft: true, hasCurves: false,
      name: '还没跑过', at: '—', epochs: d.settings.epochs || 30, best_epoch: 0,
      class_ids: ['idle'].concat(d.classes.map(function (x) { return x.id; })),
      input_size: d.features.length,
      /* 窗口帧数取自训练台算出来的划分；还没算出来就留 null，视图渲染成 —，不拿常数顶上。 */
      window: isNum(split.window) ? split.window : null, stride: isNum(d.settings.stride) ? d.settings.stride : c.STRIDE, seed: c.SEED,
      gate: c.GATE, gate_passed: false, gate_reasons: [],
      metrics: { macro_f1: null, min_recall: null, idle_false_alarm: null, per_class: [], confusion: [] },
      curves: { epochs: [], train_loss: [], val_loss: [], macro_f1: [], min_recall: [], idle_false_alarm: [] },
      split: { train: [], val: [] }, mistakes: [], files: []
    };
  }

  /* 在线开跑时，契约用上一轮的（窗口、种子、特征长度这些两边一致），曲线清空重长。 */
  function seedRun() {
    var runs = global.OD.data.runs;
    if (!runs.length) return draftRun();
    var last = runs[0];
    return {
      sample: false, draft: true, hasCurves: false,
      name: '正在训练', at: '—', epochs: global.OD.data.settings.epochs || last.epochs,
      best_epoch: 0, class_ids: last.class_ids, input_size: last.input_size,
      window: isNum(global.OD.data.split && global.OD.data.split.window) ? global.OD.data.split.window : last.window,
      stride: isNum(global.OD.data.settings.stride) ? global.OD.data.settings.stride : last.stride,
      seed: last.seed, gate: last.gate,
      gate_passed: false, gate_reasons: [],
      metrics: { macro_f1: null, min_recall: null, idle_false_alarm: null, per_class: [], confusion: [] },
      curves: { epochs: [], train_loss: [], val_loss: [], macro_f1: [], min_recall: [], idle_false_alarm: [] },
      split: last.split, mistakes: [], files: []
    };
  }

  function baseRun() {
    if (liveRun) return liveRun;
    var runs = global.OD.data.runs;
    return runs.length ? runs[0] : draftRun();
  }
  function gate() { return global.OD.data.constants.GATE; }
  function slice(arr, n) { return arr.slice(0, n); }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  function now() { return new Date().toTimeString().slice(0, 8); }
  function f3(v) { return D.fmtNum(v, 3); }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  /* progress.json 的 history 是逐轮追加的，整段换掉比增量拼接更不容易出错。 */
  function applyHistory(run, history) {
    var rows = Array.isArray(history) ? history : [];
    function col(key) {
      return rows.map(function (row) { return isNum(row[key]) ? row[key] : null; });
    }
    run.curves = {
      epochs: rows.map(function (row, i) { return isNum(row.epoch) ? row.epoch : i + 1; }),
      train_loss: col('train_loss'), val_loss: col('val_loss'),
      macro_f1: col('macro_f1'), min_recall: col('min_recall'),
      idle_false_alarm: col('idle_false_alarm'),
      f1_50: col('f1_50'), edit: col('edit'), frame_accuracy: col('frame_accuracy'),
      pred_segments: col('pred_segments'), true_segments: col('true_segments')
    };
    run.hasCurves = rows.length > 0;
    if (rows.some(function (row) { return isNum(row.f1_50); })) run.train_unit = 'timeline';
    var best = 0, seen = -Infinity;
    rows.forEach(function (row, i) {
      var score = isNum(row.f1_50) ? row.f1_50 : row.macro_f1;
      if (isNum(score) && score > seen) { seen = score; best = i + 1; }
    });
    run.best_epoch = best;
  }

  /* ---------- 视图 ---------- */

  function render() {
    global.OD.charts.reset();
    refs = {};
    var run = baseRun();

    refs.runEpochs = run.epochs;
    var host = el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '训练' }),
        el('p', { text: windowLine() + '训练看整条片子里的连续画面。验证必须留一整条没参与训练的片子，页上的分数就是这条片子切出来的结果。' })
      ]),
      run.draft && !training ? draftCallout() : null,
      participantCard(),
      splitCard(),
      controlCard(run),
      gateCards(run),
      chartsCard(run),
      logCard(run)
    ]));
    requestAnimationFrame(paint);
    return host;
  }

  /* 没跑过的时候，下面所有数字都还没有 —— 直说，并且把上面几步的阻断原因摆出来，
     不然用户只会看到一个按不动的按钮。 */
  function draftCallout() {
    var items = S.allGateItems();
    var bad = items.filter(function (i) { return i.kind === 'bad'; });
    var blockers = (global.OD.data.blockers || []).map(function (text) {
      return { kind: 'bad', text: text, detail: '训练台报的。' };
    });
    var all = bad.concat(blockers);
    if (!all.length) {
      return el('div', { class: 'callout warn sp-mb-4' }, [
        icon('info', 16),
        el('div', { text: '还没有跑过任何一轮。门禁暂时没有阻断项，点下面的"开始训练"就会真的开跑。' })
      ]);
    }
    return el('div', { class: 'callout bad sp-mb-4' }, [
      icon('alert', 16),
      el('div', null, [
        el('strong', { text: '现在开训会被挡下来。' }),
        el('ul', { class: 'sp-mt-2' }, all.slice(0, 4).map(function (item) {
          return el('li', { text: item.text });
        })),
        el('p', { class: 'sp-mt-2', text: '去"数据与标注"页把阻断项清掉；上面列不全的以训练台返回的为准。' })
      ])
    ]);
  }

  /* 窗口按秒定义，帧数由时间基换算。时间基来自片子自己的帧率，不是某个写死的常数。 */
  function windowLine() {
    var planned = recipeFromSettings();
    var s = global.OD.data.split;
    var seconds = planned.window_seconds === null ? '—' : planned.window_seconds;
    if (s && isNum(s.time_base_fps)) {
      return '时间基 ' + s.time_base_fps + ' fps，最短动作 ' + seconds + ' 秒。';
    }
    return '最短动作 ' + seconds + ' 秒。';
  }

  /* ---------- 参加训练的片子 ---------- */

  function participantCard() {
    var clips = (global.OD.data && global.OD.data.clips) || [];
    var settings = (global.OD.data && global.OD.data.settings) || {};
    var excluded = {};
    var held = {};
    (settings.train_excluded || []).forEach(function (id) { excluded[id] = true; });
    (settings.val_clip_ids || []).forEach(function (id) { held[id] = true; });
    var network = settings.network === 'mstcn' ? 'mstcn' : 'lstm';
    var tails = ((global.OD.data.split || {}).tails) || [];
    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '这次训练用哪套' })),
      el('label', { class: 'field sp-mt-3' }, [
        el('span', { class: 'tc sp-meta', text: '网络' }),
        el('select', {
          class: 'form-control', id: 'f-network', disabled: !!S.state.job, onchange: saveParticipation
        }, [
          el('option', { value: 'lstm', selected: network === 'lstm', text: 'LSTM' }),
          el('option', { value: 'mstcn', selected: network === 'mstcn', text: '因果 MS-TCN' })
        ])
      ]),
      el('p', { class: 'hint sp-mt-3', text: networkHint(network) }),
      tails.length ? el('div', { class: 'callout warn sp-mt-3' }, [
        icon('alert', 15),
        el('div', { text: tails.map(function (tail) {
          return tail.clip_id + ' 最后 ' + tail.seconds + ' 秒没涂，会当空闲学；没标完请用「不训练」剪掉。';
        }).join(' ') })
      ]) : null,
      clips.length ? el('div', { class: 'sp-mt-3' }, clips.map(function (clip) {
        var role = excluded[clip.id] ? 'out' : (held[clip.id] ? 'val' : 'train');
        return el('div', { class: 'row tight sp-mt-2' }, [
          el('span', { class: 'mono', text: clip.file || clip.id }),
          el('span', { class: 'sp-meta', text: (clip.segments ? clip.segments.length : 0) + ' 段' }),
          el('span', { class: 'spread' }),
          el('select', {
            class: 'form-control',
            'data-clip-role': clip.id,
            disabled: !!S.state.job,
            onchange: saveParticipation
          }, [
            el('option', { value: 'train', selected: role === 'train', text: '参加训练' }),
            el('option', { value: 'val', selected: role === 'val', text: '只做验证' }),
            el('option', { value: 'out', selected: role === 'out', text: '不参加' })
          ])
        ]);
      })) : el('p', { class: 'help', text: '还没有片子。' })
    ]);
  }

  function networkHint(network) {
    var net = network === 'mstcn'
      ? '因果 MS-TCN 只看已经发生的帧，往回看大约三十秒，并带相邻帧平滑。'
      : 'LSTM 只看已经发生的帧，没有相邻帧平滑。';
    return net + '每次截一整段连续画面来学，每一帧都带上文。至少把一条完整片子设成「只做验证」。';
  }

  function saveParticipation(evt) {
    var api = global.OD.api;
    if (!api.available || !S.state.projectId) {
      S.toast('离线示例不写划分。', 'warn');
      return;
    }
    var netEl = document.getElementById('f-network');
    var mode = 'clip';
    var network = netEl && netEl.value === 'mstcn' ? 'mstcn' : 'lstm';
    var pickedVal = evt && evt.target && evt.target.getAttribute && evt.target.value === 'val';
    if (pickedVal) mode = 'clip';
    var excluded = [];
    var val = [];
    document.querySelectorAll('[data-clip-role]').forEach(function (sel) {
      var id = sel.getAttribute('data-clip-role');
      if (sel.value === 'out') excluded.push(id);
      else if (sel.value === 'val' && mode === 'clip') val.push(id);
    });
    api.split(S.state.projectId, {
      action: 'participate',
      split_mode: mode,
      network: network,
      train_excluded: excluded,
      val_clip_ids: val
    }).then(afterSplit).catch(splitFailed);
  }

  /* ---------- 验证集划分 ---------- */

  /* 划分的单位是段，不是片子。提案由训练台算（同一套规则只有一份实现），
     人确认之后才写盘；片子或帧率一变，确认就自动失效、重新提案。 */
  function splitCard() {
    var s = global.OD.data.split;
    var head = el('header', null, [
      el('h2', { text: '验证集划分' }),
      el('span', { class: 'end' }, s && s.ok
        ? [(global.OD.api && global.OD.api.available) ? null : S.sampleTag('划分为示例'),
           s.confirmed ? el('span', { class: 'badge ok' }, '已确认') : el('span', { class: 'badge warn' }, '待确认'),
           s.same_source ? el('span', { class: 'badge warn', title: '训练集和验证集的段来自同一条片子' }, '同源验证') : null]
        : null)
    ]);

    if (!s) {
      return el('section', { class: 'card' }, [head,
        el('div', { class: 'callout warn' }, [
          icon('info', 15),
          el('div', { text: '离线示例没有训练台，算不出划分。在线模式下这里会列出每段进训练还是进验证，由你确认。' })
        ])
      ]);
    }
    if (!s.ok) {
      return el('section', { class: 'card' }, [head,
        el('div', { class: 'callout bad' }, [
          icon('alert', 15),
          el('div', null, [
            el('strong', { text: '这份标注还分不出训练和验证。' }),
            el('div', { class: 'sp-mt-2', text: s.reason || '原因未知' })
          ])
        ])
      ]);
    }

    var body = [
      el('p', { class: 'hint', text: splitHint(s) }),
      s.same_source ? el('div', { class: 'callout warn sp-mt-3' }, [
        icon('alert', 15),
        el('div', null, [
          el('strong', { text: '同源验证：训练集和验证集的段来自同一条片子。' }),
          el('div', { class: 'sp-mt-2', text: '这个分数看不出换一条新视频会不会被切坏。上面改成「整条片子做验证」，并把一条片子设成只做验证。' })
        ])
      ]) : (s.mode === 'clip' ? el('div', { class: 'callout ok sp-mt-3' }, [
        icon('check', 15),
        el('div', { text: '验证集是整条没参与训练的片子。页上的分数看的是换一条视频认不认得出、会不会被切碎。' })
      ]) : null),
      el('div', { class: 'grid c2 sp-gap-3 sp-mt-4' }, [
        sideCard('训练集', s.train, 'ok'),
        sideCard('验证集', s.val, 'busy')
      ]),
      s.mode === 'clip' ? null : el('div', { class: 'row sp-mt-4' }, [
        el('label', { class: 'row tight', for: 'f-split_val_count' }, [
          el('span', { class: 'tc sp-meta', text: '验证集目标窗口数' }),
          el('input', {
            class: 'form-control mono split-count', type: 'number', id: 'f-split_val_count', min: '1',
            value: String(global.OD.data.settings.split_val_count || ''), placeholder: '按 20% 自动'
          })
        ]),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: rerollSplit }, '重新提案'),
        el('span', { class: 'spread' }),
        el('button', {
          class: 'btn btn-primary btn-sm', type: 'button', disabled: !!S.state.job, onclick: confirmSplit
        }, [icon('check', 13), s.confirmed ? '再确认一次' : '确认这个划分'])
      ])
    ];
    return el('section', { class: 'card' }, [head].concat(body));
  }

  function sideCard(title, side, tone) {
    var counts = side.class_counts || [];
    if (counts.length) {
      var rows = [];
      var actions = 0;
      counts.forEach(function (row) {
        var text = row.frames + ' 帧';
        if (row.class_id !== 'idle') {
          text += ' / ' + row.segments + ' 段';
          actions += row.segments;
        }
        rows.push(el('dt', { text: className(row.class_id) }));
        rows.push(el('dd', { text: text }));
      });
      rows.push(el('dt', { text: '片子' }));
      rows.push(el('dd', { text: (side.clips || []).join('、') || '—' }));
      return el('div', { class: 'gate is-' + tone }, [
        el('div', { class: 'k' }, title),
        el('div', { class: 'v' }, [String(actions), el('small', { text: ' 段' })]),
        el('dl', { class: 'keyval sp-mt-3' }, rows)
      ]);
    }
    var classes = side.classes.map(className).join(' / ') || '—';
    return el('div', { class: 'gate is-' + tone }, [
      el('div', { class: 'k' }, title),
      el('div', { class: 'v' }, [String(side.windows), el('small', { text: ' 个窗口' })]),
      el('dl', { class: 'keyval sp-mt-3' }, [
        el('dt', { text: '段' }), el('dd', { text: side.segments + ' 段' }),
        el('dt', { text: '片子' }), el('dd', { text: side.clips.join('、') || '—' }),
        el('dt', { text: '类别' }), el('dd', { text: classes })
      ])
    ]);
  }

  function className(id) {
    if (id === 'idle') return '空闲';
    var found = null;
    global.OD.data.classes.forEach(function (c) { if (c.id === id) found = c.name; });
    return found || id;
  }

  function splitInput() {
    var api = global.OD.api;
    if (!api.available || !S.state.projectId) {
      S.toast('离线示例不算划分，也不写盘。', 'warn');
      return null;
    }
    var input = document.getElementById('f-split_val_count');
    var raw = input && input.value.trim() !== '' ? Number(input.value) : 0;
    if (raw && (!isFinite(raw) || raw < 1)) {
      S.toast('验证集窗口数要是正整数，或者留空按 20% 自动。', 'bad');
      return null;
    }
    return { api: api, valCount: raw || 0 };
  }

  function rerollSplit() {
    var ready = splitInput();
    if (!ready) return;
    ready.api.split(S.state.projectId, {
      action: 'propose',
      split_seed: (global.OD.data.settings.split_seed || 7) + 1,
      split_val_count: ready.valCount
    }).then(afterSplit).catch(splitFailed);
  }

  function confirmSplit() {
    var s = global.OD.data.split;
    if (!s || !s.ok) return;
    var ready = splitInput();
    if (!ready) return;
    ready.api.split(S.state.projectId, {
      action: 'confirm',
      split_val_count: ready.valCount,
      train: s.train_entries,
      val: s.val_entries
    }).then(afterSplit).catch(splitFailed);
  }

  function splitHint(s) {
    if (s && s.mode === 'clip') {
      return '验证集是整条片子。这条片子的段全部不进训练。帧率不同按秒对齐到最慢的那条，不用重复帧去补。';
    }
    return '划分单位是段，从同一批片子里留几段。这个分数看不出新视频会不会被切坏。';
  }

  function afterSplit(result) {
    var split = result && result.split;
    if (split && split.mode === 'clip' && split.ok) S.toast('整条片子已划开，可以开训。', 'ok');
    else if (split && split.confirmed) S.toast('划分已确认，可以开训了。', 'ok');
    else if (split && !split.ok) S.toast(split.reason || '还分不开', 'warn');
    else S.toast('已重新提案，确认之后才能开训。', 'warn');
    return global.OD.live.reload().then(function () { S.render(); });
  }

  function splitFailed(err) {
    S.toast('划分没成：' + err.message, 'bad');
  }

  var RECIPE_FIELDS = [
    { key: 'epochs', label: '训练轮数', step: '1', min: 1, max: 500, integer: true },
    { key: 'learning_rate', label: '学习率', step: '0.0001', min: 0.000001, max: 1, integer: false },
    { key: 'dropout', label: 'Dropout', step: '0.05', min: 0, max: 0.9, integer: false },
    { key: 'window_seconds', label: '最短动作秒数', hint: '短于这个时长的碎段会并进相邻的段。', step: '0.01', min: 0.04, max: 60, integer: false },
    { key: 'context_seconds', label: '上文秒数', hint: '截段时往前带上的画面。这些帧只提供上文，不算损失。', step: '1', min: 2, max: 120, integer: false },
    { key: 'target_seconds', label: '目标段秒数', hint: '一段里真正算损失的长度。', step: '1', min: 2, max: 120, integer: false },
    { key: 'batch_size', label: '每批段数', hint: '每凑满这么多段就更新一次权重。', step: '1', min: 1, max: 32, integer: true }
  ];

  function recipeFromSettings() {
    var s = (global.OD.data && global.OD.data.settings) || {};
    var split = (global.OD.data && global.OD.data.split) || {};
    return {
      epochs: isNum(s.epochs) ? s.epochs : null,
      learning_rate: isNum(s.learning_rate) ? s.learning_rate : null,
      dropout: isNum(s.dropout) ? s.dropout : null,
      window_seconds: isNum(s.window_seconds) ? s.window_seconds : (isNum(split.window_seconds) ? split.window_seconds : null),
      window: isNum(split.window) ? split.window : null,
      stride: isNum(s.stride) ? s.stride : null,
      context_seconds: isNum(s.context_seconds) ? s.context_seconds : 30,
      target_seconds: isNum(s.target_seconds) ? s.target_seconds : 30,
      batch_size: isNum(s.batch_size) ? s.batch_size : 8,
      device: '',
      device_kind: '',
      device_note: ''
    };
  }

  function framesHint(recipe) {
    return isNum(recipe.window) ? '当前片子下是 ' + recipe.window + ' 帧' : '';
  }

  function recipeText(recipe) {
    function cell(value) { return value === null || value === undefined || value === '' ? '—' : String(value); }
    var frames = isNum(recipe.window) ? '（' + recipe.window + ' 帧）' : '';
    return '训练轮数 ' + cell(recipe.epochs)
      + ' · 学习率 ' + cell(recipe.learning_rate)
      + ' · Dropout ' + cell(recipe.dropout)
      + ' · 最短动作 ' + cell(recipe.window_seconds) + ' 秒' + frames
      + ' · 上文 ' + cell(recipe.context_seconds) + ' 秒'
      + ' · 目标段 ' + cell(recipe.target_seconds) + ' 秒'
      + ' · 每批 ' + cell(recipe.batch_size) + ' 段';
  }

  function setRecipeInputs(recipe, locked) {
    RECIPE_FIELDS.forEach(function (spec) {
      var input = document.getElementById('f-train-' + spec.key);
      if (!input) return;
      if (recipe && isNum(recipe[spec.key])) input.value = recipe[spec.key];
      input.disabled = !!locked;
    });
    if (refs.windowFrames) refs.windowFrames.textContent = framesHint(recipe || {});
  }

  function readRecipeForm() {
    var patch = {};
    var problems = [];
    RECIPE_FIELDS.forEach(function (spec) {
      var input = document.getElementById('f-train-' + spec.key);
      var raw = input ? String(input.value).trim() : '';
      var value = Number(raw);
      var bad = raw === '' || !isFinite(value) || value < spec.min || value > spec.max;
      if (!bad && spec.integer && Math.round(value) !== value) bad = true;
      if (input) input.classList.toggle('is-bad', bad);
      if (bad) problems.push(spec.label + ' 要在 ' + spec.min + ' 到 ' + spec.max + (spec.integer ? ' 的整数' : '') + '之间');
      else patch[spec.key] = spec.integer ? Math.round(value) : value;
    });
    return { patch: patch, problems: problems };
  }

  function paintRecipe(active) {
    var recipe = active && liveRecipe ? liveRecipe : null;
    if (refs.recipeLabel) refs.recipeLabel.textContent = recipe ? '这一轮正在用' : '下一次开训，直接填';
    if (recipe) setRecipeInputs(recipe, true);
    if (!refs.deviceBadge) return;
    if (recipe && recipe.device) {
      refs.deviceBadge.style.display = '';
      refs.deviceBadge.textContent = recipe.device;
      refs.deviceBadge.title = recipe.device_note || '';
      refs.deviceBadge.className = 'badge ' + (recipe.device_kind === 'cuda' ? 'ok' : 'warn');
    } else {
      refs.deviceBadge.style.display = 'none';
      refs.deviceBadge.textContent = '';
      refs.deviceBadge.title = '';
    }
  }

  function recipeInputs(shown, locked) {
    refs.windowFrames = el('div', { class: 'help', text: framesHint(shown) });
    return el('div', { class: 'train-fields' }, RECIPE_FIELDS.map(function (spec) {
      var input = el('input', {
        class: 'form-control mono', type: 'number', step: spec.step, id: 'f-train-' + spec.key,
        min: String(spec.min), max: String(spec.max),
        value: shown[spec.key] === null || shown[spec.key] === undefined ? '' : shown[spec.key],
        disabled: locked
      });
      var extra = spec.key === 'window_seconds' ? refs.windowFrames : null;
      return el('label', { class: 'field' }, [
        el('span', { class: 'form-label', text: spec.label }),
        input,
        spec.hint ? el('span', { class: 'help', text: spec.hint }) : null,
        extra
      ]);
    }));
  }

  function controlCard(run) {
    refs.status = el('span', { class: 'badge ' + (run.draft ? 'plain' : 'ok') },
      training ? '进行中' : (run.draft ? '还没跑过' : '第 ' + run.epochs + ' 轮'));
    refs.statusDraft = !!run.draft;
    refs.nameInput = el('input', {
      class: 'form-control', type: 'text', id: 'f-train-display-name', maxlength: '40',
      placeholder: '例如 包装-现场1', disabled: training, 'aria-label': '模型名字'
    });
    refs.trainBtn = el('button', {
      class: 'btn btn-primary train-launch', type: 'button', disabled: training, onclick: startTrain
    }, [icon('zap', 14), training ? '正在训练' : '开始训练']);
    refs.deviceBadge = el('span', { class: 'badge plain', style: 'display:none' });
    var shown = training && liveRecipe ? liveRecipe : recipeFromSettings();
    refs.recipeLabel = el('div', { class: 'help sp-mt-3', text: training && liveRecipe ? '这一轮正在用' : '下一次开训，直接填' });
    if (training && liveRecipe && liveRecipe.device) {
      refs.deviceBadge.style.display = '';
      refs.deviceBadge.textContent = liveRecipe.device;
      refs.deviceBadge.title = liveRecipe.device_note || '';
      refs.deviceBadge.className = 'badge ' + (liveRecipe.device_kind === 'cuda' ? 'ok' : 'warn');
    }
    var previous = !training && run && !run.draft
      ? el('p', { class: 'help sp-mt-3', text: '下面的曲线是上一轮 ' + run.name + '。上面填的数是下一次开训会用的。改窗口秒数后要重新确认划分。' })
      : null;

    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '训练' }),
        el('span', { class: 'end' }, [
          run.sample ? S.sampleTag('曲线与指标为示例') : null,
          refs.status
        ])
      ]),
      el('p', { class: 'hint', text: '点一次开始训练就存一份模型。训练轮数是这份数据来回学多少遍，不是下面列表里的一条。' }),
      el('label', { class: 'field' }, [
        el('span', { class: 'form-label', text: '模型名字' }),
        refs.nameInput
      ]),
      el('div', { class: 'row sp-mt-3' }, [refs.trainBtn, refs.deviceBadge]),
      refs.recipeLabel,
      recipeInputs(shown, training),
      previous
    ]);
  }

  /* ---------- 门禁指标 ---------- */

  function gateCards(run) {
    var split = (global.OD.data && global.OD.data.split) || {};
    refs.gateHost = el('section', { class: 'card' }, el('header', null, [
      el('h2', { text: '这一轮的分数' }),
      el('span', { class: 'end' }, el('span', { class: 'badge plain', text: '只供查看，不决定能不能用' }))
    ]));
    refs.gateHost.appendChild(el('p', { class: 'hint', text: scoreNote(split) }));
    refs.gateBody = el('div', { class: 'grid c4' });
    refs.gateHost.appendChild(refs.gateBody);
    paintGates(valuesAt(run, cursor || run.epochs));
    return refs.gateHost;
  }

  function scoreNote(split) {
    var run = baseRun();
    if (run && run.hasCurves && run.train_unit !== 'timeline' && !(run.metrics && isNum(run.metrics.f1_50))) {
      return '旧训练方式，分数不能和新模型比。那时候是把每一段单独送进去认类别。';
    }
    if (split && split.ok && split.same_source) {
      return '这些分数来自同一批片子里留下的段。新训练会改成整条片子，这个分数才和新视频对得上。';
    }
    return '这些分数来自整条没参与训练的片子，和结果页拿同一条片子验证是同一套切段。';
  }

  /* 第 n 轮的各项数值。曲线还没长到那一轮就返回 null，由门禁卡渲染成"还没跑"。 */
  function valuesAt(run, n) {
    var c = run.curves;
    var i = Math.max(0, Math.min(c.macro_f1.length, n) - 1);
    var best = 0, bestValue = -Infinity;
    c.macro_f1.slice(0, i + 1).forEach(function (v, k) {
      if (isNum(v) && v > bestValue) { bestValue = v; best = k + 1; }
    });
    function at(key) { return c[key] && i >= 0 && i < c[key].length ? c[key][i] : null; }
    return {
      epoch: n, best_epoch: best,
      macro_f1: at('macro_f1'), min_recall: at('min_recall'), idle_false_alarm: at('idle_false_alarm'),
      f1_50: at('f1_50'), edit: at('edit'), frame_accuracy: at('frame_accuracy'),
      pred_segments: at('pred_segments'), true_segments: at('true_segments'),
      train_loss: at('train_loss'), val_loss: at('val_loss')
    };
  }

  function timelineScores(run) {
    return !!(run && (run.train_unit === 'timeline' || (run.metrics && isNum(run.metrics.f1_50))));
  }

  function paintGates(v) {
    if (!refs.gateBody) return;
    D.clear(refs.gateBody);
    var run = baseRun();
    var timeline = timelineScores(run) || isNum(v.f1_50);
    var cards = timeline ? [
      isNum(v.f1_50) ? {
        k: '段级 F1@50', v: f3(v.f1_50), ok: null, pct: v.f1_50 * 100, meter: true,
        sub: '整条验证片子切出来的段，对上标注的比例'
      } : pendingGate('段级 F1@50', '跑起来才有'),
      isNum(v.edit) ? {
        k: '编辑分', v: f3(v.edit), ok: null, pct: v.edit, meter: true,
        sub: '动作顺序差多少，100 是完全按标注的顺序'
      } : pendingGate('编辑分', '跑起来才有'),
      isNum(v.frame_accuracy) ? {
        k: '逐帧准确率', v: f3(v.frame_accuracy), ok: null, pct: v.frame_accuracy * 100, meter: true,
        sub: '每一帧判对的比例'
      } : pendingGate('逐帧准确率', '跑起来才有'),
      isNum(v.idle_false_alarm) ? {
        k: '空闲误报', v: f3(v.idle_false_alarm), ok: null,
        pct: Math.min(100, v.idle_false_alarm * 100), meter: true,
        sub: '空闲帧被判成动作的比例，越低越好'
      } : pendingGate('空闲误报', '跑起来才有'),
      isNum(v.pred_segments) ? {
        k: '切出 / 标注', v: String(Math.round(v.pred_segments)) + ' / ' + String(Math.round(v.true_segments || 0)),
        meter: false, ok: null, sub: '动作段数，不含空闲'
      } : pendingGate('切出 / 标注', '跑起来才有')
    ] : [
      isNum(v.macro_f1) ? {
        k: '宏平均 F1', v: f3(v.macro_f1), ok: null,
        pct: v.macro_f1 * 100, meter: true,
        sub: '各类 F1 的平均，越高越好'
      } : pendingGate('宏平均 F1', '跑起来才有'),
      isNum(v.min_recall) ? {
        k: '最低召回', v: f3(v.min_recall), ok: null,
        pct: v.min_recall * 100, meter: true,
        sub: '各类召回里最差的那个'
      } : pendingGate('最低召回', '跑起来才有'),
      isNum(v.idle_false_alarm) ? {
        k: '空闲误报', v: f3(v.idle_false_alarm), ok: null,
        pct: Math.min(100, v.idle_false_alarm * 100), meter: true,
        sub: '停机画面被判成动作的比例，越低越好'
      } : pendingGate('空闲误报', '跑起来才有'),
      v.best_epoch ? {
        k: '最佳轮次', v: v.best_epoch, unit: ' / ' + refs.runEpochs + ' 轮', meter: false, ok: true,
        sub: timeline ? '按段级 F1@50 保留的那一轮' : '按验证宏平均 F1 保留的那一轮'
      } : pendingGate('最佳轮次', '跑完才有')
    ];
    cards.forEach(function (spec) {
      refs.gateBody.appendChild(gateCard(spec));
    });
    if (refs.status && !refs.statusDraft) {
      refs.status.textContent = '第 ' + v.epoch + ' / ' + refs.runEpochs + ' 轮';
    }
  }

  /* 没有数就是没有数：灰色的"—"，不画进度条，也不假装是 0。 */
  function pendingGate(k, note) {
    return { k: k, v: '—', ok: null, meter: false, sub: note + ' · 还没跑' };
  }

  /* 阈值文字限位在 10%–90%：刻线还在真实位置，文字不会顶出卡片两端。 */
  function labelPct(pct) { return Math.max(10, Math.min(90, pct)); }

  function gateCard(spec) {
    var tone = spec.ok === null ? 'is-pending' : (spec.ok ? 'is-ok' : 'is-bad');
    return el('div', { class: 'gate ' + tone }, [
      el('div', { class: 'k' }, [spec.ok === null ? null : icon(spec.ok ? 'check' : 'alert', 13), spec.k]),
      el('div', { class: 'v' }, [String(spec.v), spec.unit ? el('small', { text: spec.unit }) : null]),
      spec.meter === false ? null : el('div', { class: 'meter sp-mt-5' }, [
        el('div', { class: 'fill ' + (spec.ok ? 'ok' : 'bad'), style: 'width:' + Math.max(1, spec.pct) + '%' }),
        el('div', { class: 'mark', style: 'left:' + spec.markPct + '%' }),
        spec.markLabel ? el('span', { class: 'mark-label', style: 'left:' + labelPct(spec.markPct) + '%', text: spec.markLabel }) : null
      ]),
      el('div', { class: 'sub', text: spec.sub })
    ]);
  }

  /* ---------- 图表 ---------- */

  function chartsCard(run) {
    /* 没有曲线就不挂图表实例：空的坐标系看着像"训练过了但一条都没动"，比占位图更容易误读。 */
    var hasCurves = !!run.hasCurves && run.curves.macro_f1.length > 0;
    var finals = !training && hasCurves;
    var perClass = run.metrics.per_class.length > 0;
    refs.nodes = hasCurves ? {
      loss: el('div', { class: 'chart' }),
      f1: el('div', { class: 'chart' }),
      recall: el('div', { class: 'chart' }),
      idle: el('div', { class: 'chart' }),
      cls: el('div', { class: 'chart tall' }),
      matrix: el('div', { class: 'chart tall' })
    } : {};
    function seriesNode(key, title, note) {
      return hasCurves ? box(title, note, refs.nodes[key]) : box(title, note, placeholder('跑起来才有曲线'));
    }
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '曲线与混淆' }),
        el('span', { class: 'end' }, run.sample ? S.sampleTag('曲线为示例') : null)
      ]),
      el('div', { class: 'chart-grid' }, [
        seriesNode('loss', '损失', '越低越好。验证损失只算分类，不含平滑'),
        seriesNode('f1', timelineScores(run) ? '段级 F1@50' : '宏平均 F1', '越高越好'),
        seriesNode('recall', timelineScores(run) ? '编辑分' : '最低召回', timelineScores(run) ? '动作顺序，越高越好' : '各类里最差的那个'),
        seriesNode('idle', '空闲误报', '越低越好'),
        box('各类精确率 / 召回 / F1', finals ? '按类别看是哪一类拖后腿' : '跑完才有',
          finals && perClass ? refs.nodes.cls : placeholder('跑完才有各类指标')),
        box('混淆矩阵', finals ? '行是真实、列是判成的' : '跑完才有',
          finals && run.metrics.confusion.length ? refs.nodes.matrix : placeholder('跑完才有混淆矩阵'))
      ])
    ]);
  }

  function box(title, unit, node) {
    return el('div', { class: 'chart-box' }, [
      el('header', null, [el('h3', { text: title }), el('span', { class: 'unit', text: unit })]),
      node
    ]);
  }

  function placeholder(text) {
    return el('div', { class: 'empty compact tall' }, [
      icon('zap', 22),
      el('span', { class: 'sp-fs-12', text: text })
    ]);
  }

  function f1Spec(run, n) {
    var timeline = run.curves.f1_50 && run.curves.f1_50.some(isNum);
    var spec = {
      name: timeline ? '段级 F1@50' : '宏平均 F1',
      x: slice(run.curves.epochs, n),
      data: slice(timeline ? run.curves.f1_50 : run.curves.macro_f1, n),
      color: global.OD.charts.colors.busy, min: 0.3, max: 1
    };
    var best = valuesAt(run, n).best_epoch;
    if (best <= n) spec.best = best;
    return spec;
  }

  function recallSpec(run, n) {
    var timeline = run.curves.edit && run.curves.edit.some(isNum);
    var data = timeline
      ? slice(run.curves.edit, n).map(function (value) { return isNum(value) ? value / 100 : null; })
      : slice(run.curves.min_recall, n);
    return {
      name: timeline ? '编辑分' : '最低召回', x: slice(run.curves.epochs, n), data: data,
      color: global.OD.charts.colors.ok, min: 0.3, max: 1
    };
  }

  function idleSpec(run, n) {
    return {
      name: '空闲误报', x: slice(run.curves.epochs, n), data: slice(run.curves.idle_false_alarm, n),
      color: global.OD.charts.colors.warn, side: 'max', min: 0, max: 0.2
    };
  }

  function paint() {
    var run = baseRun();
    refs.runEpochs = run.epochs;
    if (!run.hasCurves || !run.curves.macro_f1.length) return;
    if (!refs.nodes.loss || !document.body.contains(refs.nodes.loss)) return;
    var C = global.OD.charts, n = cursor || run.epochs;
    var inst = {
      loss: C.mount(refs.nodes.loss, C.loss(refs.nodes.loss, {
        x: slice(run.curves.epochs, n), train: slice(run.curves.train_loss, n), val: slice(run.curves.val_loss, n)
      })),
      f1: C.mount(refs.nodes.f1, C.line(refs.nodes.f1, f1Spec(run, n))),
      recall: C.mount(refs.nodes.recall, C.line(refs.nodes.recall, recallSpec(run, n))),
      idle: C.mount(refs.nodes.idle, C.line(refs.nodes.idle, idleSpec(run, n)))
    };
    /* ECharts 没加载上就整块退化成静态布局，不要留半截实例等着挨炸。 */
    refs.inst = inst.loss && inst.f1 && inst.recall && inst.idle ? inst : null;
    /* 混淆和逐类只有跑完那一轮才有；训练途中这两块保持占位。 */
    if (!training && refs.nodes.cls && run.metrics.per_class.length) {
      C.mount(refs.nodes.cls, C.perClass(refs.nodes.cls, run.metrics.per_class));
      if (run.metrics.confusion.length) {
        C.mount(refs.nodes.matrix, C.matrix(refs.nodes.matrix, {
          names: run.metrics.per_class.map(function (r) { return r.name; }),
          values: run.metrics.confusion
        }));
      }
    }
  }

  /* 训练中只更新四条随时间走的曲线。 */
  function chartsMissing() {
    var run = baseRun();
    var ready = run && run.hasCurves && run.curves && run.curves.macro_f1.length > 0;
    var mounted = refs.nodes && refs.nodes.loss && document.body.contains(refs.nodes.loss);
    return !!(ready && !mounted && S.state.view === 'train');
  }

  function updateSeries() {
    var run = baseRun();
    if (!refs.nodes.loss || !document.body.contains(refs.nodes.loss)) return;
    if (refs.inst === undefined) { paint(); return; }
    if (!refs.inst) return;
    var C = global.OD.charts, n = cursor;
    refs.inst.loss.setOption(C.loss(refs.nodes.loss, {
      x: slice(run.curves.epochs, n), train: slice(run.curves.train_loss, n), val: slice(run.curves.val_loss, n)
    }), true);
    refs.inst.f1.setOption(C.line(refs.nodes.f1, f1Spec(run, n)), true);
    refs.inst.recall.setOption(C.line(refs.nodes.recall, recallSpec(run, n)), true);
    refs.inst.idle.setOption(C.line(refs.nodes.idle, idleSpec(run, n)), true);
  }

  /* ---------- 日志 ---------- */

  function logCard() {
    refs.logHost = el('div', { class: 'logbox' });
    paintLog();
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '日志' }),
        el('span', { class: 'end' }, global.OD.data.log.length ? S.sampleTag('日志为示例') : null)
      ]),
      refs.logHost
    ]);
  }

  function paintLog() {
    if (!refs.logHost) return;
    D.clear(refs.logHost);
    var fromData = global.OD.data.log.map(function (l) {
      return { t: l[0], level: l[1], text: l[2] };
    });
    var lines = afterLog || (training ? liveLog : fromData);
    if (!lines.length) {
      refs.logHost.appendChild(el('p', { class: 'help' },
        '还没有日志。点"开始训练"以后，这里逐轮打出训练台报回来的真实数字。'));
      return;
    }
    lines.forEach(function (l) { refs.logHost.appendChild(logLine(l)); });
    refs.logHost.scrollTop = refs.logHost.scrollHeight;
  }

  function logLine(l) {
    return el('div', { class: 'l-' + (l.level || 'info') }, [
      el('span', { class: 't', text: l.t + '  ' }),
      el('span', { text: l.text })
    ]);
  }

  function pushLog(line) {
    line.t = line.t || now();
    liveLog.push(line);
    if (!refs.logHost) return;
    refs.logHost.appendChild(logLine(line));
    refs.logHost.scrollTop = refs.logHost.scrollHeight;
  }

  /* ---------- 任务 ---------- */

  function readDisplayName() {
    var input = document.getElementById('f-train-display-name');
    var name = input ? String(input.value).trim() : '';
    var bad = !name || name.length > 40 || name.indexOf('/') >= 0 || name.indexOf('\\') >= 0;
    if (input) input.classList.toggle('is-bad', bad);
    return { name: name, bad: bad };
  }

  function startTrain() {
    if (S.state.job) { S.toast('已经有一个任务在跑了，先停掉它。', 'warn'); return; }
    var named = readDisplayName();
    if (named.bad) { S.toast('先给这一次训练起个名字，不能带路径，最多 40 个字', 'bad'); return; }
    var form = readRecipeForm();
    if (form.problems.length) { S.toast(form.problems[0], 'bad'); return; }
    var api = global.OD.api;
    if (api.available && S.state.projectId) startLiveTrain(form.patch, named.name);
    else startSampleTrain();
  }

  /* 离线示例：没有后端可跑，走一遍动画把示例轮次的曲线画出来，页面挂着示例标记。 */
  function startSampleTrain() {
    var data = global.OD.data, run = data.runs[0];
    if (!run) { S.toast('示例数据里没有可跑的轮次，这条按钮只在真实训练台上管用。', 'warn'); return; }
    var total = run.epochs;
    var segs = data.clips.reduce(function (sum, c) { return sum + c.segments.length; }, 0);

    training = true;
    afterLog = null;
    liveLog = [];
    cursor = 0;
    if (refs.trainBtn) refs.trainBtn.disabled = true;
    if (refs.logHost) { D.clear(refs.logHost); }

    [
      { level: 'info', text: '读取标注：' + data.classes.length + ' 个类别、' + data.clips.length + ' 条片子、' + segs + ' 段' },
      { level: 'info', text: '窗口 ' + D.fmtNum(run.window, 0) + ' 帧、步长 ' + run.stride + '、特征长度 ' + run.input_size },
      { level: 'info', text: '划分按段：训练 ' + run.split.train.join('、') + '，验证 ' + run.split.val.join('、') + '（seed ' + run.seed + '）' },
      { level: 'info', text: '示例数据：这一次不会真的训练，曲线来自示例轮次' }
    ].forEach(pushLog);

    S.startJob({
      kind: 'train', label: '训练（示例）', total: total, unit: '轮', period: 420,
      onTick: function (done) {
        cursor = done;
        var v = valuesAt(run, done);
        paintGates(v);
        updateSeries();
        pushLog({
          level: 'info',
          text: 'epoch ' + pad2(done) + '  train ' + f3(v.train_loss) + '  val ' + f3(v.val_loss) + '  macro-F1 ' + f3(v.macro_f1) +
            (done === v.best_epoch ? '  ← 目前最好，已保存' : '')
        });
      },
      onStop: finish,
      onDone: function () {
        var v = valuesAt(run, run.epochs);
        pushLog({
          level: 'info',
          text: '已保存。宏平均 F1 ' + f3(v.macro_f1) + '、最低召回 ' + f3(v.min_recall) + '、空闲误报 ' + f3(v.idle_false_alarm)
        });
        pushLog({ level: 'info', text: '示例轮次的产物：' + run.files.join('、') });
        finish();
      }
    });
  }

  /* 在线：POST /train 会阻塞到训完，progress.json 每轮都在重写。
     轮询只喂进度和曲线，收尾一律以 POST 的返回为准。 */
  function startLiveTrain(patch, displayName) {
    var data = global.OD.data, api = global.OD.api, projectId = S.state.projectId;
    var segs = data.clips.reduce(function (sum, c) { return sum + c.segments.length; }, 0);
    var logged = 0;
    var phase = '';
    var posted = {};
    Object.keys(patch).forEach(function (key) { posted[key] = patch[key]; data.settings[key] = patch[key]; });
    posted.display_name = displayName;
    training = true;
    afterLog = null;
    liveLog = [];
    cursor = 0;
    liveRecipe = null;
    liveRun = seedRun();
    refs.runEpochs = liveRun.epochs;
    if (refs.trainBtn) refs.trainBtn.disabled = true;
    if (refs.nameInput) refs.nameInput.disabled = true;
    setRecipeInputs(null, true);
    if (refs.recipeLabel) refs.recipeLabel.textContent = '这一轮正在用';
    if (refs.logHost) { D.clear(refs.logHost); }

    [
      { level: 'info', text: '读取标注：' + data.classes.length + ' 个类别、' + data.clips.length + ' 条片子、' + segs + ' 段' },
      { level: 'info', text: '窗口 ' + D.fmtNum(liveRun.window, 0) + ' 帧、步长 ' + liveRun.stride + '、特征长度 ' + liveRun.input_size },
      { level: 'info', text: '开训：先抽取勾选片子的特征，抽完再训练。已有最新特征的直接用。' }
    ].forEach(pushLog);

    S.startJob({
      kind: 'train', label: '训练', total: liveRun.epochs, unit: '轮', period: 1500,
      live: {
        get: function () {
          return api.progress(projectId).then(function (p) {
            p = p || {};
            var features = p.features || {};
            if (p.status === 'extracting') {
              return {
                done: features.done || 0,
                total: features.total || 0,
                status: 'extracting',
                history: [],
                recipe: null
              };
            }
            return {
              done: typeof p.epoch === 'number' ? p.epoch : 0,
              total: typeof p.epochs === 'number' ? p.epochs : 0,
              status: p.status, history: p.history || [], recipe: p.recipe || null
            };
          });
        },
        run: function () { return api.train(projectId, posted); },
        onFail: function (err) {
          training = false;
          liveRecipe = null;
          setRecipeInputs(null, false);
          if (refs.nameInput) refs.nameInput.disabled = false;
          if (refs.trainBtn) refs.trainBtn.disabled = false;
          if (refs.recipeLabel) refs.recipeLabel.textContent = '下一次开训，直接填';
          pushLog({ level: 'bad', text: err.message });
        }
      },
      onTick: function (done, total, snapshot) {
        if (!snapshot) return;
        if (snapshot.status === 'extracting') {
          if (phase !== 'extracting') {
            phase = 'extracting';
            pushLog({ level: 'info', text: '正在抽取特征。这一步按帧走，抽完才进入训练。' });
          }
          if (S.state.job) {
            S.state.job.label = '抽取特征';
            S.state.job.unit = '帧';
          }
          return;
        }
        if (phase === 'extracting') {
          phase = 'training';
          pushLog({ level: 'info', text: '特征已就绪，开始训练' });
        }
        if (S.state.job) {
          S.state.job.label = '训练';
          S.state.job.unit = '轮';
        }
        if (snapshot.recipe) { liveRecipe = snapshot.recipe; paintRecipe(true); }
        if (isNum(snapshot.total) && snapshot.total > 0) liveRun.epochs = snapshot.total;
        var hist = Array.isArray(snapshot.history) ? snapshot.history : [];
        applyHistory(liveRun, hist);
        refs.runEpochs = liveRun.epochs;
        cursor = hist.length;
        if (chartsMissing()) { S.render(); return; }
        var v = valuesAt(liveRun, cursor || liveRun.epochs);
        paintGates(v);
        updateSeries();
        /* history 是整段重发的，只把新长出来的那几轮打进日志。 */
        while (logged < hist.length) {
          var row = hist[logged];
          logged += 1;
          pushLog({
            level: 'info',
            text: 'epoch ' + pad2(row.epoch) + '  train ' + f3(row.train_loss) + '  val ' + f3(row.val_loss) + '  macro-F1 ' + f3(row.macro_f1) +
              (row.epoch === liveRun.best_epoch ? '  ← 目前最好，已保存' : '')
          });
        }
      },
      onStop: function () {
        pushLog({ level: 'warn', text: '已请求停止，训练台会在这一轮跑完后收工并保留当时最好的权重' });
        finishLive(1200);
      },
      onDone: function (result) {
        var report = (result && result.report) || {};
        var m = report.metrics || {};
        pushLog({
          level: 'ok',
          text: '已保存。宏平均 F1 ' + f3(m.macro_f1) + '、最低召回 ' + f3(m.min_recall) + '、空闲误报 ' + f3(m.idle_false_alarm)
        });
        pushLog({ level: 'info', text: '结果写到 runs/' + ((result && result.run) || '新目录') + '，下面按磁盘上的内容重新画一遍' });
        finishLive(0);
      }
    });
  }

  function finish() {
    training = false;
    cursor = 0;
    afterLog = liveLog.length ? liveLog : null;
    if (global.OD.shell.state.job === null) S.render();
  }

  /* 在线收尾：等训练台把这一轮的文件写完再读盘。
     停止的时候服务端要跑完当前这一轮才落盘，所以给它一点时间。 */
  function finishLive(delay) {
    training = false;
    cursor = 0;
    liveRun = null;
    liveRecipe = null;
    afterLog = liveLog.length ? liveLog : null;
    setTimeout(function () {
      global.OD.live.reload().then(function () {
        S.render();
        var run = global.OD.data.runs[0];
        if (afterLog && run) pushLog({ level: 'info', text: '这一轮的产物：' + run.files.join('、') });
      }).catch(function (err) {
        S.render();
        S.toast('训练结束了，但重新读取工程失败：' + err.message, 'bad');
      });
    }, delay || 0);
  }

  function watch(initial) {
    if (!initial || S.state.job) return;
    if (initial.status !== 'extracting' && initial.status !== 'training') return;
    var api = global.OD.api;
    var projectId = S.state.projectId;
    if (!api.available || !projectId) return;
    var history = Array.isArray(initial.history) ? initial.history : [];
    training = true;
    afterLog = null;
    liveLog = [{ t: now(), level: 'info', text: '已接上正在进行的任务，刷新不会再重开一轮。' }];
    cursor = history.length;
    liveRecipe = initial.recipe || null;
    liveRun = seedRun();
    if (history.length) applyHistory(liveRun, history);
    var phase = initial.status === 'training' ? 'training' : '';
    var logged = history.length;
    var lastStamp = initial.updated_at || 0;
    var lastMove = Date.now();
    var closed = false;

    function snapshotOf(progress) {
      progress = progress || {};
      var stamp = progress.updated_at || 0;
      if (stamp && stamp !== lastStamp) {
        lastStamp = stamp;
        lastMove = Date.now();
      }
      var busy = progress.status === 'extracting' || progress.status === 'training';
      var stale = busy && (Date.now() - lastMove > 45000);
      var features = progress.features || {};
      if (progress.status === 'extracting') {
        return {
          done: features.done || 0,
          total: features.total || 0,
          status: 'extracting',
          history: [],
          recipe: null,
          stale: stale
        };
      }
      return {
        done: typeof progress.epoch === 'number' ? progress.epoch : history.length,
        total: typeof progress.epochs === 'number' ? progress.epochs : (liveRun.epochs || 0),
        status: progress.status,
        history: progress.history || [],
        recipe: progress.recipe || null,
        stale: stale,
        error: progress.error || ''
      };
    }

    function finishWatch(progress) {
      if (closed) return;
      closed = true;
      training = false;
      liveRun = null;
      liveRecipe = null;
      S.endJob();
      if (progress && (progress.status === 'error' || progress.status === 'stopped')) {
        S.toast(progress.error || '这一轮停了', progress.status === 'error' ? 'bad' : 'warn');
      } else if (progress && progress.status === 'done') {
        S.toast('这一轮训练已完成', 'ok');
      }
      global.OD.live.reload().catch(function () { S.render(); });
    }

    var extracting = initial.status === 'extracting';
    S.startJob({
      watch: true,
      kind: 'train',
      label: extracting ? '抽取特征' : '训练',
      total: extracting ? ((initial.features && initial.features.total) || 1) : (liveRun.epochs || 1),
      unit: extracting ? '帧' : '轮',
      period: 1000,
      live: {
        get: function () { return api.progress(projectId).then(snapshotOf); },
        run: function () { return Promise.resolve(null); }
      },
      onTick: function (done, total, snapshot) {
        if (!snapshot || closed) return;
        if (snapshot.status === 'done' || snapshot.status === 'error' || snapshot.status === 'stopped' ||
            snapshot.status === 'idle' || snapshot.status === 'features-done' || snapshot.status === 'check-done') {
          finishWatch(snapshot);
          return;
        }
        if (snapshot.status === 'extracting') {
          if (phase !== 'extracting') {
            phase = 'extracting';
            pushLog({ level: 'info', text: '正在抽取特征。抽完才进入训练。' });
          }
          if (S.state.job) {
            S.state.job.label = '抽取特征';
            S.state.job.unit = '帧';
          }
          return;
        }
        if (phase !== 'training') {
          var fromExtract = phase === 'extracting';
          phase = 'training';
          pushLog({ level: 'info', text: fromExtract ? '特征已就绪，开始训练' : '训练还在进行' });
        }
        if (S.state.job) {
          S.state.job.label = '训练';
          S.state.job.unit = '轮';
        }
        if (snapshot.recipe) { liveRecipe = snapshot.recipe; paintRecipe(true); }
        if (isNum(snapshot.total) && snapshot.total > 0) liveRun.epochs = snapshot.total;
        var hist = Array.isArray(snapshot.history) ? snapshot.history : [];
        applyHistory(liveRun, hist);
        if (refs.runEpochs !== undefined) refs.runEpochs = liveRun.epochs;
        cursor = hist.length;
        if (chartsMissing()) { S.render(); return; }
        if (refs.gateBody) {
          var values = valuesAt(liveRun, cursor || liveRun.epochs);
          paintGates(values);
          updateSeries();
        }
        while (logged < hist.length) {
          var row = hist[logged];
          logged += 1;
          pushLog({
            level: 'info',
            text: 'epoch ' + pad2(row.epoch) + '  train ' + f3(row.train_loss) + '  val ' + f3(row.val_loss) + '  macro-F1 ' + f3(row.macro_f1) +
              (row.epoch === liveRun.best_epoch ? '  ← 目前最好，已保存' : '')
          });
        }
      },
      onStop: function () { finishLive(1200); },
      onDone: function () { finishLive(0); }
    });
    S.render();
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.train = { render: render, inspector: inspector, watch: watch };

  function inspector() {
    var run = baseRun();
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '分数怎么看' }),
        el('p', null,
          '宏平均看整体，最低召回看有没有哪一类整段没认出来，空闲误报看停机时会不会乱报。这些数只供查看，模型能不能用拿到工位上试。')
      ]),
      run && run.files.length ? el('div', { class: 'inspect-block' }, [
        el('h3', { text: '这一轮的产物' }),
        el('div', null, run.files.map(function (f) {
          return el('div', { class: 'class-row' }, [el('span', { class: 'mono sp-fs-12', text: f })]);
        }))
      ]) : null,
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '为什么划分单位是段' }),
        el('p', null,
          '相邻窗口只差 8 帧，几乎一模一样。所以划分只能按整段切 —— 同一个段的窗口要么全进训练、要么全进验证，绝不能跨边，否则验证集里全是训练集的双胞胎，指标会虚高到 0.99 而工位上完全不能用。')
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '两套网络' }),
        el('p', null,
          'LSTM 是现在这套，每一帧一个类别，没有相邻帧平滑。因果 MS-TCN 只看已经发生的帧，后一阶段修边界，训练时加上相邻帧平滑，用来压类别来回跳、一段被切碎。选哪套就训哪套，各自存成一个模型。')
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '同源验证意味着什么' }),
        el('p', null,
          '如果验证段和训练段来自同一批片子，分数看不出换一条新视频会不会被切坏。把一条完整片子设成「只做验证」之后，页上的分数才是这条片子整段没参与训练的结果。')
      ])
    ]);
  }
})(window);
