/* 应用外壳：路由、左侧步骤导航、顶部唯一任务条、通知与确认对话框。
   全站的状态只有一个来源 —— 五个步骤的状态点由门禁与产物推导，
   任务条只有一个，停止入口只有一个。 */
(function (global) {
  'use strict';

  var D = global.OD.dom;
  var el = D.el, clear = D.clear, icon = D.icon, qs = D.qs;

  var STEPS = [
    { key: 'model', name: '模型与工位', idx: '1' },
    { key: 'data', name: '数据与标注', idx: '2' },
    { key: 'train', name: '训练', idx: '3' },
    { key: 'runs', name: '结果', idx: '4' }
  ];

  /* source 决定这个页面是"离线示例"还是"连着真实训练台"。
     live 时 projectId 就是接口路径里的工程名；bootError 留着服务没起时的原因。 */
  var state = {
    view: 'home', clipId: '1', job: null, segSel: null, zoom: 4, seekFrame: 0,
    source: 'offline', projectId: null, bootError: ''
  };
  var subs = [];
  var host, railHost, jobHost, crumbsHost, inspectHost, shellEl, srcTag;
  var tickTimer = null, pollTimer = null, jobStart = 0;

  function on(fn) { subs.push(fn); }
  function emit() { subs.forEach(function (fn) { fn(state); }); }

  /* ---------- 状态推导 ---------- */

  function clipById(id) {
    var clips = global.OD.data.clips;
    for (var i = 0; i < clips.length; i++) if (clips[i].id === id) return clips[i];
    return clips[0];
  }

  function segmentsOf(clip) { return clip ? clip.segments : []; }

  /* 窗口按秒定义：这条片子自己的帧率乘出来，就是它上面「段够不够长」的帧数门槛。
     训练台报的 window_seconds 优先；离线示例回落到同一个默认值。
     界面和训练用同一个秒数 —— 以前一边按源帧量、一边按重采样后量，14 段被全部误标红。 */
  function windowSeconds() {
    var s = global.OD.data && global.OD.data.split;
    return (s && typeof s.window_seconds === 'number' && s.window_seconds > 0) ? s.window_seconds : 1.92;
  }

  function windowFrames(clip) {
    var fps = clip && clip.fps > 0 ? clip.fps : 25;
    return Math.max(1, Math.round(windowSeconds() * fps));
  }

  /* 时间基由片子帧率推导，不再是让人填的设置项 —— 换相机不用改任何东西。
     左栏、模型页、模型页报告都取这一份，免得各写一套迟早分叉。
     划分还没算出来（离线示例、或片子帧率读不出）时如实写"还没算出来"，不拿常数顶上。 */
  function timeBase() {
    var split = global.OD.data && global.OD.data.split;
    var fps = split && typeof split.time_base_fps === 'number' && split.time_base_fps > 0 ? split.time_base_fps : null;
    var win = split && typeof split.window === 'number' && split.window > 0 ? split.window : null;
    var sec = split && typeof split.window_seconds === 'number' && split.window_seconds > 0 ? split.window_seconds : windowSeconds();
    if (fps === null) return { fps: null, window: win, seconds: sec, text: '还没算出来' };
    return {
      fps: fps, window: win, seconds: sec,
      text: fps.toFixed(2) + ' fps · 窗口 ' + (win === null ? '—' : win + ' 帧') + ' = ' + sec + ' 秒'
    };
  }

  /* 标注问题：段短于窗长、段间窄缝、片尾未标注（≥窗长会变成 idle）。 */
  function clipIssues(clip) {
    if (!clip || !clip.segments.length) return { short: [], narrow: [], tail: 0, windows: 0 };
    var W = windowFrames(clip);
    var segs = clip.segments.slice().sort(function (a, b) { return a.start - b.start; });
    var short = segs.filter(function (s) { return s.end - s.start < W; });
    var narrow = [];
    for (var i = 1; i < segs.length; i++) {
      var gap = segs[i].start - segs[i - 1].end;
      if (gap > 0 && gap <= 3) narrow.push({ after: segs[i - 1], before: segs[i], gap: gap });
    }
    var tail = Math.max(0, clip.frame_count - segs[segs.length - 1].end);
    var windows = segs.reduce(function (sum, s) {
      var len = s.end - s.start;
      return sum + (len >= W ? Math.floor((len - W) / global.OD.data.constants.STRIDE) + 1 : 0);
    }, 0);
    return { short: short, narrow: narrow, tail: tail, windows: windows };
  }

  function gateOf(clip) {
    var issues = clipIssues(clip);
    var W = windowFrames(clip);
    var items = [];
    /* 每条都带上片子 id，门禁点进去才能停在出问题的那条片子上，而不是永远跳第一条。 */
    function add(item) { item.clip = clip ? clip.id : null; items.push(item); }
    if (clip && clip.segments.length === 0) {
      add({ kind: 'bad', text: clip.file + ' 还没有任何标注段', detail: '没有标注就没有训练窗口。打开标注工作台，把画面停在动作开始处再点类别。', go: 'annotate' });
    }
    issues.short.forEach(function (s) {
      add({
        kind: 'warn', text: clip.file + ' 第 ' + s.start + '–' + s.end + ' 帧只有 ' + (s.end - s.start) + ' 帧',
        detail: '短于最短窗长 ' + W + ' 帧，这一段产不出任何训练窗口。', go: 'annotate'
      });
    });
    issues.narrow.forEach(function (n) {
      add({
        kind: 'warn', text: clip.file + ' 第 ' + n.after.end + '–' + n.before.start + ' 帧之间只有 ' + n.gap + ' 帧空隙',
        detail: '相邻两段贴得太近，容易把同一个动作切成两段。', go: 'annotate'
      });
    });
    if (issues.tail >= W) {
      add({
        kind: 'warn', text: clip.file + ' 片尾 ' + issues.tail + ' 帧未标注',
        detail: '超过 ' + W + ' 帧的空白会被当成空闲段参与训练，确认这里确实是停机画面。', go: 'annotate'
      });
    }
    /* 划分能不能成由训练台算（split_overview），这里不再自己按「片子条数」下判断 ——
       划分单位改成段之后，一条片子也能分开，那条规则是假的。 */
    var split = global.OD.data.split;
    if (split && !split.ok) {
      add({ kind: 'bad', text: '验证集分不出来', detail: split.reason || '训练台没有给出原因', go: 'train' });
    }
    return items;
  }

  /* 门禁要按全部片子算。以前只取 clipById('1')，接上真实工程以后会漏掉别的片子。
     同一条原因会在每条片子上都出现一次，按文字去重。 */
  function allGateItems() {
    var seen = {}, out = [];
    global.OD.data.clips.forEach(function (clip) {
      gateOf(clip).forEach(function (item) {
        if (seen[item.text]) return;
        seen[item.text] = true;
        out.push(item);
      });
    });
    return out;
  }

  function stepState(key) {
    var data = global.OD.data;
    var job = state.job;
    if (job) {
      if ((key === 'features' && job.kind === 'extract') || (key === 'train' && job.kind === 'train')) return 'busy';
    }
    if (key === 'model') {
      if (!data.probe.yolo.ok) return 'bad';
      var handsOn = !!(data.settings && data.settings.hands_in_training);
      if (handsOn && !(data.probe.hands && data.probe.hands.ok)) return 'bad';
      return 'ok';
    }
    if (key === 'data') {
      var items = allGateItems();
      if (items.some(function (i) { return i.kind === 'bad'; })) return 'bad';
      if (items.length) return 'warn';
      return 'ok';
    }
    if (key === 'features') {
      var rows = data.features.rows;
      if (rows.every(function (r) { return r.state === 'fresh'; })) return 'ok';
      if (rows.some(function (r) { return r.state === 'fresh'; })) return 'warn';
      return 'warn';
    }
    if (key === 'train' || key === 'runs') {
      var runs = data.runs;
      if (!runs.length) return null;
      return 'ok';
    }
    return null;
  }

  function stepSub(key) {
    var data = global.OD.data;
    if (key === 'model') return data.probe.yolo.ok ? data.probe.yolo.classes.length + ' 个零件类别' : '还没检查';
    if (key === 'data') {
      var issues = allGateItems();
      return issues.length ? issues.length + ' 条待处理' : '可以开训';
    }
    if (key === 'train') return data.runs.length ? '最近 ' + data.runs[0].name : '开训时自动抽特征';
    if (key === 'runs') return '保留最近 ' + data.runs.length + ' 轮';
    return '';
  }

  /* ---------- 任务条 ---------- */

  /* 两种任务：离线是前端自己走的假进度，在线是真的 POST + 轮询 progress.json。
     在线的 spec.live = { get(): Promise<{done,total,status}>, run(): Promise<结果>, onFail? }。
     进度条只信轮询，收尾只信 POST 的返回 —— 两个都不靠猜。 */
  function startJob(spec) {
    stopJobSilently();
    jobStart = Date.now();
    state.job = {
      kind: spec.kind, label: spec.label, total: spec.total,
      done: 0, unit: spec.unit || '', onTick: spec.onTick, onStop: spec.onStop, onDone: spec.onDone
    };
    if (spec.live) {
      state.job.live = spec.live;
      state.job.watch = !!spec.watch;
      renderJob();
      emit();
      pollTimer = setInterval(pollLive, spec.period || 1200);
      pollLive();
      if (!spec.watch) runLive();
      return;
    }
    var period = spec.period || 600;
    tickTimer = setInterval(function () {
      if (!state.job) return;
      state.job.done += 1;
      if (state.job.onTick) state.job.onTick(state.job.done, state.job.total);
      if (state.job.done >= state.job.total) {
        var done = state.job.onDone, label = state.job.label;
        clearInterval(tickTimer); tickTimer = null; state.job = null;
        if (done) done();
        toast(label + '已完成', 'ok');
      }
      renderJob();
      emit();
    }, period);
    renderJob();
    emit();
  }

  /* progress.json 是训练台每一拍重写的，读失败不打断任务，下一拍再来。 */
  function pollLive() {
    var job = state.job;
    if (!job || !job.live) return;
    job.live.get().then(function (snapshot) {
      if (state.job !== job) return;
      if (snapshot) {
        if (typeof snapshot.done === 'number') job.done = snapshot.done;
        if (typeof snapshot.total === 'number' && snapshot.total > 0) job.total = snapshot.total;
        job.stale = !!snapshot.stale;
        job.misses = 0;
      }
      if (job.onTick) job.onTick(job.done, job.total, snapshot);
      renderJob();
      emit();
    }).catch(function () {
      if (state.job !== job) return;
      job.misses = (job.misses || 0) + 1;
      if (job.misses >= 4) job.stale = true;
      renderJob();
    });
  }

  function runLive() {
    var job = state.job;
    job.live.run().then(function (result) {
      if (state.job !== job) return;
      var done = job.onDone, label = job.label;
      stopJobSilently(); renderJob();
      if (done) done(result);
      toast(label + '已完成', 'ok');
      emit();
    }).catch(function (err) {
      if (state.job !== job) return;
      var label = job.label, onFail = job.live.onFail;
      stopJobSilently(); renderJob();
      if (onFail) onFail(err);
      toast(label + '失败：' + err.message, 'bad');
      emit();
    });
  }

  function stopJobSilently() {
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    state.job = null;
  }

  function endJob() {
    stopJobSilently();
    renderJob();
    emit();
  }

  function requestStop() {
    if (!state.job) return;
    var job = state.job;
    confirmDialog({
      title: '停止' + job.label + '？',
      body: '已经跑到 ' + job.done + '/' + job.total + '。停止后这一轮的中间结果不会写进结果页。',
      ok: '停止',
      danger: true,
      onOk: function () {
        var label = job.label, onStop = job.onStop;
        stopJobSilently(); renderJob(); emit();
        if (onStop) onStop();
        toast('已停止' + label, 'warn');
        /* 在线任务真的跑在训练台进程里，界面停下不算停 —— 必须让服务端收工。
           停止接口不走工程锁，训练跑着也收得到。 */
        if (job.live && global.OD.api.available && state.projectId) {
          global.OD.api.stop(state.projectId).catch(function (err) {
            toast('停止请求没送到训练台：' + err.message, 'bad');
          });
        }
      }
    });
  }

  function renderJob() {
    if (!jobHost) return;
    clear(jobHost);
    var job = state.job;
    jobHost.className = 'jobbar ' + (job ? 'is-busy' : 'is-idle');
    jobHost.appendChild(el('span', { class: 'dot ' + (job ? 'busy' : 'ok') }));
    if (!job) {
      jobHost.appendChild(el('span', { class: 'label', text: '空闲' }));
      return;
    }
    var pct = job.total > 0 ? Math.max(0, Math.min(100, Math.round((job.done / job.total) * 100))) : 0;
    jobHost.appendChild(el('span', { class: 'label', text: job.stale ? '训练台已断开' : job.label }));
    jobHost.appendChild(el('div', { class: 'progress bar', role: 'progressbar', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 },
      el('div', { class: 'progress-bar busy', style: 'width:' + pct + '%' })));
    jobHost.appendChild(el('span', { class: 'label mono', text: job.done + '/' + job.total + ' ' + job.unit }));
    jobHost.appendChild(el('span', { class: 'label mono', text: ((Date.now() - jobStart) / 1000).toFixed(0) + 's' }));
    jobHost.appendChild(el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: requestStop },
      [icon('stop', 13), '停止']));
  }

  /* ---------- 顶部与导航 ---------- */

  var VIEW_TITLE = {
    home: '全部工程', model: '模型与工位', data: '数据与标注', features: '特征',
    train: '训练', runs: '结果', annotate: '标注工作台'
  };

  /* 数据源徽标：这个页面到底是"连着训练台"还是"离线示例"，必须一眼看得出来，
     否则用户会把示例里的数字当成自己工程跑出来的。 */
  function renderSource() {
    if (!srcTag) return;
    var live = state.source === 'live';
    /* 连不上的原因由 live 模块拿着，外壳不复制一份，免得两处会不一致。 */
    var err = state.bootError || (global.OD.live && global.OD.live.error ? global.OD.live.error() : '');
    srcTag.className = 'badge ' + (live ? 'ok' : 'plain');
    srcTag.textContent = live ? '在线 · ' + state.projectId : '离线示例';
    srcTag.title = live
      ? '数据来自本机训练台的 ' + state.projectId + ' 工程，写操作会真的落盘'
      : (err ? '训练台没连上：' + err : '双击打开，没有后端，写操作不会落盘');
    /* 页签标题也跟着工程走：写死一个工程名，切过去之后页签就在说谎。 */
    document.title = '时序训练台 · ' + global.OD.data.project.id + (live ? '' : '（示例）');
  }

  function renderTop() {
    clear(crumbsHost);
    renderSource();
    var clip = clipById(state.clipId);
    crumbsHost.appendChild(el('a', { href: '#/home', text: '全部工程' }));
    crumbsHost.appendChild(el('span', { class: 'sep', text: '/' }));
    crumbsHost.appendChild(el('span', { class: 'mid' }, [
      el('a', { href: '#/model', text: global.OD.data.project.id }),
      el('span', { class: 'sep', text: '/' })
    ]));
    if (state.view === 'annotate') {
      crumbsHost.appendChild(el('span', { class: 'mid' }, [
        el('a', { href: '#/data', text: '数据与标注' }),
        el('span', { class: 'sep', text: '/' })
      ]));
      crumbsHost.appendChild(el('span', { class: 'here', text: clip.file }));
    } else {
      crumbsHost.appendChild(el('span', { class: 'here', text: VIEW_TITLE[state.view] }));
    }
  }

  function renderRail() {
    if (!railHost) return;
    clear(railHost);
    railHost.appendChild(el('div', { class: 'rail-title', text: '工程' }));
    STEPS.forEach(function (step) {
      var st = stepState(step.key);
      var current = state.view === step.key || (step.key === 'data' && state.view === 'annotate');
      railHost.appendChild(el('button', {
        class: 'step', type: 'button', 'data-state': st || 'none',
        'aria-current': current ? 'true' : 'false',
        onclick: function () { go(step.key); }
      }, [
        el('span', { class: 'idx', text: st === 'ok' ? '✓' : (st === 'bad' ? '!' : step.idx) }),
        el('span', { class: 'name', text: step.name }),
        el('span', { class: 'sr', text: '状态：' + stateLabel(st) }),
        el('span', { class: 'sub', text: stepSub(step.key) })
      ]));
    });

    railHost.appendChild(el('div', { class: 'rail-sep' }));
    railHost.appendChild(el('div', { class: 'rail-title', text: '片子' }));
    global.OD.data.clips.forEach(function (clip) {
      var st = clip.segments.length ? 'ok' : 'none';
      railHost.appendChild(el('button', {
        class: 'step flat', type: 'button', 'data-state': st,
        'aria-current': state.view === 'annotate' && state.clipId === clip.id ? 'true' : 'false',
        onclick: function () { go('annotate', { clipId: clip.id }); }
      }, [
        el('span', { class: 'idx' }, icon('film', 13)),
        el('span', { class: 'name', text: clip.file }),
        el('span', { class: 'sub', text: clip.segments.length + ' 段 · ' + clip.frame_count + ' 帧' })
      ]));
    });

    var s = global.OD.data.settings;
    railHost.appendChild(el('div', { class: 'rail-foot' }, el('dl', null, [
      el('dt', { text: '时间基' }), el('dd', { text: timeBase().text }),
      el('dt', { text: '零件类别' }), el('dd', { text: (s.yolo_classes || []).length + ' 个' }),
      el('dt', { text: '特征长度' }), el('dd', { text: global.OD.data.features.length })
    ])));
  }

  function stateLabel(st) {
    return { ok: '通过', warn: '有警告', bad: '未通过', busy: '进行中' }[st] || '未开始';
  }

  /* ---------- 路由 ---------- */

  function go(view, params) {
    if (params && params.clipId) state.clipId = params.clipId;
    state.seekFrame = params && params.frame !== undefined ? params.frame : 0;
    state.view = view;
    if (view !== 'annotate') state.segSel = null;
    var hash = '#/' + view + (view === 'annotate' ? '/' + state.clipId : '');
    if (global.location.hash !== hash) { global.location.hash = hash; return; }
    render();
  }

  function parseHash() {
    var raw = (global.location.hash || '').replace(/^#\/?/, '');
    var parts = raw.split('/').filter(Boolean);
    if (!parts.length) return { view: 'home' };
    if (parts[0] === 'annotate') return { view: 'annotate', clipId: parts[1] || '1' };
    if (parts[0] === 'features') return { view: 'train' };
    if (VIEW_TITLE[parts[0]]) return { view: parts[0] };
    return { view: 'home' };
  }

  function render() {
    var route = parseHash();
    state.view = route.view;
    if (route.clipId) state.clipId = route.clipId;

    renderTop();
    renderRail();
    renderJob();

    var view = global.OD.views[state.view] || global.OD.views.home;
    clear(host);
    var node = view.render();
    if (node) host.appendChild(node);

    clear(inspectHost);
    var side = view.inspector ? view.inspector() : null;
    shellEl.classList.toggle('has-inspect', !!side);
    if (side) inspectHost.appendChild(side);
  }

  /* ---------- 通知与确认 ---------- */

  /* 通知：可关闭，鼠标悬停或键盘聚焦时暂停自动消失，
     免得正在读的时候它自己淡掉。 */
  var TOAST_MS = 2600;

  function toast(message, kind) {
    var hostEl = qs('#toast-host');
    if (!hostEl) return;
    var timer = null, gone = false;
    var node = el('div', { class: 'toast ' + (kind || ''), role: 'status' }, [
      icon(kind === 'bad' ? 'alert' : (kind === 'ok' ? 'check' : 'info'), 16),
      el('span', { text: message }),
      el('button', { class: 'x', type: 'button', 'aria-label': '关闭这条提示', onclick: dismiss }, icon('close', 13))
    ]);
    function dismiss() {
      if (gone) return;
      gone = true;
      if (timer) { clearTimeout(timer); timer = null; }
      node.style.transition = 'opacity var(--t) var(--ease)';
      node.style.opacity = '0';
      setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 200);
    }
    function hold() { if (timer) { clearTimeout(timer); timer = null; } }
    function release() { if (!gone && !timer) timer = setTimeout(dismiss, TOAST_MS); }
    node.addEventListener('mouseenter', hold);
    node.addEventListener('mouseleave', release);
    node.addEventListener('focusin', hold);
    node.addEventListener('focusout', release);
    hostEl.appendChild(node);
    release();
  }

  function confirmDialog(spec) {
    var dlg = el('dialog', { class: 'dlg' }, el('div', { class: 'dialog' }, [
      el('h2', { text: spec.title }),
      el('p', { text: spec.body }),
      el('div', { class: 'acts' }, [
        el('button', { class: 'btn btn-ghost', type: 'button', onclick: function () { dlg.close(); } }, '取消'),
        el('button', {
          class: 'btn ' + (spec.danger ? 'btn-danger' : 'btn-primary'), type: 'button',
          onclick: function () { dlg.close(); spec.onOk(); }
        }, spec.ok || '确定')
      ])
    ]));
    document.body.appendChild(dlg);
    dlg.addEventListener('close', function () { if (dlg.parentNode) dlg.parentNode.removeChild(dlg); });
    dlg.showModal();
  }

  /* 示例数据标记：非真实数值的地方必须挂上它。 */
  function sampleTag(text) {
    return el('span', { class: 'sample', title: '示例数据，不是真实训练结果' }, [icon('flask', 11), text || '示例']);
  }

  /* ---------- 挂载 ---------- */

  function mount() {
    shellEl = qs('#shell');
    host = qs('#view-host');
    railHost = qs('#rail');
    jobHost = qs('#jobbar');
    crumbsHost = qs('#crumbs');
    inspectHost = qs('#inspector');
    srcTag = qs('#src-tag');
    global.addEventListener('hashchange', render);
    render();
  }

  global.OD.shell = {
    mount: mount, go: go, on: on, emit: emit, render: render,
    toast: toast, confirm: confirmDialog, sampleTag: sampleTag,
    startJob: startJob, endJob: endJob, state: state,
    clipById: clipById, clipIssues: clipIssues, gateOf: gateOf,
    windowSeconds: windowSeconds, windowFrames: windowFrames, timeBase: timeBase,
    allGateItems: allGateItems, stepState: stepState, segmentsOf: segmentsOf
  };
})(window);
