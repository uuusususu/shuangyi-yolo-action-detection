/* 应用外壳：路由、左侧步骤导航、顶部唯一任务条、通知与确认对话框。
   全站的状态只有一个来源 —— 五个步骤的状态点由门禁与产物推导，
   任务条只有一个，停止入口只有一个。 */
(function (global) {
  'use strict';

  var D = global.OD.dom;
  var el = D.el, clear = D.clear, icon = D.icon, qs = D.qs;

  /* 两组步骤：零件模型（YOLO）在前，动作模型（LSTM）在后。模型库是两组之间的交接点：
     零件训完导出进库，动作训练从库里选零件模型。 */
  var GROUPS = [
    { key: 'yolo', name: '零件模型', steps: [
      { key: 'pdata', name: '零件数据', icon: 'db' },
      { key: 'palabel', name: '零件标注', icon: 'box' },
      { key: 'ptrain', name: '零件训练', icon: 'cpu' },
      { key: 'pvalidate', name: '零件验证', icon: 'chart' },
      { key: 'library', name: '模型库', icon: 'box' }
    ] },
    { key: 'action', name: '动作模型', steps: [
      { key: 'model', name: '模型与工位', icon: 'rocket' },
      { key: 'data', name: '数据与标注', icon: 'film' },
      { key: 'train', name: '动作训练', icon: 'cpu' },
      { key: 'runs', name: '结果', icon: 'chart' }
    ] }
  ];
  var STEPS = GROUPS.reduce(function (all, g) { return all.concat(g.steps); }, []);
  var YOLO_STEPS = { pdata: true, palabel: true, ptrain: true, pvalidate: true, library: true };
  /* 说明栏收起与否按用户上次的选择记住。 */
  var INSPECT_KEY = 'od.inspector.collapsed';

  /* source 决定这个页面是"离线示例"还是"连着真实训练台"。
     live 时 projectId 就是接口路径里的工程名；bootError 留着服务没起时的原因。 */
  var state = {
    view: 'home', clipId: '1', job: null, segSel: null, zoom: 4, seekFrame: 0,
    source: 'offline', projectId: null, bootError: ''
  };
  var subs = [];
  var host, railHost, jobHost, crumbsHost, inspectHost, shellEl, srcTag;
  var tickTimer = null, pollTimer = null, jobStart = 0;
  var renderedView = null, renderedHash = '';

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
    if (YOLO_STEPS[key]) return global.OD.yolo ? global.OD.yolo.stepState(key) : 'none';
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
    if (YOLO_STEPS[key]) return global.OD.yolo ? global.OD.yolo.stepSub(key) : '';
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

  function fmtClock(seconds) {
    seconds = Math.max(0, Math.round(seconds));
    var m = Math.floor(seconds / 60), s = seconds % 60;
    if (m >= 60) return Math.floor(m / 60) + ' 时 ' + (m % 60) + ' 分';
    return m ? m + ' 分 ' + s + ' 秒' : s + ' 秒';
  }

  /* 训练页在内容区内显示任务面板，其他页面使用底部任务栏。
     动作训练、抽特征走 state.job；零件训练由零件模块报上来，复用同一套进度与停止入口。 */
  function renderJob() {
    if (!jobHost) return;
    clear(jobHost);
    var inline = state.view === 'train' ? document.getElementById('train-jobbar') : null;
    if (inline) clear(inline);
    var job = state.job;
    var yolo = !job && global.OD.yolo && global.OD.yolo.activeJob ? global.OD.yolo.activeJob() : null;
    if (!job && !yolo && global.OD.parts) yolo = global.OD.parts.activeJob();
    if (!job && !yolo && global.OD.pvalidate) yolo = global.OD.pvalidate.activeJob();
    if (!job && !yolo && global.OD.transfer && global.OD.transfer.activeJob) yolo = global.OD.transfer.activeJob();
    var target = inline || jobHost;
    jobHost.className = 'dock ' + (!inline && (job || yolo) ? 'is-busy' : 'is-idle');
    if (inline) {
      inline.className = 'train-job-panel ' + (job || yolo ? 'is-busy' : 'is-idle');
      inline.parentNode.classList.toggle('is-running', !!(job || yolo));
    }
    if (!job && !yolo) {
      if (inline) inline.appendChild(el('p', { class: 'help',
        text: '当前没有正在运行的任务。开始训练后，这里会显示抽特征和训练的实时进度；训练结果和日志在下方查看。' }));
      return;
    }

    var label, done, total, unit, elapsed, onStop, onOpen;
    if (job) {
      label = job.stale ? '训练台已断开' : job.label;
      done = job.done; total = job.total; unit = job.unit;
      elapsed = (Date.now() - jobStart) / 1000;
      onStop = requestStop;
      onOpen = job.kind === 'train' ? function () { go('train'); } : (job.kind === 'check' ? function () { go('runs'); } : null);
    } else {
      label = (yolo.label || '零件训练') + ' · ' + yolo.name;
      done = yolo.done; total = yolo.total; unit = yolo.unit || '轮';
      elapsed = yolo.elapsed;
      onStop = yolo.stop;
      onOpen = yolo.open || function () { go('ptrain'); };
    }
    var pct = total > 0 ? Math.max(0, Math.min(100, (done / total) * 100)) : 0;
    /* 剩余时间按已用时间线性外推；进度太少时估不准，干脆不写。 */
    var eta = elapsed && done > 0 && total > done && pct >= 3 ? elapsed * (total - done) / done : null;

    target.appendChild(el('div', { class: 'dock-row' }, [
      el('span', { class: 'dock-spin', 'aria-hidden': 'true' }),
      el('div', { class: 'dock-main' }, [
        el('div', { class: 'dock-head' }, [
          el('strong', { text: label }),
          el('span', { class: 'dock-num', text: done + ' / ' + (total || '—') + ' ' + unit }),
          el('span', { class: 'dock-pct', text: Math.floor(pct) + '%' })
        ]),
        el('div', { class: 'dock-track', role: 'progressbar', 'aria-valuenow': Math.round(pct), 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': label },
          el('div', { class: 'dock-fill', style: 'width:' + pct.toFixed(1) + '%' }))
      ]),
      el('div', { class: 'dock-meta' }, [
        el('span', null, ['已用 ', el('b', { text: elapsed ? fmtClock(elapsed) : '—' })]),
        el('span', null, ['剩余 ', el('b', { text: eta === null ? '估算中' : '约 ' + fmtClock(eta) })])
      ]),
      el('div', { class: 'dock-acts' }, [
        onOpen ? el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: onOpen }, '查看详情') : null,
        el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: onStop }, [icon('stop', 13), '停止'])
      ])
    ]));
  }

  /* ---------- 顶部与导航 ---------- */

  var VIEW_TITLE = {
    home: '全部工程', model: '模型与工位', data: '数据与标注', features: '特征',
    train: '动作训练', runs: '结果', annotate: '标注工作台',
    pdata: '零件数据', palabel: '零件标注', ptrain: '零件训练', pvalidate: '零件验证', library: '模型库'
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
    if (state.view === 'home') {
      crumbsHost.appendChild(el('span', { class: 'here', text: '全部工程' }));
      return;
    }
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

  var partsPrefetchedProject = null;
  function renderRail() {
    if (!railHost) return;
    clear(railHost);
    if (global.OD.yolo) global.OD.yolo.prefetch();
    if (global.OD.parts && state.projectId && partsPrefetchedProject !== state.projectId) {
      partsPrefetchedProject = state.projectId;
      global.OD.parts.prefetch().then(function () {
      renderJob();
      if (state.view === 'pdata' || state.view === 'palabel') render();
      });
    }
    /* 当前工程：点一下回到全部工程，在那里切换或新建。 */
    railHost.appendChild(el('button', {
      class: 'project-pick', type: 'button', title: '切换工程',
      'aria-current': state.view === 'home' ? 'true' : 'false',
      onclick: function () { go('home'); }
    }, [
      el('span', { class: 'project-avatar', text: (global.OD.data.project.id || '?').slice(0, 1).toUpperCase() }),
      el('span', { class: 'project-text' }, [
        el('span', { class: 'project-label', text: '当前工程' }),
        el('span', { class: 'project-name', text: global.OD.data.project.id })
      ]),
      icon('chev-right', 14)
    ]));

    GROUPS.forEach(function (group) {
      railHost.appendChild(el('div', { class: 'rail-title rail-group', 'data-mod': group.key, text: group.name }));
      group.steps.forEach(function (step) {
        var st = stepState(step.key);
        var current = state.view === step.key || (step.key === 'data' && state.view === 'annotate');
        var sub = stepSub(step.key);
        railHost.appendChild(el('button', {
          class: 'step', type: 'button', 'data-state': st || 'none',
          'aria-current': current ? 'true' : 'false', title: step.name + (sub ? '：' + sub : ''),
          onclick: function () { go(step.key); }
        }, [
          el('span', { class: 'idx' }, icon(step.icon, 16)),
          el('span', { class: 'name', text: step.name }),
          el('span', { class: 'sr', text: '状态：' + stateLabel(st) }),
          st === 'bad' || st === 'warn' || st === 'busy' ? el('span', { class: 'state-dot ' + st, 'aria-hidden': 'true' }) : null
        ]));
      });
    });

    /* 视频列表：直接点进标注。和 HUB 侧栏里的数据集清单一样，只放名字和数量。 */
    var clips = global.OD.data.clips;
    railHost.appendChild(el('div', { class: 'rail-title', text: '视频 · ' + clips.length }));
    var list = el('div', { class: 'clip-nav' });
    clips.forEach(function (clip) {
      list.appendChild(el('button', {
        class: 'clip-link', type: 'button',
        'aria-current': state.view === 'annotate' && state.clipId === clip.id ? 'true' : 'false',
        onclick: function () { go('annotate', { clipId: clip.id }); }
      }, [
        el('span', { class: 'clip-name', text: clip.file }),
        el('sup', { class: clip.segments.length ? '' : 'is-empty', text: String(clip.segments.length) })
      ]));
    });
    railHost.appendChild(list);
  }

  function stateLabel(st) {
    return { ok: '通过', warn: '有警告', bad: '未通过', busy: '进行中' }[st] || '未开始';
  }

  /* ---------- 路由 ---------- */

  function go(view, params) {
    if (view !== renderedView && !canLeaveView()) return;
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
    if (renderedView && route.view !== renderedView && !canLeaveView()) {
      state.view = renderedView;
      if (global.location.hash !== renderedHash) global.location.replace(renderedHash);
      return;
    }
    state.view = route.view;
    if (route.clipId) state.clipId = route.clipId;

    renderTop();
    renderRail();

    var view = global.OD.views[state.view] || global.OD.views.home;
    clear(host);
    var node = view.render();
    if (node) host.appendChild(node);
    renderedView = state.view;
    renderedHash = global.location.hash;
    renderJob();

    clear(inspectHost);
    var side = view.inspector ? view.inspector() : null;
    var collapsed = inspectorCollapsed();
    shellEl.classList.toggle('has-inspect', !!side && !collapsed);
    if (side) inspectHost.appendChild(side);
    renderInspectToggle(!!side, collapsed);
  }

  function canLeaveView() {
    var previous = renderedView && global.OD.views[renderedView];
    return !previous || !previous.beforeLeave || previous.beforeLeave() !== false;
  }

  /* ---------- 说明栏收起 ---------- */

  function inspectorCollapsed() {
    try { return global.localStorage.getItem(INSPECT_KEY) === '1'; } catch (e) { return false; }
  }

  function setInspectorCollapsed(value) {
    try { global.localStorage.setItem(INSPECT_KEY, value ? '1' : '0'); } catch (e) { /* 隐私模式下存不了，就只在本次生效 */ }
    render();
  }

  /* 顶栏右侧的开关：工作区要宽的时候（标注、曲线）把说明栏收起来。没有说明栏的页面不显示。 */
  function renderInspectToggle(hasSide, collapsed) {
    var btn = qs('#inspect-toggle');
    if (!btn) return;
    btn.hidden = !hasSide;
    btn.setAttribute('aria-pressed', collapsed ? 'false' : 'true');
    btn.title = collapsed ? '显示说明栏' : '收起说明栏';
    btn.textContent = collapsed ? '说明 ›' : '‹ 收起说明';
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

  /* 高清图表灯箱：支持上一张/下一张、缩放与原图下载。 */
  function lightboxDialog(items, startIndex) {
    if (!items || !items.length) return;
    var index = Math.max(0, Math.min(items.length - 1, startIndex || 0));
    var scale = 1;
    var titleEl = el('strong', { class: 'lb-title' });
    var countEl = el('span', { class: 'mono sp-fs-12 sp-dim' });
    var zoomEl = el('span', { class: 'mono sp-fs-12', text: '100%' });
    var img = el('img', { class: 'lb-img', alt: '' });
    var dl = el('a', { class: 'btn btn-ghost btn-sm', download: '', text: '下载原图' });
    var prevBtn = el('button', { class: 'btn btn-secondary btn-sm', type: 'button', onclick: function () { step(-1); } }, '‹ 上一张');
    var nextBtn = el('button', { class: 'btn btn-secondary btn-sm', type: 'button', onclick: function () { step(1); } }, '下一张 ›');
    var viewport = el('div', { class: 'lb-viewport', onwheel: function (e) {
      e.preventDefault();
      setScale(e.deltaY < 0 ? scale * 1.2 : scale / 1.2);
    } }, img);

    function setScale(next) {
      scale = Math.max(0.25, Math.min(5, next));
      img.style.transform = 'scale(' + scale.toFixed(3) + ')';
      zoomEl.textContent = Math.round(scale * 100) + '%';
    }
    function sync() {
      var cur = items[index];
      scale = 1;
      img.style.transform = 'scale(1)';
      zoomEl.textContent = '100%';
      titleEl.textContent = cur.label || cur.name || '图表预览';
      countEl.textContent = (index + 1) + ' / ' + items.length;
      img.src = cur.url;
      dl.href = cur.url;
      dl.setAttribute('download', cur.name || 'chart.png');
      prevBtn.disabled = items.length <= 1;
      nextBtn.disabled = items.length <= 1;
    }
    function step(delta) {
      index = (index + delta + items.length) % items.length;
      sync();
    }
    var dlg = el('dialog', { class: 'dlg lb-dlg', onclick: function (e) { if (e.target === dlg) dlg.close(); } }, el('div', { class: 'lb-shell' }, [
      el('header', { class: 'lb-head' }, [
        titleEl,
        countEl,
        el('div', { class: 'lb-tools' }, [
          items.length > 1 ? prevBtn : null,
          items.length > 1 ? nextBtn : null,
          el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { setScale(scale / 1.25); } }, '−'),
          zoomEl,
          el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { setScale(scale * 1.25); } }, '＋'),
          el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { setScale(1); } }, '1:1'),
          dl,
          el('button', { class: 'btn btn-secondary btn-sm', type: 'button', onclick: function () { dlg.close(); } }, '关闭')
        ])
      ]),
      viewport
    ]));
    dlg.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    });
    dlg.addEventListener('close', function () { if (dlg.parentNode) dlg.parentNode.removeChild(dlg); });
    document.body.appendChild(dlg);
    sync();
    dlg.showModal();
  }

  var serverInfoCache = null;
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { toast('已复制网址：' + text, 'ok'); })
        .catch(function () { fallbackCopy(text); });
    } else fallbackCopy(text);
  }
  function fallbackCopy(text) {
    var ta = el('textarea', { value: text, style: 'position:fixed;left:-9999px' });
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('已复制网址：' + text, 'ok'); } catch (e) { toast('请手动选中复制：' + text, 'info'); }
    document.body.removeChild(ta);
  }

  var unlockDialogOpen = false;
  function promptUnlock() {
    if (unlockDialogOpen || !global.OD.api || !global.OD.api.authUnlock) return;
    unlockDialogOpen = true;
    var errNode = el('p', { class: 'help bad sp-mt-2', text: '' });
    var pinInput = el('input', {
      type: 'password', class: 'form-control mono', placeholder: '请输入本机设置的内网访问口令（PIN）',
      autocomplete: 'current-password'
    });
    var submitBtn = el('button', { class: 'btn btn-primary', type: 'submit', text: '解锁进入' });
    var form = el('form', {
      class: 'dialog', style: 'max-width:440px',
      onsubmit: function (e) {
        e.preventDefault();
        var pin = pinInput.value.trim();
        if (!pin) { errNode.textContent = '请输入访问口令'; return; }
        submitBtn.disabled = true;
        errNode.textContent = '';
        global.OD.api.authUnlock(pin).then(function () {
          dlg.close();
          toast('内网口令校验通过', 'ok');
          loadServerInfo().then(function () {
            if (global.OD.live && global.OD.live.boot) {
              global.OD.live.boot().then(function () { render(); });
            } else {
              render();
            }
          });
        }).catch(function (err) {
          errNode.textContent = err.message || '口令不正确';
          submitBtn.disabled = false;
          pinInput.focus();
        });
      }
    }, [
      el('h2', { text: '内网访问口令验证' }),
      el('p', { text: '训练台主机已启用内网访问保护，请输入主机设置的 PIN 口令继续访问（本机 127.0.0.1 免密）。' }),
      el('div', { class: 'sp-mt-3' }, pinInput),
      errNode,
      el('div', { class: 'acts sp-mt-3' }, [
        el('button', { class: 'btn btn-ghost', type: 'button', onclick: function () { dlg.close(); } }, '暂不解锁'),
        submitBtn
      ])
    ]);
    var dlg = el('dialog', { class: 'dlg' }, form);
    dlg.addEventListener('close', function () {
      unlockDialogOpen = false;
      if (dlg.parentNode) dlg.parentNode.removeChild(dlg);
    });
    document.body.appendChild(dlg);
    dlg.showModal();
    pinInput.focus();
  }

  function randomPin() {
    var out = '';
    for (var i = 0; i < 6; i++) out += String(Math.floor(Math.random() * 10));
    return out;
  }

  function showLanDialog() {
    var info = serverInfoCache || {};
    var urls = info.lan_urls || [];
    var isLocal = info.is_local_client !== false;
    var pinInput = el('input', {
      type: 'text', class: 'form-control mono',
      placeholder: info.pin_enabled ? '输入新口令可更换（4~32 位）' : '留空为免密；输入 4~32 位开启口令保护'
    });
    var roCheck = el('input', {
      type: 'checkbox', checked: !!info.readonly,
      onchange: function () {
        global.OD.api.authConfig({ readonly: roCheck.checked }).then(function () {
          toast(roCheck.checked ? '已开启内网只读模式：远端电脑仅可浏览与下载' : '已关闭内网只读模式：远端电脑可上传与操作', 'ok');
          loadServerInfo();
        }).catch(function (err) {
          roCheck.checked = !roCheck.checked;
          toast('保存失败：' + err.message, 'bad');
        });
      }
    });

    var securityBox = isLocal ? el('div', { class: 'lan-security-box sp-mt-3' }, [
      el('div', { class: 'lan-sec-head' }, [
        el('strong', { text: '内网安全保护（仅本机可配置）' }),
        el('span', { class: 'badge ' + (info.pin_enabled ? 'ok' : 'plain'), text: info.pin_enabled ? '已开启 PIN 口令' : '未设口令' })
      ]),
      el('div', { class: 'lan-pin-row sp-mt-2' }, [
        pinInput,
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { pinInput.value = randomPin(); } }, '随机 6 位'),
        el('button', {
          class: 'btn btn-primary btn-sm', type: 'button',
          onclick: function () {
            var val = pinInput.value.trim();
            if (!val) { toast('先输入或随机生成 4~32 位口令', 'warn'); return; }
            global.OD.api.authConfig({ pin: val }).then(function () {
              toast('已启用内网访问口令（PIN）', 'ok');
              dlg.close();
              loadServerInfo().then(showLanDialog);
            }).catch(function (err) { toast('设置失败：' + err.message, 'bad'); });
          }
        }, '启用/更换口令'),
        info.pin_enabled ? el('button', {
          class: 'btn btn-danger btn-sm', type: 'button',
          onclick: function () {
            global.OD.api.authConfig({ clear_pin: true }).then(function () {
              toast('已清除内网访问口令', 'warn');
              dlg.close();
              loadServerInfo().then(showLanDialog);
            }).catch(function (err) { toast('清除失败：' + err.message, 'bad'); });
          }
        }, '关闭口令') : null
      ]),
      el('label', { class: 'parts-inline sp-mt-2' }, [
        roCheck,
        '开启内网只读模式（远端电脑仅查看页面/图表与下载 ZIP，禁止修改数据、删除或启停训练）'
      ])
    ]) : el('div', { class: 'lan-security-box sp-mt-3' }, [
      el('div', { class: 'lan-sec-head' }, [
        el('strong', { text: '当前内网连接状态' }),
        el('span', { class: 'badge ' + (info.readonly ? 'warn' : 'ok'), text: info.readonly ? '只读模式' : '完整读写' })
      ]),
      el('p', { class: 'help sp-mt-2', text: '安全策略仅支持在运行训练台的本机（127.0.0.1）修改。' }),
      info.pin_enabled ? el('div', { class: 'row sp-mt-2' }, [
        el('button', {
          class: 'btn btn-ghost btn-sm', type: 'button',
          onclick: function () {
            global.OD.api.authLock().then(function () {
              dlg.close();
              toast('已退出当前内网授权会话', 'info');
              loadServerInfo().then(promptUnlock);
            });
          }
        }, '锁定并退出授权')
      ]) : null
    ]);

    var dlg = el('dialog', { class: 'dlg', onclick: function (e) { if (e.target === dlg) dlg.close(); } }, el('div', { class: 'dialog', style: 'max-width:580px' }, [
      el('h2', { text: '内网访问与跨机数据推送' }),
      el('p', { text: info.lan_enabled
        ? '训练台已开启内网监听。将采集电脑通过网线直连或接入同一局域网后，在浏览器打开下方任一网址，即可直接拖拽上传图片、视频或数据集 ZIP 压缩包：'
        : '当前运行在仅本机回环模式（127.0.0.1）。使用默认启动脚本（不带 --local）即可开启内网访问。' }),
      urls.length ? el('div', { class: 'lan-url-list sp-mt-3' }, urls.map(function (u) {
        return el('div', { class: 'lan-url-row' }, [
          el('code', { class: 'mono', text: u }),
          el('button', { class: 'btn btn-primary btn-sm', type: 'button', onclick: function () { copyText(u); } }, '复制网址')
        ]);
      })) : (info.lan_enabled ? el('div', { class: 'callout warn sp-mt-3' }, [
        icon('info', 15),
        el('div', { text: '暂未检测到活动局域网网卡 IP。插上网线或连接路由器后点击“刷新检测”即可看到内网 IP。' })
      ]) : null),
      securityBox,
      el('div', { class: 'callout info sp-mt-3' }, [
        icon('info', 15),
        el('div', { text: '若另一台电脑打不开网址，请在运行训练台的电脑上双击运行「放行内网端口.bat」（放行 Windows 防火墙 TCP ' + (info.port || 8771) + ' 入站端口）。' })
      ]),
      el('div', { class: 'acts sp-mt-3' }, [
        el('button', { class: 'btn btn-ghost', type: 'button', onclick: function () { dlg.close(); loadServerInfo(true).then(showLanDialog); } }, '刷新检测'),
        el('button', { class: 'btn btn-primary', type: 'button', onclick: function () { dlg.close(); } }, '知道了')
      ])
    ]));
    dlg.addEventListener('close', function () { if (dlg.parentNode) dlg.parentNode.removeChild(dlg); });
    document.body.appendChild(dlg);
    dlg.showModal();
  }

  function loadServerInfo() {
    var btn = qs('#lan-btn');
    if (!btn || !global.OD.api || !global.OD.api.available || !global.OD.api.serverInfo) return Promise.resolve(null);
    return global.OD.api.serverInfo().then(function (info) {
      serverInfoCache = info;
      btn.hidden = false;
      var urls = (info && info.lan_urls) || [];
      var tags = [];
      if (info && info.pin_enabled) tags.push('PIN');
      if (info && info.readonly) tags.push('只读');
      var suffix = tags.length ? (' · ' + tags.join('/')) : '';
      if (info && info.lan_enabled && urls.length) {
        var shortHost = urls[0].replace(/^https?:\/\//, '').replace(/\/$/, '');
        btn.textContent = '内网 · ' + shortHost + suffix;
        btn.title = '点击查看并复制内网访问网址与安全设置';
      } else if (info && info.lan_enabled) {
        btn.textContent = '内网已开' + suffix;
      } else {
        btn.textContent = '仅本机';
      }
      if (info && info.auth_required) promptUnlock();
      return info;
    }).catch(function () { return null; });
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
    var toggle = qs('#inspect-toggle');
    if (toggle) toggle.addEventListener('click', function () { setInspectorCollapsed(!inspectorCollapsed()); });
    var lanBtn = qs('#lan-btn');
    if (lanBtn) lanBtn.addEventListener('click', showLanDialog);
    loadServerInfo();
    global.addEventListener('hashchange', render);
    /* 已用时间每秒走一格；没有任务时什么也不画。 */
    setInterval(function () { if (state.job) renderJob(); }, 1000);
    render();
  }

  /* 切到另一个工程时调用。上一个工程的进度轮询还在问旧工程，选中的片子 id、
     选中的段在新工程里不存在，留着会让标注页指错片子、把旧工程的进度显示到新工程上。 */
  function resetForProject() {
    stopJobSilently();
    renderJob();
    var clips = (global.OD.data && global.OD.data.clips) || [];
    state.clipId = clips.length ? clips[0].id : '';
    state.segSel = null;
    state.seekFrame = 0;
    state.paintClass = null;
    if (global.OD.yolo) global.OD.yolo.reset();
    if (global.OD.parts) global.OD.parts.reset();
    if (global.OD.pvalidate) global.OD.pvalidate.reset();
    if (global.OD.transfer && global.OD.transfer.reset) global.OD.transfer.reset();
  }

  global.OD.shell = {
    resetForProject: resetForProject, renderRail: renderRail, renderJob: renderJob,
    mount: mount, go: go, on: on, emit: emit, render: render,
    toast: toast, confirm: confirmDialog, lightbox: lightboxDialog, showLan: showLanDialog, promptUnlock: promptUnlock,
    serverInfo: function () { return serverInfoCache; }, sampleTag: sampleTag,
    startJob: startJob, endJob: endJob, state: state,
    clipById: clipById, clipIssues: clipIssues, gateOf: gateOf,
    windowSeconds: windowSeconds, windowFrames: windowFrames, timeBase: timeBase,
    allGateItems: allGateItems, stepState: stepState, segmentsOf: segmentsOf
  };
})(window);
