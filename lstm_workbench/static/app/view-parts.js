/* 零件素材、审核发布、专业 OBB 标注画布与 Meta SAM 3 / SAM 2.1 / YOLOE 智能自动标注。
   草稿先留在本机，再按素材修订号串行保存；训练进行中彻底禁用所有智能标注以保障 ≥8GB 显存。 */
(function (global) {
  'use strict';
  var D = global.OD.dom, el = D.el, S = global.OD.shell, A = global.OD.api;
  var state = {
    pid: null, data: null, loading: false, error: '', selected: null,
    filter: '', status: '', group: '', category: '',
    splits: {}, checked: {}, quality: null, models: [], runs: [],
    draft: null, saving: false, saveAgain: false, timer: null, editor: null, refs: {},
    job: null, jobTimer: null,
    /* 新增：画布模式、页内素材接入、Meta SAM 智能状态与三合一自动标注 */
    toolMode: 'select', fitMode: 'obb',
    mediaClips: [], showIngest: false,
    smart: null, smartLoading: false, smartTried: false, smartPanelOpen: false,
    smartSelectedModel: '', smartUploading: '',
    smartPollTimer: null, smartEncodedId: '',
    autoTab: 'zeroshot',
    zsEngine: 'yoloe_vp', zsModel: '', zsConf: '0.25', zsPrompts: {}, zsExemplar: null,
    trackModel: '', trackMaxFrames: '120',
    customUrlModel: 'sam3.pt', customUrl: ''
  };
  var STATUS = { unlabeled: '未标注', pending: '待审核', reviewed: '已审核', negative: '确认无目标' };
  var generation = 0;

  function context() { return { project: pid(), generation: generation }; }
  function currentContext(c) { return c.project === pid() && c.project === state.pid && c.generation === generation; }
  function action(c, request, success, prefix) {
    return Promise.resolve().then(function () { if (currentContext(c)) return request(); })
      .then(function (result) { if (currentContext(c) && result !== undefined) return success(result); })
      .catch(function (e) { if (currentContext(c)) S.toast((prefix || '') + e.message, 'bad'); });
  }
  function refreshRelated(c) {
    if (!currentContext(c)) return Promise.resolve();
    if (global.OD.pvalidate && global.OD.pvalidate.invalidate) global.OD.pvalidate.invalidate(c.project);
    if (global.OD.yolo && global.OD.yolo.refresh) return global.OD.yolo.refresh(c.project);
    return Promise.resolve();
  }
  function pid() { return S.state.projectId; }
  function live() { return global.OD.live.isLive() && !!pid(); }
  function clone(v) { return JSON.parse(JSON.stringify(v)); }
  function norm(r) { return r.parts || r.snapshot || r; }
  function btn(text, fn, kind, attrs) {
    return el('button', Object.assign({ type: 'button', class: 'btn btn-sm ' + (kind || 'btn-secondary'), onclick: fn }, attrs || {}), text);
  }
  function page(title, lead, body) {
    return el('div', { class: 'view is-on parts-view' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [el('div', { class: 'mod-tag', 'data-mod': 'yolo', text: '零件' }), el('h1', { text: title }), el('p', { text: lead })])
    ].concat(body)));
  }
  function current() { return state.data && (state.data.items || []).find(function (x) { return x.id === state.selected; }); }
  function draftKey() { return 'parts-draft:' + state.pid + ':' + state.selected; }
  function remember() { if (!state.draft) return; try { localStorage.setItem(draftKey(), JSON.stringify(state.draft)); } catch (e) { /* 存储不可用时继续依赖后端保存 */ } }
  function forgetDraft() { try { localStorage.removeItem(draftKey()); } catch (e) { /* 私密浏览 */ } }
  function dirty() {
    var item = current();
    return !!(item && state.draft && (
      JSON.stringify(item.boxes) !== JSON.stringify(state.draft.boxes) ||
      item.status !== state.draft.status ||
      item.source_group !== state.draft.source_group
    ));
  }
  function setDraft(item) {
    state.draft = item ? { boxes: clone(item.boxes || []), status: item.status, source_group: item.source_group, revision: item.revision } : null;
    if (!item) return;
    try {
      var stored = JSON.parse(localStorage.getItem(draftKey()) || 'null');
      if (stored) {
        state.draft = stored;
        if (stored.revision !== item.revision) {
          state.refs.conflict = true;
          S.toast('发现旧修订的本机草稿，已保留。请核对后重新载入或另行保存。', 'warn');
        } else {
          S.toast('已恢复这张素材尚未保存的草稿', 'warn');
        }
      }
    } catch (e) { /* 损坏草稿不影响磁盘标注 */ }
  }
  function fmtBytes(n) {
    if (!n || n <= 0) return '';
    if (n >= 1073741824) return (n / 1073741824).toFixed(2) + ' GB';
    if (n >= 1048576) return (n / 1048576).toFixed(1) + ' MB';
    return Math.round(n / 1024) + ' KB';
  }
  function copyText(text) {
    try {
      if (global.navigator && global.navigator.clipboard && global.navigator.clipboard.writeText) {
        global.navigator.clipboard.writeText(text).then(function () { S.toast('已复制安装命令到剪贴板', 'ok'); });
        return;
      }
    } catch (e) { /* 降级提示 */ }
    S.toast('请手动复制命令：' + text, 'warn');
  }

  function load(force) {
    if (!live()) return Promise.resolve();
    if (state.pid !== pid()) { reset(); state.pid = pid(); }
    if (state.loading) return state.loading;
    var id = pid(), task = context();
    state.error = '';
    var clipsReq = A.partsMediaClips ? A.partsMediaClips(id).catch(function () { return { clips: [] }; }) : Promise.resolve({ clips: [] });
    state.loading = Promise.all([A.parts(id), A.yolo(id), A.models(), A.yoloValidations(id), clipsReq]).then(function (out) {
      if (!currentContext(task)) return;
      state.data = norm(out[0]);
      state.runs = out[1].runs || [];
      state.models = ((out[2].yolo || {}).files || []).map(function (m) { return m.name; });
      state.mediaClips = (out[4] && out[4].clips) || [];
      state.error = '';
      var running = (out[3].evaluations || []).find(function (j) { return j.status === 'running'; });
      if (running) { state.job = running; watchJob(running.id); }
      if (!state.selected && state.data.items.length) state.selected = state.data.items[0].id;
      assignSplits();
      if (!state.draft) setDraft(current());
    }).catch(function (err) {
      if (currentContext(task)) state.error = err.message;
    }).finally(function () {
      if (currentContext(task)) state.loading = false;
    });
    return state.loading;
  }

  function ensure() {
    if (!live()) return false;
    if (state.pid === pid() && state.data) return true;
    if (state.pid === pid() && state.error) return false;
    if (!state.loading) load().then(function () { if (['pdata', 'palabel'].indexOf(S.state.view) >= 0) S.render(); });
    return false;
  }
  function loadingBody() {
    return [el('p', { class: 'help', text: state.error || '正在读取…' }), state.error ? btn('重试读取', function () { load(true).then(function () { S.render(); }); }, 'btn-primary') : null];
  }
  function assignSplits() {
    var groups = Array.from(new Set((state.data.items || []).map(function (i) { return i.source_group; }))).sort(), next = {};
    groups.forEach(function (g, i) { next[g] = state.splits[g] || (i === 1 ? 'val' : 'train'); });
    state.splits = next;
    if (state.group && groups.indexOf(state.group) < 0) state.group = '';
  }
  function statusHint(text, bad) {
    var hint = state.refs.saveHint;
    if (hint) { hint.textContent = text; hint.className = 'parts-save-hint ' + (bad ? 'text-danger' : ''); }
  }
  function change(boxes) {
    if (!state.draft) return;
    state.draft.boxes = clone(boxes);
    state.draft.status = 'pending';
    remember();
    statusHint('有改动，稍后自动保存');
    schedule();
    paintBoxList();
  }
  function schedule() { clearTimeout(state.timer); state.timer = setTimeout(function () { save().catch(function () { }); }, 900); }
  function save(status) {
    var callTask = context();
    clearTimeout(state.timer);
    if (status && state.draft) state.draft.status = status;
    if (!state.draft || !current()) return Promise.resolve();
    remember();
    if (state.saving) {
      state.saveAgain = true;
      return state.saving.then(function () { if (currentContext(callTask)) return save(status); });
    }
    if (!dirty()) return Promise.resolve();
    var project = pid(), task = context(), id = state.selected, item = current(), payload = clone(state.draft), serial = JSON.stringify(state.draft);
    statusHint('正在保存…');
    state.saving = A.partsSave(project, id, payload).then(function (raw) {
      if (!currentContext(task) || state.selected !== id) return;
      var result = raw.item || raw, ix = state.data.items.findIndex(function (x) { return x.id === id; });
      state.data.items[ix] = Object.assign({}, item, result, { boxes: result.boxes || payload.boxes });
      if (raw.item && Number.isInteger(raw.revision)) state.data.revision = raw.revision;
      if (JSON.stringify(state.draft) === serial) {
        state.draft = { boxes: clone(state.data.items[ix].boxes), status: state.data.items[ix].status, source_group: state.data.items[ix].source_group, revision: state.data.items[ix].revision };
        forgetDraft();
      } else {
        state.draft.revision = state.data.items[ix].revision; remember(); state.saveAgain = true;
      }
      statusHint(state.saveAgain ? '还有新改动，继续保存…' : '已保存 · ' + (STATUS[state.data.items[ix].status] || state.data.items[ix].status));
      state.refs.conflict = false;
      paintItemRows();
    }).catch(function (err) {
      if (!currentContext(task)) return;
      state.saveAgain = false;
      statusHint('保存失败：' + err.message + '。草稿已保留。', true);
      S.toast('零件标注保存失败：' + err.message, 'bad');
      throw err;
    }).finally(function () {
      if (currentContext(task)) { state.saving = false; if (state.saveAgain) { state.saveAgain = false; schedule(); } }
    });
    return state.saving;
  }

  function select(id) {
    if (id === state.selected) return;
    var task = context();
    save().then(function () {
      if (!currentContext(task) || dirty()) return;
      state.selected = id;
      state.refs.conflict = false;
      state.smartEncodedId = '';
      setDraft(current());
      S.render();
      preEncodeCurrent();
    }).catch(function () { });
  }
  function refresh() {
    var task = context();
    save().then(function () {
      if (!currentContext(task) || dirty()) return;
      state.draft = null;
      return load(true);
    }).then(function () { if (currentContext(task)) S.render(); }).catch(function () { });
  }
  function applySnapshot(result) {
    var snapshot = norm(result), wasDirty = dirty();
    if (state.data && Number.isInteger(snapshot.revision) && snapshot.revision < state.data.revision) return;
    state.data = snapshot;
    state.quality = null;
    assignSplits();
    if (!current() && state.data.items.length) state.selected = state.data.items[0].id;
    if (!wasDirty) setDraft(current());
    S.render();
  }
  function formField(label, node) { return el('label', { class: 'parts-field' }, [el('span', { text: label }), node]); }
  function input(value, attrs) { return el('input', Object.assign({ class: 'form-control', value: value || '' }, attrs || {})); }
  function selectBox(options, value, onchange, label) {
    var s = el('select', { class: 'form-select', 'aria-label': label || '', onchange: onchange }, options.map(function (o) {
      return el('option', { value: o[0], text: o[1] });
    }));
    s.value = value;
    return s;
  }

  /* ---------- 智能标注（Meta SAM 文件夹模型选择、加载与停止控制） ---------- */

  var SMART_PREF_KEY = 'od_smart_preferred_model';
  function getPreferredSmartModel() {
    try { return (typeof localStorage !== 'undefined' && localStorage.getItem(SMART_PREF_KEY)) || ''; } catch (e) { return ''; }
  }
  function setPreferredSmartModel(name) {
    try {
      if (typeof localStorage === 'undefined') return;
      if (name) localStorage.setItem(SMART_PREF_KEY, name);
      else localStorage.removeItem(SMART_PREF_KEY);
    } catch (e) { }
  }
  function localSmartModels(info) {
    return ((info && info.available_models) || []).filter(function (m) {
      return m.kind === 'sam_family' && m.local;
    });
  }
  function defaultSlotForModel(m) {
    if (!m) return 'interactive';
    var caps = m.capabilities || [];
    if (caps.indexOf('interactive') >= 0) return 'interactive';
    if (m.family === 'yoloe') return 'yoloe_vp';
    return 'auto';
  }

  function smartReadyForInteractive() {
    return !!(state.smart && !state.smart.training_locked && state.smart.state === 'ready' && state.smart.active_slot === 'interactive');
  }

  function applySmartStatus(info, openOnUnavailable) {
    if (!info) return;
    state.smart = info;
    var locals = localSmartModels(info);
    if (info.active_model) {
      state.smartSelectedModel = info.active_model;
    } else if (!state.smartSelectedModel || !locals.some(function (m) { return m.name === state.smartSelectedModel; })) {
      var pref = getPreferredSmartModel();
      var prefHit = pref && locals.some(function (m) { return m.name === pref; });
      state.smartSelectedModel = prefHit ? pref : (locals.length ? locals[0].name : '');
    }
    if (info.training_locked) {
      state.smartPanelOpen = false;
      if (state.toolMode === 'smart' || state.toolMode === 'exemplar') state.toolMode = 'select';
    } else if (openOnUnavailable && info.show_model_panel) {
      state.smartPanelOpen = true;
    }
    var hasDl = info.downloads && Object.keys(info.downloads).some(function (k) { return info.downloads[k] && info.downloads[k].status === 'downloading'; });
    var hasBatch = info.batch_job && ['running', 'queued'].indexOf(info.batch_job.status) >= 0;
    if (hasDl || hasBatch) scheduleSmartPoll();
  }

  function scheduleSmartPoll() {
    clearTimeout(state.smartPollTimer);
    if (!A.smartStatus || !live()) return;
    var task = context();
    state.smartPollTimer = setTimeout(function () {
      if (!currentContext(task) || S.state.view !== 'palabel') return;
      var prevBatch = state.smart && state.smart.batch_job ? state.smart.batch_job.status : '';
      A.smartStatus(task.project, false).then(function (info) {
        if (!currentContext(task)) return;
        applySmartStatus(info, false);
        var curBatch = info.batch_job ? info.batch_job.status : '';
        if ((prevBatch === 'running' || prevBatch === 'queued') && ['done', 'failed', 'stopped'].indexOf(curBatch) >= 0) {
          var j = info.batch_job;
          var msg = curBatch === 'done'
            ? ('智能批量标注完成：写入 ' + (j.applied || 0) + ' 张待审核，跳过 ' + (j.skipped || 0) + ' 张')
            : ('智能批量标注结束：' + (j.error || curBatch));
          S.toast(msg, curBatch === 'done' ? 'ok' : 'warn');
          if (!dirty()) { state.draft = null; load(true).then(function () { if (currentContext(task) && S.state.view === 'palabel') S.render(); }); }
          else S.render();
          return;
        }
        if (S.state.view === 'palabel') S.render();
      }).catch(function () { });
    }, 1300);
  }

  function autoInitSmart() {
    if (!A.smartStatus || !A.smartActivate || !live() || state.smartTried || state.smartLoading) return;
    var task = context();
    state.smartTried = true;
    state.smartLoading = true;
    A.smartStatus(task.project, false).then(function (info) {
      if (!currentContext(task)) return;
      applySmartStatus(info, true);
      if (info.training_locked) { S.render(); return; }
      if (info.state === 'ready') {
        S.render();
        if (info.active_slot === 'interactive') preEncodeCurrent();
        return;
      }
      // 若用户上次选过的模型就在本地 models/sam 文件夹中则优先拉起；否则默认尝试 sam3.pt
      var locals = localSmartModels(info);
      var pref = getPreferredSmartModel();
      var chosen = pref ? locals.filter(function (m) { return m.name === pref; })[0] : null;
      var targetModel = chosen ? chosen.name : 'sam3.pt';
      var targetSlot = chosen ? defaultSlotForModel(chosen) : 'interactive';
      return A.smartActivate(task.project, { model: targetModel, slot: targetSlot, auto_default: true }).then(function (res) {
        if (!currentContext(task)) return;
        applySmartStatus(res, true);
        if (res.state === 'ready') {
          S.toast('已自动加载 ' + (res.active_model || targetModel) + (res.active_slot === 'interactive' ? '（按 S 进入智能点选）' : ''), 'ok');
          if (res.active_slot === 'interactive') preEncodeCurrent();
        }
        S.render();
      });
    }).catch(function (err) {
      if (!currentContext(task)) return;
      state.smart = Object.assign({}, state.smart || {}, {
        state: 'unavailable',
        show_model_panel: true,
        reason: err.message || '无法连接智能标注服务'
      });
      state.smartPanelOpen = true;
      S.render();
    }).finally(function () {
      if (currentContext(task)) state.smartLoading = false;
    });
  }

  function activateSmartModel(modelName, slot) {
    if (!A.smartActivate || !live() || !modelName) return;
    var task = context();
    state.smartLoading = true;
    state.smartSelectedModel = modelName;
    S.render();
    A.smartActivate(task.project, { model: modelName, slot: slot || 'auto' }).then(function (res) {
      if (!currentContext(task)) return;
      applySmartStatus(res, true);
      if (res.state === 'ready') {
        setPreferredSmartModel(res.active_model || modelName);
        state.smartPanelOpen = false;
        S.toast('已加载并启用：' + res.active_model + '（' + (res.device || 'GPU') + (res.half ? ' FP16' : '') + '）', 'ok');
        if (res.active_slot === 'interactive') {
          state.toolMode = 'smart';
          preEncodeCurrent();
        }
      } else {
        S.toast(res.reason || '模型加载未成功', 'warn');
      }
      S.render();
    }).catch(function (err) {
      if (currentContext(task)) { S.toast('加载模型失败：' + err.message, 'bad'); S.render(); }
    }).finally(function () {
      if (currentContext(task)) state.smartLoading = false;
    });
  }

  function stopSmartModel(modelLabel) {
    if (!A.smartUnload || !live()) return;
    var task = context();
    var targetLabel = modelLabel || (state.smart && state.smart.active_model) || '智能标注模型';
    A.smartUnload(task.project, '已手动停止模型 ' + targetLabel).then(function (s) {
      if (!currentContext(task)) return;
      applySmartStatus(s, false);
      state.smartEncodedId = '';
      if (state.toolMode === 'smart') state.toolMode = 'select';
      S.toast('已停止 ' + targetLabel + ' 并释放显存', 'ok');
      S.render();
    }).catch(function (err) {
      if (currentContext(task)) S.toast('停止模型失败：' + err.message, 'bad');
    });
  }

  function openSamFolder() {
    if (!A.smartModelsOpenDir) return;
    A.smartModelsOpenDir().then(function (res) {
      S.toast('已打开模型文件夹：' + (res && res.dir ? res.dir : 'models/sam'), 'ok');
    }).catch(function (err) {
      S.toast(err.message, 'warn');
    });
  }

  function uploadSamModelFile(file) {
    if (!A.smartModelsUpload || !file) return;
    var task = context();
    state.smartUploading = '正在保存 ' + file.name + ' (0%)…';
    S.render();
    A.smartModelsUpload(file, function (ratio) {
      if (!currentContext(task)) return;
      state.smartUploading = '正在保存 ' + file.name + ' (' + Math.round(ratio * 100) + '%)…';
      S.render();
    }).then(function (res) {
      if (!currentContext(task)) return;
      state.smartUploading = '';
      applySmartStatus(res, false);
      if (res && res.uploaded_model) state.smartSelectedModel = res.uploaded_model;
      S.toast('已保存模型到 models/sam/' + ((res && res.uploaded_model) || file.name) + '，可直接选择加载', 'ok');
      if (global.OD.live && global.OD.live.refreshLibrary) global.OD.live.refreshLibrary();
      S.render();
    }).catch(function (err) {
      if (!currentContext(task)) return;
      state.smartUploading = '';
      S.toast('保存模型失败：' + err.message, 'bad');
      S.render();
    });
  }

  function preEncodeCurrent() {
    if (!A.smartEncode || !smartReadyForInteractive()) return;
    var item = current();
    if (!item || state.smartEncodedId === item.id) return;
    var task = context(), targetId = item.id;
    state.smartEncodedId = targetId;
    A.smartEncode(task.project, targetId).catch(function () {
      if (currentContext(task) && state.smartEncodedId === targetId) state.smartEncodedId = '';
    });
  }

  function runSmartPrompt(prompt) {
    if (!A.smartPredict || !live()) return;
    var item = current();
    if (!item) return;
    if (!smartReadyForInteractive()) {
      state.smartPanelOpen = true;
      S.toast('智能标注模型尚未就绪，请先在可用模型面板中选择或下载', 'warn');
      S.render();
      return;
    }
    var task = context();
    if (state.editor) state.editor.setSmartBusy(true);
    A.smartPredict(task.project, {
      item_id: item.id,
      points: prompt.points,
      labels: prompt.labels,
      bbox: prompt.bbox,
      fit_mode: state.fitMode
    }).then(function (res) {
      if (!currentContext(task) || !state.editor) return;
      state.editor.setSmartBusy(false);
      state.editor.setSmartPreview({
        points: res.points,
        polygon: res.polygon,
        score: res.score,
        source: res.source
      });
      statusHint('智能预测完成（' + (res.elapsed_ms || 0) + 'ms，按 Enter 确认入框，右键补负点微调）');
    }).catch(function (err) {
      if (!currentContext(task)) return;
      if (state.editor) state.editor.setSmartBusy(false);
      S.toast(err.message, 'bad');
      if (A.smartStatus) {
        A.smartStatus(task.project, false).then(function (info) {
          if (currentContext(task)) { applySmartStatus(info, true); S.render(); }
        }).catch(function () { });
      }
    });
  }

  /* ---------- 素材管理页（pdata） ---------- */

  function pipelineGuide() {
    var steps = [
      ['pdata', '1. 接入素材 / 拉取 ul://', '拉取云端或跨网线上传图片视频'],
      ['palabel', '2. OBB 智能标注与审核', 'SAM 3 点选 / 零样本批量标 / 审核发布'],
      ['ptrain', '3. YOLO 训练与导出', '查看曲线与混淆矩阵，导出 ONNX'],
      ['data', '4. 动作视频标注确认', '标注工艺段并确认空闲区间'],
      ['train', '5. LSTM 时序训练', '提取连续特征并训练动作模型']
    ];
    return el('section', { class: 'card pipeline-guide' }, [
      el('div', { class: 'pipeline-steps' }, steps.map(function (st) {
        var cur = S.state.view === st[0];
        return el('button', { type: 'button', class: 'pipeline-step' + (cur ? ' is-active' : ''), onclick: function () { S.go(st[0]); } }, [
          el('strong', { text: st[1] }),
          el('small', { text: st[2] })
        ]);
      }))
    ]);
  }

  function management() {
    var task = context();
    var source = input('', { id: 'parts-source', placeholder: '图片、文件夹或视频的训练机本地完整路径' }), group = input('', { placeholder: '留空按原文件或目录分组' }), interval = input('30', { type: 'number', min: '1', max: '100000' }), limit = input('300', { type: 'number', min: '1', max: '10000' }), dedupe = el('input', { type: 'checkbox', checked: true });
    var importButton = btn('导入素材', function () {
      if (!source.value.trim()) { S.toast('先填写素材路径', 'warn'); return; }
      importButton.disabled = true;
      importButton.text = '正在导入…';
      action(task, function () { return A.partsImport(task.project, { source: source.value.trim(), source_group: group.value.trim(), interval: Number(interval.value), max_frames: Number(limit.value), dedupe: dedupe.checked }); }, applySnapshot)
        .finally(function () { if (currentContext(task)) { importButton.disabled = false; importButton.text = '导入素材'; } });
    }, 'btn-primary');
    var classes = el('textarea', { class: 'form-control', rows: '3', 'aria-label': '零件类别，每行一个', placeholder: '每行一个类别；类别顺序决定标签编号' }, (state.data.classes || []).join('\n'));
    var items = state.data.items || [], counts = { unlabeled: 0, pending: 0, reviewed: 0, negative: 0 };
    items.forEach(function (i) { counts[i.status] = (counts[i.status] || 0) + 1; });
    var body = [
      pipelineGuide(),
      el('section', { class: 'card' }, [
        el('header', null, [el('h2', { text: '零件标注与数据发布' }), el('span', { class: 'end' }, btn('打开智能标注画布', function () { S.go('palabel'); }, 'btn-primary'))]),
        el('p', { class: 'help', text: items.length + ' 张素材 · 未标注 ' + counts.unlabeled + ' · 待审核 ' + counts.pending + ' · 已审核 ' + counts.reviewed + ' · 确认无目标 ' + counts.negative }),
        el('p', { class: 'help', text: '图片抽样用于零件检测；不会改变动作视频的连续特征。所有自动标注结果均以“待审核”入库，人工确认后方可发布。' })
      ]),
      global.OD.transfer ? global.OD.transfer.renderHub() : null,
      el('details', { class: 'card', open: !global.OD.transfer }, [
        el('summary', { style: 'font-weight:600;cursor:pointer', text: '从训练机本地磁盘路径直接导入图片、目录或视频' }),
        el('div', { class: 'parts-import-grid sp-mt-3' }, [formField('素材路径', source), formField('拍摄来源组', group), formField('视频每隔多少帧抽一张', interval), formField('最多导入帧数', limit)]),
        el('div', { class: 'parts-toolbar' }, [el('label', { class: 'parts-inline' }, [dedupe, '去除重复或相似帧']), importButton])
      ]),
      el('section', { class: 'card' }, [
        el('h2', { text: '类别顺序' }), classes,
        el('div', { class: 'parts-toolbar' }, btn('保存类别', function () {
          action(task, function () {
            return save().then(function () {
              if (currentContext(task)) return A.partsClasses(task.project, classes.value.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean), state.data.revision);
            });
          }, applySnapshot);
        })),
        el('p', { class: 'help', text: '编号从 0 开始。已有标注后不要重排类别；已发布版本保留自己的类别顺序。' })
      ]),
      publishCard()
    ];
    if (state.quality) body.push(qualityCard());
    return body;
  }

  function publishCard() {
    var task = context();
    assignSplits();
    var groups = Array.from(new Set((state.data.items || []).map(function (i) { return i.source_group; }))).sort();
    var versions = state.data.versions || [];
    return el('section', { class: 'card' }, [
      el('h2', { text: '按拍摄来源划分并发布' }),
      el('p', { class: 'help', text: '同一原视频或拍摄批次整体进入一个集合，至少需要独立的训练与验证来源。所有图片都应审核，空图请明确确认无目标。' }),
      el('div', { class: 'parts-group-list' }, groups.map(function (g) {
        return el('div', { class: 'parts-group-row' }, [
          el('span', { text: g, title: g }),
          selectBox([['train', '训练集'], ['val', '验证集'], ['test', '独立测试集']], state.splits[g], function (e) { state.splits[g] = e.target.value; }, '来源 ' + g + ' 的用途')
        ]);
      })),
      el('div', { class: 'parts-toolbar' }, [
        btn('检查数据质量', function () { action(task, function () { return A.partsQuality(task.project); }, function (q) { state.quality = q; S.render(); }); }),
        btn('发布固定数据集版本', function () {
          action(task, function () {
            return save().then(function () {
              if (currentContext(task)) { assignSplits(); return A.partsPublish(task.project, { splits: clone(state.splits), val_ratio: 0.2, test_ratio: 0.1 }); }
            });
          }, function (v) {
            S.toast('已发布数据版本 ' + (v.id || v.version || ''), 'ok');
            state.draft = null;
            return Promise.all([load(true), refreshRelated(task)]).then(function () { if (currentContext(task)) S.render(); });
          }, '发布失败：');
        }, 'btn-primary')
      ]),
      versions.length ? el('div', { class: 'parts-versions' }, versions.map(function (v) {
        return el('div', null, [
          el('code', { text: v.id || v.version || v.name }),
          el('span', { text: ' · ' + ((v.items || []).length || v.item_count || v.images || '') + ' 张 · ' + (v.created_at ? new Date(v.created_at * 1000).toLocaleString() : '') })
        ]);
      })) : el('p', { class: 'help', text: '尚未发布版本。发布后可在零件训练页选用，后续标注修改不会影响已发布数据。' })
    ]);
  }

  function qualityCard() {
    var q = state.quality, issues = q.issues || [];
    return el('section', { class: 'card' }, [
      el('h2', { text: q.ok ? '质量检查通过' : '需要处理的问题' }),
      issues.length ? el('ul', { class: 'parts-issues' }, issues.map(function (issue) {
        return el('li', null, [issue.item_id ? btn('定位素材', function () { openItem(issue.item_id); }) : null, el('span', { text: (issue.code ? issue.code + '：' : '') + issue.message })]);
      })) : el('p', { class: 'help', text: '未发现阻断问题。仍需确认拍摄来源与人工审核质量。' })
    ]);
  }

  function renderData(legacyNode) {
    if (!live()) return page('零件素材与标注', '连接本机训练台后使用。', []);
    if (!ensure()) return page('零件素材与标注', '素材、审核与数据发布', loadingBody());
    var body = management();
    if (legacyNode) body.push(el('details', { class: 'card parts-legacy' }, [el('summary', { text: '导入外部已标注数据集（兼容旧流程）' }), legacyNode]));
    return page('零件素材与标注', '从原始素材到可追溯的数据集版本。', body);
  }

  /* ---------- 零件标注页（palabel）辅助动作 ---------- */

  function filtered() {
    return (state.data.items || []).filter(function (i) {
      return (!state.filter || (i.name + ' ' + i.source).toLowerCase().indexOf(state.filter.toLowerCase()) >= 0) &&
        (!state.status || i.status === state.status) &&
        (!state.group || i.source_group === state.group) &&
        (state.category === '' || (i.boxes || []).some(function (b) { return b.class_id === Number(state.category); }));
    });
  }

  function stepItem(delta, onlyUnreviewed) {
    var list = filtered();
    if (!list.length) return;
    var idx = list.findIndex(function (x) { return x.id === state.selected; });
    if (idx < 0) { select(list[0].id); return; }
    if (onlyUnreviewed) {
      for (var k = idx + 1; k < list.length; k++) {
        if (list[k].status !== 'reviewed' && list[k].status !== 'negative') { select(list[k].id); return; }
      }
      for (var m = 0; m < idx; m++) {
        if (list[m].status !== 'reviewed' && list[m].status !== 'negative') { select(list[m].id); return; }
      }
      S.toast('当前筛选列表已全部审核完成！', 'ok');
      return;
    }
    var nextIdx = idx + delta;
    if (nextIdx >= 0 && nextIdx < list.length) select(list[nextIdx].id);
  }

  function copyPrevFrameBoxes() {
    var list = filtered();
    var idx = list.findIndex(function (x) { return x.id === state.selected; });
    if (idx <= 0) { S.toast('当前已经是第一张，没有上一帧可复制', 'warn'); return; }
    var prev = list[idx - 1];
    if (!prev.boxes || !prev.boxes.length) { S.toast('上一张素材没有标注框', 'warn'); return; }
    var copied = clone(prev.boxes).map(function (b) { b.source = 'copy'; delete b.confidence; return b; });
    if (state.editor) state.editor.addBoxes(copied, false);
    else if (state.draft) change((state.draft.boxes || []).concat(copied));
    S.toast('已复制上一帧 ' + copied.length + ' 个框', 'ok');
  }

  function reviewAndNext() {
    if (!state.draft || !state.draft.boxes || !state.draft.boxes.length) {
      S.toast('当前没有标注框；若画面确实无零件请按 E 确认无目标', 'warn');
      return;
    }
    save('reviewed').then(function () {
      paintItemRows();
      S.toast('已审核并跳转下一张', 'ok');
      stepItem(1, true);
    }).catch(function () { });
  }

  function negativeAndNext() {
    if (!state.draft) return;
    if (state.draft.boxes && state.draft.boxes.length) {
      S.toast('画面仍有标注框，请先删除后再确认无目标', 'warn');
      return;
    }
    save('negative').then(function () {
      paintItemRows();
      S.toast('已确认无目标并跳转下一张', 'ok');
      stepItem(1, true);
    }).catch(function () { });
  }

  function quickAddClass(rawName) {
    var name = String(rawName || '').trim();
    if (!name) { S.toast('请输入新类别名称', 'warn'); return Promise.resolve(); }
    var existing = state.data.classes || [];
    if (existing.indexOf(name) >= 0) { S.toast('类别“' + name + '”已存在', 'warn'); return Promise.resolve(); }
    var task = context(), nextClasses = existing.concat([name]);
    return action(task, function () {
      return save().then(function () {
        if (currentContext(task)) return A.partsClasses(task.project, nextClasses, state.data.revision);
      });
    }, function (snap) {
      applySnapshot(snap);
      if (state.editor) {
        state.editor.setClasses(state.data.classes);
        state.editor.setClass(state.data.classes.length - 1);
      }
      S.toast('已新增类别：' + name, 'ok');
    });
  }

  /* ---------- 智能状态条与可用模型面板 ---------- */

  function renderSmartBanner() {
    if (!A.smartStatus) return null;
    var info = state.smart;
    if (!info) {
      return el('section', { class: 'card smart-bar' }, [
        el('div', { class: 'smart-bar-row' }, [
          el('strong', { text: state.smartLoading ? '正在自动检测并拉起默认 Meta SAM 3…' : '智能标注状态未同步' }),
          btn('刷新检测', function () { state.smartTried = false; autoInitSmart(); })
        ])
      ]);
    }
    if (info.training_locked) {
      return el('section', { class: 'card smart-bar is-locked' }, [
        el('div', { class: 'smart-bar-row' }, [
          el('strong', { text: '训练进行中：已自动停止并禁用所有智能标注（含 CPU 模式）以保障 ≥8GB 显卡训练稳定' }),
          el('span', { class: 'help', text: '当前可照常使用纯人工画框（W / R）、改框与审核；训练结束后点右侧刷新恢复默认 SAM 3。' }),
          btn('检查训练状态', function () {
            var task = context();
            A.smartStatus(task.project, false).then(function (s) {
              if (!currentContext(task)) return;
              applySmartStatus(s, true);
              if (!s.training_locked) { state.smartTried = false; autoInitSmart(); }
              else S.render();
            });
          })
        ])
      ]);
    }

    var isReady = info.state === 'ready';
    var isIdle = info.state === 'idle';
    var locals = localSmartModels(info);
    var selectedName = state.smartSelectedModel || info.active_model || (locals.length ? locals[0].name : '');
    var selectedModel = locals.filter(function (m) { return m.name === selectedName; })[0] || locals[0] || null;
    if (selectedModel) state.smartSelectedModel = selectedModel.name;

    var vramText = (info.vram && info.vram.cuda && info.vram.total_mb)
      ? ('GPU ' + (info.vram.gpu || '') + ' · 剩余 ' + (info.vram.free_mb || 0) + ' / ' + info.vram.total_mb + ' MiB')
      : (info.env && info.env.ok ? 'CPU 模式' : '未检测到 YOLO Python 环境');
    var badgeText = state.smartLoading
      ? '正在加载模型…'
      : (isReady
        ? ('运行中：' + info.active_model + '（' + (info.device || 'GPU') + (info.half ? ' FP16' : '') + ' · ' + info.active_slot + '）')
        : (isIdle
          ? (locals.length ? '未加载模型 · 请选择文件夹内模型加载使用' : '未加载模型 · 请先下载或放入模型到 models/sam')
          : (locals.length ? '当前未加载 · 请从下方选择已保存模型加载使用' : 'models/sam 文件夹暂无权重 · 请放入或一键下载模型')));

    var topControls = [];
    if (locals.length) {
      var picker = selectBox(
        locals.map(function (m) {
          return [m.name, m.name + (m.active ? '（运行中）' : (' · ' + fmtBytes(m.bytes)))];
        }),
        selectedModel ? selectedModel.name : '',
        function (e) { state.smartSelectedModel = e.target.value; S.render(); },
        '选择 models/sam 文件夹中的模型'
      );
      picker.className = 'form-select smart-model-select';
      topControls.push(picker);
      if (!selectedModel || !selectedModel.active) {
        topControls.push(btn(isReady ? '切换加载' : '加载并使用', function () {
          if (!selectedModel) return;
          activateSmartModel(selectedModel.name, defaultSlotForModel(selectedModel));
        }, 'btn-primary', { disabled: state.smartLoading || !selectedModel }));
      }
    }
    if (isReady) {
      topControls.push(btn('停止模型', function () {
        stopSmartModel(info.active_model);
      }, 'btn-danger', { disabled: state.smartLoading }));
    } else if (!locals.length && A.smartModelsOpenDir) {
      topControls.push(btn('打开模型文件夹', openSamFolder, 'btn-secondary'));
    }
    topControls.push(btn(
      state.smartPanelOpen ? '收起模型文件夹' : ('管理模型文件夹 (本地 ' + locals.length + ')'),
      function () { state.smartPanelOpen = !state.smartPanelOpen; S.render(); },
      isReady || locals.length ? 'btn-secondary' : 'btn-primary'
    ));

    var dotCls = isReady ? 'is-ok' : (state.smartLoading ? 'is-busy' : (isIdle ? 'is-idle' : 'is-warn'));
    var barCls = isReady ? 'is-ready' : (isIdle ? 'is-idle' : 'is-warn');
    var rows = [
      el('div', { class: 'smart-bar-row' }, [
        el('span', { class: 'smart-dot ' + dotCls }),
        el('strong', { text: badgeText }),
        el('span', { class: 'help', text: vramText }),
        el('span', { class: 'sp-ml-auto smart-bar-actions' }, topControls)
      ])
    ];

    if (info.reason && !isReady && info.reason_code !== 'stopped') {
      rows.push(el('p', { class: 'smart-reason', text: info.reason }));
    }

    if (state.smartPanelOpen) {
      rows.push(renderAvailableModelsPanel(info));
    }
    return el('section', { class: 'card smart-bar ' + barCls }, rows);
  }

  function renderAvailableModelsPanel(info) {
    var task = context();
    var env = info.env || {};
    var depHints = [];
    if (env.ok && !env.clip && env.clip_install_cmd) {
      depHints.push(el('div', { class: 'smart-dep-tip' }, [
        el('span', { text: '提示：当前 YOLO 环境未安装 clip（仅影响 SAM 3 文本提示词模式；交互点选、视觉示例与视频追踪均不受影响）。安装命令：' }),
        el('code', { text: env.clip_install_cmd }),
        btn('复制命令', function () { copyText(env.clip_install_cmd); })
      ]));
    }
    if (env.ok && !env.mobileclip && env.mobileclip_install_cmd) {
      depHints.push(el('div', { class: 'smart-dep-tip' }, [
        el('span', { text: '提示：未安装 mobileclip（仅影响 YOLOE 文本提示模式；YOLOE 视觉示例框免安装直接可用）。安装命令：' }),
        el('code', { text: env.mobileclip_install_cmd }),
        btn('复制命令', function () { copyText(env.mobileclip_install_cmd); })
      ]));
    }

    var models = info.available_models || [];
    var cards = models.map(function (m) {
      var dl = m.download;
      var dlBusy = dl && dl.status === 'downloading';
      var actionBtns = [];
      if (m.kind === 'yolo_preannotate') {
        actionBtns.push(btn('去 YOLO 预标注使用', function () {
          state.autoTab = 'yolo';
          state.smartPanelOpen = false;
          S.render();
        }));
      } else if (m.local) {
        if (m.active) {
          actionBtns.push(el('span', { class: 'badge ok', text: '运行中' }));
          actionBtns.push(btn('停止此模型', function () {
            stopSmartModel(m.name);
          }, 'btn-danger', { disabled: state.smartLoading }));
        } else {
          actionBtns.push(btn('加载并使用', function () {
            activateSmartModel(m.name, defaultSlotForModel(m));
          }, 'btn-primary', { disabled: state.smartLoading }));
        }
        if (m.family === 'yoloe') {
          actionBtns.push(btn('用于零样本批量标', function () {
            state.autoTab = 'zeroshot';
            state.zsModel = m.name;
            state.zsEngine = 'yoloe_vp';
            state.smartPanelOpen = false;
            S.toast('已选中 ' + m.name + '，可在下方“智能自动标注”面板发起零样本批量标', 'ok');
            S.render();
          }));
        }
      } else if (m.downloadable && A.smartDownload) {
        actionBtns.push(btn(dlBusy ? ('下载中 ' + (dl.percent || 0) + '%') : '下载到文件夹', function () {
          action(task, function () { return A.smartDownload(task.project, m.name, ''); }, function () {
            S.toast('开始下载 ' + m.name + ' 到 models/sam/', 'ok');
            scheduleSmartPoll();
            A.smartStatus(task.project, false).then(function (s) { if (currentContext(task)) { applySmartStatus(s, false); S.render(); } });
          });
        }, 'btn-primary', { disabled: dlBusy }));
      } else if (m.gated) {
        actionBtns.push(el('span', { class: 'help', text: '请将权重放入 ' + (info.sam_dir || 'models/sam') + '，或点上方「上传权重」/下方直链下载' }));
      }

      var statusTag = m.active ? '运行中' : (m.local ? ('已保存在文件夹 · ' + fmtBytes(m.bytes)) : (m.gated ? '需手动放入/上传' : '未下载'));
      return el('div', { class: 'smart-model-card' + (m.active ? ' is-active' : '') + (m.local ? ' is-local' : '') }, [
        el('div', { class: 'smart-model-head' }, [
          el('strong', { text: m.label || m.name }),
          el('span', { class: 'smart-model-meta', text: '≈' + (m.vram_gb || 1) + 'G 显存 · ' + statusTag })
        ]),
        el('p', { class: 'help', text: m.note || '' }),
        dlBusy ? el('div', { class: 'progress sp-mt-2' }, el('div', { class: 'progress-bar busy', style: 'width:' + (dl.percent || 5) + '%' })) : null,
        dl && dl.status === 'failed' ? el('p', { class: 'text-danger', text: '下载失败：' + dl.error }) : null,
        el('div', { class: 'parts-toolbar', style: 'margin:6px 0 0' }, actionBtns)
      ]);
    });

    var uploadInput = el('input', {
      type: 'file', accept: '.pt,.pth', style: 'display:none',
      onchange: function (e) {
        var file = e.target.files && e.target.files[0];
        if (file) uploadSamModelFile(file);
        e.target.value = '';
      }
    });
    var urlInput = input(state.customUrl, {
      placeholder: '可选：填入内网或自有镜像的 sam3.pt 直链 (http/https) 一键拉取到 models/sam/',
      oninput: function (e) { state.customUrl = e.target.value; }
    });
    return el('div', { class: 'smart-panel' }, [
      el('div', { class: 'smart-panel-head' }, [
        el('div', null, [
          el('strong', { text: 'Meta / 智能标注模型文件夹：' }),
          el('code', { class: 'mono', text: info.sam_dir || 'models/sam' })
        ]),
        el('div', { class: 'smart-bar-actions' }, [
          uploadInput,
          A.smartModelsOpenDir ? btn('打开模型文件夹', openSamFolder) : null,
          A.smartModelsUpload ? btn(state.smartUploading || '上传模型 (.pt/.pth)', function () { uploadInput.click(); }, 'btn-secondary', { disabled: !!state.smartUploading }) : null,
          btn('刷新扫描文件夹', function () {
            A.smartStatus(task.project, true).then(function (s) { if (currentContext(task)) { applySmartStatus(s, false); S.render(); } });
          })
        ])
      ])
    ].concat(depHints).concat([
      el('div', { class: 'smart-model-grid' }, cards),
      A.smartDownload ? el('div', { class: 'parts-toolbar', style: 'margin-top:10px' }, [
        urlInput,
        btn('从自定义直链下载 sam3.pt', function () {
          if (!state.customUrl.trim()) { S.toast('请先粘贴可访问的 HTTP/HTTPS 直链', 'warn'); return; }
          action(task, function () { return A.smartDownload(task.project, 'sam3.pt', state.customUrl.trim()); }, function () {
            S.toast('已启动 sam3.pt 下载任务', 'ok');
            scheduleSmartPoll();
          });
        })
      ]) : null
    ]));
  }

  /* ---------- 页内快捷导入素材 / 从工程视频抽帧抽屉 ---------- */

  function renderQuickIngest() {
    var task = context();
    var fileInput = el('input', {
      type: 'file', multiple: true, accept: 'image/*,video/*,.zip', style: 'display:none',
      onchange: function (e) {
        var files = Array.from(e.target.files || []);
        if (!files.length || !A.uploadParts) return;
        var p = Promise.resolve();
        files.forEach(function (file) {
          p = p.then(function () {
            S.toast('正在上传 ' + file.name + '…', 'ok');
            return A.uploadParts(task.project, file, { interval: 30, maxFrames: 300, dedupe: true });
          }).then(function (snap) {
            if (currentContext(task)) applySnapshot(snap);
          });
        });
        p.then(function () { if (currentContext(task)) { state.showIngest = false; S.toast('素材上传并入库完成', 'ok'); S.render(); } })
          .catch(function (err) { if (currentContext(task)) S.toast('上传失败：' + err.message, 'bad'); });
      }
    });

    var clipSel = selectBox(
      state.mediaClips.length
        ? state.mediaClips.map(function (c) { return [c.file, c.file + ' (' + fmtBytes(c.bytes) + ')']; })
        : [['', '当前工程 media/ 暂无视频']],
      state.mediaClips.length ? state.mediaClips[0].file : '',
      null,
      '选择当前工程视频'
    );
    var ivInput = input('30', { type: 'number', min: '1', max: '10000' });
    var maxInput = input('120', { type: 'number', min: '1', max: '2000' });

    return el('section', { class: 'card smart-ingest' }, [
      el('div', { class: 'smart-panel-head' }, [
        el('strong', { text: '在标注页直接接入素材（上传图片/视频/ZIP，或从本工程动作视频抽帧）' }),
        state.data.items.length ? btn('收起', function () { state.showIngest = false; S.render(); }) : null
      ]),
      el('div', { class: 'parts-toolbar' }, [
        fileInput,
        btn('选择图片 / 视频 / ZIP 上传', function () { fileInput.click(); }, 'btn-primary'),
        el('span', { class: 'help', text: '支持多张图片、MP4/AVI 视频自动抽帧或 YOLO ZIP 数据集' })
      ]),
      A.partsImportClip ? el('div', { class: 'parts-toolbar' }, [
        formField('本工程 media/ 视频', clipSel),
        formField('隔几帧抽 1 张', ivInput),
        formField('最多抽帧数', maxInput),
        btn('从工程视频一键抽帧', function () {
          if (!clipSel.value) { S.toast('当前工程 media/ 里还没有视频，请先上传视频或去动作素材页传片', 'warn'); return; }
          action(task, function () {
            return A.partsImportClip(task.project, { file: clipSel.value, interval: Number(ivInput.value) || 30, max_frames: Number(maxInput.value) || 120, dedupe: true });
          }, function (snap) {
            state.showIngest = false;
            applySnapshot(snap);
            S.toast('已从工程视频抽帧入库，可直接开始标注', 'ok');
          }, '抽帧失败：');
        }, 'btn-primary', { disabled: !state.mediaClips.length })
      ]) : null
    ]);
  }

  /* ---------- 左侧素材列表 ---------- */

  function itemList() {
    var search = input(state.filter, { placeholder: '搜索名称或来源', oninput: function (e) { state.filter = e.target.value; paintItemRows(); } });
    var groups = Array.from(new Set(state.data.items.map(function (i) { return i.source_group; }))).sort();
    var list = el('div', { class: 'parts-item-list' });
    state.refs.list = list;
    var controls = el('div', { class: 'parts-list-filters' }, [
      search,
      selectBox([['', '全部标注状态']].concat(Object.keys(STATUS).map(function (k) { return [k, STATUS[k]]; })), state.status, function (e) { state.status = e.target.value; paintItemRows(); }, '按标注状态筛选'),
      selectBox([['', '全部来源']].concat(groups.map(function (g) { return [g, g]; })), state.group, function (e) { state.group = e.target.value; paintItemRows(); }, '按拍摄来源筛选'),
      selectBox([['', '全部类别']].concat(state.data.classes.map(function (n, i) { return [String(i), n]; })), state.category, function (e) { state.category = e.target.value; paintItemRows(); }, '按零件类别筛选')
    ]);
    setTimeout(paintItemRows, 0);
    return el('aside', { class: 'parts-assets' }, [controls, list]);
  }

  function paintItemRows() {
    var host = state.refs.list;
    if (!host || !host.isConnected) return;
    D.clear(host);
    filtered().forEach(function (i) {
      var check = el('input', { type: 'checkbox', checked: !!state.checked[i.id], 'aria-label': '选择 ' + i.name, onchange: function (e) { state.checked[i.id] = e.target.checked; } });
      host.appendChild(el('div', { class: 'parts-item ' + (i.id === state.selected ? 'is-selected' : '') }, [
        check,
        el('button', { type: 'button', class: 'parts-item-main', onclick: function () { select(i.id); } }, [
          el('img', { src: A.partsImageUrl(pid(), i.id), loading: 'lazy', alt: '', width: '64', height: '48' }),
          el('span', null, [
            el('strong', { text: i.name }),
            el('small', { text: (STATUS[i.status] || i.status) + ' · ' + ((i.boxes || []).length) + ' 框' }),
            el('small', { text: i.source_group, title: i.source_group })
          ])
        ])
      ]));
    });
    if (!host.children.length) host.appendChild(el('p', { class: 'help', text: '没有符合筛选条件的素材' }));
  }

  /* ---------- 中央专业 OBB 画布 + 右侧属性与行内类别 ---------- */

  function editorPanel() {
    var item = current();
    if (!item) {
      return el('div', { class: 'parts-empty-wrap' }, [
        el('div', { class: 'empty' }, [
          el('h2', { text: '还没有零件素材' }),
          el('p', { class: 'help', text: '可直接在本页上传图片/视频，或从本工程已有动作视频一键抽帧开始标注。' }),
          btn('打开素材页管理', function () { S.go('pdata'); })
        ]),
        renderQuickIngest()
      ]);
    }
    if (!state.draft) setDraft(item);
    var locked = !!(state.smart && state.smart.training_locked);
    var hint = el('span', { class: 'parts-save-hint', text: dirty() ? '本机有未保存草稿' : '已保存 · ' + STATUS[item.status] });
    state.refs.saveHint = hint;
    var zoom = el('span', { class: 'mono', text: '100%' });

    var modeBtns = {};
    function modeButton(key, label, disabled) {
      var active = state.toolMode === key || (key === 'draw' && state.toolMode === 'hbb');
      var b = btn(label, function () {
        if (key === 'smart') {
          if (locked) { S.toast('训练进行中，已禁用智能标注', 'warn'); return; }
          if (!smartReadyForInteractive()) {
            state.smartPanelOpen = true;
            S.toast('请先在可用模型面板中加载或下载 SAM 3 / SAM 2.1 / MobileSAM', 'warn');
            S.render();
            return;
          }
          preEncodeCurrent();
        }
        state.toolMode = key;
        if (state.editor) state.editor.setMode(key);
        Object.keys(modeBtns).forEach(function (k) {
          modeBtns[k].className = 'btn btn-sm ' + (k === key ? 'btn-primary' : 'btn-secondary');
        });
      }, active ? 'btn-primary' : 'btn-secondary', { disabled: !!disabled });
      modeBtns[key] = b;
      return b;
    }

    var stage = el('div', { class: 'parts-stage' });
    var category = selectBox(
      state.data.classes.length ? state.data.classes.map(function (n, i) { return [String(i), (i + 1 <= 9 ? '[' + (i + 1) + '] ' : '') + n]; }) : [['0', '（请先新增类别）']],
      '0',
      function (e) { if (state.editor) state.editor.setClass(e.target.value); },
      '当前框的零件类别'
    );
    state.refs.category = category;
    var boxes = el('div', { class: 'parts-boxes' });
    state.refs.boxes = boxes;
    var name = el('strong', { text: item.name });

    var editor = new global.OD.obb.Editor(stage, {
      src: A.partsImageUrl(pid(), item.id),
      boxes: state.draft.boxes,
      classes: state.data.classes,
      mode: locked && state.toolMode === 'smart' ? 'select' : state.toolMode,
      onChange: change,
      onSelect: function (index, box) {
        if (box && state.refs.category) state.refs.category.value = String(box.class_id);
        paintBoxList();
      },
      onModeChange: function (m) {
        state.toolMode = m;
        Object.keys(modeBtns).forEach(function (k) {
          if (modeBtns[k]) modeBtns[k].className = 'btn btn-sm ' + (k === m ? 'btn-primary' : 'btn-secondary');
        });
      },
      onSmartPrompt: runSmartPrompt,
      onExemplarPicked: function (bbox) {
        state.zsExemplar = { item_id: item.id, item_name: item.name, bbox: bbox, class_id: Number(category.value || 0) };
        S.toast('已框选视觉示例（类别：' + (state.data.classes[state.zsExemplar.class_id] || state.zsExemplar.class_id) + '），可直接启动零样本批量标', 'ok');
        S.render();
      },
      onError: function (message) { S.toast(message, 'bad'); },
      onSave: function () { save().catch(function () { }); },
      onZoom: function (z) { zoom.textContent = Math.round(z * 100) + '%'; }
    });
    state.editor = editor;
    setTimeout(paintBoxList, 0);

    var newClsInput = input('', {
      placeholder: '输入新类别名回车添加',
      onkeydown: function (e) {
        if (e.key === 'Enter') { e.preventDefault(); quickAddClass(newClsInput.value); }
      }
    });

    var smartSubBar = (!locked && state.toolMode === 'smart') ? el('div', { class: 'smart-canvas-bar' }, [
      el('span', { class: 'help', text: 'SAM 交互：左键点击零件(正点) · 右键排除背景(负点) · 拖拽画粗框' }),
      selectBox([['obb', '拟合旋转框 OBB'], ['hbb', '拟合水平框 HBB']], state.fitMode, function (e) { state.fitMode = e.target.value; }, '掩码拟合形状'),
      btn('确认入框 Enter', function () {
        if (!editor.acceptSmartPreview()) S.toast('请先在图上点击零件生成预览掩码', 'warn');
      }, 'btn-primary'),
      btn('清除提示点 Esc', function () { editor.clearSmart(); })
    ]) : null;

    var list = filtered();
    var posIdx = list.findIndex(function (x) { return x.id === item.id; });
    var posText = posIdx >= 0 ? ('第 ' + (posIdx + 1) + ' / ' + list.length + ' 张') : '';

    return el('section', { class: 'parts-editor' }, [
      el('header', { class: 'parts-editor-head' }, [
        name,
        el('span', { class: 'help', text: posText + ' · ' + item.width + ' × ' + item.height + (item.frame_index !== null && item.frame_index !== undefined ? ' · 原视频帧 ' + item.frame_index : '') }),
        hint
      ]),
      el('div', { class: 'parts-toolbar' }, [
        modeButton('select', '选择 V', false),
        modeButton('draw', '水平框 W', false),
        modeButton('obb3', '3点旋转框 R', false),
        modeButton('smart', '智能点选 S', locked),
        btn('上一张 A', function () { stepItem(-1, false); }),
        btn('下一张 D', function () { stepItem(1, false); }),
        btn('复制上帧 C', copyPrevFrameBoxes),
        btn('撤销', function () { editor.undo(); }),
        btn('重做', function () { editor.redo(); }),
        btn('复制框', function () { editor.duplicate(); }),
        btn('删除框', function () { editor.remove(); }),
        btn('−', function () { editor.setZoom(editor.zoom / 1.2); }),
        zoom,
        btn('＋', function () { editor.setZoom(editor.zoom * 1.2); }),
        btn('100%', function () { editor.setZoom(1); })
      ]),
      smartSubBar,
      el('div', { class: 'parts-canvas-grid' }, [
        el('div', { class: 'parts-stage-scroll' }, stage),
        el('aside', { class: 'parts-properties' }, [
          formField('当前/选中框类别（快捷键 1-9）', category),
          el('div', { class: 'smart-add-cls' }, [
            newClsInput,
            btn('+ 类别', function () { quickAddClass(newClsInput.value); })
          ]),
          el('div', { class: 'parts-toolbar' }, [
            btn('↶ 5°', function () { editor.rotateBy(-5); }),
            btn('↷ 5°', function () { editor.rotateBy(5); }),
            btn('缩小', function () { editor.scaleBy(0.9); }),
            btn('放大', function () { editor.scaleBy(1.1); })
          ]),
          el('h3', { text: '当前标注框 (' + (state.draft.boxes || []).length + ')' }),
          boxes,
          formField('拍摄来源组', input(state.draft.source_group, {
            onchange: function (e) { state.draft.source_group = e.target.value; state.draft.status = 'pending'; remember(); schedule(); }
          })),
          el('p', { class: 'help', text: '快捷键：A/D 切图 · 1-9 切类 · V/W/R/S 切工具 · Space 审核并下一张 · E 确认无目标 · C 复制上帧 · 滚轮缩放 · 中键拖拽平移。' }),
          el('div', { class: 'parts-review-actions' }, [
            btn('保存待审核', function () { save('pending').then(paintItemRows).catch(function () { }); }),
            btn('审核并下一张 (Space)', reviewAndNext, 'btn-primary'),
            btn('确认无目标 (E)', function () {
              if (state.draft.boxes.length) { S.toast('画面仍有框，请先删除或继续审核', 'warn'); return; }
              if (global.confirm('确认已检查整张图片，画面中没有需要检测的零件？')) negativeAndNext();
            })
          ]),
          btn('重新载入磁盘版本', function () {
            if (dirty() && !global.confirm('放弃这张素材的本机草稿，重新载入已保存版本？')) return;
            forgetDraft(); state.draft = null; load(true).then(function () { setDraft(current()); S.render(); });
          })
        ])
      ])
    ]);
  }

  function paintBoxList() {
    var host = state.refs.boxes;
    if (!host || !state.draft) return;
    D.clear(host);
    state.draft.boxes.forEach(function (b, i) {
      var src = b.source && b.source !== 'manual' ? (' [' + b.source + (typeof b.confidence === 'number' ? ' ' + Math.round(b.confidence * 100) + '%' : '') + ']') : '';
      host.appendChild(btn((i + 1) + '. ' + (state.data.classes[b.class_id] || b.class_id) + src, function () {
        state.editor.selected = i; state.editor.changedSelection(); state.editor.paint();
      }, state.editor && state.editor.selected === i ? 'btn-primary' : 'btn-secondary'));
    });
    if (!state.draft.boxes.length) host.appendChild(el('p', { class: 'help', text: '没有标注框。按 W/R 手动画框，或按 S 用 Meta SAM 点选。' }));
  }

  /* ---------- 三合一智能自动标注面板（零样本 + 连续帧追踪 + 自训 YOLO） ---------- */

  function stopJob() {
    return A.stop(pid()).then(function () { S.toast('已请求停止零件任务', 'warn'); }).catch(function (e) { S.toast('停止失败：' + e.message, 'bad'); });
  }
  function activeJob() {
    var job = state.job;
    if (state.pid === pid() && job && ['running', 'queued', 'preparing'].indexOf(job.status) >= 0) {
      return { name: job.id, label: job.mode === 'preannotate' ? '零件预标注' : '零件验证', done: job.done || 0, total: job.total || (job.items || []).length, unit: '张', elapsed: Math.max(0, Date.now() / 1000 - (job.created_at || Date.now() / 1000)), stop: stopJob, open: function () { S.go(job.mode === 'preannotate' ? 'palabel' : 'pvalidate'); } };
    }
    var sj = state.smart && state.smart.batch_job;
    if (state.pid === pid() && sj && ['running', 'queued'].indexOf(sj.status) >= 0) {
      return { name: sj.id, label: sj.engine === 'video_track' ? '连续帧追踪' : '零样本自动标', done: sj.done || 0, total: sj.total || 1, unit: '张', elapsed: Math.max(0, Date.now() / 1000 - (sj.started_at || Date.now() / 1000)), stop: stopJob, open: function () { S.go('palabel'); } };
    }
    return null;
  }

  function renderAutoLabelHub() {
    var task = context();
    var locked = !!(state.smart && state.smart.training_locked);
    var sj = state.smart && state.smart.batch_job;
    var sjRunning = !!(sj && ['running', 'queued'].indexOf(sj.status) >= 0);

    var tabs = el('div', { class: 'smart-tabs' }, [
      btn('1. 零样本冷启动批量标（SAM 3 / YOLOE）', function () { state.autoTab = 'zeroshot'; S.render(); }, state.autoTab === 'zeroshot' ? 'btn-primary' : 'btn-secondary'),
      btn('2. 连续视频帧追踪传播（SAM 3 / SAM 2.1）', function () { state.autoTab = 'track'; S.render(); }, state.autoTab === 'track' ? 'btn-primary' : 'btn-secondary'),
      btn('3. 自训练 YOLO-OBB 预标注', function () { state.autoTab = 'yolo'; S.render(); }, state.autoTab === 'yolo' ? 'btn-primary' : 'btn-secondary')
    ]);

    var bodyNode;
    if (state.autoTab === 'zeroshot') {
      var localModels = ((state.smart && state.smart.available_models) || []).filter(function (m) {
        return m.local && (m.family === 'yoloe' || m.family === 'sam3');
      });
      var engineSel = selectBox([
        ['yoloe_vp', 'YOLOE 视觉示例框（免 CLIP，推荐冷启动）'],
        ['sam3_semantic', 'SAM 3 语义文本/示例（需 sam3.pt + clip）'],
        ['yoloe_text', 'YOLOE 英文文本提示（需 mobileclip）']
      ], state.zsEngine, function (e) { state.zsEngine = e.target.value; S.render(); }, '零样本引擎');

      var modelOpts = localModels.length
        ? localModels.map(function (m) { return [m.name, m.name + ' (' + m.family + ')']; })
        : [['yoloe-26s-seg.pt', 'yoloe-26s-seg.pt（请先在上方模型面板下载）'], ['sam3.pt', 'sam3.pt（请先放入 models/sam/）']];
      var chosenModel = state.zsModel || modelOpts[0][0];
      var modelSel = selectBox(modelOpts, chosenModel, function (e) { state.zsModel = e.target.value; }, '零样本模型');
      var confInput = input(state.zsConf, { type: 'number', min: '0.01', max: '1', step: '0.05', oninput: function (e) { state.zsConf = e.target.value; } });
      var fitSel = selectBox([['obb', '输出旋转框 OBB'], ['hbb', '输出水平框 HBB']], state.fitMode, function (e) { state.fitMode = e.target.value; }, '输出形状');

      var promptInputs = (state.zsEngine !== 'yoloe_vp') ? el('div', { class: 'smart-prompts-grid' }, (state.data.classes || []).map(function (clsName, idx) {
        return formField('类别 ' + idx + '（' + clsName + '）英文提示词', input(state.zsPrompts[idx] || '', {
          placeholder: '例如 metal heatsink / screw',
          oninput: function (e) { state.zsPrompts[idx] = e.target.value; }
        }));
      })) : null;

      var exText = state.zsExemplar
        ? ('已选示例：' + state.zsExemplar.item_name + ' · 类别 ' + (state.data.classes[state.zsExemplar.class_id] || state.zsExemplar.class_id))
        : '尚未框选视觉示例（点击右侧按钮后在画布上拖框标 1 个样例零件）';

      bodyNode = el('div', { class: 'smart-tab-body' }, [
        el('p', { class: 'help', text: '处理勾选素材（未勾选则处理所有未审核图）。结果统一存为“待审核”，已审核与确认无目标素材自动跳过。' }),
        el('div', { class: 'parts-toolbar' }, [
          engineSel, modelSel, formField('置信度', confInput), fitSel,
          btn('在当前画布框选视觉示例', function () {
            if (!state.editor) { S.toast('请先打开一张图片作为视觉示例参考图', 'warn'); return; }
            state.editor.setMode('exemplar');
            S.toast('请在画布上拖动框选 1 个样例零件', 'ok');
          }, 'btn-secondary', { disabled: locked }),
          btn('开始零样本批量标', function () {
            if (!A.smartBatchZeroshot) return;
            var ids = Object.keys(state.checked).filter(function (k) { return state.checked[k]; });
            var prompts = Object.keys(state.zsPrompts).map(function (k) {
              return { class_id: Number(k), text: String(state.zsPrompts[k] || '').trim() };
            }).filter(function (p) { return !!p.text; });
            action(task, function () {
              return save().then(function () {
                return A.smartBatchZeroshot(task.project, {
                  engine: state.zsEngine,
                  model: modelSel.value,
                  items: ids,
                  prompts: prompts,
                  visual_exemplar: state.zsExemplar,
                  conf: Number(state.zsConf) || 0.25,
                  fit_mode: state.fitMode
                });
              });
            }, function () {
              S.toast('零样本批量自动标注已启动', 'ok');
              scheduleSmartPoll();
              A.smartStatus(task.project, false).then(function (s) { if (currentContext(task)) { applySmartStatus(s, false); S.render(); } });
            });
          }, 'btn-primary', { disabled: locked || sjRunning }),
          sjRunning ? btn('停止智能任务', function () { A.smartStop(task.project).then(function () { S.toast('已请求停止', 'warn'); }); }) : null
        ]),
        el('p', { class: 'help', text: exText }),
        promptInputs
      ]);
    } else if (state.autoTab === 'track') {
      var trackModels = ((state.smart && state.smart.available_models) || []).filter(function (m) {
        return m.local && (m.family === 'sam3' || m.family === 'sam2');
      });
      var tOpts = trackModels.length
        ? trackModels.map(function (m) { return [m.name, m.name]; })
        : [['', '暂无本地 SAM 3 / SAM 2.1（请先在上方模型面板下载 sam2.1_b.pt）']];
      var tSel = selectBox(tOpts, state.trackModel || tOpts[0][0], function (e) { state.trackModel = e.target.value; }, '视频追踪模型');
      var maxF = input(state.trackMaxFrames, { type: 'number', min: '1', max: '600', oninput: function (e) { state.trackMaxFrames = e.target.value; } });
      bodyNode = el('div', { class: 'smart-tab-body' }, [
        el('p', { class: 'help', text: '将当前选中图作为首帧种子，向后按每块 ≤40 帧（8GB 显存安全窗口）自动追踪同拍摄来源组的后续连续帧并拟合 OBB。' }),
        el('div', { class: 'parts-toolbar' }, [
          tSel,
          formField('最多向后追踪帧数', maxF),
          btn('从当前帧向后追踪传播', function () {
            if (!A.smartBatchTrack || !current()) return;
            action(task, function () {
              return save().then(function () {
                return A.smartBatchTrack(task.project, {
                  seed_item_id: state.selected,
                  model: tSel.value,
                  max_frames: Number(state.trackMaxFrames) || 120,
                  fit_mode: state.fitMode
                });
              });
            }, function () {
              S.toast('连续帧追踪传播已启动', 'ok');
              scheduleSmartPoll();
              A.smartStatus(task.project, false).then(function (s) { if (currentContext(task)) { applySmartStatus(s, false); S.render(); } });
            });
          }, 'btn-primary', { disabled: locked || sjRunning }),
          sjRunning ? btn('停止追踪', function () { A.smartStop(task.project).then(function () { S.toast('已请求停止', 'warn'); }); }) : null
        ])
      ]);
    } else {
      bodyNode = preannotationBody(locked);
    }

    var progressBlock = (sj && (sj.status === 'running' || sj.status === 'done' || sj.status === 'failed' || sj.status === 'stopped'))
      ? el('div', { class: 'sp-mt-2' }, [
        sjRunning ? el('div', { class: 'progress' }, el('div', { class: 'progress-bar busy', style: 'width:' + Math.min(100, Math.round((sj.done || 0) * 100 / Math.max(1, sj.total || 1))) + '%' })) : null,
        el('p', { class: 'help', text: '最近智能批量任务：' + sj.id + ' (' + sj.engine + ') · ' + ({ running: '进行中 ' + (sj.done || 0) + '/' + (sj.total || 0), done: '已完成，写入 ' + (sj.applied || 0) + ' 张 / 跳过 ' + (sj.skipped || 0) + ' 张', failed: '失败：' + (sj.error || ''), stopped: '已停止' }[sj.status] || sj.status) })
      ])
      : null;

    return el('details', { class: 'card parts-preannotate', open: !!(activeJob() || sjRunning) }, [
      el('summary', { text: '智能自动标注（零样本冷启动 · 连续帧追踪 · 自训 YOLO-OBB 预标注）' }),
      tabs,
      bodyNode,
      progressBlock
    ]);
  }

  function preannotationBody(locked) {
    var task = context();
    var run = selectBox([['', '选择 PT 训练轮次']].concat(state.runs.map(function (r) { return [r.name, r.name]; })), '', null, '预标注使用的轮次');
    var model = selectBox([['best.pt', '本轮 best.pt'], ['last.pt', '本轮 last.pt']].concat(state.models.map(function (m) { return [m, m]; })), 'best.pt', null, '预标注模型');
    var conf = input('0.25', { type: 'number', min: '0.01', max: '1', step: '0.05' });
    var device = selectBox([['0', 'GPU 0'], ['cpu', 'CPU']], '0', null, '预标注计算设备');
    var j = state.job, jTotal = j ? (j.total || (j.items || []).length) : 0, jDone = j ? (j.done || 0) : 0;
    return el('div', { class: 'smart-tab-body' }, [
      el('p', { class: 'help', text: '处理勾选素材；未勾选时只处理当前图片。结果保存为待审核，已有已审核或无目标确认的图片将跳过。' }),
      el('div', { class: 'parts-toolbar' }, [
        run, model, formField('置信度', conf), device,
        btn('开始 YOLO 预标注', function () {
          var ids = Object.keys(state.checked).filter(function (k) { return state.checked[k]; });
          if (!ids.length && state.selected) ids = [state.selected];
          if (!ids.length) { S.toast('先选择素材', 'warn'); return; }
          save().then(function () {
            if (!currentContext(task)) return;
            return A.yoloPreannotate(task.project, { run: run.value, model: model.value, items: ids, conf: Number(conf.value), device: device.value, imgsz: 640 });
          }).then(function (job) {
            if (!currentContext(task) || !job) return;
            state.job = job; watchJob(job.id); S.toast('预标注已开始，请等待结果', 'ok'); S.renderJob(); S.render();
          }).catch(function (e) { if (currentContext(task)) S.toast(e.message, 'bad'); });
        }, 'btn-primary', { disabled: locked }),
        activeJob() ? btn('停止零件任务', stopJob) : null
      ]),
      j && j.status === 'running' && jTotal > 0 ? el('div', { class: 'progress sp-mt-2' }, el('div', { class: 'progress-bar busy', style: 'width:' + Math.min(100, Math.round(jDone * 100 / jTotal)) + '%' })) : null,
      el('p', { class: 'help', text: j ? '最近 YOLO 预标注：' + j.id + ' · ' + ({ running: '正在推理 (' + jDone + '/' + jTotal + ' 张)', done: '已完成', failed: '失败', stopped: '已停止' }[j.status] || j.status) : '' })
    ]);
  }

  function watchJob(id) {
    clearTimeout(state.jobTimer);
    var project = pid(), task = context();
    state.jobTimer = setTimeout(function () {
      if (!currentContext(task)) return;
      A.yoloValidation(project, id).then(function (job) {
        if (!currentContext(task)) return;
        state.job = job; S.renderJob();
        if (['running', 'queued', 'preparing'].indexOf(job.status) >= 0) watchJob(id);
        else {
          var skipped = job.skipped || [], reasons = Object.values(job.skip_reasons || {});
          var outcome = job.mode === 'preannotate'
            ? '预标注完成：已处理 ' + (job.applied || []).length + ' 张，跳过 ' + skipped.length + ' 张' + (reasons.length ? '。' + Array.from(new Set(reasons)).join('；') : '')
            : '零件验证完成，请检查结果';
          S.toast(job.status === 'done' ? outcome : '零件任务结束：' + (job.error || job.status), job.status === 'done' && !skipped.length ? 'ok' : 'warn');
          if (!dirty()) { state.draft = null; load(true).then(function () { if (currentContext(task) && S.state.view === 'palabel') S.render(); }); }
        }
      }).catch(function (e) { if (currentContext(task)) S.toast('零件任务状态读取失败：' + e.message, 'bad'); });
    }, 1200);
  }

  function render() {
    if (!live()) return page('零件 OBB 标注', '连接本机训练台后使用。', []);
    if (!ensure()) return page('零件 OBB 标注', '素材与标注', loadingBody());
    state.refs = {};
    // 进入标注页时自动检测训练状态并默认拉起 SAM 3（不可用时自动弹出可用模型清单）
    autoInitSmart();
    return page('零件 OBB 标注', '支持人工 3 点旋转框、Meta SAM 3 / SAM 2.1 点选秒出框、零样本批量标与连续帧追踪。', [
      renderSmartBanner(),
      el('div', { class: 'parts-toolbar' }, [
        btn('素材、质检与发布', function () { S.go('pdata'); }),
        btn(state.showIngest ? '收起导入素材' : '+ 上传素材 / 视频抽帧', function () { state.showIngest = !state.showIngest; S.render(); }, 'btn-primary'),
        btn('刷新素材', refresh),
        btn('勾选筛选结果', function () { filtered().forEach(function (i) { state.checked[i.id] = true; }); paintItemRows(); }),
        btn('清空勾选', function () { state.checked = {}; paintItemRows(); }),
        btn('删除当前素材', deleteCurrent)
      ]),
      state.showIngest ? renderQuickIngest() : null,
      el('div', { class: 'parts-workspace' }, [itemList(), editorPanel()]),
      renderAutoLabelHub()
    ]);
  }

  function deleteCurrent() {
    var item = current(), project = pid(), task = context();
    if (!item) return;
    if (!global.confirm('从工作区删除“' + item.name + '”？已发布版本保留。当前标注会先保存，保存失败时不会删除。')) return;
    save().then(function () {
      if (!currentContext(task) || dirty()) return null;
      return A.partsDelete(project, item.id, current().revision);
    }).then(function (result) {
      if (!result || !currentContext(task)) return;
      forgetDraft(); state.draft = null; state.selected = null; delete state.checked[item.id];
      applySnapshot(result); S.toast('素材已从工作区删除，已发布版本保留', 'ok');
    }).catch(function (e) { if (currentContext(task)) S.toast('无法删除素材：' + e.message + '。当前现场保留。', 'bad'); });
  }

  function openItem(id) {
    var project = pid(), task = context();
    var pending = state.pid === project && state.data ? save() : Promise.resolve();
    return pending.then(function () {
      if (!currentContext(task) || dirty()) return false;
      state.draft = null;
      return load(true).then(function () {
        if (!currentContext(task)) return false;
        if (state.error) { S.toast('素材刷新失败：' + state.error, 'bad'); return false; }
        state.selected = id;
        var item = current();
        if (!item) { S.toast('这张素材已不在当前工作区，请刷新素材列表', 'warn'); return false; }
        setDraft(item); S.go('palabel'); return true;
      });
    }).catch(function () { return false; });
  }

  function beforeNavigate() {
    if (state.saving || dirty()) {
      save().catch(function () { });
      S.toast('正在保存当前零件标注，请保存完成后再切换页面。', 'warn');
      return false;
    }
    // 离开零件标注页时释放常驻智能标注显存，下次进入再按需自动拉起默认 SAM 3
    if (S.state.view === 'palabel') {
      clearTimeout(state.smartPollTimer);
      state.smartTried = false;
      state.smartEncodedId = '';
      if (A.smartUnload && state.smart && state.smart.state === 'ready' && !(state.smart.batch_job && state.smart.batch_job.status === 'running')) {
        A.smartUnload(pid(), '离开零件标注页自动释放显存').catch(function () { });
      }
    }
    return true;
  }

  function reset() {
    generation++;
    clearTimeout(state.timer); clearTimeout(state.jobTimer); clearTimeout(state.smartPollTimer);
    state.job = null; state.pid = null; state.data = null; state.loading = false; state.selected = null;
    state.draft = null; state.saving = false; state.saveAgain = false; state.refs = {}; state.editor = null;
    state.checked = {}; state.quality = null; state.splits = {}; state.filter = ''; state.status = '';
    state.group = ''; state.category = ''; state.mediaClips = []; state.showIngest = false;
    state.smart = null; state.smartLoading = false; state.smartTried = false; state.smartPanelOpen = false;
    state.smartEncodedId = ''; state.zsExemplar = null;
  }

  /* 全局标注页高频快捷键（非输入框聚焦时触发） */
  global.addEventListener('keydown', function (e) {
    if (!S || !S.state || S.state.view !== 'palabel' || !state.data || !current()) return;
    var tag = e.target && e.target.tagName ? e.target.tagName.toUpperCase() : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) return;
    var ctrl = e.ctrlKey || e.metaKey;
    var k = (e.key || '').toLowerCase();
    if (e.code === 'Space' || (ctrl && e.key === 'Enter')) {
      e.preventDefault();
      reviewAndNext();
      return;
    }
    if (ctrl || e.altKey) return;
    if (k === 'a' || e.key === 'ArrowLeft') { e.preventDefault(); stepItem(-1, false); return; }
    if (k === 'd' || e.key === 'ArrowRight') { e.preventDefault(); stepItem(1, false); return; }
    if (k === 'c') { e.preventDefault(); copyPrevFrameBoxes(); return; }
    if (k === 'e') { e.preventDefault(); negativeAndNext(); return; }
    if (k >= '1' && k <= '9') {
      var idx = Number(k) - 1;
      if (idx < (state.data.classes || []).length) {
        e.preventDefault();
        if (state.refs.category) state.refs.category.value = String(idx);
        if (state.editor) state.editor.setClass(idx);
        S.toast('当前类别：' + (idx + 1) + '. ' + state.data.classes[idx], 'ok');
      }
      return;
    }
    if (k === 's' && state.editor) {
      e.preventDefault();
      if (state.smart && state.smart.training_locked) { S.toast('训练进行中，已禁用智能标注', 'warn'); return; }
      if (!smartReadyForInteractive()) { state.smartPanelOpen = true; S.toast('请先选择或下载可用智能标注模型', 'warn'); S.render(); return; }
      state.toolMode = 'smart';
      state.editor.setMode('smart');
      preEncodeCurrent();
    }
  });

  global.addEventListener('beforeunload', function (e) { if (dirty() || state.saving) { remember(); e.preventDefault(); e.returnValue = ''; } });
  global.OD.parts={renderData:renderData,reset:reset,openItem:openItem,beforeNavigate:beforeNavigate,activeJob:activeJob,prefetch:function(){if(live())return load(true);return Promise.resolve();}};
  global.OD.views.palabel = { render: render, beforeLeave: beforeNavigate, inspector: function () { return null; } };
})(window);
