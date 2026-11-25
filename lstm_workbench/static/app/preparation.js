/* 特征准备独立轮询，不重建页面，也不占用前台训练任务状态。 */
(function (global) {
  'use strict';
  var el = global.OD.dom.el, S = global.OD.shell;
  var busy = false, polling = false;
  var labels = { ready: '已就绪', pending: '待准备', queued: '排队中', preparing: '准备中', paused: '已暂停', failed: '失败', excluded: '未勾选' };
  function status() { return global.OD.data.preparation; }
  function paintCell(node) {
    var rows = (status() && status().clips) || [];
    var row = rows.filter(function (r) { return r.clip_id === node.getAttribute('data-feature-status'); })[0];
    node.textContent = row ? (labels[row.state] || row.state) : '待检查';
    node.className = 'badge ' + (row && row.state === 'ready' ? 'ok' : row && row.state === 'failed' ? 'bad' : 'plain');
    node.title = row && row.error || '';
  }
  function cell(id) {
    var node = el('span', { class: 'badge plain', 'data-feature-status': id });
    paintCell(node);
    return node;
  }
  function action(which) {
    if (busy || !global.OD.api.available || !S.state.projectId) return;
    var projectId = S.state.projectId;
    busy = true; paint();
    global.OD.api.prepare(projectId, which).then(function (value) {
      if (S.state.projectId === projectId) global.OD.data.preparation = value;
    }).catch(function (err) { S.toast('特征准备：' + err.message, 'bad'); }).finally(function () { busy = false; paint(); });
  }
  function render() {
    var node = el('section', { class: 'card feature-preparation', 'aria-label': '特征准备', 'data-preparation': '' }, [
      el('header', null, [el('h2', { text: '特征准备' }), el('span', { class: 'badge plain', 'data-prep-count': '' })]),
      el('p', { class: 'help sp-mt-3', text: '保存标注后自动准备已勾选的视频。有效特征直接复用，修改动作标注不会重抽。' }),
      el('p', { class: 'help sp-mt-2', 'data-prep-detail': '', role: 'status' }),
      el('div', { class: 'progress sp-mt-2', role: 'progressbar', 'aria-label': '后台特征准备进度', 'aria-valuemin': 0, 'aria-valuemax': 100 },
        el('div', { class: 'progress-bar', 'data-prep-fill': '' })),
      el('div', { class: 'row sp-mt-3' }, [
        el('button', { class: 'btn btn-secondary btn-sm', type: 'button', 'data-prep-start': '', onclick: function () { action('prepare'); } }, '准备全部勾选视频'),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', 'data-prep-pause': '', onclick: function () { action('pause'); } }, '暂停准备')
      ])
    ]);
    paintCard(node);
    return node;
  }
  function paintCard(node) {
    var p = status(), active = p && p.active;
    node.querySelector('[data-prep-count]').textContent = p ? p.ready + ' / ' + p.total + ' 条就绪' : '等待连接';
    var message = p ? p.blocked_reason : '连接训练台后显示准备状态';
    if (!message && p) {
      if (p.paused) message = '后台准备已暂停；训练时仍会补齐所需特征。';
      else if (active) message = active.clip_id + '：' + active.done + ' / ' + (active.total || '—') + ' 帧';
      else if (p.failed) message = p.failed + ' 条准备失败，点击重试。';
      else if (p.queued) message = p.queued + ' 条等待准备，当前任务结束后继续。';
      else message = p.ready === p.total && p.total ? '全部已就绪，训练可直接加载缓存。' : '保存当前视频标注或点击准备按钮即可提前处理。';
    }
    node.querySelector('[data-prep-detail]').textContent = message;
    var preparing = p && active && p.clips.some(function (row) { return row.clip_id === active.clip_id && row.state === 'preparing'; });
    var part = preparing && active.total ? Math.min(1, active.done / active.total) : 0;
    var percent = p && p.total ? Math.min(100, 100 * (p.ready + part) / p.total) : 0;
    node.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', Math.round(percent));
    node.querySelector('[data-prep-fill]').style.width = percent + '%';
    var start = node.querySelector('[data-prep-start]');
    start.disabled = busy || !p || !p.total || !!p.blocked_reason;
    start.textContent = p && p.paused ? '继续准备' : p && p.failed ? '重试未就绪视频' : '准备全部勾选视频';
    node.querySelector('[data-prep-pause]').disabled = busy || !p || p.paused || !(p.active || p.queued);
  }
  function paint() {
    document.querySelectorAll('[data-preparation]').forEach(paintCard);
    document.querySelectorAll('[data-feature-status]').forEach(paintCell);
    var train = global.OD.views && global.OD.views.train;
    if (train && train.preparationUpdated) train.preparationUpdated();
  }
  setInterval(function () {
    var projectId = S.state.projectId;
    if (polling || !global.OD.api.available || !projectId) return;
    polling = true;
    global.OD.api.preparationState(projectId).then(function (value) {
      if (S.state.projectId !== projectId) return;
      global.OD.data.preparation = value;
      paint();
    }).catch(function () {
      if (S.state.projectId === projectId) document.querySelectorAll('[data-prep-detail]').forEach(function (node) { node.textContent = '无法读取准备状态，正在重连…'; });
    }).finally(function () { polling = false; });
  }, 1200);
  global.OD.preparation = { render: render, cell: cell };
})(window);
