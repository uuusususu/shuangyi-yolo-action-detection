/* 零件模型（YOLO）三页：零件数据、零件训练、模型库。
   训练跑在训练台起的独立进程里；这里只发请求、轮询进度、画曲线。
   复用全站同一套卡片、表格、徽章、曲线组件，不另起风格。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;

  /* 本页自己的状态。切工程时整体丢弃（projectId 变了就重拉）。 */
  var cache = { projectId: null, info: null, env: null, detail: null, log: [], logNext: 0, loading: false, error: '' };
  var pollTimer = null;
  var generation = 0, loadSerial = 0, detailSerial = 0;

  function api() { return global.OD.api; }
  function pid() { return S.state.projectId; }
  function live() { return global.OD.live.isLive() && !!pid(); }

  function offlineCard() {
    return el('div', { class: 'callout warn' }, [
      icon('info', 16),
      el('div', { text: '零件模型训练需要连着训练台。现在是离线示例，这一页没有数据。' })
    ]);
  }

  /* ---------- 数据加载 ---------- */

  function load(force) {
    if (!live()) return Promise.resolve(null);
    if (cache.projectId !== pid()) {
      generation++;
      cache = { projectId: pid(), info: null, env: cache.env, detail: null, log: [], logNext: 0, loading: false, error: '' };
    }
    if (cache.loading && !force) return Promise.resolve(cache.info);
    cache.loading = true;
    var id = pid(), epoch = generation, serial = ++loadSerial;
    return api().yolo(id).then(function (info) {
      if (id !== pid() || epoch !== generation || serial !== loadSerial) return null;
      cache.info = info;
      cache.error = '';
      var current = info.active || (info.runs[0] && info.runs[0].name) || '';
      if (current && (!cache.detail || cache.detail.name !== current || info.active)) {
        return loadRun(current).then(function () { return info; });
      }
      return info;
    }).catch(function (err) {
      if (id === pid() && epoch === generation && serial === loadSerial) cache.error = err.message;
    }).then(function (result) {
      if (id === pid() && epoch === generation && serial === loadSerial) cache.loading = false;
      return result;
    });
  }

  function loadRun(name) {
    var id = pid(), epoch = generation, serial = ++detailSerial;
    var switched = !cache.detail || cache.detail.name !== name;
    if (switched) { cache.log = []; cache.logNext = 0; }
    return Promise.all([
      api().yoloRun(id, name),
      api().yoloLog(id, name, cache.logNext)
    ]).then(function (out) {
      if (id !== pid() || epoch !== generation || serial !== detailSerial) return;
      cache.detail = out[0];
      cache.log = cache.log.concat(out[1].lines || []).slice(-600);
      cache.logNext = out[1].next || cache.logNext;
    });
  }

  function loadEnv(refresh) {
    if (!live()) return Promise.resolve(null);
    if (cache.env && !refresh) return Promise.resolve(cache.env);
    return api().yoloEnv(refresh).then(function (env) { cache.env = env; return env; })
      .catch(function (err) { cache.env = { ok: false, error: err.message }; return cache.env; });
  }

  /* 训练中每 2 秒刷新一次。底部任务面板在任何页面都要跟着走，所以切到别的页也继续轮询；
     只有停在零件训练页时才重画进度卡片、曲线和日志（不整页重画，表单里正在填的不会丢）。 */
  function watch() {
    if (pollTimer) return;
    pollTimer = setInterval(function () {
      if (!cache.info || !cache.info.active) { stopWatch(); return; }
      load(true).then(function () {
        S.renderJob();
        if (S.state.view === 'ptrain') paintLive();
        if (!cache.info || !cache.info.active) {
          stopWatch();
          var d = cache.detail;
          if (d) S.toast('零件模型训练' + statusText(d.status), d.status === 'done' ? 'ok' : 'warn');
          if (['pdata', 'ptrain', 'library'].indexOf(S.state.view) >= 0) S.render();
          else S.renderRail();
        }
      });
    }, 2000);
  }

  /* 报给底部任务面板：正在训练的那一轮跑到第几轮、用了多久。 */
  function activeJob() {
    if (!cache.info || !cache.info.active || cache.projectId !== pid()) return null;
    var d = cache.detail && cache.detail.name === cache.info.active ? cache.detail : null;
    var started = d && d.started_at;
    return {
      name: cache.info.active,
      done: d ? d.epoch || 0 : 0,
      total: d ? d.epochs || 0 : 0,
      elapsed: started ? Math.max(0, Date.now() / 1000 - started) : 0,
      stop: doStop
    };
  }

  function stopWatch() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

  function statusText(status) {
    return { training: '进行中', done: '已完成', failed: '失败', stopped: '已停止' }[status] || status || '—';
  }
  function statusKind(status) {
    return { training: 'busy', done: 'ok', failed: 'bad', stopped: 'warn' }[status] || 'plain';
  }

  function fmt(value, digits) { return D.fmtNum(value, digits === undefined ? 3 : digits); }
  function when(seconds) {
    if (!seconds) return '—';
    var d = new Date(seconds * 1000);
    function p(n) { return (n < 10 ? '0' : '') + n; }
    return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes());
  }

  /* 首次进入或切工程时先拉数据，拉完再画一次。 */
  function ensure() {
    if (!live()) return false;
    if (cache.projectId === pid() && cache.info) return true;
    if (cache.projectId === pid() && cache.error) return false;
    if (!cache.loading) {
      Promise.all([load(true), loadEnv(false)]).then(function () {
        if (['pdata', 'ptrain', 'library'].indexOf(S.state.view) >= 0) S.render();
      });
    }
    return false;
  }

  function loadingView(title) {
    return page(title, '', [el('div', { class: 'card' }, [
      el('p', { class: 'help', text: cache.error || '正在读取…' }),
      cache.error ? el('button', {type:'button',class:'btn btn-primary',onclick:function(){
        var project=pid();load(true).then(function(){if(project===pid())S.render();});
      }}, '重试读取') : null
    ])]);
  }

  function page(title, lead, body) {
    return el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('div', { class: 'mod-tag', 'data-mod': 'yolo', text: '零件' }),
        el('h1', { text: title }),
        lead ? el('p', { text: lead }) : null
      ])
    ].concat(body)));
  }

  /* ============ 零件数据 ============ */

  function renderData() {
    return global.OD.parts ? global.OD.parts.renderData(renderLegacyData()) : renderLegacyData();
  }

  function renderLegacyData() {
    if (!live()) return page('零件数据', '', [offlineCard()]);
    if (!ensure()) return loadingView('零件数据');
    var ds = cache.info.dataset || {};
    var body = [];

    body.push(el('section', { class: 'card' }, [
      el('header', null, [el('h2', { text: '导入数据集' })]),
      el('p', { class: 'help', text: '导入外部 Ultralytics 数据集：选择 data.yaml 或所在目录，整份复制进工程。旧格式缺少 val 时按图片抽取验证集，无法保证同源隔离；正式比较请在原生标注工作区按来源发布固定版本。' }),
      el('div', { class: 'row sp-mt-3' }, [
        el('input', { class: 'form-control mono', id: 'yolo-src', placeholder: '例如 E:\\datasets\\sanreipian\\data.yaml' }),
        el('button', { class: 'btn btn-primary', type: 'button', id: 'yolo-import', onclick: doImport }, ds.exists ? '重新导入' : '导入')
      ]),
      ds.exists ? el('p', { class: 'help sp-mt-2', text: '重新导入会替换本工程现有的零件图片和标注，已训练的轮次保留。' }) : null
    ]));

    if (!ds.exists) {
      body.push(el('div', { class: 'callout warn' }, [icon('info', 16), el('div', { text: '这个工程还没有零件数据。导入后才能训练零件模型。' })]));
      return page('外部数据集导入', '兼容已有 YOLO 数据集；新素材请在零件标注工作区完成审核与发布。', body);
    }

    var total = (ds.class_counts || []).reduce(function (a, b) { return a + b; }, 0) || 1;
    var low = [];
    body.push(el('section', { class: 'card' }, [
      el('header', null, [el('h2', { text: '数据概况' }), el('span', { class: 'end' }, el('span', { class: 'badge plain mono', text: ds.task }))]),
      el('div', { class: 'stat-row' }, [
        stat('训练图片', ds.train_images), stat('验证图片', ds.val_images),
        stat('类别', (ds.names || []).length), stat('标注框', ds.boxes)
      ]),
      el('div', { class: 'tbl-wrap sp-mt-3' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [el('th', { text: '#' }), el('th', { text: '类别' }), el('th', { text: '标注框' }), el('th', { text: '占比' })])),
        el('tbody', null, (ds.names || []).map(function (name, index) {
          var count = (ds.class_counts || [])[index] || 0;
          var share = count / total;
          if (count < 50) low.push(name);
          return el('tr', null, [
            el('td', { class: 'mono', text: index }),
            el('td', { text: name }),
            el('td', { class: 'mono', text: count }),
            el('td', null, el('div', { class: 'share' }, [
              el('span', { class: 'share-bar', style: 'width:' + Math.max(2, Math.round(share * 100)) + '%' }),
              el('span', { class: 'mono sp-fs-12', text: Math.round(share * 100) + '%' })
            ]))
          ]);
        }))
      ])),
      ds.source ? el('p', { class: 'help sp-mt-2 mono', text: '来源：' + ds.source }) : null
    ]));
    if (low.length) {
      body.push(el('div', { class: 'callout warn' }, [icon('alert', 15), el('div', {
        text: low.join('、') + ' 标注框少于 50 个，这些类别很可能检不准。少的类别多补一些图。'
      })]));
    }
    if (ds.val_images < 20) {
      body.push(el('div', { class: 'callout warn' }, [icon('alert', 15), el('div', {
        text: '验证图片只有 ' + ds.val_images + ' 张，mAP 会很不稳定。验证集最好有 50 张以上。'
      })]));
    }
    return page('外部数据集导入', '兼容已有 YOLO 数据集；来源不明确的验证分数只作参考。', body);
  }

  function stat(label, value) {
    return el('div', { class: 'stat' }, [el('div', { class: 'stat-v mono', text: value === undefined || value === null ? '—' : String(value) }), el('div', { class: 'stat-k', text: label })]);
  }

  function doImport() {
    var input = document.getElementById('yolo-src');
    var source = input ? input.value.trim() : '';
    if (!source) { S.toast('先填数据集路径', 'warn'); return; }
    var button = document.getElementById('yolo-import');
    if (button) { button.disabled = true; button.textContent = '导入中…'; }
    api().yoloImport(pid(), source).then(function () {
      S.toast('零件数据已导入', 'ok');
      return load(true);
    }).catch(function (err) {
      S.toast('导入失败：' + err.message, 'bad');
    }).then(function () { S.render(); });
  }

  /* ============ 零件训练 ============ */

  var FIELDS = [
    { key: 'epochs', label: '轮数', value: 100, help: '数据少时 100～200。' },
    { key: 'imgsz', label: '输入尺寸', value: 640, help: '和产线推理一致，默认 640。' },
    { key: 'batch', label: '批大小', value: 16, help: '显存不够就减半；填 -1 让 YOLO 自动选。' },
    { key: 'device', label: '设备', value: '0', help: '0 = 第一块显卡；cpu = 不用显卡（很慢）。' }
  ];

  function renderTrain() {
    if (!live()) return page('零件训练', '', [offlineCard()]);
    if (!ensure()) return loadingView('零件训练');
    var info = cache.info;
    var ds = info.dataset || {};
    var versions = (info.versions || []).slice().sort(function (a, b) { return (b.created_at || 0) - (a.created_at || 0); });
    var selectedVersion = cache.selectedVersion;
    if (selectedVersion === undefined || (!ds.exists && !versions.some(function(v){return v.id===selectedVersion;}))) selectedVersion = versions.length ? versions[0].id : '';
    cache.selectedVersion = selectedVersion;
    var selectedData = versions.filter(function (v) { return v.id === selectedVersion; })[0];
    var env = cache.env || {};
    var running = !!info.active;
    var body = [];

    if (!cache.env) {
      // 第一次探测要启动 YOLO 的 Python 并导入 torch，几秒钟，期间不报红。
      body.push(el('div', { class: 'callout info' }, [icon('info', 16), el('div', { text: '正在检测 YOLO 训练环境…' })]));
      loadEnv(false).then(function () { if (S.state.view === 'ptrain') S.render(); });
    } else if (!env.ok) {
      body.push(el('div', { class: 'callout bad' }, [icon('alert', 16), el('div', null, [
        el('strong', { text: 'YOLO 训练环境不可用。 ' }), env.error || '正在检测…',
        el('div', { class: 'sp-mt-2' }, el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () {
          loadEnv(true).then(function () { S.render(); });
        } }, '重新检测'))
      ])]));
    }
    if (!ds.exists && !versions.length) {
      body.push(el('div', { class: 'callout warn' }, [icon('info', 16), el('div', null, [
        '还没有零件数据。',
        el('button', { class: 'btn btn-ghost btn-sm sp-ml-2', type: 'button', onclick: function () { S.go('pdata'); } }, '去导入')
      ])]));
    }

    var task = selectedVersion ? 'obb' : (ds.task || 'obb');
    var defaults = (info.defaults && info.defaults.model) || {};
    var form = el('section', { class: 'card' }, [
      el('header', null, [el('h2', { text: '训练参数' }),
        el('span', { class: 'end' }, env.ok ? el('span', { class: 'badge ' + (env.cuda ? 'ok' : 'warn'), text: env.cuda ? env.gpu : '没有可用显卡' }) : null)]),
      el('div', { class: 'grid c2' }, [
        el('div', { class: 'field' }, [
          el('label', { class: 'form-label', for: 'y-dataset-version', text: '数据集版本' }),
          el('select', { class: 'form-select mono', id: 'y-dataset-version', disabled: running, onchange: function (event) { cache.selectedVersion = event.target.value; } },
            (ds.exists ? [el('option', { value: '', selected: !selectedVersion, text: '外部导入数据（兼容模式）' })] : []).concat(versions.map(function (version) {
              return el('option', { value: version.id, selected: version.id === selectedVersion, text: version.id + ' · ' + ((version.classes || []).length) + ' 类' });
            }))),
          el('div', { class: 'help', text: selectedData ? '使用已审核且按来源分组的固定快照，后续标注修改不影响本轮数据。' : '外部数据将为本轮冻结快照；来源不明时无法保证训练与验证独立。' })
        ]),
        el('div', { class: 'field' }, [
          el('label', { class: 'form-label', for: 'y-model', text: '底模' }),
          el('input', { class: 'form-control mono', id: 'y-model', value: (cache.form || {}).model === undefined ? defaults[task] || '' : cache.form.model, disabled: running, list: 'y-bases', oninput:function(e){cache.form=cache.form||{};cache.form.model=e.target.value;} }),
          el('datalist', { id: 'y-bases' }, ((info.defaults && info.defaults.bases) || []).map(function (name) {
            return el('option', { value: name });
          })),
          el('div', { class: 'help', text: '可选 models\\yolo_base 里已有的底模（不联网），也可以填官方名（首次要联网下载）或本机 .pt 路径（在旧模型上接着训）。' })
        ])
      ].concat(FIELDS.map(function (f) {
        return el('div', { class: 'field' }, [
          el('label', { class: 'form-label', for: 'y-' + f.key, text: f.label }),
          el('input', { class: 'form-control mono', id: 'y-' + f.key, value: (cache.form || {})[f.key] === undefined ? (f.key === 'device' && env.ok && !env.cuda ? 'cpu' : f.value) : cache.form[f.key], disabled: running, oninput:function(e){cache.form=cache.form||{};cache.form[f.key]=e.target.value;} }),
          el('div', { class: 'help', text: f.help })
        ]);
      }))),
      el('details', { class: 'sp-mt-3' }, [
        el('summary', { text: '高级参数' }),
        el('div', { class: 'field sp-mt-2' }, [
          el('input', { class: 'form-control mono', id: 'y-extra', value:(cache.form || {}).extra || '', placeholder: '例如 lr0=0.005 mosaic=0.5 patience=50', disabled: running, oninput:function(e){cache.form=cache.form||{};cache.form.extra=e.target.value;} }),
          el('div', { class: 'help', text: 'Ultralytics 训练参数，空格分隔的 key=value。数据、模型、任务、续训和输出目录由训练台管理，不能在此覆盖。' })
        ])
      ]),
      el('div', { class: 'row sp-mt-3' }, [
        el('button', {
          class: 'btn btn-primary', type: 'button', id: 'y-start',
          disabled: running || (!ds.exists && !versions.length) || !env.ok, onclick: doTrain
        }, running ? '训练中…' : '开始训练'),
        running ? el('button', { class: 'btn btn-danger', type: 'button', onclick: doStop }, [icon('stop', 14), ' 停止']) : null
      ])
    ]);
    body.push(form);

    body.push(el('div', { id: 'y-live' }, liveSection()));
    body.push(historySection());
    if (running) watch();
    return page('零件训练', '训练零件检测模型。训完导出 ONNX 进模型库，动作训练和产线都从库里选。', body);
  }

  function readParams() {
    var params = { task: (cache.info.dataset || {}).task || 'obb' };
    var version = document.getElementById('y-dataset-version');
    params.dataset_version = version ? version.value : '';
    if (params.dataset_version) params.task = 'obb';
    var model = document.getElementById('y-model');
    params.model = model ? model.value.trim() : '';
    FIELDS.forEach(function (f) {
      var node = document.getElementById('y-' + f.key);
      params[f.key] = node ? node.value.trim() : f.value;
    });
    var extra = document.getElementById('y-extra');
    params.extra = extra ? extra.value.trim() : '';
    return params;
  }

  function doTrain() {
    var params = readParams();
    var button = document.getElementById('y-start');
    if (button) { button.disabled = true; button.textContent = '启动中…'; }
    api().yoloTrain(pid(), params).then(function () {
      S.toast('零件模型开始训练', 'ok');
      cache.detail = null;
      return load(true);
    }).catch(function (err) {
      S.toast('启动失败：' + err.message, 'bad');
    }).then(function () { S.render(); });
  }

  function doStop() {
    S.confirm({
      title: '停止零件模型训练？',
      body: '已经保存的 best.pt 会留着，可以照样导出。',
      ok: '停止', danger: true,
      onOk: function () {
        api().stop(pid()).then(function () {
          S.toast('已发出停止', 'warn');
          setTimeout(function () { load(true).then(function () { S.render(); }); }, 1500);
        }).catch(function (err) { S.toast('停止失败：' + err.message, 'bad'); });
      }
    });
  }

  var artifactFilter = 'all';

  function fmtSpan(sec) {
    if (sec === null || sec === undefined || !isFinite(sec)) return '—';
    var s = Math.max(0, Math.round(sec));
    var m = Math.floor(s / 60), r = s % 60;
    if (m >= 60) return Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
    return m ? (m + 'm ' + r + 's') : (r + 's');
  }

  function artifactUrl(runName, item) {
    return api().yoloFileUrl(pid(), runName, item.name) + '?v=' + (item.mtime || 0);
  }

  function artifactsGallery(d) {
    var artifacts = d.artifacts || [];
    var valPairs = d.val_pairs || [];
    if (!artifacts.length && !valPairs.length) {
      return el('div', { class: 'callout info sp-mt-3' }, [
        icon('info', 15),
        el('div', { text: d.status === 'training'
          ? '正在等待 YOLO 生成首轮数据分布图（labels.jpg、train_batch0.jpg）；训练完成后会自动展示混淆矩阵、PR/F1 曲线和验证集真值对比图。'
          : '本轮目录下暂无可视化图表文件。' })
      ]);
    }

    var lightboxItems = artifacts.map(function (a) {
      return { name: a.name, label: a.label || a.name, url: artifactUrl(d.name, a), category: a.category };
    });

    function openLightboxByName(fname) {
      if (!S.lightbox) return;
      var idx = 0;
      for (var i = 0; i < lightboxItems.length; i++) {
        if (lightboxItems[i].name === fname) { idx = i; break; }
      }
      S.lightbox(lightboxItems, idx);
    }

    var tabs = [
      ['all', '全部图表 (' + artifacts.length + ')'],
      ['eval', '评估与混淆矩阵'],
      ['val_pair', '验证集真值 vs 预测对比 (' + valPairs.length + ')'],
      ['data_train', '标签分布与增强批次']
    ];

    var tabBar = el('div', { class: 'seg-group', role: 'group', 'aria-label': '筛选官方训练图表' }, tabs.map(function (t) {
      return el('button', {
        type: 'button', 'aria-pressed': String(artifactFilter === t[0]),
        onclick: function () { artifactFilter = t[0]; paintLive(); }
      }, t[1]);
    }));

    var content = [];

    if ((artifactFilter === 'all' || artifactFilter === 'val_pair') && valPairs.length) {
      content.push(el('div', { class: 'yolo-val-pairs sp-mt-3' }, valPairs.map(function (pair) {
        var v = '?v=' + (pair.mtime || 0);
        var lblUrl = pair.labels ? api().yoloFileUrl(pid(), d.name, pair.labels) + v : '';
        var predUrl = pair.pred ? api().yoloFileUrl(pid(), d.name, pair.pred) + v : '';
        return el('div', { class: 'yolo-pair-card' }, [
          el('div', { class: 'yolo-pair-head' }, [
            el('strong', { text: '验证集批次 #' + pair.batch + ' · 真值与模型旋转框预测对照' }),
            el('span', { class: 'sp-fs-12 sp-dim', text: '点击任意图片可高清缩放查看细节' })
          ]),
          el('div', { class: 'yolo-pair-grid' }, [
            lblUrl ? el('figure', { class: 'yolo-plot-item', onclick: function () { openLightboxByName(pair.labels); } }, [
              el('img', { src: lblUrl, loading: 'lazy', alt: pair.labels }),
              el('figcaption', null, [el('span', { class: 'badge ok', text: '人工标注真值' }), el('span', { class: 'mono sp-fs-11', text: pair.labels })])
            ]) : null,
            predUrl ? el('figure', { class: 'yolo-plot-item', onclick: function () { openLightboxByName(pair.pred); } }, [
              el('img', { src: predUrl, loading: 'lazy', alt: pair.pred }),
              el('figcaption', null, [el('span', { class: 'badge busy', text: '模型预测结果' }), el('span', { class: 'mono sp-fs-11', text: pair.pred })])
            ]) : null
          ])
        ]);
      })));
    }

    if (artifactFilter !== 'val_pair') {
      var shown = artifacts.filter(function (a) {
        if (artifactFilter === 'all') return a.category !== 'val_batch';
        if (artifactFilter === 'eval') return a.category === 'eval';
        if (artifactFilter === 'data_train') return a.category === 'data' || a.category === 'train_batch';
        return true;
      });
      if (shown.length) {
        content.push(el('div', { class: 'yolo-plot-grid sp-mt-3' }, shown.map(function (a) {
          var url = artifactUrl(d.name, a);
          var catBadge = { eval: ['ok', '评估'], data: ['plain', '分布'], train_batch: ['warn', '增强'], val_batch: ['busy', '验证'] }[a.category] || ['plain', '图表'];
          return el('figure', { class: 'yolo-plot-item', onclick: function () { openLightboxByName(a.name); } }, [
            el('img', { src: url, loading: 'lazy', alt: a.label || a.name }),
            el('figcaption', null, [
              el('span', { class: 'badge ' + catBadge[0], text: catBadge[1] }),
              el('strong', { text: a.label || a.name }),
              el('small', { class: 'mono', text: D.fmtBytes ? D.fmtBytes(a.bytes) : a.name })
            ])
          ]);
        })));
      } else if (artifactFilter !== 'all') {
        content.push(el('p', { class: 'help sp-mt-3', text: '当前分类下还没有生成对应的图表（评估曲线与混淆矩阵通常在训练收尾验证时生成）。' }));
      }
    }

    return el('div', { class: 'yolo-gallery-wrap sp-mt-4' }, [
      el('div', { class: 'yolo-gallery-head' }, [
        el('h3', { text: 'Ultralytics 官方诊断与评估图表（点击放大缩放）' }),
        tabBar
      ])
    ].concat(content));
  }

  /* 进度、曲线、官方可视化画廊、日志。训练中定时只重画这一块。 */
  function liveSection() {
    var d = cache.detail;
    if (!d) return el('section', { class: 'card' }, el('p', { class: 'help', text: '还没有训练记录。' }));
    var pct = d.epochs ? Math.min(100, Math.round(100 * d.epoch / d.epochs)) : 0;
    var series = d.series || {};
    var keys = Object.keys(series);
    function pick(prefix) { for (var i = 0; i < keys.length; i++) if (keys[i].indexOf(prefix) === 0) return series[keys[i]]; return []; }
    var x = (series.epoch || []).map(function (v, i) { return v === null ? i + 1 : v; });
    var charts = el('div', { class: 'chart-grid sp-mt-3' }, [
      chartBox('验证 mAP 精度', function (node) {
        return lineChart(node, x, [['mAP50', pick('metrics/mAP50(')], ['mAP50-95', pick('metrics/mAP50-95(')]], 0, 1);
      }),
      chartBox('精确率与召回率 (P / R)', function (node) {
        return lineChart(node, x, [['Precision', pick('metrics/precision(')], ['Recall', pick('metrics/recall(')]], 0, 1);
      }),
      chartBox('定位 / 分类 / 角度分布损失', function (node) {
        return lineChart(node, x, [
          ['训练 box', pick('train/box_loss')], ['训练 cls', pick('train/cls_loss')], ['训练 dfl', pick('train/dfl_loss')],
          ['验证 box', pick('val/box_loss')], ['验证 cls', pick('val/cls_loss')], ['验证 dfl', pick('val/dfl_loss')]
        ]);
      }),
      chartBox('学习率衰减曲线', function (node) {
        return lineChart(node, x, [['lr/pg0', pick('lr/pg0')], ['lr/pg1', pick('lr/pg1')]]);
      })
    ]);
    var timeText = d.status === 'training'
      ? (fmtSpan(d.elapsed_seconds) + (d.eta_seconds ? ' / 剩 ' + fmtSpan(d.eta_seconds) : ''))
      : fmtSpan(d.elapsed_seconds);

    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '第 ' + d.name + ' 轮' }),
        el('span', { class: 'end' }, el('span', { class: 'badge ' + statusKind(d.status), text: statusText(d.status) }))
      ]),
      el('div', { class: 'progress-line' }, [
        el('div', { class: 'progress', role: 'progressbar', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 },
          el('div', { class: 'progress-bar ' + statusKind(d.status), style: 'width:' + pct + '%' })),
        el('span', { class: 'mono sp-fs-12', text: d.epoch + ' / ' + (d.epochs || '—') + ' 轮 (' + pct + '%)' })
      ]),
      el('div', { class: 'stat-row sp-mt-3' }, [
        stat('最佳轮次', d.best_epoch || '—'),
        stat('mAP50', fmt(d.map50)),
        stat('mAP50-95', fmt(d.map5095)),
        stat('精确率 P', fmt(d.precision)),
        stat('召回率 R', fmt(d.recall)),
        stat(d.status === 'training' ? '耗时 / 预计剩余' : '总训练耗时', timeText),
        stat('ONNX', d.exported ? '已导出' : '—')
      ]),
      d.error ? el('div', { class: 'callout bad sp-mt-3' }, [icon('alert', 15), el('div', { class: 'mono sp-fs-12', text: d.error })]) : null,
      exportBar(d),
      x.length ? charts : el('p', { class: 'help sp-mt-3', text: '第一轮跑完后出实时折线图。YOLO 首轮会先扫描缓存图片并生成下方的数据分布与批次增强图。' }),
      artifactsGallery(d),
      el('details', { class: 'sp-mt-3' }, [
        el('summary', { text: '训练追溯与数据指纹' }),
        el('pre', { class: 'logbox', text: JSON.stringify({
          dataset_version: (d.dataset || {}).version || (d.params || {}).dataset_version || '',
          dataset: d.dataset || d.data_snapshot || d.traceability || null,
          model: d.model || d.model_snapshot || null,
          environment: d.environment || null,
          params: d.params || {}, has_last: !!d.has_last, resumable: !!d.resumable
        }, null, 2) })
      ]),
      el('details', { class: 'sp-mt-3', open: d.status === 'failed' || d.status === 'training' ? true : null }, [
        el('summary', { text: '训练实时终端日志' }),
        el('pre', { class: 'logbox', id: 'y-log', text: cache.log.slice(-200).join('\n') || '（等待子进程输出…）' })
      ])
    ]);
  }

  function exportBar(d) {
    if (!d.has_best || d.status === 'training') return null;
    var exported = d.exported;
    var inUse = exported && (global.OD.data.settings || {}).yolo_path === exported.file;
    return el('div', { class: 'callout ' + (exported ? 'ok' : 'info') + ' sp-mt-3' }, [
      icon(exported ? 'check' : 'info', 15),
      el('div', { class: 'grow' }, exported
        ? ['已导出 ', el('span', { class: 'mono', text: exported.file }), inUse ? '，本工程正在用这个零件模型。' : '，还没在本工程启用。']
        : '训练结束。导出 ONNX 后才能给动作训练和产线用。'),
      el('div', { class: 'row tight' }, [
        exported ? null : el('button', { class: 'btn btn-primary btn-sm', type: 'button', onclick: function (e) { doExport(d.name, e.target); } }, '导出 ONNX'),
        exported && !inUse ? el('button', { class: 'btn btn-primary btn-sm', type: 'button', onclick: function (e) { doUse(d.name, e.target); } }, '在本工程启用') : null
      ])
    ]);
  }

  function doExport(run, button) {
    if (button) { button.disabled = true; button.textContent = '导出中…'; }
    api().yoloExport(pid(), run).then(function (out) {
      S.toast('已导出 ' + out.export.file, 'ok');
      return load(true).then(function () { return global.OD.live.refreshLibrary(); });
    }).catch(function (err) {
      S.toast('导出失败：' + err.message, 'bad');
    }).then(function () { S.render(); });
  }

  function doUse(run, button) {
    if (button) { button.disabled = true; button.textContent = '检查中…'; }
    api().yoloUse(pid(), run).then(function () {
      S.toast('本工程已换用新的零件模型。动作特征会按新模型重新抽。', 'ok');
      return global.OD.live.reload();
    }).catch(function (err) {
      S.toast('启用失败：' + err.message, 'bad');
    }).then(function () { S.render(); });
  }

  function historySection() {
    var runs = (cache.info && cache.info.runs) || [];
    if (!runs.length) return null;
    var current = cache.detail && cache.detail.name;
    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '历史轮次' })),
      el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, ['轮次', '状态', '轮数', 'mAP50', 'mAP50-95', 'ONNX', ''].map(function (t) { return el('th', { text: t }); }))),
        el('tbody', null, runs.map(function (r) {
          return el('tr', { class: r.name === current ? 'is-on' : null }, [
            el('td', { class: 'mono', text: r.name }),
            el('td', null, el('span', { class: 'badge ' + statusKind(r.status), text: statusText(r.status) })),
            el('td', { class: 'mono', text: r.epoch + '/' + (r.epochs || '—') }),
            el('td', { class: 'mono', text: fmt(r.map50) }),
            el('td', { class: 'mono', text: fmt(r.map5095) }),
            el('td', { class: 'mono sp-fs-12', text: r.exported ? r.exported.file : '—' }),
            el('td', null, el('div', { class: 'row tight' }, [
              el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () {
                loadRun(r.name).then(function () { S.render(); });
              } }, '查看'),
              r.has_last ? el('button', { class: 'btn btn-ghost btn-sm', type: 'button', disabled: !r.resumable || !!cache.info.active,
                title: r.resumable ? '从 last.pt 恢复优化器和原数据版本' : '已完成或缺少可恢复的训练状态', onclick: function () { doResume(r.name); } }, '从 last.pt 续训') : null
            ]))
          ]);
        }))
      ]))
    ]);
  }

  function doResume(name) {
    S.confirm({ title: '从 last.pt 恢复训练？', body: '继续这一轮的原模型、参数和固定数据版本。新标注不会加入这次续训。', ok: '继续训练', onOk: function () {
      api().yoloResume(pid(), name, {}).then(function () {
        S.toast('已恢复零件模型训练', 'ok');
        cache.detail = null;
        return load(true);
      }).catch(function (err) { S.toast('续训失败：' + err.message, 'bad'); }).then(function () { S.render(); });
    } });
  }

  /* 曲线用全站同一个挂载器：主题色、窗口缩放、离开页面时释放，和动作训练页一致。 */
  function chartBox(title, build) {
    var node = el('div', { class: 'chart' });
    var box = el('div', { class: 'chart-box' }, [el('header', null, el('h3', { text: title })), node]);
    setTimeout(function () {
      if (!node.isConnected) return;
      global.OD.charts.mount(node, build(node));
    }, 0);
    return box;
  }

  function lineChart(node, x, lines, min, max) {
    var C = global.OD.charts.colors;
    var palette = [C.busy, C.ok, C.warn, C.cats[1], C.cats[3] || '#14b8a6', C.cats[4] || '#ec4899'];
    var ink = C.muted, grid = C.line;
    return {
      animation: false,
      grid: { left: 44, right: 12, top: 28, bottom: 28 },
      legend: { top: 0, right: 0, textStyle: { color: ink, fontSize: 11 }, itemWidth: 10, itemHeight: 3, icon: 'roundRect' },
      tooltip: { trigger: 'axis', backgroundColor: C.card, borderColor: grid, textStyle: { color: C.ink, fontSize: 12 } },
      xAxis: { type: 'category', data: x, axisLine: { lineStyle: { color: grid } }, axisLabel: { color: ink, fontSize: 11 } },
      yAxis: { type: 'value', min: min, max: max, splitLine: { lineStyle: { color: grid } }, axisLabel: { color: ink, fontSize: 11 } },
      series: lines.filter(function (l) { return l[1] && l[1].length; }).map(function (l, i) {
        return { type: 'line', name: l[0], data: l[1], smooth: true, showSymbol: false, lineStyle: { width: 2, color: palette[i % palette.length] }, itemStyle: { color: palette[i % palette.length] } };
      })
    };
  }

  /* 训练中定时刷新：只换掉进度块和历史表，表单不动。 */
  function paintLive() {
    var host = document.getElementById('y-live');
    if (!host) return;
    D.clear(host);
    host.appendChild(liveSection());
    var log = document.getElementById('y-log');
    if (log) log.scrollTop = log.scrollHeight;
  }

  /* ============ 模型库 ============ */

  var samLib = { info: null, loading: false, busy: false, uploading: '', timer: null };

  function loadSamLib(refresh) {
    if (!api().smartModelsStatus || !global.OD.live.isLive()) return Promise.resolve(null);
    if (samLib.loading && !refresh) return Promise.resolve(samLib.info);
    samLib.loading = true;
    return api().smartModelsStatus(!!refresh).then(function (info) {
      samLib.info = info;
      samLib.loading = false;
      var hasDl = info && info.downloads && Object.keys(info.downloads).some(function (k) {
        return info.downloads[k] && info.downloads[k].status === 'downloading';
      });
      clearTimeout(samLib.timer);
      if (hasDl && S.state.view === 'library') {
        samLib.timer = setTimeout(function () {
          if (S.state.view === 'library') loadSamLib(false).then(function () { if (S.state.view === 'library') S.render(); });
        }, 1500);
      }
      return info;
    }).catch(function () {
      samLib.loading = false;
      return samLib.info;
    });
  }

  function renderSamLibraryCard(library) {
    var samMeta = (library && library.sam) || {};
    if (global.OD.live.isLive() && !samLib.info && !samLib.loading && api().smartModelsStatus) {
      loadSamLib(false).then(function () { if (S.state.view === 'library') S.render(); });
    }
    var info = samLib.info;
    var samDir = (info && info.sam_dir) || samMeta.dir || 'models/sam';
    var locked = !!(info && info.training_locked);
    var activeModel = (info && info.state === 'ready' && info.active_model) || '';
    var models = ((info && info.available_models) || []).filter(function (m) { return m.kind === 'sam_family'; });
    if (!models.length && samMeta.files && samMeta.files.length) {
      models = samMeta.files.map(function (f) { return Object.assign({ kind: 'sam_family', local: true, active: false }, f); });
    }
    var localCount = models.filter(function (m) { return m.local; }).length;

    var uploadInput = el('input', {
      type: 'file', accept: '.pt,.pth', style: 'display:none',
      onchange: function (e) {
        var file = e.target.files && e.target.files[0];
        if (!file || !api().smartModelsUpload) return;
        samLib.uploading = '正在保存 ' + file.name + ' (0%)…';
        S.render();
        api().smartModelsUpload(file, function (ratio) {
          samLib.uploading = '正在保存 ' + file.name + ' (' + Math.round(ratio * 100) + '%)…';
          S.render();
        }).then(function (res) {
          samLib.uploading = '';
          samLib.info = res;
          S.toast('已保存模型到 ' + samDir + '\\' + ((res && res.uploaded_model) || file.name), 'ok');
          return global.OD.live.refreshLibrary();
        }).catch(function (err) {
          samLib.uploading = '';
          S.toast('保存模型失败：' + err.message, 'bad');
        }).then(function () { if (S.state.view === 'library') S.render(); });
        e.target.value = '';
      }
    });

    function doActivateSam(modelName) {
      if (!api().smartModelsActivate || samLib.busy) return;
      samLib.busy = true;
      S.render();
      api().smartModelsActivate({ model: modelName, slot: 'auto' }).then(function (res) {
        samLib.info = res;
        if (res && res.state === 'ready') {
          try { localStorage.setItem('od_smart_preferred_model', res.active_model || modelName); } catch (e) { }
          S.toast('已加载并启用 ' + res.active_model + '（' + (res.device || 'GPU') + (res.half ? ' FP16' : '') + '）', 'ok');
        } else {
          S.toast((res && res.reason) || '模型未能启动', 'warn');
        }
      }).catch(function (err) {
        S.toast('加载模型失败：' + err.message, 'bad');
      }).then(function () {
        samLib.busy = false;
        if (S.state.view === 'library') S.render();
      });
    }

    function doStopSam(label) {
      if (!api().smartModelsUnload || samLib.busy) return;
      samLib.busy = true;
      S.render();
      api().smartModelsUnload('在模型库手动停止 ' + (label || activeModel || '模型')).then(function (res) {
        samLib.info = res;
        S.toast('已停止 ' + (label || activeModel || '模型') + ' 并释放显存', 'ok');
      }).catch(function (err) {
        S.toast('停止模型失败：' + err.message, 'bad');
      }).then(function () {
        samLib.busy = false;
        if (S.state.view === 'library') S.render();
      });
    }

    var cards = models.map(function (m) {
      var dl = m.download;
      var dlBusy = dl && dl.status === 'downloading';
      var btns = [];
      if (m.local) {
        if (m.active) {
          btns.push(el('span', { class: 'badge ok', text: '运行中' }));
          btns.push(el('button', {
            class: 'btn btn-danger btn-sm', type: 'button', disabled: samLib.busy || locked,
            onclick: function () { doStopSam(m.name); }
          }, '停止此模型'));
        } else {
          btns.push(el('button', {
            class: 'btn btn-primary btn-sm', type: 'button', disabled: samLib.busy || locked,
            onclick: function () { doActivateSam(m.name); }
          }, activeModel ? '切换加载' : '加载并使用'));
        }
      } else if (m.downloadable && api().smartModelsDownload) {
        btns.push(el('button', {
          class: 'btn btn-primary btn-sm', type: 'button', disabled: dlBusy || locked,
          onclick: function () {
            api().smartModelsDownload(m.name, '').then(function () {
              S.toast('开始下载 ' + m.name + ' 到 ' + samDir, 'ok');
              return loadSamLib(false);
            }).catch(function (err) {
              S.toast('下载失败：' + err.message, 'bad');
            }).then(function () { if (S.state.view === 'library') S.render(); });
          }
        }, dlBusy ? ('下载中 ' + (dl.percent || 0) + '%') : '下载到文件夹'));
      } else if (m.gated) {
        btns.push(el('span', { class: 'help', text: '请将权重放入上方文件夹或点「上传模型」' }));
      }
      var statusText = m.active ? '运行中' : (m.local ? ('已保存 · ' + D.fmtBytes(m.bytes)) : (m.gated ? '需手动放入/上传' : '未下载'));
      return el('div', { class: 'smart-model-card' + (m.active ? ' is-active' : '') + (m.local ? ' is-local' : '') }, [
        el('div', { class: 'smart-model-head' }, [
          el('strong', { text: m.label || m.name }),
          el('span', { class: 'smart-model-meta', text: '≈' + (m.vram_gb || 1) + 'G 显存 · ' + statusText })
        ]),
        el('p', { class: 'help', text: m.note || '' }),
        dlBusy ? el('div', { class: 'progress sp-mt-2' }, el('div', { class: 'progress-bar busy', style: 'width:' + (dl.percent || 5) + '%' })) : null,
        dl && dl.status === 'failed' ? el('p', { class: 'text-danger', text: '下载失败：' + dl.error }) : null,
        el('div', { class: 'parts-toolbar', style: 'margin:6px 0 0' }, btns)
      ]);
    });

    var statusBadge = locked
      ? el('span', { class: 'badge bad', text: '训练中已锁定' })
      : (activeModel
        ? el('span', { class: 'badge ok mono', text: '运行中：' + activeModel })
        : el('span', { class: 'badge plain', text: '未加载 · 文件夹已存 ' + localCount + ' 个' }));

    return el('section', { class: 'card sp-mt-4' }, [
      el('header', null, [
        el('span', { class: 'mod-tag', 'data-mod': 'yolo', text: 'Meta' }),
        el('h2', { text: 'Meta SAM / 智能标注模型' }),
        el('span', { class: 'end' }, statusBadge)
      ]),
      el('div', { class: 'row sp-mt-2 sp-mb-3', style: 'flex-wrap:wrap;gap:8px' }, [
        el('span', { class: 'help mono', text: '保存目录：' + samDir }),
        el('span', { class: 'spread' }),
        uploadInput,
        activeModel ? el('button', {
          class: 'btn btn-danger btn-sm', type: 'button', disabled: samLib.busy,
          onclick: function () { doStopSam(activeModel); }
        }, '停止当前模型') : null,
        api().smartModelsOpenDir ? el('button', {
          class: 'btn btn-ghost btn-sm', type: 'button',
          onclick: function () {
            api().smartModelsOpenDir().then(function (r) {
              S.toast('已打开模型文件夹：' + ((r && r.dir) || samDir), 'ok');
            }).catch(function (err) { S.toast(err.message, 'warn'); });
          }
        }, '打开模型文件夹') : null,
        api().smartModelsUpload ? el('button', {
          class: 'btn btn-ghost btn-sm', type: 'button', disabled: !!samLib.uploading,
          onclick: function () { uploadInput.click(); }
        }, samLib.uploading || '上传模型 (.pt/.pth)') : null,
        el('button', {
          class: 'btn btn-ghost btn-sm', type: 'button',
          onclick: function () {
            Promise.all([global.OD.live.refreshLibrary(), loadSamLib(true)]).then(function () { S.render(); });
          }
        }, '重新扫描')
      ]),
      el('p', { class: 'help sp-mb-3', text: '所有 Meta SAM 3 / SAM 2.1 / MobileSAM / YOLOE 权重统一保存在上方文件夹中，可自由选择加载任意已保存模型使用，用完随时点击「停止此模型」释放显存。' }),
      cards.length
        ? el('div', { class: 'smart-model-grid' }, cards)
        : el('p', { class: 'help', text: samLib.loading ? '正在扫描模型文件夹…' : '暂无可展示模型，点击「打开模型文件夹」放入 .pt/.pth 文件后重新扫描。' })
    ]);
  }

  function renderLibrary() {
    var library = global.OD.data.library || {};
    var files = (library.yolo && library.yolo.files) || [];
    var current = (global.OD.data.settings || {}).yolo_path || '';
    var body = [];
    body.push(el('div', { class: 'row sp-mb-3' }, [
      el('span', { class: 'help mono', text: (library.yolo && library.yolo.dir) || '' }),
      el('span', { class: 'spread' }),
      el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () {
        Promise.all([global.OD.live.refreshLibrary(), loadSamLib(true)]).then(function () { S.render(); });
      } }, '重新扫描')
    ]));
    if (!files.length) {
      body.push(el('div', { class: 'callout warn' }, [icon('info', 16), el('div', { text: '模型库里还没有零件模型。在「零件训练」训完后点「导出 ONNX」，或者把 .onnx 文件放进上面这个目录再重新扫描。' })]));
    } else {
      body.push(el('div', { class: 'grid c3' }, files.map(function (item) {
        var origin = item.origin || {};
        var mine = current && (current === item.path || current === item.name);
        var users = item.used_by || [];
        return el('article', { class: 'card model-card' + (mine ? ' is-current' : '') }, [
          el('header', null, [
            el('span', { class: 'mod-tag', 'data-mod': 'yolo', text: '零件' }),
            el('h3', { class: 'mono', text: item.name }),
            mine ? el('span', { class: 'end' }, el('span', { class: 'badge ok', text: '本工程在用' })) : null
          ]),
          el('dl', { class: 'keyval' }, [
            el('dt', { text: '来源' }), el('dd', { text: origin.project ? (origin.project + ' / 第 ' + origin.run + ' 轮') : '手动放入' }),
            el('dt', { text: '输入尺寸' }), el('dd', { class: 'mono', text: origin.imgsz || '—' }),
            el('dt', { text: '大小' }), el('dd', { class: 'mono', text: D.fmtBytes(item.bytes) }),
            el('dt', { text: '更新时间' }), el('dd', { class: 'mono', text: item.mtime || '—' }),
            el('dt', { text: '指纹' }), el('dd', { class: 'mono', text: origin.sha256 ? origin.sha256.slice(0, 12) : '—' }),
            el('dt', { text: '使用的工程' }), el('dd', { text: users.length ? users.join('、') : '没有' })
          ]),
          mine ? null : el('div', { class: 'row sp-mt-3' }, el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () {
            S.go('model');
            S.toast('在「模型与工位」的下拉框里选 ' + item.name + '，再点检查并保存。', 'info');
          } }, '在本工程使用'))
        ]);
      })));
    }
    body.push(renderSamLibraryCard(library));
    var hands = library.hands || {};
    body.push(el('section', { class: 'card sp-mt-4' }, [
      el('header', null, [el('span', { class: 'mod-tag', 'data-mod': 'action', text: '手部' }), el('h2', { text: '手部模型' }),
        el('span', { class: 'end' }, el('span', { class: 'badge ' + (hands.exists ? 'ok' : 'warn'), text: hands.exists ? (hands.source === 'library' ? '库内' : 'MediaPipe 内置') : '缺文件' }))]),
      el('p', { class: 'help mono', text: hands.dir || '' })
    ]));
    return page('模型库', '所有工程共用一份。零件模型、Meta SAM 智能标注模型与手部模型都在这里统一管理。', body);
  }

  /* ---------- 步骤栏状态 ---------- */

  function stepState(key) {
    if (!live() || cache.projectId !== pid() || !cache.info) return 'none';
    var info = cache.info;
    if (key === 'pdata') return info.dataset && info.dataset.exists ? 'ok' : 'none';
    if (key === 'ptrain') {
      if (info.active) return 'busy';
      var last = info.runs[0];
      if (!last) return 'none';
      return last.status === 'done' ? 'ok' : (last.status === 'failed' ? 'bad' : 'warn');
    }
    if (key === 'library') return (global.OD.data.settings || {}).yolo_path ? 'ok' : 'none';
    return 'none';
  }

  function stepSub(key) {
    if (!live()) return '需要连着训练台';
    var info = cache.projectId === pid() ? cache.info : null;
    if (key === 'pdata') return info && info.dataset && info.dataset.exists ? (info.dataset.train_images + ' 张训练图') : '还没导入';
    if (key === 'ptrain') {
      if (!info) return '—';
      if (info.active) return '训练中';
      return info.runs.length ? ('最近 ' + info.runs[0].name) : '还没训练';
    }
    if (key === 'library') {
      var p = (global.OD.data.settings || {}).yolo_path;
      return p ? String(p).split(/[\\/]/).pop() : '还没选零件模型';
    }
    return '';
  }

  function prefetch() {
    if (live() && (cache.projectId !== pid() || !cache.info) && !cache.loading) {
      Promise.all([load(true), loadEnv(false)]).then(function () {
        if (cache.info && cache.info.active) watch();
        if (['pdata', 'ptrain', 'library'].indexOf(S.state.view) >= 0) S.render();
        else { S.renderRail(); S.renderJob(); }
      });
    }
  }

  function inspectorData() {
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [el('h3', { text: '标注与发布' }), el('p', { text: '在零件标注页导入原图或视频，编辑旋转框并确认审核；通过质检后按来源发布固定数据版本。也保留外部 YOLO 数据集导入。' })]),
      el('div', { class: 'inspect-block' }, [el('h3', { text: '数据多少够用' }), el('p', { text: '每个类别至少几百个框；产线上会出现的角度、光照、遮挡都要有。验证集要和训练集来自不同批次的产品，mAP 才说明问题。' })])
    ]);
  }

  function inspectorTrain() {
    var env = cache.env || {};
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [el('h3', { text: '训练环境' }), env.ok ? el('dl', { class: 'keyval' }, [
        el('dt', { text: 'Ultralytics' }), el('dd', { class: 'mono', text: env.ultralytics }),
        el('dt', { text: 'PyTorch' }), el('dd', { class: 'mono', text: env.torch }),
        el('dt', { text: '显卡' }), el('dd', { text: env.cuda ? env.gpu : '无' }),
        el('dt', { text: 'Python' }), el('dd', { class: 'mono sp-fs-11', text: env.python_path })
      ]) : el('p', { text: env.error || '检测中…' })]),
      el('div', { class: 'inspect-block' }, [el('h3', { text: '和动作训练的关系' }), el('p', { text: '显卡只有一张，零件训练时不能同时抽特征或训练动作。换了零件模型以后，动作特征会按新模型重新抽，动作模型也要重训。' })])
    ]);
  }

  function inspectorLibrary() {
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [el('h3', { text: '产线怎么用' }), el('p', { text: '产线程序加载的是这里的 .onnx。把选定的文件拷到产线的 config 目录，并在产线配置页选它。' })])
    ]);
  }

  function applySuggestedParams(suggested) {
    if (!suggested) return;
    if (cache.projectId !== pid()) cache.projectId = pid();
    cache.form = cache.form || {};
    ['model', 'batch', 'epochs', 'imgsz', 'device'].forEach(function (k) {
      if (suggested[k] !== undefined && suggested[k] !== '') cache.form[k] = String(suggested[k]);
    });
    if (suggested.task === 'obb') cache.selectedVersion = '';
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.pdata = { render: renderData, inspector: inspectorData };
  global.OD.views.ptrain = { render: renderTrain, inspector: inspectorTrain };
  global.OD.views.library = { render: renderLibrary, inspector: inspectorLibrary };
  global.OD.yolo = { stepState: stepState, stepSub: stepSub, prefetch: prefetch, activeJob: activeJob,
    applySuggestedParams: applySuggestedParams,
    refresh:function (project) { if (!live() || project !== pid()) return Promise.resolve(); return load(true); },
    reset:function () { generation++; stopWatch(); cache = { projectId: null, info: null, env: cache.env, detail: null, log: [], logNext: 0, loading: false, error: '' }; } };
})(window);
