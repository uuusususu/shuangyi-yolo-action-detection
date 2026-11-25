/* 数据集与零件素材统一传输中心：
   1. 智能解析 Ultralytics 平台命令（export ULTRALYTICS_API_KEY + yolo train data=ul://...）或直链，后台拉取并生成标准 .zip；
   2. 浏览器拖拽/内网网线直连分块上传（支持多图、视频抽帧、数据集 .zip/.ndjson），实时显示上传速率与进度；
   3. 已生成的 .zip 归档列表，支持一键下载到浏览器。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;

  var state = {
    projectId: null,
    tab: 'remote',          /* 'remote' | 'upload' */
    commandText: '',
    cred: null,             /* {has_key, masked, source} */
    sourceGroup: '',
    interval: 30,
    maxFrames: 300,
    dedupe: true,
    job: null,              /* 服务端异步拉取任务 */
    archives: [],           /* 已生成的 zip 列表 */
    uploading: null,        /* {label, pct, speedBps, doneFiles, totalFiles} */
    pollTimer: null
  };

  function api() { return global.OD.api; }
  function pid() { return S.state.projectId; }
  function live() { return global.OD.live && global.OD.live.isLive() && !!pid(); }

  /* 客户端即时拆解命令，粘贴瞬间就给用户可视化反馈，不必等网络请求 */
  function parseLocal(raw) {
    var text = String(raw || '').trim();
    if (!text) return null;
    var keyMatch = text.match(/(?:export|set|\$env:)?\s*ULTRALYTICS_API_KEY\s*=\s*['"]?([^\s'";]+)['"]?/i)
      || text.match(/(?:yolo|ul)\s+login\s+['"]?([A-Za-z0-9_\-]{16,128})['"]?/i);
    var apiKey = keyMatch ? keyMatch[1] : '';
    var joined = text.replace(/\\\r?\n/g, ' ');
    var kv = {};
    joined.split(/\r?\n|&&|;/).forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed || /^(?:export|set|\$env:)/i.test(trimmed)) return;
      trimmed.split(/\s+/).forEach(function (tok, idx) {
        var eq = tok.indexOf('=');
        if (eq > 0 && !/^(?:https?|ul):\/\//i.test(tok)) {
          var k = tok.slice(0, eq).toLowerCase();
          var v = tok.slice(eq + 1).replace(/^['"]|['"]$/g, '');
          if (k && v) kv[k] = v;
        } else if ((idx === 1 || idx === 2) && /^(?:obb|detect|segment)$/i.test(tok) && !kv.task) {
          kv.task = tok.toLowerCase();
        }
      });
    });
    var data = kv.data || '';
    var model = kv.model || '';
    if (!data) {
      var m = text.match(/(ul:\/\/[^\s'";]+|https?:\/\/[^\s'";]+)/i);
      if (m) data = m[1];
    }
    if (data && /^https?:\/\/platform\.ultralytics\.com\//i.test(data)) {
      data = 'ul://' + data.replace(/^https?:\/\/platform\.ultralytics\.com\//i, '').replace(/^\/+|\/+$/g, '');
    }
    var params = {};
    ['task', 'batch', 'epochs', 'imgsz', 'device', 'project'].forEach(function (k) {
      if (kv[k] !== undefined) params[k] = kv[k];
    });
    if (!params.task && /-obb/i.test(model)) params.task = 'obb';
    return { apiKey: apiKey, data: data, model: model, params: params };
  }

  function maskKey(k) {
    if (!k) return '';
    return k.length <= 10 ? (k.slice(0, 2) + '••••' + k.slice(-2)) : (k.slice(0, 6) + '••••••••' + k.slice(-4));
  }

  function fmtSpeed(bps) {
    if (!bps || bps <= 0) return '';
    if (bps >= 1048576) return (bps / 1048576).toFixed(1) + ' MB/s';
    if (bps >= 1024) return Math.round(bps / 1024) + ' KB/s';
    return bps + ' B/s';
  }

  function phaseLabel(phase) {
    return {
      starting: '准备中',
      auth: '平台鉴权',
      model: '下载底模',
      manifest: '拉取标注清单',
      downloading: '下载压缩包',
      images: '并发拉取图片',
      extracting: '安全解压校验',
      zip: '生成 ZIP 压缩包',
      import: '写入当前工程',
      done: '已完成',
      stopped: '已取消',
      failed: '失败'
    }[phase] || phase || '处理中';
  }

  function ensureLoaded() {
    if (!live() || !api().transferState) return;
    var currentPid = pid();
    if (state.projectId !== currentPid) {
      stopPoll();
      state.projectId = currentPid;
      state.job = null;
      state.archives = [];
    }
    if (!state.cred && api().ulCredentials) {
      api().ulCredentials().then(function (c) {
        state.cred = c;
        paintDynamic();
      }).catch(function () {});
    }
    api().transferState(currentPid).then(function (res) {
      if (currentPid !== pid()) return;
      state.archives = (res && res.archives) || [];
      if (res && res.status && res.status !== 'idle') {
        state.job = res;
        if (res.status === 'running') startPoll();
      }
      paintDynamic();
      S.renderJob();
    }).catch(function () {});
  }

  function startPoll() {
    if (state.pollTimer) return;
    state.pollTimer = setInterval(function () {
      if (!live()) { stopPoll(); return; }
      var currentPid = pid();
      api().transferState(currentPid).then(function (res) {
        if (currentPid !== pid()) return;
        var wasRunning = state.job && state.job.status === 'running';
        state.job = res;
        state.archives = (res && res.archives) || state.archives;
        paintDynamic();
        S.renderJob();
        if (!res || res.status !== 'running') {
          stopPoll();
          if (wasRunning && res) {
            if (res.status === 'done') {
              S.toast(res.message || '数据集、底模与训练参数已自动加载完成', 'ok');
              refreshAllAfterImport(currentPid, res);
            } else if (res.status === 'failed') {
              S.toast(res.message || ('拉取失败：' + res.error), 'bad');
            } else if (res.status === 'stopped') {
              S.toast('已取消下载/导入任务', 'warn');
            }
          }
        }
      }).catch(function () {});
    }, 900);
  }

  function stopPoll() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function applyParsedParamsToYolo(job) {
    if (!job || !job.parsed || !global.OD.yolo || !global.OD.yolo.applySuggestedParams) return;
    var p = Object.assign({}, job.parsed.params || {});
    if (job.downloaded_model) p.model = job.downloaded_model;
    else if (job.parsed.model_filename) p.model = job.parsed.model_filename;
    global.OD.yolo.applySuggestedParams(p);
  }

  function refreshAllAfterImport(projectId, completedJob) {
    var tasks = [];
    if (global.OD.parts && global.OD.parts.prefetch) tasks.push(global.OD.parts.prefetch());
    if (global.OD.yolo && global.OD.yolo.refresh) tasks.push(global.OD.yolo.refresh(projectId));
    Promise.all(tasks).then(function () {
      if (completedJob) applyParsedParamsToYolo(completedJob);
      if (['pdata', 'palabel', 'ptrain'].indexOf(S.state.view) >= 0) S.render();
      else S.renderRail();
    }).catch(function () {});
  }

  function startRemoteTransfer() {
    if (!live()) { S.toast('请先连接训练台并选择工程', 'warn'); return; }
    var raw = state.commandText.trim();
    if (!raw) { S.toast('请先粘贴 Ultralytics 命令、ul:// 链接或下载地址', 'warn'); return; }
    var currentPid = pid();
    api().startTransfer(currentPid, {
      source: raw,
      api_key: '',
      remember_key: true,
      target: 'both',
      download_model: true,
      source_group: state.sourceGroup.trim(),
      interval: Number(state.interval) || 30,
      max_frames: Number(state.maxFrames) || 300,
      dedupe: state.dedupe
    }).then(function (job) {
      state.job = job;
      if ((parseLocal(raw) || {}).apiKey && api().ulCredentials) {
        api().ulCredentials().then(function (c) { state.cred = c; paintDynamic(); }).catch(function () {});
      }
      paintDynamic();
      S.renderJob();
      startPoll();
    }).catch(function (err) {
      S.toast('无法启动拉取：' + err.message, 'bad');
    });
  }

  function cancelActiveTransfer() {
    if (!live()) return;
    api().cancelTransfer(pid()).then(function (job) {
      state.job = job;
      paintDynamic();
      S.renderJob();
    }).catch(function (err) { S.toast('取消失败：' + err.message, 'bad'); });
  }

  function handleFilesUpload(fileList) {
    if (!live()) { S.toast('请先连接训练台并选择工程', 'warn'); return; }
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return;
    var currentPid = pid();
    var startedAt = Date.now();
    var bytesCompleted = 0;

    function next(idx) {
      if (idx >= files.length) {
        state.uploading = null;
        paintDynamic();
        S.toast('已成功上传并导入 ' + files.length + ' 个文件', 'ok');
        refreshAllAfterImport(currentPid);
        return;
      }
      var file = files[idx];
      var isDatasetArchive = /\.(zip|ndjson)$/i.test(file.name);
      state.uploading = {
        label: '正在上传 ' + file.name + ' (' + (idx + 1) + '/' + files.length + ')',
        pct: 0,
        speedBps: 0,
        doneFiles: idx,
        totalFiles: files.length
      };
      paintDynamic();

      var onProg = function (ratio) {
        var elapsed = Math.max(0.1, (Date.now() - startedAt) / 1000);
        var currentBytes = bytesCompleted + file.size * ratio;
        state.uploading = {
          label: '正在推送 ' + file.name + ' (' + (idx + 1) + '/' + files.length + ')',
          pct: Math.round(((idx + ratio) / files.length) * 100),
          speedBps: Math.round(currentBytes / elapsed),
          doneFiles: idx,
          totalFiles: files.length
        };
        paintDynamic();
      };

      var req = isDatasetArchive
        ? api().uploadYoloDataset(currentPid, file, onProg, true)
        : api().uploadParts(currentPid, file, {
          onProgress: onProg,
          sourceGroup: state.sourceGroup.trim(),
          interval: Number(state.interval) || 30,
          maxFrames: Number(state.maxFrames) || 300,
          dedupe: state.dedupe
        });

      req.then(function (res) {
        bytesCompleted += file.size;
        if (res && res.downloads) state.archives = res.downloads;
        next(idx + 1);
      }).catch(function (err) {
        state.uploading = null;
        paintDynamic();
        S.toast('上传 ' + file.name + ' 失败：' + err.message, 'bad');
      });
    }

    next(0);
  }

  function renderParsedPreview() {
    var parsed = parseLocal(state.commandText);
    if (!parsed || (!parsed.data && !parsed.model && !parsed.apiKey)) return null;
    var badges = [];
    if (parsed.apiKey) badges.push(el('span', { class: 'badge ok mono', text: 'API Key: ' + maskKey(parsed.apiKey) }));
    else if (parsed.data && /^ul:\/\//i.test(parsed.data) && !(state.cred && state.cred.has_key)) {
      badges.push(el('span', { class: 'badge warn', text: '未检测到 API Key，请一并粘贴 export ULTRALYTICS_API_KEY="..."' }));
    }
    if (parsed.data) badges.push(el('span', { class: 'badge busy mono', text: '数据集: ' + parsed.data }));
    if (parsed.model) badges.push(el('span', { class: 'badge plain mono', text: '底模: ' + parsed.model }));
    Object.keys(parsed.params).forEach(function (k) {
      badges.push(el('span', { class: 'badge plain mono', text: k + '=' + parsed.params[k] }));
    });
    return el('div', { class: 'transfer-parsed sp-mt-2' }, [
      el('span', { class: 'sp-fs-12 sp-dim', text: '已自动识别：' })
    ].concat(badges));
  }

  function renderProgressPanel() {
    if (state.uploading) {
      var u = state.uploading;
      var spd = fmtSpeed(u.speedBps);
      return el('div', { class: 'transfer-progress is-busy sp-mt-3' }, [
        el('div', { class: 'transfer-prog-head' }, [
          el('span', { class: 'badge busy', text: '跨机/本地推送中' }),
          el('strong', { text: u.label }),
          el('span', { class: 'mono sp-ml-auto', text: (spd ? spd + ' · ' : '') + u.pct + '%' })
        ]),
        el('div', { class: 'progress sp-mt-2' },
          el('div', { class: 'progress-bar busy', style: 'width:' + u.pct + '%' }))
      ]);
    }
    var j = state.job;
    if (!j || j.status === 'idle') return null;
    var isRun = j.status === 'running';
    var pct = j.total > 0 ? Math.min(100, Math.round((j.done / j.total) * 100)) : 0;
    if (j.status === 'done') pct = 100;
    var kind = j.status === 'done' ? 'ok' : (j.status === 'failed' ? 'bad' : (j.status === 'stopped' ? 'warn' : 'busy'));
    var metaParts = [];
    if (j.bytes_done > 0) metaParts.push((j.bytes_done / 1048576).toFixed(1) + ' MB');
    if (isRun && j.speed_bps > 0) metaParts.push(fmtSpeed(j.speed_bps));
    if (isRun && j.eta_seconds) metaParts.push('剩余约 ' + j.eta_seconds + ' 秒');
    if (j.total > 0 && j.unit) metaParts.push(j.done + ' / ' + j.total + ' ' + j.unit);

    return el('div', { class: 'transfer-progress is-' + kind + ' sp-mt-3' }, [
      el('div', { class: 'transfer-prog-head' }, [
        el('span', { class: 'badge ' + kind, text: phaseLabel(j.phase) }),
        el('strong', { text: j.message || '' }),
        el('span', { class: 'mono sp-fs-12 sp-ml-auto', text: metaParts.join(' · ') }),
        isRun ? el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: cancelActiveTransfer }, '取消') : null
      ]),
      el('div', { class: 'progress sp-mt-2' },
        el('div', { class: 'progress-bar ' + kind, style: 'width:' + pct + '%' })),
      j.zip_file && j.status === 'done' ? el('div', { class: 'row tight sp-mt-2' }, [
        el('span', { class: 'sp-fs-12', text: '已生成标准 YOLO 压缩包：' }),
        el('code', { class: 'mono', text: j.zip_file + (j.zip_bytes ? ' (' + D.fmtBytes(j.zip_bytes) + ')' : '') }),
        el('a', {
          class: 'btn btn-primary btn-sm',
          href: api().yoloDownloadUrl(pid(), j.zip_file),
          download: j.zip_file,
          text: '下载生成的 ZIP'
        }),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { S.go('ptrain'); } }, '去零件训练 ›')
      ]) : null
    ]);
  }

  function renderArchives() {
    if (!state.archives || !state.archives.length) return null;
    return el('div', { class: 'transfer-archives sp-mt-3' }, [
      el('div', { class: 'sp-fs-12 sp-dim sp-mb-2', text: '本工程已生成的标准数据集 ZIP 归档（点击可直接下载备份或分发）：' }),
      el('div', { class: 'transfer-archive-list' }, state.archives.slice(0, 6).map(function (item) {
        return el('a', {
          class: 'transfer-archive-pill',
          href: api().yoloDownloadUrl(pid(), item.name),
          download: item.name,
          title: '点击下载 ' + item.name
        }, [
          icon('upload', 13),
          el('span', { class: 'mono', text: item.name }),
          el('small', { text: D.fmtBytes(item.bytes) })
        ]);
      }))
    ]);
  }

  function paintDynamic() {
    var progHost = document.getElementById('transfer-dynamic-status');
    if (progHost) {
      D.clear(progHost);
      var p = renderProgressPanel();
      if (p) progHost.appendChild(p);
      var a = renderArchives();
      if (a) progHost.appendChild(a);
    }
    var parsedHost = document.getElementById('transfer-parsed-host');
    if (parsedHost) {
      D.clear(parsedHost);
      var node = renderParsedPreview();
      if (node) parsedHost.appendChild(node);
    }
    var credBadge = document.getElementById('transfer-cred-badge');
    if (credBadge) {
      var parsed = parseLocal(state.commandText);
      var c = state.cred;
      if (parsed && parsed.apiKey) {
        credBadge.className = 'badge ok mono';
        credBadge.textContent = '将自动保存 Key: ' + maskKey(parsed.apiKey);
      } else if (c && c.has_key) {
        credBadge.className = 'badge ok mono';
        credBadge.textContent = '已记住 Key: ' + c.masked;
      } else {
        credBadge.className = 'badge plain';
        credBadge.textContent = '命令含 Key 时自动提取保存';
      }
    }
  }

  function renderHub(options) {
    ensureLoaded();
    var opts = options || {};
    var running = !!(state.job && state.job.status === 'running') || !!state.uploading;

    var tabBar = el('div', { class: 'seg-group', role: 'group', 'aria-label': '选择数据接入方式' }, [
      el('button', {
        type: 'button', 'aria-pressed': String(state.tab === 'remote'),
        onclick: function () { state.tab = 'remote'; S.render(); }
      }, '粘贴 ul:// 训练命令或下载链接'),
      el('button', {
        type: 'button', 'aria-pressed': String(state.tab === 'upload'),
        onclick: function () { state.tab = 'upload'; S.render(); }
      }, '拖拽 / 跨网线上传文件与 ZIP')
    ]);

    var parsedNow = parseLocal(state.commandText);
    var credBadgeText = parsedNow && parsedNow.apiKey
      ? ('将自动保存 Key: ' + maskKey(parsedNow.apiKey))
      : (state.cred && state.cred.has_key ? ('已记住 Key: ' + state.cred.masked) : '命令含 Key 时自动提取保存');
    var credBadgeCls = 'badge ' + ((parsedNow && parsedNow.apiKey) || (state.cred && state.cred.has_key) ? 'ok mono' : 'plain');

    var remotePane = el('div', { class: 'sp-mt-3' }, [
      el('label', { class: 'parts-field' }, [
        el('div', { class: 'transfer-cmd-head' }, [
          el('span', { text: '直接粘贴整段 Ultralytics 命令、ul:// 链接或 ZIP/NDJSON 地址' }),
          el('span', { id: 'transfer-cred-badge', class: credBadgeCls, text: credBadgeText })
        ]),
        el('textarea', {
          class: 'form-control mono transfer-cmd-input', rows: '4',
          placeholder: '直接粘贴整段命令即可，其余全部自动加载，例如：\nexport ULTRALYTICS_API_KEY="ul_xxxx"\nyolo train model=ul://ultralytics/yolo26/yolo26n-obb data=ul://unicorn-kingfisher/datasets/sanreipian task=obb batch=70',
          value: state.commandText,
          oninput: function (e) { state.commandText = e.target.value; paintDynamic(); },
          onkeydown: function (e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !running) {
              e.preventDefault();
              startRemoteTransfer();
            }
          }
        })
      ]),
      el('div', { id: 'transfer-parsed-host' }, renderParsedPreview()),
      el('div', { class: 'parts-toolbar sp-mt-3' }, [
        el('button', {
          type: 'button', class: 'btn btn-primary', disabled: running,
          onclick: startRemoteTransfer
        }, running ? '正在自动拉取加载…' : '一键解析并自动加载'),
        el('span', { class: 'sp-fs-12 sp-dim', text: '自动保存 Key · 拉取数据集与底模 · 生成 ZIP · 导入画布 · 回填训练参数（Ctrl+Enter 快捷开始）' })
      ])
    ]);

    var fileInput = el('input', {
      type: 'file', multiple: true, class: 'sr', id: 'transfer-upload-input',
      accept: '.zip,.ndjson,image/*,video/*,.jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv',
      onchange: function () {
        if (fileInput.files && fileInput.files.length) {
          handleFilesUpload(fileInput.files);
          fileInput.value = '';
        }
      }
    });
    var dropZone = el('label', {
      class: 'empty drop compact drop-row sp-mt-3', for: 'transfer-upload-input',
      ondragover: function (e) { e.preventDefault(); dropZone.classList.add('is-over'); },
      ondragleave: function () { dropZone.classList.remove('is-over'); },
      ondrop: function (e) {
        e.preventDefault(); dropZone.classList.remove('is-over');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
          handleFilesUpload(e.dataTransfer.files);
        }
      }
    }, [
      icon('upload', 22),
      el('div', null, [
        el('strong', { text: '拖拽图片、采集视频或数据集 .zip / .ndjson 到此处，或点击选择文件' }),
        el('p', { text: '支持内网网线直连电脑直接从浏览器高速分块推送：含 data.yaml 的 .zip 自动导入为数据集；普通图片与视频自动抽帧去重进入标注工作区。' })
      ]),
      fileInput
    ]);

    var uploadPane = el('div', null, [
      dropZone,
      el('div', { class: 'parts-import-grid sp-mt-3' }, [
        el('label', { class: 'parts-field' }, [
          el('span', { text: '拍摄来源组（图片/视频上传时使用）' }),
          el('input', { class: 'form-control', placeholder: '留空按文件名自动分组', value: state.sourceGroup, oninput: function (e) { state.sourceGroup = e.target.value; } })
        ]),
        el('label', { class: 'parts-field' }, [
          el('span', { text: '视频抽帧间隔' }),
          el('input', { type: 'number', class: 'form-control', min: '1', max: '10000', value: state.interval, oninput: function (e) { state.interval = e.target.value; } })
        ]),
        el('label', { class: 'parts-field' }, [
          el('span', { text: '单视频最多抽帧' }),
          el('input', { type: 'number', class: 'form-control', min: '1', max: '10000', value: state.maxFrames, oninput: function (e) { state.maxFrames = e.target.value; } })
        ]),
        el('label', { class: 'parts-field' }, [
          el('span', { text: '重复画面过滤' }),
          el('label', { class: 'parts-inline sp-mt-2' }, [
            el('input', { type: 'checkbox', checked: state.dedupe, onchange: function (e) { state.dedupe = e.target.checked; } }),
            '开启相似帧去重'
          ])
        ])
      ])
    ]);

    return el('section', { class: 'card transfer-card' }, [
      el('header', null, [
        el('h2', { text: opts.title || '云端拉取 (ul://) · 跨网线上传 · 生成 ZIP' }),
        el('span', { class: 'end' }, tabBar)
      ]),
      state.tab === 'remote' ? remotePane : uploadPane,
      el('div', { id: 'transfer-dynamic-status' }, [renderProgressPanel(), renderArchives()])
    ]);
  }

  function activeJob() {
    var j = state.job;
    if (!live() || state.projectId !== pid() || !j || j.status !== 'running') return null;
    return {
      name: phaseLabel(j.phase),
      label: '数据拉取与打包',
      done: j.done || 0,
      total: j.total || 100,
      unit: j.unit || '%',
      elapsed: j.elapsed || Math.max(0, Date.now() / 1000 - (j.started_at || Date.now() / 1000)),
      stop: cancelActiveTransfer,
      open: function () { S.go('pdata'); }
    };
  }

  function reset() {
    stopPoll();
    state.projectId = null;
    state.job = null;
    state.archives = [];
    state.uploading = null;
  }

  global.OD.transfer = {
    renderHub: renderHub,
    activeJob: activeJob,
    reset: reset
  };
})(window);
