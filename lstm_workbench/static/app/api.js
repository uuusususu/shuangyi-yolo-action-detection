/* 真实训练台的 HTTP 客户端（app.py，只监听 127.0.0.1:8771）。
   路径一律相对，所以只有页面本身也是训练台托管的时候才连得上 ——
   双击打开是 file://，跨源请求会被浏览器掐掉，这里直接判定不可用。
   用 XHR 而不是 fetch：上传要进度，且 file:// 下能同步判定。 */
(function (global) {
  'use strict';

  var available = global.location.protocol !== 'file:';

  function offlineError() { return new Error('离线原型没有后端，这一步不会写盘'); }

  function send(method, path, options) {
    options = options || {};
    if (!available) return Promise.reject(offlineError());
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      xhr.open(method, path, true);
      /* 抽特征和训练是阻塞的，普通请求的超时不能套上去，调用方显式给 0。 */
      xhr.timeout = options.timeout === undefined ? 30000 : options.timeout;
      if (options.filename) xhr.setRequestHeader('X-Filename', options.filename);
      if (options.replace) xhr.setRequestHeader('X-Replace', '1');
      if (options.resetCheck) xhr.setRequestHeader('X-Check-Reset', '1');
      if (options.json !== undefined) xhr.setRequestHeader('Content-Type', 'application/json');
      if (xhr.upload && options.onProgress) {
        xhr.upload.addEventListener('progress', function (e) {
          if (e.lengthComputable) options.onProgress(e.loaded / e.total);
        });
      }
      xhr.addEventListener('load', function () {
        var body = null;
        try { body = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch (e) { body = null; }
        if (xhr.status >= 200 && xhr.status < 300) { resolve(body); return; }
        /* 服务端每条错误都带 error 字段（含 409 的"正在跑任务"），原样透出去。 */
        reject(new Error((body && body.error) || ('请求失败：HTTP ' + xhr.status)));
      });
      xhr.addEventListener('error', function () { reject(new Error('连不上训练台')); });
      xhr.addEventListener('timeout', function () { reject(new Error('请求超时')); });
      xhr.addEventListener('abort', function () { reject(new Error('请求被取消')); });
      xhr.send(options.json !== undefined ? JSON.stringify(options.json) : (options.body || null));
    });
  }

  function base(projectId) { return '/api/projects/' + encodeURIComponent(projectId); }

  global.OD = global.OD || {};
  global.OD.api = {
    available: available,

    /* GET /api/projects → {projects:[名字]} */
    projects: function () {
      return send('GET', '/api/projects').then(function (body) { return (body && body.projects) || []; });
    },

    /* GET /api/models → 全局模型库现状：{root, yolo:{dir,exists,files[]}, hands:{dir,exists}}。
       所有工程共用一份库，所以这个接口不带工程 id。 */
    models: function () { return send('GET', '/api/models'); },

    /* GET /api/projects/<id> → 工程全貌：settings / annotations / runs / comparable_runs / features / blockers / warnings */
    project: function (projectId) { return send('GET', base(projectId)); },

    /* GET /api/projects/<id>/progress → progress.json，没有这个文件时是 {status:"idle"} */
    progress: function (projectId) { return send('GET', base(projectId) + '/progress'); },

    /* GET /api/projects/<id>/runs/<name> → {name, files, report?, contract?, split?} */
    run: function (projectId, name) { return send('GET', base(projectId) + '/runs/' + encodeURIComponent(name)); },

    /* 结果页拿这一轮的权重看一条新视频。视频只进这一轮的 check 目录。 */
    checkStatus: function (projectId, name) {
      return send('GET', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check');
    },
    uploadCheck: function (projectId, name, file) {
      return send('POST', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check/media', {
        filename: file.name, body: file, timeout: 0, resetCheck: true
      });
    },
    runCheck: function (projectId, name) {
      return send('POST', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check', { json: {}, timeout: 0 });
    },
    checkClip: function (projectId, name, clipId) {
      return send('POST', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check/clip', {
        json: { clip_id: clipId }, timeout: 0
      });
    },
    checkFileUrl: function (projectId, name, file) {
      return base(projectId) + '/runs/' + encodeURIComponent(name) + '/check/file/' + encodeURIComponent(file);
    },
    checkOverlay: function (projectId, name) {
      return send('GET', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check/overlay', { timeout: 0 });
    },
    fillOverlay: function (projectId, name) {
      return send('POST', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check/overlay', { json: {}, timeout: 0 });
    },
    deleteCheck: function (projectId, name) {
      return send('DELETE', base(projectId) + '/runs/' + encodeURIComponent(name) + '/check');
    },

    /* 用某一轮的权重给当前片子画段。段先留在页面上，保存标注才写盘。 */
    prelabel: function (projectId, runName, clipId) {
      return send('POST', base(projectId) + '/runs/' + encodeURIComponent(runName) + '/prelabel', {
        json: { clip_id: clipId }, timeout: 0
      });
    },

    /* 素材直链。app.py 的 _send_project_file 支持 Range，video 能拖动进度。 */
    mediaUrl: function (projectId, file) {
      return base(projectId) + '/file/' + String(file).split('/').map(encodeURIComponent).join('/');
    },

    /* POST /api/projects → 201 {id}；重名是 409 */
    createProject: function (projectId, settings) {
      return send('POST', '/api/projects', { json: { id: projectId, settings: settings || {} } });
    },

    /* POST /api/projects/<id>/media，文件名走 X-Filename 头，正文是裸字节。
       storedName 跟 file.name 分开传：一组连续图要收进 images-<时间戳>/ 子目录。 */
    uploadMedia: function (projectId, file, storedName, onProgress, replace) {
      return send('POST', base(projectId) + '/media', {
        filename: storedName || file.name, body: file, onProgress: onProgress, timeout: 0,
        replace: !!replace
      });
    },

    /* 删掉一条片子的视频、标注和特征。已经训完的轮次留着。 */
    deleteClip: function (projectId, clipId) {
      return send('DELETE', base(projectId) + '/clips/' + encodeURIComponent(clipId));
    },

    /* 连续图整组传完后收成一条片子。 */
    adoptClip: function (projectId, fileName, replace) {
      return send('POST', base(projectId) + '/clips', { json: { file: fileName, replace: !!replace } });
    },

    /* 标注整体覆盖写。服务端要求 classes 与 clips 两个列表都在。 */
    saveAnnotations: function (projectId, payload) {
      return send('POST', base(projectId) + '/annotations', { json: payload });
    },

    /* 选定模型并真加载一次；指纹由服务端写，页面传的不算数 */
    patchModel: function (projectId, patch) {
      return send('POST', base(projectId) + '/model', { json: patch, timeout: 0 });
    },

    /* 验证集划分：提案或确认。划分单位是段，不是片子。
       propose: {split_seed, split_val_count} → 换种子/换目标窗口数，旧确认自动作废
       confirm: {train, val} 形如 [[clip_id, segment_id], ...] */
    split: function (projectId, body) {
      return send('POST', base(projectId) + '/split', { json: body || { action: 'propose' } });
    },

    extract: function (projectId, clipIds) {
      return send('POST', base(projectId) + '/features', { json: { clips: clipIds || [] }, timeout: 0 });
    },

    revealFeature: function (projectId, clipId) {
      return send('POST', base(projectId) + '/features/reveal', { json: { clip_id: clipId } });
    },

    train: function (projectId, patch) {
      return send('POST', base(projectId) + '/train', { json: patch || {}, timeout: 0 });
    },

    /* 停止不走工程锁，训练跑着的时候也收得到 */
    stop: function (projectId) {
      return send('POST', base(projectId) + '/stop', { json: {} });
    }
  };
})(window);
