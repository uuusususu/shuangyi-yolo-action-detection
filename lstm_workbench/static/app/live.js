/* 在线模式：把真实训练台（app.py + 8771）的载荷翻译成视图吃的 global.OD.data。
   页面由训练台托管时 boot() 先拉一次工程，成功就整体换掉示例数据；
   双击打开（file://）时 api.available 为假，一套示例数据原样留着，什么都不换。

   两条硬约束：
   1. constants 必须沿用同一个对象。view-annotate.js 在脚本加载时就抓了它的引用，
      换掉的话窗长、步长、随机种子会变成一个读不到的对象。
   2. 真实接口没给的字段一律留 null，不许拿示例值或 0 顶上 —— 视图必须显式渲染成 —。 */
(function (global) {
  'use strict';

  var api = global.OD.api;

  var IDLE = 'idle';
  var IDLE_NAME = '空闲';
  var RUN_DETAIL_LIMIT = 5;

  var source = 'offline';
  var projectId = null;
  var projectNames = [];
  var bootError = '';
  /* 全局模型库清单。读不到就是 null —— 和「库是空的」是两回事，模型页分开写。 */
  var library = null;

  function isLive() { return source === 'live'; }
  function error() { return bootError; }
  function projects() { return projectNames.slice(); }
  function current() { return projectId; }

  /* ---------- 小工具 ---------- */

  function arr(value) { return Array.isArray(value) ? value : []; }

  /* 数字就是数字，不是数字一律 null —— 视图把 null 渲染成 —，不做任何补零。 */
  function num(value, fallback) {
    if (typeof value === 'number' && isFinite(value)) return value;
    if (fallback === undefined) return null;
    return fallback;
  }

  /* 轮次目录名 20260924-104233 → 2026-09-24 10:42 */
  function stamp(name) {
    var m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(String(name));
    return m ? m[1] + '-' + m[2] + '-' + m[3] + ' ' + m[4] + ':' + m[5] : String(name);
  }

  /* 曲线里宏平均 F1 最高的那一轮，就是 trainer.py 存 best_lstm.pt 的依据。 */
  function bestEpoch(history) {
    var best = 0, seen = -Infinity;
    history.forEach(function (row, i) {
      var v = num(row.macro_f1);
      if (v !== null && v > seen) { seen = v; best = num(row.epoch, i + 1); }
    });
    return best;
  }

  function series(history, key) {
    return history.map(function (row) { return num(row[key]); });
  }

  function nameMap(annotations) {
    var map = {};
    map[IDLE] = IDLE_NAME;
    arr(annotations && annotations.classes).forEach(function (c) { map[c.id] = c.name; });
    return map;
  }

  /* ---------- 轮次 ---------- */

  /* ctx 由 toData 现算：不能在映射途中回头读 global.OD.data，那时它还是上一份数据。 */
  function toRun(name, detail, ctx) {
    var report = detail.report || {};
    var contract = detail.contract || {};
    var metrics = report.metrics || {};
    var history = arr(report.history);
    var constants = ctx.constants;
    var names = ctx.names;
    var fileById = ctx.fileById;

    function label(id) { return names[id] || id; }

    /* split.json 的 train / val 是 [[clip_id, segment_id], ...] —— 划分单位是段。
       界面要的是「涉及哪些片子」和「各多少段」，所以在这里收敛一次。 */
    var split = detail.split || {};
    function clipIds(entries) {
      var seen = [];
      arr(entries).forEach(function (item) {
        var clipId = Array.isArray(item) ? item[0] : item;
        if (clipId && seen.indexOf(clipId) < 0) seen.push(clipId);
      });
      return seen;
    }
    var trainIds = clipIds(split.train);
    var valIds = clipIds(split.val);

    var reasons = [];
    if (report.reason) reasons.push(String(report.reason));
    if (!report.usable && !report.reason && report.stopped) reasons.push('这一轮被中途停止，没有写完整的报告');

    return {
      sample: false,
      name: name,
      display_name: contract.display_name || '',
      at: stamp(name),
      hasCurves: history.length > 0,
      epochs: num(contract.epochs, history.length) || ctx.settings.epochs,
      best_epoch: bestEpoch(history),
      class_ids: arr(contract.class_ids).length ? contract.class_ids : [IDLE],
      input_size: num(contract.input_size, ctx.featureLength),
      /* 契约里一定有 window（trainer.py 必填参数）。真缺了就用这个工程当前窗口顶上 ——
         那是一个真实算出来的帧数，不是某个写死的常数。 */
      window: num(contract.window, ctx.splitWindow),
      stride: num(contract.stride, constants.STRIDE),
      window_seconds: num(contract.window_seconds),
      time_base_fps: num(contract.time_base_fps),
      same_source: contract.same_source_validation === true,
      network: contract.network === 'mstcn' ? 'mstcn' : 'lstm',
      smoothing: contract.smoothing === true,
      train_unit: contract.train_unit || '',
      seed: num(contract.seed, constants.SEED),
      gate: constants.GATE,
      gate_passed: !!report.usable,
      gate_reasons: reasons,
      metrics: {
        macro_f1: num(metrics.macro_f1),
        min_recall: num(metrics.min_recall),
        idle_false_alarm: num(metrics.idle_false_alarm),
        f1_50: num(metrics.f1_50),
        edit: num(metrics.edit),
        frame_accuracy: num(metrics.frame_accuracy),
        pred_segments: num(metrics.pred_segments),
        true_segments: num(metrics.true_segments),
        per_class: arr(metrics.per_class).map(function (row) {
          return {
            id: row.class_id, name: label(row.class_id),
            support: num(row.support), precision: num(row.precision),
            recall: num(row.recall), f1: num(row.f1)
          };
        }),
        confusion: arr(metrics.confusion)
      },
      curves: {
        epochs: history.map(function (row, i) { return num(row.epoch, i + 1); }),
        train_loss: series(history, 'train_loss'),
        val_loss: series(history, 'val_loss'),
        macro_f1: series(history, 'macro_f1'),
        min_recall: series(history, 'min_recall'),
        idle_false_alarm: series(history, 'idle_false_alarm'),
        f1_50: series(history, 'f1_50'),
        edit: series(history, 'edit'),
        frame_accuracy: series(history, 'frame_accuracy'),
        pred_segments: series(history, 'pred_segments'),
        true_segments: series(history, 'true_segments')
      },
      /* split.json 记的是片子 id，视图要的是文件名。 */
      split: {
        train: trainIds.map(function (id) { return fileById[id] || id; }),
        val: valIds.map(function (id) { return fileById[id] || id; })
      },
      /* trainer.py 的 mistakes 只有 clip_id / truth / predicted / start / end，没有置信度。 */
      mistakes: arr(report.mistakes).map(function (m) {
        return {
          clip: fileById[m.clip_id] || m.clip_id,
          start: num(m.start, 0),
          end: num(m.end, 0),
          truth: label(m.truth),
          pred: label(m.predicted !== undefined ? m.predicted : m.pred),
          confidence: null
        };
      }),
      files: arr(detail.files)
    };
  }

  /* ---------- 手部模型状态 ---------- */

  /* 手部后端的"已加载"以训练台启动时真加载的结果为准（library.hands.load），
     它比 settings 里有没有指纹可靠：指纹只有手动点过「检查并保存」才有，
     而模型能不能加载是启动时就能知道的事实。
     预热还没跑完是 loading，跑失败是 failed 并带原因 —— 都不能当成"已加载"。
     预热没开（idle，比如测试里起的服务）才退回只看指纹。 */
  function handsProbe(settings, complexity) {
    var load = library && library.hands && library.hands.load;
    var state = load && load.state;
    if (state === 'ready') {
      return {
        ok: true,
        complexity: num(load.model_complexity, complexity),
        sha256: load.sha256 || settings.hands_sha256 || '',
        source: load.source || '', dir: load.dir || '', paths: arr(load.paths),
        loadState: 'ready'
      };
    }
    if (state === 'loading') {
      return { ok: false, complexity: complexity, sha256: settings.hands_sha256 || '', source: '', dir: '', paths: [], loadState: 'loading' };
    }
    if (state === 'failed') {
      return {
        ok: false, complexity: complexity, sha256: '', source: load.source || '', dir: load.dir || '', paths: [],
        error: load.error || '手部模型加载失败', loadState: 'failed'
      };
    }
    return settings.hands_sha256
      ? { ok: true, complexity: complexity, sha256: settings.hands_sha256, source: '', dir: '', paths: [], loadState: 'idle' }
      : { ok: false, complexity: complexity, sha256: '', source: '', dir: '', paths: [], loadState: 'idle' };
  }

  var pollTimer = null;
  var POLL_LIMIT = 40;

  /* 预热是后台线程，第一次打开页面时可能还在 loading。隔一秒问一次，
     变成 ready / failed 就整体重画；最多问 POLL_LIMIT 次，别无限轮询。 */
  function watchHandsLoad(remaining) {
    if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null; }
    var load = library && library.hands && library.hands.load;
    if (!load || load.state !== 'loading' || remaining <= 0) return;
    pollTimer = setTimeout(function () {
      pollTimer = null;
      refreshLibrary().then(function () {
        var now = library && library.hands && library.hands.load;
        if (now && now.state !== 'loading') {
          reload().catch(function () {});
        } else {
          watchHandsLoad(remaining - 1);
        }
      });
    }, 1000);
  }

  /* ---------- 整体映射 ---------- */

  function toData(payload, details) {
    var settings = payload.settings || {};
    var annotations = payload.annotations || {};
    var names = nameMap(annotations);

    /* 素材是不是还在：feature_store.summary() 每条都会试一次 find_media，
       找不到就把原因写进 note。note 非空 = 素材没了。 */
    var noteById = {};
    arr(payload.features && payload.features.clips).forEach(function (row) {
      if (row.note) noteById[row.clip_id] = row.note;
    });

    var fileById = {};
    var clips = arr(annotations.clips).map(function (raw) {
      fileById[raw.id] = raw.file || raw.id;
      return {
        id: raw.id,
        file: raw.file || raw.id,
        source: 'real',
        hasMedia: !noteById[raw.id],
        frame_count: num(raw.frame_count, 0),
        fps: num(raw.fps, 0),
        /* 分辨率只用于显示。真实接口不给就是 null，视图渲染成 —，不拿一个默认尺寸顶上。 */
        width: num(raw.width),
        height: num(raw.height),
        bytes: null,          /* 真实接口不给文件大小 */
        note: noteById[raw.id] || '',
        segments: arr(raw.segments).map(function (s) {
          return { id: s.id, class_id: s.class_id, start: s.start, end: s.end };
        }),
        excluded: arr(raw.excluded).map(function (s) {
          return { id: s.id, start: s.start, end: s.end };
        })
      };
    });

    /* 有文件且指纹还对得上是最新；有文件但指纹变了是已失效；没有文件是未抽取。 */
    var stats = arr(payload.features && payload.features.clips);
    var rows = stats.map(function (row) {
      var mediaMissing = !!row.note;
      var state = row.fresh ? 'fresh' : (row.file ? 'stale' : 'missing');
      return {
        clip_id: row.clip_id,
        state: state,
        mediaMissing: mediaMissing,
        frames: num(row.frames),
        feature_length: num(row.feature_length),
        file: row.file || '',
        bytes: num(row.bytes),
        extracted_at: row.extracted_at || null,
        reason: null
      };
    });

    /* 抽取质量：每帧的 no_part / no_hand 计数都在清单里，能加就加出真实比例。 */
    var sumFrames = 0, sumNoPart = 0, sumNoHand = 0, complete = stats.length > 0;
    stats.forEach(function (row) {
      if (typeof row.frames !== 'number' || typeof row.no_part_frames !== 'number' || typeof row.no_hand_frames !== 'number') {
        complete = false;
        return;
      }
      sumFrames += row.frames;
      sumNoPart += row.no_part_frames;
      sumNoHand += row.no_hand_frames;
    });
    var quality = complete && sumFrames > 0
      ? { no_part_ratio: sumNoPart / sumFrames, no_hand_ratio: sumNoHand / sumFrames, frames: sumFrames }
      : null;

    var cachedLength = null;
    stats.forEach(function (row) { if (typeof row.feature_length === 'number') cachedLength = row.feature_length; });

    /* 零件类别来自模型，不是人填的 —— settings.yolo_classes 只有真探测过才有值。
       长度以训练台为准：抽过就用清单里真抽出来的那个，没抽过就用后端按模型类别算好的。
       公式不再在客户端留第二份副本 —— 两份迟早会分叉。 */
    var featureLength = cachedLength !== null ? cachedLength : num(payload.features && payload.features.length, 0);

    /* 手部复杂度不在 settings 里，但每一轮的 contract.json 都记了。有轮次就有真值。 */
    var handsComplexity = null;
    details.forEach(function (detail) {
      var value = detail && detail.contract && detail.contract.hands_model_complexity;
      if (handsComplexity === null && typeof value === 'number') handsComplexity = value;
    });

    var ctx = {
      names: names, fileById: fileById, settings: settings,
      featureLength: featureLength, constants: global.OD.data.constants,
      splitWindow: num(payload.split && payload.split.window)
    };

    var runs = details.map(function (detail) { return toRun(detail.name, detail, ctx); });

    return {
      /* 同一个对象，不能换 —— view-annotate.js 在加载时就把它抓走了。 */
      constants: global.OD.data.constants,
      project: { id: payload.id, sample: false, created_at: null, updated_at: null },
      classes: arr(annotations.classes).map(function (c) { return { id: c.id, name: c.name }; }),
      clips: clips,
      settings: settings,
      library: library,
      probe: {
        sample: false,
        /* 指纹非空 = 上一次"检查并保存"真的加载成功过；这里只是还原，不重新探测。
           手部复杂度 settings 里没有，从最近一轮的 contract.json 里捡回来；一轮都没跑过就是 null。
           手部的 source / paths 也还原不了：settings 只存指纹，浏览器刷新后不知道上次加载的是
           哪几个文件，所以留空 —— 模型页会写明「检查并保存」之后才有文件清单，不拿库路径顶上。 */
        yolo: settings.yolo_sha256
          ? { ok: true, path: settings.yolo_path, classes: arr(settings.yolo_classes), sha256: settings.yolo_sha256 }
          : { ok: false, path: settings.yolo_path || '', classes: [], sha256: '' },
        hands: handsProbe(settings, handsComplexity)
      },
      features: {
        sample: false,
        code_version: null,
        length: featureLength,
        blocked_reason: (payload.features && payload.features.blocked_reason) || '',
        rows: rows,
        quality: quality
      },
      runs: runs,
      log: [],
      fingerprints: { yolo: settings.yolo_sha256 || '', hands: settings.hands_sha256 || '' },
      /* 训练台自己报的阻断与警告，逐条原样带到数据页 —— 比如段交叠，客户端推不出来。 */
      blockers: arr(payload.blockers),
      warnings: arr(payload.warnings),
      /* 验证集划分。单位是段：train/val 都是 [[clip_id, segment_id], ...]。
         confirmed 为假表示这是提案，训练会被拒到确认之后。 */
      split: payload.split || null
    };
  }

  /* ---------- 载入 ---------- */

  function apply(id, payload, details) {
    global.OD.data = toData(payload, details);
    source = 'live';
    projectId = id;
    if (global.OD.shell && global.OD.shell.state) {
      global.OD.shell.state.source = 'live';
      global.OD.shell.state.projectId = id;
    }
  }

  function loadProject(id) {
    return api.project(id).then(function (payload) {
      /* payload.runs 是升序的，最近的在最后；只拉最近几轮的详情，够结果页对比用。 */
      var recent = arr(payload.runs).slice(-RUN_DETAIL_LIMIT).reverse();
      return Promise.all(recent.map(function (name) {
        return api.run(id, name).catch(function () { return { name: name, files: [] }; });
      })).then(function (details) {
        apply(id, payload, details);
        return payload;
      });
    });
  }

  /* 写操作之后重新拉一遍：磁盘已经变了，界面必须跟着真的数据走，不能凭返回值拼。 */
  function reload() {
    if (!isLive()) return Promise.resolve(null);
    return loadProject(projectId).then(function (payload) {
      if (global.OD.shell) { global.OD.shell.render(); }
      return payload;
    });
  }

  /* 建了工程或者删了目录之后重新拉名单：磁盘变了，列表必须跟着变。 */
  function refreshProjects() {
    if (!api.available) return Promise.resolve(projectNames.slice());
    return api.projects().then(function (names) {
      projectNames = names;
      return names;
    });
  }

  /* 模型库清单。往库里丢了新模型之后靠"重新扫描"再拉一遍。
     拉不到就置空而不是保留旧值 —— 宁可显示成读不到，也不显示一份过期的清单。 */
  function refreshLibrary() {
    if (!api.available) return Promise.resolve(null);
    return api.models().then(function (info) {
      library = info;
      return info;
    }).catch(function () {
      library = null;
      return null;
    });
  }

  function mount() { global.OD.shell.mount(); }

  function boot() {
    /* 双击打开：没有后端可谈，示例数据原样用，视图里的写操作会自己说明不写盘。 */
    if (!api.available) { mount(); return Promise.resolve(false); }
    return api.projects().then(function (names) {
      projectNames = names;
      if (!names.length) {
        bootError = '训练台里一个工程都还没有';
        mount();
        return false;
      }
      return refreshLibrary().then(function () { return loadProject(names[0]); }).then(function () {
        mount();
        watchHandsLoad(POLL_LIMIT);
        api.progress(projectId).then(function (progress) {
          var train = global.OD.views && global.OD.views.train;
          if (train && train.watch) train.watch(progress);
        }).catch(function () {});
        return true;
      });
    }).catch(function (err) {
      /* 服务没起、端口被占、返回不是 JSON —— 一律退回离线，把原因留在顶栏上。 */
      bootError = err.message;
      source = 'offline';
      mount();
      return false;
    });
  }

  global.OD.live = {
    boot: boot, reload: reload, loadProject: loadProject, refreshProjects: refreshProjects,
    refreshLibrary: refreshLibrary,
    isLive: isLive, error: error, projects: projects, current: current,
    stamp: stamp
  };
})(window);
