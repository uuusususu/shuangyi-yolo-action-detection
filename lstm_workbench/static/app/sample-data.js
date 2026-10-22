/* 演示数据。
   真实部分：工程内 projects/test 的类别、片子元数据与 14 段标注（原样搬来），
   以及真实 settings.json 的工位参数。
   示例部分：片子的第 2/3 行、缓存与训练轮次 —— 凡是示例都带 sample 标记，
   页面必须显式渲染该标记，不把它当成真实结果。 */
(function (global) {
  'use strict';

  /* 窗口按秒定义，帧数由时间基换算 —— 和训练台的 timeline.window_frames 是同一条规则，
     前端不留第二份公式。示例片子帧率一致，所以时间基就是它自己的帧率，窗口 12 帧。 */
  var TIME_BASE_FPS = 6.284931723151575;
  var WINDOW_SECONDS = 1.92;
  var WINDOW = Math.max(1, Math.round(WINDOW_SECONDS * TIME_BASE_FPS));
  var STRIDE = 8;
  var SEED = 7;

  function windowsIn(start, end) {
    var len = end - start;
    return len >= WINDOW ? Math.floor((len - WINDOW) / STRIDE) + 1 : 0;
  }

  /* 真实：projects/test/annotations.json */
  var CLASSES = [
    { id: 'c1790157821705', name: '包装' },
    { id: 'c1790158334094', name: '封装' },
    { id: 'c1790158389521', name: '检测配件' }
  ];

  var REAL_SEGMENTS = [
    ['s1790157837393', 'c1790157821705', 2, 25],
    ['s1790158337170', 'c1790158334094', 38, 54],
    ['s1790158392774', 'c1790158389521', 55, 67],
    ['s1790158473592', 'c1790157821705', 78, 102],
    ['s1790158497395', 'c1790158334094', 105, 123],
    ['s1790158522811', 'c1790158389521', 132, 144],
    ['s1790158849144', 'c1790157821705', 178, 190],
    ['s1790158875862', 'c1790158334094', 192, 205],
    ['s1790158900946', 'c1790158389521', 215, 228],
    ['s1790158928934', 'c1790157821705', 250, 262],
    ['s1790158948787', 'c1790158334094', 265, 282],
    ['s1790158973742', 'c1790158389521', 295, 318],
    ['s1790159007901', 'c1790157821705', 352, 365],
    ['s1790159021774', 'c1790158334094', 368, 385]
  ];

  var YOLO_SHA = '6f1c2ad4e97b30c8155ea2f4d1c0b73e9a8d2456fb0c1e7d3a5948c2be0f7d16';
  var HANDS_SHA = 'b83d5f0a71c946e2d3f8a05b6c2e4179ad3b8e05f1c7269d40ab83e5c9d2017f';

  /* 真实：projects/test/settings.json。模型路径、两个指纹和类别是示例值 —— 真实那份还没选模型，
     这里填上库里的 daoredian，好让模型页演示"已选定"的样子。类别就是 daoredian.onnx 里那两个。 */
  var SETTINGS = {
    yolo_path: 'models/yolo/daoredian.onnx',
    yolo_conf_threshold: 0.8,
    yolo_sha256: YOLO_SHA,
    yolo_classes: ['导热垫', '散热片'],
    hands_sha256: HANDS_SHA,
    hands_in_training: false,
    window_seconds: WINDOW_SECONDS,
    time_base_fps: TIME_BASE_FPS,
    epochs: 30,
    learning_rate: 0.001,
    dropout: 0.2,
    stride: STRIDE,
    station_debounce_frames: 3,
    station_min_confidence: 0.5
  };

  var SAMPLE_SHA_2 = '2c7e9b41d05a38f6b1e4c2907d5f8a36c0b47e19d8253af6c1b0e9d47a3f5820';
  var SAMPLE_SHA_3 = '9a0d3c75e1b842f6a5c9703d2e8b14f7560c93a2d1e84b70f3c5926ae8d4013b';

  /* 工程内真实的 539 帧 / 6.28fps / 2048×1200 片子，视频随原型一起交付。 */
  var CLIPS = [
    {
      id: '1', file: '1.mp4', source: 'real', hasMedia: true,
      frame_count: 539, fps: 6.284931723151575, width: 2048, height: 1200, bytes: 13433998,
      segments: REAL_SEGMENTS.map(function (s) { return { id: s[0], class_id: s[1], start: s[2], end: s[3] }; })
    },
    {
      id: '2', file: '2.mp4', source: 'sample', hasMedia: false,
      frame_count: 612, fps: 6.312, width: 2048, height: 1200, bytes: 15120442,
      segments: [
        { id: 'x1', class_id: 'c1790157821705', start: 12, end: 88 },
        { id: 'x2', class_id: 'c1790158334094', start: 96, end: 171 },
        { id: 'x3', class_id: 'c1790158389521', start: 180, end: 246 },
        { id: 'x4', class_id: 'c1790157821705', start: 262, end: 340 },
        { id: 'x5', class_id: 'c1790158334094', start: 355, end: 428 },
        { id: 'x6', class_id: 'c1790158389521', start: 441, end: 508 },
        { id: 'x7', class_id: 'c1790157821705', start: 520, end: 596 }
      ]
    },
    {
      id: '3', file: '3.mp4', source: 'sample', hasMedia: false,
      frame_count: 486, fps: 6.281, width: 2048, height: 1200, bytes: 11980335, segments: []
    }
  ];

  /* 示例划分。单位是段，和训练台一样 —— 真实的划分由训练台 POST /split 算出来给人确认，
     离线模式没有后端，所以这里按一条写死的规则算：先给每个类别各挑一段保证两边都有，
     再按窗口数补到约 20%。数字是从上面这些段真算出来的，不是手填的，界面上也标了「示例」。 */
  function sideOf(units) {
    var clipNames = [], classIds = [];
    units.forEach(function (u) {
      if (clipNames.indexOf(u.clip) < 0) clipNames.push(u.clip);
      if (classIds.indexOf(u.class_id) < 0) classIds.push(u.class_id);
    });
    return {
      segments: units.length, clips: clipNames, classes: classIds,
      windows: units.reduce(function (sum, u) { return sum + u.windows; }, 0)
    };
  }

  function buildSplit() {
    var units = [];
    CLIPS.forEach(function (clip) {
      clip.segments.forEach(function (seg) {
        var n = windowsIn(seg.start, seg.end);
        if (n > 0) units.push({ clip: clip.file, class_id: seg.class_id, windows: n });
      });
    });
    var total = units.reduce(function (sum, u) { return sum + u.windows; }, 0);
    /* 一段都装不下窗口时也要给一张完整的卡：返回 ok:false 的划分，而不是 null ——
       下面日志那几行会读它的窗口数，null 会让整份示例数据在加载时就抛。 */
    if (!total) {
      return {
        ok: false, reason: '示例数据里没有一段装得下 ' + WINDOW_SECONDS + ' 秒的窗口，分不出训练和验证。',
        confirmed: false, time_base_fps: TIME_BASE_FPS, window: WINDOW, window_seconds: WINDOW_SECONDS,
        same_source: false, train: sideOf([]), val: sideOf([]), val_share: 0
      };
    }

    var taken = {}, val = [];
    CLASSES.forEach(function (c) {
      for (var i = 0; i < units.length; i++) {
        if (!taken[i] && units[i].class_id === c.id) { taken[i] = true; val.push(units[i]); return; }
      }
    });
    var valWindows = val.reduce(function (sum, u) { return sum + u.windows; }, 0);
    var goal = Math.max(valWindows, Math.round(total * 0.2));
    for (var j = 0; j < units.length && valWindows < goal; j++) {
      if (taken[j]) continue;
      taken[j] = true; val.push(units[j]); valWindows += units[j].windows;
    }

    var trainSide = sideOf(units.filter(function (u, i) { return !taken[i]; }));
    var valSide = sideOf(val);
    return {
      ok: true, reason: '', confirmed: true,
      time_base_fps: TIME_BASE_FPS, window: WINDOW, window_seconds: WINDOW_SECONDS,
      same_source: trainSide.clips.some(function (f) { return valSide.clips.indexOf(f) >= 0; }),
      train: trainSide, val: valSide, val_share: valWindows / total
    };
  }

  var SPLIT = buildSplit();

  /* 全局模型库：所有工程共用一份，换模型是换库里的文件。
     这份清单是示例，但文件名和字节数与本机 models/ 里的实际文件一致。 */
  var LIBRARY = {
    root: 'lstm_workbench/models',
    yolo: {
      dir: 'lstm_workbench/models/yolo',
      exists: true,
      files: [
        { name: 'daoredian.onnx', path: 'lstm_workbench/models/yolo/daoredian.onnx', bytes: 10024049, mtime: '2026-09-07 16:28' },
        { name: 'jiaodian3.onnx', path: 'lstm_workbench/models/yolo/jiaodian3.onnx', bytes: 39243437, mtime: '2026-07-22 16:55' },
        { name: 'san92 (1).onnx', path: 'lstm_workbench/models/yolo/san92 (1).onnx', bytes: 10023281, mtime: '2026-07-21 16:05' }
      ]
    },
    hands: { dir: 'lstm_workbench/models/hands', exists: true }
  };

  /* 模型页探测结果。演示的是已配置状态：YOLO 和手部都从库里读，
     hands.paths 列的就是真正被读的那几个文件。 */
  var PROBE = {
    sample: true,
    yolo: {
      ok: true,
      path: SETTINGS.yolo_path,
      classes: ['导热垫', '散热片'],
      sha256: YOLO_SHA
    },
    hands: {
      ok: true,
      complexity: 0,
      sha256: HANDS_SHA,
      source: 'library',
      dir: LIBRARY.hands.dir,
      paths: [
        { name: 'palm_detection_lite.tflite', path: LIBRARY.hands.dir + '\\palm_detection\\palm_detection_lite.tflite', bytes: 1985440 },
        { name: 'hand_landmark_lite.tflite', path: LIBRARY.hands.dir + '\\hand_landmark\\hand_landmark_lite.tflite', bytes: 2071597 },
        { name: 'hand_landmark_tracking_cpu.binarypb', path: LIBRARY.hands.dir + '\\hand_landmark\\hand_landmark_tracking_cpu.binarypb', bytes: 2901 }
      ]
    }
  };

  var FEATURE_CODE_VERSION = '1';
  var FEATURE_LENGTH = SETTINGS.yolo_classes.length * 5 + 126; /* 2×5 + 126 = 136 */

  /* 缓存：一条最新、一条因阈值改变而失效、一条从未抽取。 */
  var FEATURES = {
    sample: true,
    code_version: FEATURE_CODE_VERSION,
    length: FEATURE_LENGTH,
    blocked_reason: '',
    rows: [
      { clip_id: '1', state: 'fresh', frames: 539, extracted_at: '2026-09-24 10:31', reason: null },
      {
        clip_id: '2', state: 'stale', frames: 612, extracted_at: '2026-09-23 16:02',
        reason: { key: 'yolo_conf_threshold', from: '0.80', to: '0.70' }
      },
      { clip_id: '3', state: 'missing', frames: 486, extracted_at: null, reason: null }
    ]
  };

  var GATE = { macro_f1: 0.90, min_recall: 0.85, idle_false_alarm: 0.10 };

  /* 训练曲线用固定种子生成，形状稳定、可复现；数值一律按示例标注。 */
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function curve(seed, epochs, final, start, noise, floor) {
    var rand = rng(seed);
    var out = [];
    for (var i = 0; i < epochs; i++) {
      var t = i / Math.max(1, epochs - 1);
      var base = start + (final - start) * (1 - Math.pow(1 - t, 1.8));
      var wobble = (rand() - 0.5) * noise * (1 - t * 0.6);
      var v = base + wobble;
      if (floor !== undefined) v = Math.max(floor, v);
      out.push(Math.round(v * 1000) / 1000);
    }
    return out;
  }

  function buildRun(spec) {
    var epochs = spec.epochs;
    var train = curve(spec.seed, epochs, spec.train_final, 1.32, 0.05, 0.02);
    var val = curve(spec.seed + 11, epochs, spec.val_final, 1.44, 0.11, 0.03);
    /* 让验证损失在末端抬头，复现"训练还在降、验证开始涨"的过拟合形状。 */
    for (var i = epochs - 4; i < epochs; i++) val[i] = Math.round((val[i] + (i - (epochs - 5)) * 0.02) * 1000) / 1000;
    return {
      sample: true,
      name: spec.name,
      at: spec.at,
      epochs: epochs,
      best_epoch: spec.best_epoch,
      class_ids: ['idle'].concat(CLASSES.map(function (c) { return c.id; })),
      input_size: FEATURE_LENGTH,
      window: WINDOW,
      window_seconds: WINDOW_SECONDS,
      time_base_fps: TIME_BASE_FPS,
      /* 示例轮次是按整条片子划分的（片子 2、3 进训练，片子 1 进验证），不是同源验证。 */
      same_source: false,
      stride: STRIDE,
      seed: SEED,
      gate: GATE,
      gate_passed: spec.passed,
      gate_reasons: spec.reasons,
      metrics: {
        macro_f1: spec.macro_f1,
        min_recall: spec.min_recall,
        idle_false_alarm: spec.idle_false_alarm,
        per_class: spec.per_class,
        confusion: spec.confusion
      },
      curves: { epochs: range(1, epochs), train_loss: train, val_loss: val, macro_f1: spec.f1_curve, min_recall: spec.recall_curve, idle_false_alarm: spec.idle_curve },
      split: spec.split,
      mistakes: spec.mistakes || [],
      files: ['best_lstm.pt', 'contract.json', 'metrics.json', 'split.json', 'log.txt']
    };
  }

  function range(a, b) { var out = []; for (var i = a; i <= b; i++) out.push(i); return out; }

  function perClass(rows) {
    return rows.map(function (r) {
      return {
        id: r[0], name: r[1], support: r[2], precision: r[3], recall: r[4], f1: r[5]
      };
    });
  }

  /* 运行 A：三期都过门禁 —— 混淆矩阵与逐类指标互相对得上。 */
  var RUN_A = buildRun({
    name: '20260924-1042', at: '2026-09-24 10:42', seed: 101, epochs: 30, best_epoch: 24,
    train_final: 0.08, val_final: 0.21,
    macro_f1: 0.938, min_recall: 0.886, idle_false_alarm: 0.013,
    f1_curve: curve(201, 30, 0.938, 0.52, 0.06),
    recall_curve: curve(202, 30, 0.886, 0.44, 0.07),
    idle_curve: curve(203, 30, 0.013, 0.31, 0.06, 0.005),
    passed: true, reasons: [],
    per_class: perClass([
      ['idle', '空闲', 96, 0.979, 0.979, 0.979],
      ['c1790157821705', '包装', 64, 0.953, 0.953, 0.953],
      ['c1790158334094', '封装', 52, 0.906, 0.923, 0.914],
      ['c1790158389521', '检测配件', 44, 0.929, 0.886, 0.907]
    ]),
    confusion: [
      [94, 2, 0, 0],
      [1, 61, 2, 0],
      [0, 1, 48, 3],
      [1, 0, 4, 39]
    ],
    split: { train: ['2.mp4', '3.mp4'], val: ['1.mp4'] },
    mistakes: [
      { clip: '1.mp4', start: 295, end: 318, truth: '检测配件', pred: '封装', confidence: 0.61 },
      { clip: '1.mp4', start: 368, end: 385, truth: '封装', pred: '检测配件', confidence: 0.57 }
    ]
  });

  var RUN_B = buildRun({
    name: '20260923-1631', at: '2026-09-23 16:31', seed: 102, epochs: 30, best_epoch: 19,
    train_final: 0.12, val_final: 0.29,
    macro_f1: 0.912, min_recall: 0.812, idle_false_alarm: 0.071,
    f1_curve: curve(211, 30, 0.912, 0.47, 0.07),
    recall_curve: curve(212, 30, 0.812, 0.39, 0.08),
    idle_curve: curve(213, 30, 0.071, 0.38, 0.07, 0.005),
    passed: false, reasons: ['最低召回 0.812 低于 0.85'],
    per_class: perClass([
      ['idle', '空闲', 96, 0.962, 0.948, 0.955],
      ['c1790157821705', '包装', 64, 0.925, 0.906, 0.915],
      ['c1790158334094', '封装', 52, 0.882, 0.865, 0.873],
      ['c1790158389521', '检测配件', 44, 0.841, 0.812, 0.826]
    ]),
    confusion: [
      [91, 3, 1, 1],
      [2, 58, 3, 1],
      [1, 2, 45, 4],
      [2, 1, 5, 36]
    ],
    split: { train: ['2.mp4', '3.mp4'], val: ['1.mp4'] },
    mistakes: [
      { clip: '1.mp4', start: 55, end: 67, truth: '检测配件', pred: '封装', confidence: 0.66 },
      { clip: '1.mp4', start: 215, end: 228, truth: '检测配件', pred: '包装', confidence: 0.59 }
    ]
  });

  var RUN_C = buildRun({
    name: '20260922-0915', at: '2026-09-22 09:15', seed: 103, epochs: 24, best_epoch: 11,
    train_final: 0.19, val_final: 0.41,
    macro_f1: 0.868, min_recall: 0.774, idle_false_alarm: 0.126,
    f1_curve: curve(221, 24, 0.868, 0.43, 0.08),
    recall_curve: curve(222, 24, 0.774, 0.35, 0.09),
    idle_curve: curve(223, 24, 0.126, 0.44, 0.08, 0.005),
    passed: false, reasons: ['宏平均 F1 0.868 低于 0.90', '最低召回 0.774 低于 0.85', '空闲误报 0.126 高于 0.10'],
    per_class: perClass([
      ['idle', '空闲', 88, 0.901, 0.875, 0.888],
      ['c1790157821705', '包装', 60, 0.879, 0.850, 0.864],
      ['c1790158334094', '封装', 48, 0.845, 0.812, 0.828],
      ['c1790158389521', '检测配件', 40, 0.812, 0.774, 0.792]
    ]),
    confusion: [
      [77, 5, 3, 3],
      [3, 51, 4, 2],
      [2, 3, 39, 4],
      [3, 2, 4, 31]
    ],
    split: { train: ['1.mp4', '2.mp4'], val: ['3.mp4'] },
    mistakes: [
      { clip: '3.mp4', start: 96, end: 171, truth: '封装', pred: '检测配件', confidence: 0.52 }
    ]
  });

  var RUNS = [RUN_A, RUN_B, RUN_C];

  var LOG_LINES = [
    ['10:42:01', 'info', '读取标注：3 个类别、3 条片子、21 段'],
    ['10:42:01', 'info', '特征缓存：1 条最新、1 条失效、1 条缺失'],
    ['10:42:06', 'info', '用 12.4s 重抽 2.mp4 的特征（612 帧）'],
    ['10:42:11', 'info', '可用窗口：训练 ' + SPLIT.train.windows + '、验证 ' + SPLIT.val.windows + '（按段划分，seed ' + SEED + '）'],
    ['10:42:11', 'info', '窗口 ' + WINDOW_SECONDS + ' 秒（' + TIME_BASE_FPS.toFixed(2) + 'fps 下 ' + WINDOW + ' 帧）、步长 ' + STRIDE + '、特征长度 ' + FEATURE_LENGTH],
    ['10:42:12', 'info', '开始训练：30 轮，全量批，Adam lr=1e-3'],
    ['10:42:19', 'info', 'epoch 05  train 0.412  val 0.503  macro-F1 0.681'],
    ['10:42:26', 'info', 'epoch 10  train 0.244  val 0.331  macro-F1 0.789'],
    ['10:42:33', 'warn', 'epoch 14  验证损失连续 4 轮未降，继续观察'],
    ['10:42:38', 'info', 'epoch 19  train 0.121  val 0.244  macro-F1 0.902'],
    ['10:42:43', 'info', 'epoch 24  train 0.084  val 0.212  macro-F1 0.938  ← 目前最好，已保存'],
    ['10:42:49', 'warn', 'epoch 27  验证损失抬头而训练损失仍在降，按验证宏平均 F1 保留第 24 轮'],
    ['10:42:52', 'info', 'epoch 30  train 0.080  val 0.211  macro-F1 0.931'],
    ['10:42:52', 'ok', '门禁通过：宏平均 F1 0.938、最低召回 0.886、空闲误报 0.013'],
    ['10:42:52', 'info', '写入 runs/20260924-1042：best_lstm.pt、contract.json、metrics.json、split.json、log.txt']
  ];

  global.OD = global.OD || {};
  global.OD.data = {
    constants: { STRIDE: STRIDE, SEED: SEED, GATE: GATE },
    project: { id: '包装线-01', sample: true, created_at: '2026-09-23 17:24', updated_at: '2026-09-24 10:42' },
    classes: CLASSES,
    clips: CLIPS,
    settings: SETTINGS,
    split: SPLIT,
    library: LIBRARY,
    probe: PROBE,
    features: FEATURES,
    runs: RUNS,
    log: LOG_LINES,
    fingerprints: { yolo: YOLO_SHA, hands: HANDS_SHA, alt2: SAMPLE_SHA_2, alt3: SAMPLE_SHA_3 }
  };
})(window);
