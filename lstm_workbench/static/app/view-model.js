/* 模型与工位。字段集合与真实训练台一一对应，指纹只由"检查并保存"写入。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;

  var RULES = {
    yolo_conf_threshold: { min: 0, max: 1, label: 'YOLO 置信度阈值', msg: '必须在 0 到 1 之间。' },
    epochs: { min: 1, max: 500, label: '训练轮数', msg: '必须在 1 到 500 之间。' },
    learning_rate: { min: 0.000001, max: 1, label: '学习率', msg: '必须在 0.000001 到 1 之间。' },
    dropout: { min: 0, max: 0.9, label: 'Dropout', msg: '必须在 0 到 0.9 之间。' },
    window_seconds: { min: 0.04, max: 60, label: '窗口秒数', msg: '必须在 0.04 到 60 秒之间。' },
    stride: { min: 1, max: 64, label: '步长', msg: '必须在 1 到 64 帧之间。' },
    station_debounce_frames: { min: 1, max: 120, label: '连续确认帧数', msg: '必须在 1 到 120 之间。' },
    station_min_confidence: { min: 0, max: 1, label: '目标通过阈值', msg: '必须在 0 到 1 之间。' }
  };

  function numField(key, label, step, help) {
    var s = global.OD.data.settings;
    var input = el('input', {
      class: 'form-control mono', type: 'number', step: step || '1', id: 'f-' + key,
      value: s[key] === null || s[key] === undefined ? '' : s[key], 'aria-describedby': 'h-' + key
    });
    var hint = el('div', { class: 'help', id: 'h-' + key, text: help || '' });
    input.addEventListener('input', function () {
      hint.className = 'help'; hint.textContent = help || '';
      input.classList.remove('is-bad');
    });
    return el('div', { class: 'field' }, [el('label', { class: 'form-label', for: 'f-' + key, text: label }), input, hint]);
  }

  function fieldValue(key) {
    var input = document.getElementById('f-' + key);
    return input ? input.value : '';
  }

  /* 零件类别跟着模型走 —— 它定下特征行里「框」那一段有哪些格子、按什么顺序排。
     这一栏只读：手填过一次就可能和模型对不上，而对不上不报错，只会让那一段永远全零。 */
  function partClassesBlock(y) {
    var classes = (y && y.classes) || [];
    var body;
    if (!classes.length) {
      body = el('div', { class: 'callout warn sp-mt-2' }, [
        icon('info', 15),
        el('div', null, [
          el('strong', { text: '还没有读到类别。' }),
          el('div', { class: 'sp-mt-2', text: '换上面的 ONNX 会重新加载这个模型，类别在这里刷新。没有类别就抽不了特征。' })
        ])
      ]);
    } else {
      body = el('div', null, [
        el('div', { class: 'row sp-mt-2' }, classes.map(function (name, index) {
          return el('span', { class: 'badge plain mono', title: '特征行里第 ' + (index + 1) + ' 个 5 数块' }, name);
        })),
        el('div', { class: 'help', text: '一共 ' + classes.length + ' 类，顺序就是模型自带的顺序。换模型会重新读一遍。' })
      ]);
    }
    return el('div', { class: 'field', id: 'part-classes' }, [
      el('div', { class: 'form-label', text: '零件类别（来自模型，不可编辑）' }),
      body
    ]);
  }

  function showClassesLoading() {
    var host = document.getElementById('part-classes');
    if (!host) return;
    D.clear(host);
    host.appendChild(el('div', { class: 'form-label', text: '零件类别（来自模型，不可编辑）' }));
    host.appendChild(el('div', { class: 'callout warn sp-mt-2' }, [
      icon('info', 15),
      el('div', { text: '正在加载这个 ONNX，读完才会换成它自己的类别。' })
    ]));
  }

  /* 手部复杂度没有落进 settings，只有"检查并保存"那一次真的探测才拿得到。
     没探过就是 —，不许拿 0 顶上。0 是 lite、1 是 full，跟 model_registry 的约定一致。 */
  function complexityText(value) {
    if (typeof value !== 'number') return '—';
    return value + (value === 0 ? '（lite 手掌 + 关键点）' : '（full 手掌 + 关键点）');
  }

  function validate() {
    var problems = [];
    Object.keys(RULES).forEach(function (key) {
      var input = document.getElementById('f-' + key);
      var hint = document.getElementById('h-' + key);
      var rule = RULES[key];
      var raw = input.value.trim();
      var value = Number(raw);
      var bad = raw === '' || isNaN(value) || value < rule.min || value > rule.max;
      input.classList.toggle('is-bad', bad);
      hint.className = 'help' + (bad ? ' bad' : '');
      hint.textContent = bad ? rule.msg : (RULES[key].ok || '');
      if (bad) problems.push(key);
    });
    return problems;
  }

  function render() {
    var data = global.OD.data, probe = data.probe;
    var st = S.stepState('model');

    var host = el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '模型与工位' }),
        el('p', { text: 'YOLO 和手部模型只在这里检查，不重训。两个模型都来自工程根的全局模型库，所有工程共用一份；检查出来的指纹会写进每轮训练的契约文件，锁定这一轮用的是哪几个文件。' })
      ]),
      st === 'bad' ? el('div', { class: 'callout bad sp-mb-4' }, [
        icon('alert', 16),
        el('div', null, [el('strong', { text: '还有后端没就绪。 ' }), '训练前每一项都要检查通过；没通过的项不会写指纹，特征缓存也就不会建立。'])
      ]) : null,
      el('div', { class: 'grid c2' }, [yoloCard(probe), handsCard(probe)]),
      el('div', { class: 'grid c2' }, [stationCard(), trainCard()]),
      el('div', { class: 'card' }, [
        el('header', null, [
          el('h2', { text: '检查并保存' }),
          el('span', { class: 'end' }, probe.sample ? S.sampleTag('探测结果为示例') : null)
        ]),
        el('p', { class: 'hint', text: '点一次会真的去加载一遍 ONNX 和手部模型：拿到类别名单和文件指纹再落盘。页面传上来的指纹一律不算数。' }),
        el('div', { class: 'row' }, el('button', {
          class: 'btn btn-primary', type: 'button', id: 'check-btn', onclick: runCheck
        }, [icon('check', 14), '检查并保存'])),
        el('div', { id: 'check-report', class: 'report' })
      ])
    ]));

    return host;
  }

  /* 模型库里的 ONNX 清单。库读不到（双击打开的离线页，或 /api/models 没起）时是 null，
     和「库是空的」不是一回事，提示语分开写。 */
  function libraryFiles() {
    var library = global.OD.data.library;
    return (library && library.yolo && library.yolo.files) ? library.yolo.files : [];
  }

  /* 存着的那个文件可能已经不在库里了。仍然列出来 —— 否则一保存就把用户的选定悄悄换掉。 */
  function yoloChoices() {
    var saved = global.OD.data.settings.yolo_path || '';
    var files = libraryFiles();
    var choices = files.map(function (item) {
      return { value: item.path, label: item.name + ' · ' + D.fmtBytes(item.bytes) + ' · ' + item.mtime };
    });
    if (saved && !files.some(function (item) { return item.path === saved; })) {
      choices.unshift({ value: saved, label: saved + '（已不在库里）' });
    }
    if (!saved && !choices.length) choices.unshift({ value: '', label: '（还没选定）' });
    return choices;
  }

  function yoloPicker() {
    var choices = yoloChoices();
    var select = el('select', { class: 'form-select mono', id: 'f-yolo_path' },
      choices.map(function (c) { return el('option', { value: c.value, text: c.label }); }));
    var saved = global.OD.data.settings.yolo_path || '';
    select.value = saved || (choices[0] ? choices[0].value : '');
    select.addEventListener('change', function () {
      var chosen = selectedYoloPath();
      if (!chosen || chosen === String(saved || '')) return;
      runCheck();
    });
    return select;
  }

  function rescan() {
    var live = global.OD.live;
    if (!live.isLive()) {
      S.toast('离线原型没有后端，扫不到磁盘上的模型库', 'warn');
      return;
    }
    var button = document.getElementById('rescan-btn');
    if (button) button.disabled = true;
    live.refreshLibrary().then(function (info) {
      var count = info && info.yolo && info.yolo.files ? info.yolo.files.length : 0;
      S.toast(count ? '模型库里有 ' + count + ' 个 ONNX' : '模型库里一个 ONNX 都没有', count ? 'ok' : 'warn');
      global.OD.shell.render();
    }).catch(function (err) {
      S.toast('重新扫描失败：' + err.message, 'bad');
      if (button) button.disabled = false;
    });
  }

  /* 库是空的、或压根读不到，都要说清楚该往哪放文件。 */
  function libraryNote() {
    if (libraryFiles().length) return null;
    var library = global.OD.data.library;
    var dir = library && library.yolo ? library.yolo.dir : '';
    return el('div', { class: 'callout warn sp-mt-3' }, [
      icon('info', 15),
      el('div', null, library
        ? ['模型库里还没有 ONNX 文件。把 .onnx 放进 ', el('span', { class: 'mono', text: dir }), '，再点「重新扫描」。']
        : '读不到模型库：页面是双击打开的（没有后端），或者训练台的 /api/models 还没起。')
    ]);
  }

  function yoloCard(probe) {
    var y = probe.yolo;
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: 'YOLO OBB 后端' }),
        el('span', { class: 'end' }, el('span', { class: 'badge ' + (y.ok ? 'ok' : (y.error ? 'bad' : 'warn')) }, y.ok ? '已加载' : (y.error ? '未通过' : '未检查')))
      ]),
      el('div', { class: 'field' }, [
        el('label', { class: 'form-label', for: 'f-yolo_path', text: 'ONNX 模型（全局模型库，所有工程共用）' }),
        yoloPicker(),
        el('div', { class: 'row sp-mt-2' }, [
          el('button', { class: 'btn btn-ghost btn-sm', type: 'button', id: 'rescan-btn', onclick: rescan }, '重新扫描'),
          el('div', { class: 'help', text: '换一个模型会马上重新加载，类别跟着换成新模型里的。' })
        ])
      ]),
      libraryNote(),
      el('div', { class: 'grid c2 sp-gap-3' }, [
        numField('yolo_conf_threshold', '置信度阈值', '0.05'),
        partClassesBlock(y)
      ]),
      y.ok ? el('dl', { class: 'keyval' }, [
        el('dt', { text: '本次加载' }), el('dd', { title: y.path, text: y.path || '—' }),
        el('dt', { text: '文件指纹' }), el('dd', { title: y.sha256, text: D.shortHash(y.sha256, 12) })
      ]) : el('div', { class: 'callout ' + (y.error ? 'bad' : 'warn') }, [
        icon(y.error ? 'alert' : 'info', 15),
        el('div', null, [
          el('strong', { text: y.error ? '这次没加载成功，指纹不会写上。 ' : '还没检查这个 ONNX。 ' }),
          el('div', { class: 'sp-mt-2', text: y.error || '上面已经选中库里的模型。点「检查并保存」，类别和指纹才会出现。' })
        ])
      ])
    ]);
  }

  /* 手部模型这次从哪读：库齐就用库，缺一个文件就整体退回 MediaPipe 包里那份。 */
  function handsSourceText(h) {
    if (h.source === 'library') return '工作台 models/hands';
    if (h.source === 'bundled') return 'MediaPipe 内置';
    return '—';
  }

  /* paths 只在「检查并保存」之后才有 —— 那时才真的加载过，所以只显示这一份清单，
     不拿库路径顶上：库在、但这次加载的是内置模型的话，写库路径就是假的。 */
  function handsFiles(h) {
    if (!h.paths || !h.paths.length) {
      /* 启动加载中或失败时没有路径可列，不重复解释；其余情况才是"还没加载过"。 */
      if (h.loadState === 'loading' || h.loadState === 'failed') return null;
      return el('div', { class: 'hint sp-mt-3', text: '这几个文件的实际路径在「检查并保存」之后显示：那一次才真的加载过它们。' });
    }
    return el('div', { class: 'tbl-wrap sp-mt-3' }, el('table', { class: 'tbl' }, [
      el('thead', null, el('tr', null, [
        el('th', { text: '文件' }), el('th', { text: '字节' }), el('th', { text: '绝对路径' })
      ])),
      el('tbody', null, h.paths.map(function (item) {
        return el('tr', null, [
          el('td', null, [
            item.name,
            /* 计算图不参与指纹，但少了它 MediaPipe 建不起图，单独标出来。 */
            /\.binarypb$/.test(item.name) ? el('span', { class: 'badge plain sp-ml-2', text: '计算图' }) : null
          ]),
          el('td', { class: 'mono sp-dim', text: String(item.bytes) }),
          el('td', { class: 'mono sp-dim', text: item.path })
        ]);
      }))
    ]));
  }

  function handsCard(probe) {
    var h = probe.hands || {};
    var lib = (global.OD.data.library && global.OD.data.library.hands) || {};
    var paths = (h.paths && h.paths.length) ? h.paths : (lib.files || []);
    var source = h.source || lib.source || '';
    var present = paths.length > 0;
    var complexity = typeof h.complexity === 'number' ? h.complexity : lib.complexity;
    var shown = {
      ok: !!h.ok,
      loadState: h.loadState || '',
      source: source,
      dir: h.dir || lib.dir || '',
      complexity: complexity,
      sha256: h.sha256 || '',
      paths: paths,
      error: h.error || ''
    };
    var loadState = shown.loadState;
    var badge = shown.ok ? '已加载'
      : (loadState === 'loading' ? '加载中…'
        : (loadState === 'failed' ? '加载失败' : (present ? '库里已有' : '未通过')));
    var badgeClass = shown.ok ? 'ok'
      : (loadState === 'loading' ? 'warn'
        : (loadState === 'failed' ? 'bad' : (present ? 'warn' : 'bad')));
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '手部后端' }),
        el('span', { class: 'end' }, [
          shown.source ? el('span', { class: 'badge ' + (shown.source === 'library' ? 'ok' : 'warn') }, handsSourceText(shown)) : null,
          el('span', { class: 'badge ' + badgeClass }, badge)
        ])
      ]),
      el('p', { class: 'hint', text: 'MediaPipe 的 Hands() 没有模型路径参数，只认自己包里的目录。所以给库的时候先把库镜像进缓存、再把资源根指过去 —— 下面这几条就是库里真正会被读的那几个文件。' }),
      el('dl', { class: 'keyval' }, [
        el('dt', { text: '读取来源' }), el('dd', { text: handsSourceText(shown), title: shown.dir || '' }),
        el('dt', { text: '模型复杂度' }), el('dd', { text: complexityText(shown.complexity) }),
        el('dt', { text: '每只手' }), el('dd', { text: '21 点 × 3 坐标 = 63' }),
        el('dt', { text: '两只手' }), el('dd', { text: '126 个特征位' }),
        el('dt', { text: '模型指纹' }), el('dd', { title: shown.sha256, text: shown.sha256 ? D.shortHash(shown.sha256, 12) : '点「检查并保存」后写入' })
      ]),
      handsFiles(shown),
      loadState === 'loading'
        ? el('div', { class: 'hint sp-mt-3', text: '训练台启动时在后台加载手部模型，第一次需要几秒。加载完页面会自动刷新。' })
        : null,
      shown.error ? el('div', { class: 'callout bad sp-mt-3' }, [icon('alert', 15), el('div', { text: shown.error })]) : null
    ]);
  }

  function stationCard() {
    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '时间基与部署参数' })),
      el('p', { class: 'hint', text: '时间基不用填：它由片子的帧率推导，全部片子一致时就是它，不一致时训练台会拒绝，不会替你挑一个。画面尺寸也不用填 —— 特征按每一帧自己的宽高归一化，和分辨率无关，所以没有任何尺寸门禁。' }),
      el('dl', { class: 'keyval sp-mb-3' }, [
        el('dt', { text: '时间基与窗口' }), el('dd', { class: 'left', text: S.timeBase().text })
      ]),
      el('div', { class: 'grid c2 sp-gap-3' }, [
        numField('station_debounce_frames', '连续确认帧数', '1', '只写进契约给部署端核对，不参与训练。'),
        numField('station_min_confidence', '目标通过阈值', '0.05', '只写进契约给部署端核对，不参与训练。')
      ])
    ]);
  }

  function featureLengthText() {
    var classes = (global.OD.data.settings && global.OD.data.settings.yolo_classes) || [];
    if (!classes.length && !(global.OD.data.features && global.OD.data.features.length)) return '还没选模型';
    var hands = !!(global.OD.data.settings && global.OD.data.settings.hands_in_training);
    var parts = classes.length * 5;
    var total = parts + (hands ? 126 : 0);
    return total + ' = 零件类别数×5' + (hands ? ' + 126' : '（不含手部）');
  }

  function trainCard() {
    var c = global.OD.data.constants;
    var tb = S.timeBase();
    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '训练超参' })),
      el('p', { class: 'hint', text: '这些会写进这一轮的训练。点训练时会自动抽特征，不用单独抽取。层数和隐层保持 2 层、128。改窗口秒数会让已确认的划分作废，要重新确认。' }),
      el('label', { class: 'check sp-mt-3', for: 'f-hands_in_training' }, [
        el('input', {
          type: 'checkbox', id: 'f-hands_in_training',
          checked: !!global.OD.data.settings.hands_in_training
        }),
        el('span', { text: '手部参与训练' }),
        el('span', { class: 'hint', text: '关掉后只使用 YOLO 的零件框和时间顺序，不加载手部模型。打开后同一帧再接手部 21 点。切换后下次训练会重抽特征。' })
      ]),
      el('div', { class: 'grid c2 sp-gap-3' }, [
        numField('epochs', '训练轮数', '1'),
        numField('learning_rate', '学习率', '0.0001'),
        numField('dropout', 'Dropout', '0.05', '越大越不容易死记这一条片子，推理时不起作用。'),
        numField('window_seconds', '窗口秒数', '0.01', '按片子帧率换算成帧数。'),
        numField('stride', '步长（帧）', '1', '窗口每次向前滑多少帧。')
      ]),
      el('dl', { class: 'keyval' }, [
        el('dt', { text: '窗长' }), el('dd', { text: tb.window === null ? '—' : tb.window + ' 帧 = ' + tb.seconds + ' 秒' }),
        el('dt', { text: '步长' }), el('dd', { text: (global.OD.data.settings.stride || c.STRIDE) + ' 帧' }),
        el('dt', { text: '网络' }), el('dd', { text: 'LSTM 2 层 · hidden 128（固定）' }),
        el('dt', { text: '划分随机种子' }), el('dd', { text: c.SEED }),
        el('dt', { text: '特征长度' }), el('dd', { text: featureLengthText() })
      ])
    ]);
  }

  /* 表单里能改的项，一次全带上。服务端 settings.clean() 会逐项校验，
     不在 EDITABLE_KEYS 里的键会被 400 拒掉 —— 所以别多传。 */
  /* 以当前选中的 option 为准。select.value 对不上任一 option 时会变成空字符串，
     不能拿这个空字符串去覆盖已经保存的模型。 */
  function selectedYoloPath() {
    var select = document.getElementById('f-yolo_path');
    var chosen = '';
    if (select && select.selectedIndex >= 0 && select.options[select.selectedIndex]) {
      var option = select.options[select.selectedIndex];
      chosen = String(option.value || option.getAttribute('value') || '').trim();
    }
    if (!chosen && select) chosen = String(select.value || '').trim();
    if (chosen) return chosen;
    var saved = global.OD.data.settings && global.OD.data.settings.yolo_path;
    return String(saved || '').trim();
  }

  function collectPatch() {
    function numberValue(key) {
      var raw = fieldValue(key).trim();
      return raw === '' ? null : Number(raw);
    }
    return {
      station_debounce_frames: numberValue('station_debounce_frames'),
      station_min_confidence: numberValue('station_min_confidence'),
      yolo_path: selectedYoloPath(),
      yolo_conf_threshold: numberValue('yolo_conf_threshold'),
      epochs: numberValue('epochs'),
      learning_rate: numberValue('learning_rate'),
      dropout: numberValue('dropout'),
      window_seconds: numberValue('window_seconds'),
      stride: numberValue('stride'),
      hands_in_training: !!(document.getElementById('f-hands_in_training') || {}).checked
    };
  }

  function reportTable(rows) {
    return el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
      el('thead', null, el('tr', null, [el('th', { text: '检查项' }), el('th', { text: '结果' }), el('th', { text: '说明' })])),
      el('tbody', null, rows.map(function (r) {
        return el('tr', null, [
          el('td', { text: r[0] }),
          el('td', null, el('span', { class: 'badge ' + (r[1] ? 'ok' : 'bad') }, r[1] ? '通过' : '未通过')),
          el('td', { class: 'mono sp-dim', text: r[2] })
        ]);
      }))
    ]));
  }

  /* 探测结果落进 probe：刚刚真的加载过，这一份比从 settings 还原出来的新。 */
  function absorb(result) {
    var data = global.OD.data;
    var s = result.settings || data.settings;
    var y = result.yolo || {};
    var h = result.hands || {};
    data.settings = s;
    data.probe = {
      sample: false,
      yolo: y.ok
        ? { ok: true, path: s.yolo_path, classes: y.classes || [], sha256: y.sha256 || '' }
        : { ok: false, path: s.yolo_path || '', classes: [], sha256: s.yolo_sha256 || '', error: y.error || '' },
      hands: h.ok
        ? {
          ok: true,
          complexity: typeof h.model_complexity === 'number' ? h.model_complexity : null,
          sha256: h.sha256 || '',
          /* 服务端刚加载过哪几个文件，它说了算 —— 这是唯一一次拿到真实路径的机会。 */
          source: h.source || '', dir: h.dir || '', paths: h.paths || []
        }
        : {
          ok: false,
          complexity: data.probe.hands.complexity, sha256: data.probe.hands.sha256,
          source: h.source || '', dir: h.dir || '', paths: [], error: h.error || ''
        }
    };
  }

  function runCheck() {
    var report = document.getElementById('check-report');
    D.clear(report);
    var problems = validate();
    if (problems.length) {
      report.appendChild(el('div', { class: 'callout bad' }, [
        icon('alert', 15),
        el('div', null, [el('strong', { text: '校验没过，什么都没写进文件。 ' }), problems.length + ' 个字段不合法，改完再点一次。'])
      ]));
      S.toast('校验失败，设置没有被写入', 'bad');
      return;
    }

    var patch = collectPatch();
    var api = global.OD.api;
    if (!api.available || !S.state.projectId) {
      /* 离线：这一步真的做不了，把能自己算的算给它看，别假装写过了。
         零件类别这次不来自表单（表单里已经没有它了），取的是这份示例数据自带的模型类别。 */
      var names = (global.OD.data.probe.yolo && global.OD.data.probe.yolo.classes) || [];
      report.appendChild(el('div', { class: 'callout warn' }, [
        icon('info', 15),
        el('div', { text: '离线原型没有后端：不会真的加载 ONNX，也不会写 settings.json。下面只是把表单里的值按同样的规则核对了一遍。' })
      ]));
      report.appendChild(reportTable([
        ['零件类别', names.length > 0, names.length ? names.join(' / ') : '示例数据里还没读到模型类别'],
        ['时间基与窗口', true, S.timeBase().text]
      ]));
      S.toast('离线模式：只核对了表单，没有写盘', 'warn');
      return;
    }

    var button = document.getElementById('check-btn');
    if (button) button.disabled = true;
    showClassesLoading();
    report.appendChild(el('div', { class: 'help', text: '正在加载模型…' }));

    api.patchModel(S.state.projectId, patch).then(function (result) {
      D.clear(report);
      absorb(result);
      var y = result.yolo || {}, h = result.hands || {};
      var s = result.settings || {};
      var allOk = !!y.ok && !!h.ok;

      report.appendChild(el('div', { class: 'callout ' + (allOk ? 'ok' : 'bad') }, [
        icon(allOk ? 'check' : 'alert', 15),
        el('div', {
          text: allOk
            ? '全部通过，指纹已由训练台写回 settings.json。'
            : '有项目没过。设置已经存下去了，但没过指纹的项不会建立特征缓存。'
        })
      ]));
      report.appendChild(reportTable([
        ['YOLO ONNX 加载', !!y.ok, y.ok ? '读到 ' + (y.classes || []).length + ' 个模型类别' : (y.error || '加载失败')],
        ['手部模型加载', !!h.ok, h.ok
          ? '复杂度 ' + (typeof h.model_complexity === 'number' ? h.model_complexity : '—') +
            '，从' + (h.source === 'library' ? '模型库' : 'MediaPipe 内置包') + '读取'
          : (h.error || '加载失败')],
        ['时间基与窗口', true, S.timeBase().text],
        ['零件类别', (y.classes || []).length > 0,
          (y.classes || []).length ? '特征行按这 ' + y.classes.length + ' 类排：' + y.classes.join(' / ') : '没有类别，抽不了特征']
      ]));
      if (button) button.disabled = false;
      S.toast(allOk ? '检查通过，指纹已保存' : '检查完成，有项目没过', allOk ? 'ok' : 'warn');
      /* 指纹已经变了，整个界面按磁盘上的新设置重画一遍。 */
      global.OD.live.reload().catch(function (err) { S.toast('刷新工程失败：' + err.message, 'bad'); });
    }).catch(function (err) {
      D.clear(report);
      if (button) button.disabled = false;
      report.appendChild(el('div', { class: 'callout bad' }, [
        icon('alert', 15),
        el('div', null, [el('strong', { text: '训练台拒绝了这次改动。 ' }), err.message])
      ]));
      S.toast('检查没成功：' + err.message, 'bad');
    });
  }

  function inspector() {
    var c = global.OD.data.constants;
    var tb = S.timeBase();
    var partClasses = (global.OD.data.probe.yolo && global.OD.data.probe.yolo.classes) || [];
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '会写进契约的东西' }),
        el('dl', { class: 'keyval' }, [
          el('dt', { text: 'yolo_sha256' }), el('dd', { text: D.shortHash(global.OD.data.probe.yolo.sha256, 8) }),
          el('dt', { text: 'hands_sha256' }), el('dd', { text: D.shortHash(global.OD.data.probe.hands.sha256, 8) }),
          el('dt', { text: '手部复杂度' }), el('dd', { text: global.OD.data.probe.hands.complexity === null || global.OD.data.probe.hands.complexity === undefined ? '—' : global.OD.data.probe.hands.complexity }),
          el('dt', { text: '去抖帧数' }), el('dd', { text: global.OD.data.settings.station_debounce_frames }),
          el('dt', { text: '通过阈值' }), el('dd', { text: global.OD.data.settings.station_min_confidence })
        ])
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '特征长度怎么来的' }),
        el('p', null,
          '每个零件类 5 个数（置信度、中心 x/y、框宽高，都按画面宽高归一），当前模型 ' + partClasses.length + ' 个类 = ' +
          (partClasses.length * 5) + '；两只手 2×21×3 = 126。合计 ' + global.OD.data.features.length + '。' +
          (partClasses.length ? '类别和顺序都取自模型，换模型才会变。' : '还没读到模型类别。'))
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '窗口规则' }),
        el('p', null,
          '窗长 ' + (tb.window === null ? '—' : tb.window + ' 帧 = ' + tb.seconds + ' 秒') +
          '、步长 ' + c.STRIDE + ' 帧，窗口必须完全落在一段标注里。验证集按段划分，确认之后才开训，随机种子 ' + c.SEED + '。')
      ])
    ]);
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.model = { render: render, inspector: inspector };
})(window);
