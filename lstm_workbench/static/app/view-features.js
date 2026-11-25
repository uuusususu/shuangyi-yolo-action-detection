/* 特征：缓存状态、失效原因、抽取进度。
   关键是把"为什么作废"写清楚 —— 换了 YOLO、改了阈值、换了模型（类别跟着变），都会让旧特征不再算数。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;
  var refs = {};

  var KEY_PARTS = [
    ['feature_code_version', '特征代码版本'],
    ['yolo_sha256', 'YOLO 文件指纹'],
    ['yolo_conf_threshold', '置信度阈值'],
    ['yolo_classes', '零件类别（来自模型）'],
    ['media_signature', '素材签名'],
    ['frame_count', '帧数']
  ];

  var picked = {};

  function syncPicked(rows) {
    var seen = {};
    rows.forEach(function (row) {
      seen[row.clip_id] = true;
      if (!Object.prototype.hasOwnProperty.call(picked, row.clip_id)) {
        picked[row.clip_id] = row.state !== 'fresh' && !row.mediaMissing;
      }
    });
    Object.keys(picked).forEach(function (id) { if (!seen[id]) delete picked[id]; });
  }

  function pickedIds() {
    return Object.keys(picked).filter(function (id) { return picked[id]; });
  }

  function pickedCount() { return pickedIds().length; }

  function paintExtractButton() {
    if (!refs.startBtn) return;
    var blocked = global.OD.data.features.blocked_reason || '';
    var chosen = pickedCount();
    var job = S.state.job && S.state.job.kind === 'extract';
    refs.startBtn.disabled = !!job || !!blocked || !chosen;
    refs.startBtn.lastChild.textContent = blocked ? '还没有模型类别' : (chosen ? '抽取勾选的 ' + chosen + ' 条' : '先勾选片子');
  }
  var STATE_TEXT = { fresh: '最新', stale: '已失效', missing: '未抽取' };
  var STATE_CLASS = { fresh: 'ok', stale: 'warn', missing: 'plain' };

  function render() {
    var data = global.OD.data;
    syncPicked(data.features.rows);
    refs = {};
    var job = S.state.job && S.state.job.kind === 'extract';

    var host = el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '特征' }),
        el('p', { text: '特征长度 F = 零件类别数 × 5 + 126。类别由模型决定，顺序也是模型自带的顺序。框和手取自同一帧。换了模型或改了阈值，旧特征会自动作废重抽。' })
      ]),
      summaryCard(data, job),
      cacheCard(data),
      statsCard()
    ]));
    return host;
  }

  function summaryCard(data, job) {
    var fresh = data.features.rows.filter(function (r) { return r.state === 'fresh'; }).length;
    var total = data.features.rows.length;
    /* 没有模型类别时后端会拒掉抽取，这里先把按钮按住并写清楚原因，不让人点了才吃一个 400。 */
    var blocked = data.features.blocked_reason || '';
    var chosen = pickedCount();
    refs.startBtn = el('button', {
      class: 'btn btn-primary', type: 'button', disabled: !!job || !!blocked || !chosen,
      onclick: startExtract
    }, [icon('zap', 14), blocked ? '还没有模型类别' : (chosen ? '抽取勾选的 ' + chosen + ' 条' : '先勾选片子')]);
    refs.progressWrap = el('div', { class: 'sp-mt-4' }, job ? progressBody(S.state.job.done, S.state.job.total) : null);

    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '抽取' }),
        el('span', { class: 'end' }, [
          el('span', { class: 'badge ' + (blocked ? 'bad' : fresh === total ? 'ok' : 'warn') }, blocked ? '缺模型' : fresh + '/' + total + ' 条最新'),
          data.features.sample ? S.sampleTag('缓存状态为示例') : null
        ])
      ]),
      el('p', { class: 'hint', text: '每条片子的特征存成 features/<片子>.npy，旁边记一份清单。缓存 key 由下面六项拼出来，只要有一项变了就重抽。' }),
      blocked ? el('div', { class: 'callout warn sp-mt-3' }, [
        icon('alert', 15),
        el('div', null, [el('strong', { text: '抽不了。' }), el('div', { class: 'sp-mt-2', text: blocked })])
      ]) : null,
      el('div', { class: 'row' }, [refs.startBtn, el('span', { class: 'tc sp-meta' }, '停止在右上角的任务条上，抽取和训练共用一个入口。')]),
      refs.progressWrap
    ]);
  }

  function progressBody(done, total, detail) {
    var safeTotal = total > 0 ? total : 1;
    var pct = Math.max(0, Math.min(100, Math.round((done / safeTotal) * 100)));
    var clipLine = '';
    if (detail && detail.clip_total) {
      var name = detail.clip_id || '';
      var clips = (global.OD.data && global.OD.data.clips) || [];
      clips.forEach(function (clip) { if (clip.id === detail.clip_id) name = clip.file || clip.id; });
      clipLine = name + '  ' + (detail.clip_done || 0) + ' / ' + detail.clip_total + ' 帧';
    }
    return el('div', null, [
      el('div', { class: 'row sp-mb-2' }, [
        el('span', { class: 'sp-note', text: '正在抽取' }),
        el('span', { class: 'spread' }),
        el('span', { class: 'mono sp-meta', text: (clipLine ? clipLine + ' · ' : '') + done + ' / ' + safeTotal + ' 帧 · ' + pct + '%' })
      ]),
      el('div', { class: 'progress', role: 'progressbar', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 },
        el('div', { class: 'progress-bar busy', style: 'width:' + pct + '%' })),
      el('p', { class: 'help sp-mt-2', text: '逐帧跑 YOLO 与手部模型，进度按帧往前走。中途停止不会写坏已有的 .npy。' })
    ]);
  }

  function shapeText(row) {
    if (typeof row.frames === 'number' && typeof row.feature_length === 'number') {
      return row.frames + ' × ' + row.feature_length;
    }
    return '—';
  }

  function fileButton(row) {
    if (!row.file) return el('span', { class: 'sp-dim', text: '—' });
    return el('button', {
      class: 'btn btn-ghost btn-sm mono', type: 'button', title: '在文件夹里显示这个文件',
      onclick: function () {
        var api = global.OD.api;
        if (!api.available || !S.state.projectId) {
          S.toast('离线原型没有后端，打不开文件夹。', 'warn');
          return;
        }
        api.revealFeature(S.state.projectId, row.clip_id).then(function (res) {
          if (res && res.remote && res.download) {
            var a = document.createElement('a');
            a.href = res.download;
            a.download = row.file || (row.clip_id + '.npy');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            S.toast('当前为内网远端访问，已开始下载特征文件 ' + (row.file || ''), 'info');
          }
        }).catch(function (err) {
          S.toast('打不开文件夹：' + err.message, 'bad');
        });
      }
    }, row.file);
  }

  function cacheCard(data) {
    refs.rows = {};
    var body = el('tbody', null, data.features.rows.map(function (row) {
      var clip = null;
      data.clips.forEach(function (c) { if (c.id === row.clip_id) clip = c; });
      var name = clip ? clip.file : row.clip_id;
      var box = el('input', {
        type: 'checkbox', checked: !!picked[row.clip_id], disabled: !!row.mediaMissing,
        'aria-label': '抽取 ' + name,
        onchange: function () { picked[row.clip_id] = box.checked; paintExtractButton(); }
      });
      var tr = el('tr', null, [
        el('td', null, box),
        el('td', null, [el('span', { class: 'mono', text: name }), clip && !clip.hasMedia && clip.source !== 'real' ? el('span', { class: 'sp-ml-2' }, S.sampleTag()) : null]),
        el('td', null, fileButton(row)),
        el('td', { class: 'n mono', text: row.bytes ? D.fmtBytes(row.bytes) : '—' }),
        el('td', { class: 'n mono', text: shapeText(row) }),
        el('td', { class: 'mono sp-meta', text: row.extracted_at || '—' }),
        el('td', null, [el('span', { class: 'badge ' + STATE_CLASS[row.state], text: STATE_TEXT[row.state] })])
      ]);
      refs.rows[row.clip_id] = { tr: tr, row: row };
      return tr;
    }));

    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '缓存' })),
      el('p', { class: 'hint', text: '勾选要抽的片子。点文件名会在文件夹里定位这个 .npy。' }),
      el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [
          el('th', { text: '' }), el('th', { text: '片子' }), el('th', { text: '文件' }),
          el('th', { class: 'n', text: '大小' }), el('th', { class: 'n', text: '形状' }),
          el('th', { text: '时间' }), el('th', { text: '状态' })
        ])),
        body
      ]))
    ]);
  }

  function statsCard() {
    refs.stats = el('section', { class: 'card' });
    paintStats();
    return refs.stats;
  }

  /* 质量两条：每帧有没有检到零件 / 有没有检到手。
     在线时是清单里逐条的计数加出来的真实比例；离线示例没人报过，就直说没有。 */
  function paintStats() {
    if (!refs.stats) return;
    var data = global.OD.data;
    var q = data.features.quality;
    D.clear(refs.stats);
    refs.stats.appendChild(el('header', null, [
      el('h2', { text: '抽取质量' }),
      el('span', { class: 'end' }, q ? null : S.sampleTag('没有真实统计') )
    ]));
    if (!q) {
      refs.stats.appendChild(el('div', { class: 'empty compact' }, [
        icon('zap', 22),
        el('strong', { text: '还没有可看的统计' }),
        el('p', { text: '训练台会在每次抽取后报出逐帧的"没检到零件 / 没检到手"计数。抽过一轮、并且缓存是最新的，这里才有数。' })
      ]));
      return;
    }
    var rows = [
      { k: '没检到零件', v: q.no_part_ratio, max: 0.5, hint: '零件框全为 0 的帧占比。高了说明模型没检出零件：置信度阈值太高，或者这类零件压根没进画面。' },
      { k: '没检到手', v: q.no_hand_ratio, max: 0.5, hint: '两只手都没检到的帧占比。手在画面外或遮挡严重时会升高。' }
    ];
    refs.stats.appendChild(el('div', { class: 'grid c2' }, rows.map(function (r) {
      var bad = r.v > 0.3;
      return el('div', { class: 'gate ' + (bad ? 'is-bad' : 'is-ok') }, [
        el('div', { class: 'k', text: r.k }),
        el('div', { class: 'v', text: D.fmtNum(r.v * 100, 1) + '%' }),
        el('div', { class: 'meter sp-mt-2' },
          el('div', { class: 'fill ' + (bad ? 'bad' : 'ok'), style: 'width:' + Math.min(100, (r.v / r.max) * 100) + '%' })),
        el('div', { class: 'sub', text: r.hint })
      ]);
    })));
    refs.stats.appendChild(el('p', { class: 'help sp-mt-3', text: '按 ' + D.fmtInt(q.frames) + ' 帧统计。' }));
  }

  /* ---------- 抽取任务 ---------- */

  function startExtract() {
    var ids = pickedIds();
    if (!ids.length) { S.toast('先勾选要抽取的片子。', 'warn'); return; }
    if (S.state.job) { S.toast('已经有一个任务在跑了，先停掉它。', 'warn'); return; }

    var api = global.OD.api;
    var projectId = S.state.projectId;
    if (!api.available || !projectId) {
      S.toast('离线原型没有后端，抽不了特征。用 python -m lstm_workbench.app 起训练台，再从它给的地址打开这个页面。', 'warn');
      return;
    }

    var frames = 0;
    var clips = (global.OD.data && global.OD.data.clips) || [];
    ids.forEach(function (id) {
      clips.forEach(function (clip) { if (clip.id === id) frames += clip.frame_count || 0; });
    });
    var total = frames || ids.length;
    S.startJob({
      kind: 'extract', label: '抽取特征', total: total, unit: '帧', period: 1000,
      live: {
        get: function () {
          return api.progress(projectId).then(function (p) {
            var f = (p && p.features) || {};
            return {
              done: f.done || 0,
              total: f.total || 0,
              status: p && p.status,
              clips: f.clips || [],
              clip_id: f.clip_id || '',
              clip_done: f.clip_done || 0,
              clip_total: f.clip_total || 0
            };
          });
        },
        run: function () { return api.extract(projectId, ids); }
      },
      onTick: function (done, tot, snapshot) {
        if (refs.progressWrap) {
          D.clear(refs.progressWrap);
          refs.progressWrap.appendChild(progressBody(done, tot || total, snapshot));
        }
        ((snapshot && snapshot.clips) || []).forEach(function (raw) {
          if (!raw.fresh) return;
          var row = findRow(raw.clip_id);
          if (row && row.state !== 'fresh') { row.state = 'fresh'; row.reason = null; repaintRow(row); }
        });
      },
      onStop: function () {
        S.render();
        /* 服务端在下一帧才看到停止信号，稍等一下再读盘，免得读到半路的状态。 */
        setTimeout(function () { global.OD.live.reload().catch(function () {}); }, 800);
      },
      onDone: function (result) {
        if (result && result.stopped) {
          S.toast('抽取已停止：' + (result.error || ''), 'warn');
          global.OD.live.reload().catch(function () {});
          return;
        }
        paintStats();
        S.render();
        global.OD.live.reload().catch(function () {});
      }
    });
  }

  function findRow(clipId) {
    var rows = global.OD.data.features.rows;
    for (var i = 0; i < rows.length; i++) if (rows[i].clip_id === clipId) return rows[i];
    return null;
  }

  function repaintRow(row) {
    var ref = refs.rows && refs.rows[row.clip_id];
    if (!ref) return;
    var cells = ref.tr.children;
    D.clear(cells[6]); cells[6].appendChild(el('span', { class: 'badge ok', text: '最新' }));
  }

  /* 六项里前四项是工程级的，后两项（素材签名、帧数）是每条片子各算一份 ——
     右边那一列取当前选中片子的值，标题里也这么说，免得被当成全工程一个数。 */
  function inspector() {
    var data = global.OD.data;
    var clip = S.clipById(S.state.clipId);
    var s = data.settings;
    var perClip = clip ? clip.file + ' · ' + D.fmtBytes(clip.bytes) : '—';
    var values = {
      feature_code_version: data.features.code_version,
      yolo_sha256: D.shortHash(data.probe.yolo.sha256, 6),
      yolo_conf_threshold: D.fmtNum(s.yolo_conf_threshold, 2),
      yolo_classes: (s.yolo_classes || []).join(','),
      media_signature: perClip,
      frame_count: clip ? D.fmtInt(clip.frame_count) : '—'
    };
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '缓存 key 的六项' }),
        el('p', { class: 'help', text: '后两项是每条片子各一份，这里列的是当前这条。' }),
        el('div', null, KEY_PARTS.map(function (p) {
          var value = values[p[0]];
          if (value === null || value === undefined || value === '') value = '—';
          return el('div', { class: 'class-row' }, [
            el('span', { class: 'sp-fs-12', text: p[1] }),
            el('span', { class: 'n', title: String(value), text: String(value) })
          ]);
        }))
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '为什么这么设计' }),
        el('p', null,
          '训练和推理必须吃同一套特征。只按时间戳判断新旧，换了模型还会拿旧特征去训练，指标会好看得莫名其妙。六项里任意一项变了就重抽，代价是多等几分钟，换的是结果可信。')
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '特征长度' }),
        el('dl', { class: 'keyval' }, [
          el('dt', { text: '零件框' }), el('dd', { text: (s.yolo_classes || []).length + ' × 5 = ' + ((s.yolo_classes || []).length * 5) }),
          el('dt', { text: '手部' }), el('dd', { text: '2 × 21 × 3 = 126' }),
          el('dt', { text: '合计' }), el('dd', { text: data.features.length })
        ])
      ])
    ]);
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.features = { render: render, inspector: inspector };
})(window);
