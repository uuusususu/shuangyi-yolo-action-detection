/* 数据与标注：拖放上传、片子表、开训门禁。门禁每条都能点，直接跳到出问题的地方。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;
  var view = 'cards';

  function render() {
    var data = global.OD.data;

    var host = el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '数据与标注' }),
        el('p', { text: '放入视频或一组按序号命名的连续图。点开一条片子，把画面停在动作开始那一帧，再点类别。段要装得下 ' + S.windowSeconds() + ' 秒的窗口才产得出训练样本。' })
      ]),
      uploadCard(),
      el('section', { class: 'card' }, [
        el('header', null, [
          el('h2', { text: '片子' }),
          el('span', { class: 'end' }, [
            el('div', { class: 'seg-group', role: 'group', 'aria-label': '切换片子显示方式' }, [
              el('button', { type: 'button', 'aria-pressed': String(view === 'cards'), onclick: function () { view = 'cards'; S.render(); } }, '卡片'),
              el('button', { type: 'button', 'aria-pressed': String(view === 'table'), onclick: function () { view = 'table'; S.render(); } }, '表格')
            ]),
            el('span', { class: 'badge plain' }, data.clips.length + ' 条')
          ])
        ]),
        data.clips.length ? (view === 'cards' ? clipCards(data) : clipTable(data)) : el('div', { class: 'empty' }, [
          icon('film', 28),
          el('strong', { text: '一条片子都没有' }),
          el('p', { text: '把相机录下来的 mp4 拖到上面的框里，或者点"选择文件"。' })
        ])
      ]),
      gateCard()
    ]));

    return host;
  }

  /* 上传是真的写盘。文件名走 X-Filename 头 —— 训练台按这个名字给片子起 id：
     单文件用原文件名（id 就是主名），一次多个图按 static/app.js 的约定收进 images-<时间戳>/ 目录。
     选完立刻刷新工程，片子表跟着磁盘走。 */
  function uploadCard() {
    var api = global.OD.api;
    var input = el('input', {
      type: 'file', multiple: true, accept: 'video/*,image/*', class: 'sr',
      id: 'clip-file', onchange: function () { if (input.files.length) accept(input.files); }
    });
    var zone = el('label', {
      class: 'empty drop', for: 'clip-file',
      ondragover: function (e) { e.preventDefault(); zone.classList.add('is-over'); },
      ondragleave: function () { zone.classList.remove('is-over'); },
      ondrop: function (e) {
        e.preventDefault(); zone.classList.remove('is-over');
        if (e.dataTransfer && e.dataTransfer.files.length) accept(e.dataTransfer.files);
      }
    }, [
      icon('upload', 28),
      el('strong', { text: '把片子和连续图拖到这里' }),
      el('p', { text: 'mp4 / mov 每条各自成一条片子，一次选多条也分开列。图片请按 0001.jpg、0002.jpg 这样连续命名，整组一起选。' }),
      input
    ]);

    function accept(files) {
      if (!api.available) {
        S.toast('离线原型没有后端，这 ' + files.length + ' 个文件不会写盘。要真上传，用 python -m lstm_workbench 起训练台，再从它给的地址打开这个页面。', 'warn');
        return;
      }
      if (!S.state.projectId) { S.toast('还没有打开工程，先回"全部工程"里选一个。', 'bad'); return; }
      var list = Array.prototype.slice.call(files);
      var hits = list.filter(function (file) {
        if (isImage(file) && list.filter(isImage).length > 1) return false;
        return (global.OD.data.clips || []).some(function (clip) { return clip.file === file.name; });
      });
      if (hits.length) {
        S.confirm({
          title: hits.length === 1 ? ('替换 ' + hits[0].name + '？') : ('替换 ' + hits.length + ' 条同名片子？'),
          body: '工程里已经有这个文件名。替换会删掉它上面的标注和已抽出的特征。已经训完的轮次还留着。',
          danger: true,
          ok: '替换',
          onOk: function () { startUpload(list, true); }
        });
        return;
      }
      startUpload(list, false);
    }

    function isImage(file) {
      var name = String(file.name || '').toLowerCase();
      return /\.(jpg|jpeg|png|bmp)$/.test(name) || String(file.type || '').indexOf('image/') === 0;
    }

    function startUpload(list, replace) {
      var images = list.filter(isImage);
      var group = images.length > 1 ? ('images-' + Date.now()) : '';
      var done = 0, failed = [], imageOk = 0;

      function storedName(file) {
        return group && isImage(file) ? group + '/' + file.name : file.name;
      }

      function step(index) {
        if (index >= list.length) {
          if (failed.length === list.length) throw new Error(failed[0]);
          return null;
        }
        var file = list[index];
        var grouped = !!(group && isImage(file));
        return api.uploadMedia(S.state.projectId, file, storedName(file), null, replace && !grouped).then(function () {
          done = index + 1;
          if (grouped) imageOk += 1;
          return step(index + 1);
        }).catch(function (err) {
          failed.push(file.name + '：' + err.message);
          done = index + 1;
          return step(index + 1);
        });
      }

      S.startJob({
        kind: 'upload', label: '上传', total: list.length, unit: '个文件', period: 400,
        live: {
          get: function () { return Promise.resolve({ done: done, total: list.length, status: 'uploading' }); },
          run: function () { return Promise.resolve(step(0)); }
        },
        onTick: function (d, total) {
          if (S.state.job) S.state.job.label = '上传 ' + Math.min(d + 1, total) + '/' + total;
        },
        onDone: function () {
          var adopted = group && imageOk > 0
            ? api.adoptClip(S.state.projectId, group, false)
            : Promise.resolve();
          adopted.catch(function (err) {
            failed.push(err.message);
          }).then(function () {
            return global.OD.live.reload();
          }).then(function () {
            if (failed.length) S.toast(failed.length + ' 个没放进片子：' + failed[0], 'bad');
            else if (replace) S.toast('已替换，这条片子的标注已清空', 'ok');
            else S.toast('已放进工程，每条视频各自一条片子', 'ok');
          }).catch(function (err) { S.toast('文件写完了，但刷新工程失败：' + err.message, 'bad'); });
        }
      });
    }

    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '放入片子' }),
        el('span', { class: 'end' }, global.OD.api.available ? null : S.sampleTag('离线不写盘'))
      ]),
      zone
    ]);
  }

  /* 两种"这条片子有问题"要分清：示例行是原型自带的假数据，
     真实行缺素材是磁盘上文件真的没了 —— 后者带着训练台给的原因。 */
  function mediaTag(clip) {
    if (clip.hasMedia) return null;
    if (clip.source === 'real') {
      return el('span', { class: 'badge bad', title: clip.note || '' }, '素材缺失');
    }
    return S.sampleTag();
  }

  function clipState(clip) {
    if (clip.segments.length === 0) return { cls: 'plain', text: '待标注' };
    var issues = S.clipIssues(clip);
    if (issues.windows === 0) return { cls: 'bad', text: '标注过短' };
    if (issues.short.length || issues.narrow.length || issues.tail >= S.windowFrames(clip)) return { cls: 'warn', text: '有警告' };
    return { cls: 'ok', text: '可训练' };
  }

  function clipMeta(clip) {
    var issues = S.clipIssues(clip);
    return [
      ['时长', D.fmtDuration(clip.frame_count, clip.fps)],
      ['帧数', D.fmtInt(clip.frame_count)],
      ['帧率', D.fmtNum(clip.fps, 2) + ' fps'],
      ['分辨率', D.fmtNum(clip.width, 0) + '×' + D.fmtNum(clip.height, 0)],
      ['文件大小', D.fmtBytes(clip.bytes)],
      ['标注段', clip.segments.length + ' 段'],
      ['可训练窗口', issues.windows + ' 个']
    ];
  }

  function removeClip(clip) {
    if (!global.OD.api.available || !S.state.projectId) {
      S.toast('离线原型没有后端，删不了片子。', 'warn');
      return;
    }
    S.confirm({
      title: '删除 ' + clip.file + '？',
      body: '视频、这条片子上的标注和已抽出的特征会一起去掉。已经训完的轮次还留着。下一次训练要重新确认划分。',
      danger: true,
      ok: '删除',
      onOk: function () {
        global.OD.api.deleteClip(S.state.projectId, clip.id).then(function () {
          return global.OD.live.reload();
        }).then(function () {
          if (S.state.clipId === clip.id) {
            S.state.clipId = (global.OD.data.clips[0] && global.OD.data.clips[0].id) || '';
          }
          S.toast('已删除 ' + clip.file, 'ok');
        }).catch(function (err) { S.toast('删除失败：' + err.message, 'bad'); });
      }
    });
  }

  function clipCards(data) {
    return el('div', { class: 'grid c3' }, data.clips.map(function (clip) {
      var st = clipState(clip);
      var issues = S.clipIssues(clip);
      return el('article', { class: 'card' }, [
        el('header', null, [
          el('h2', { class: 'mono', text: clip.file }),
          el('span', { class: 'end' }, [
            mediaTag(clip),
            el('span', { class: 'badge ' + st.cls }, st.text)
          ])
        ]),
        el('dl', { class: 'keyval' }, clipMeta(clip).map(function (row) {
          return [el('dt', { text: row[0] }), el('dd', { text: row[1] })];
        }).reduce(function (a, b) { return a.concat(b); }, [])),
        issues.windows === 0 && clip.segments.length ? el('div', { class: 'help bad sp-mt-3' },
          '没有一段长到 ' + S.windowFrames(clip) + ' 帧（' + S.windowSeconds() + ' 秒），这条片子现在产不出训练样本。') : null,
        el('div', { class: 'row sp-mt-4' }, [
          el('button', {
            class: 'btn btn-primary btn-sm', type: 'button', disabled: !clip.hasMedia,
            title: clip.hasMedia ? '' : (clip.note || '这条示例片没有随原型附带素材'),
            onclick: function () { S.go('annotate', { clipId: clip.id }); }
          }, [icon('film', 13), '打开标注']),
          el('button', {
            class: 'btn btn-danger btn-sm', type: 'button',
            onclick: function () { removeClip(clip); }
          }, [icon('trash', 13), '删除'])
        ])
      ]);
    }));
  }

  function clipTable(data) {
    return el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
      el('thead', null, el('tr', null, [
        el('th', { text: '片子' }), el('th', { class: 'n', text: '帧数' }), el('th', { class: 'n opt', text: '帧率' }),
        el('th', { class: 'opt', text: '分辨率' }), el('th', { class: 'n opt', text: '大小' }),
        el('th', { class: 'n', text: '标注段' }), el('th', { class: 'n', text: '可训练窗口' }),
        el('th', { text: '状态' }), el('th', { text: '' })
      ])),
      el('tbody', null, data.clips.map(function (clip) {
        var st = clipState(clip), issues = S.clipIssues(clip);
        return el('tr', null, [
          el('td', null, [el('span', { class: 'mono', text: clip.file }), clip.hasMedia ? null : el('span', { class: 'sp-ml-2' }, mediaTag(clip))]),
          el('td', { class: 'n', text: D.fmtInt(clip.frame_count) }),
          el('td', { class: 'n opt', text: D.fmtNum(clip.fps, 2) }),
          el('td', { class: 'mono opt', text: clip.width + '×' + clip.height }),
          el('td', { class: 'n mono opt', text: D.fmtBytes(clip.bytes) }),
          el('td', { class: 'n', text: clip.segments.length }),
          el('td', { class: 'n', text: issues.windows }),
          el('td', null, el('span', { class: 'badge ' + st.cls }, st.text)),
          el('td', null, el('div', { class: 'row' }, [
            el('button', {
              class: 'btn btn-ghost btn-sm', type: 'button', disabled: !clip.hasMedia,
              title: clip.hasMedia ? '' : (clip.note || '这条示例片没有随原型附带素材'),
              onclick: function () { S.go('annotate', { clipId: clip.id }); }
            }, '打开标注'),
            el('button', {
              class: 'btn btn-danger btn-sm', type: 'button',
              onclick: function () { removeClip(clip); }
            }, '删除')
          ]))
        ]);
      }))
    ]));
  }

  /* 门禁有两个来源：
     1. 页面自己能从标注推出来的（段太短、片尾空白、片子不够），每条都能点进去改；
     2. 训练台 service.annotation_blockers() 报的，客户端推不出来（没有片子、段交叠、划分凑不出来），只能原样展示。 */
  function gateCard() {
    var data = global.OD.data;
    var serverBad = (data.blockers || []).map(function (text) {
      return { kind: 'bad', text: text, detail: '训练台自己报的，改完标注后重试。', go: null };
    });
    var serverWarn = (data.warnings || []).map(function (text) {
      return { kind: 'warn', text: text, detail: '训练台自己报的，可以带着跑。', go: null };
    });
    var items = S.allGateItems().concat(serverBad, serverWarn);
    var bad = items.filter(function (i) { return i.kind === 'bad'; });
    var warn = items.filter(function (i) { return i.kind !== 'bad'; });

    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '开训门禁' }),
        el('span', { class: 'end' }, [
          el('span', { class: 'badge bad' }, bad.length + ' 条阻断'),
          el('span', { class: 'badge warn' }, warn.length + ' 条警告')
        ])
      ]),
      el('p', { class: 'hint', text: '阻断项不清掉就没法开训；警告项可以带着跑，但结果里会体现出来。能点的直接跳到要改的地方。' }),
      bad.length ? el('div', { class: 'sp-mb-4' }, [
        el('div', { class: 'rail-title flat', text: '阻断' })
      ].concat(bad.map(issueRow))) : null,
      warn.length ? el('div', null, [
        el('div', { class: 'rail-title flat', text: '警告' })
      ].concat(warn.map(issueRow))) : el('div', { class: 'callout ok' }, [icon('check', 15), el('div', { text: '没有待处理项，可以直接去抽特征。' })])
    ]);
  }

  /* 没有跳转目标的（训练台报的那些）渲染成静态行，不做成按钮 —— 点了没反应比没有按钮更糟。 */
  function issueRow(item) {
    var body = [
      icon(item.kind === 'bad' ? 'alert' : 'info', 16),
      el('span', { class: 't', text: item.text }),
      item.go ? icon('arrow-right', 14) : null,
      el('span', { class: 'd', text: item.detail })
    ];
    if (!item.go) return el('div', { class: 'issue static ' + item.kind }, body);
    return el('button', {
      class: 'issue ' + item.kind, type: 'button',
      onclick: function () { S.go(item.go, item.go === 'annotate' ? { clipId: item.clip || S.state.clipId } : undefined); }
    }, body);
  }

  function inspector() {
    var data = global.OD.data;
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '标注规则' }),
        el('dl', { class: 'keyval' }, [
          el('dt', { text: '最短段长' }), el('dd', { text: S.windowSeconds() + ' 秒（按每条片子自己的帧率换算）' }),
          el('dt', { text: '窗口步长' }), el('dd', { text: global.OD.data.constants.STRIDE + ' 帧' }),
          el('dt', { text: '划分单位' }), el('dd', { text: '按段 · seed ' + global.OD.data.constants.SEED })
        ])
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '类别' }),
        el('p', { class: 'sp-meta sp-mb-2', text: '增加、改名或删除。改名会留下原来的段；删除会去掉这一类在所有片子上的标注。空闲不用建，它是段与段之间空出来的。' }),
        global.OD.classEditor()
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '划分按段，不按片子' }),
        el('p', null,
          '同一个段的窗口要么全进训练、要么全进验证，不会跨边 —— 相邻窗口只差 8 帧，跨边的验证集等于训练集的双胞胎。' +
          '训练和验证里都必须出现每一个类别。只有一条片子也能分开，但两边的段来自同一次录制，分数是乐观的，结果页会标出来。')
      ])
    ]);
  }

  function classCount(id) {
    return global.OD.data.clips.reduce(function (sum, clip) {
      return sum + (clip.segments || []).filter(function (s) { return s.class_id === id; }).length;
    }, 0);
  }

  function saveClasses(classes) {
    var api = global.OD.api;
    if (!api.available || !S.state.projectId) {
      S.toast('离线原型没有后端，类别不会写盘。', 'warn');
      return;
    }
    var ids = {};
    classes.forEach(function (c) { ids[c.id] = true; });
    var payload = {
      classes: classes.map(function (c) { return { id: c.id, name: c.name }; }),
      clips: global.OD.data.clips.map(function (clip) {
        return {
          id: clip.id,
          file: clip.file,
          frame_count: clip.frame_count,
          fps: clip.fps,
          width: clip.width,
          height: clip.height,
          segments: (clip.segments || []).filter(function (s) { return ids[s.class_id]; }).map(function (s) {
            return { id: s.id, class_id: s.class_id, start: s.start, end: s.end };
          }),
          excluded: (clip.excluded || []).map(function (s) {
            return { id: s.id, start: s.start, end: s.end };
          })
        };
      })
    };
    api.saveAnnotations(S.state.projectId, payload).then(function () {
      return global.OD.live.reload();
    }).then(function () {
      S.toast('类别已保存', 'ok');
    }).catch(function (err) {
      S.toast('类别没保存：' + err.message, 'bad');
    });
  }

  function classEditor() {
    var data = global.OD.data;
    var draft = el('input', {
      class: 'form-control', type: 'text', placeholder: '新类别名称', 'aria-label': '新类别名称'
    });
    function add() {
      var name = draft.value.trim();
      if (!name) { S.toast('先写上类别名称。', 'bad'); return; }
      if (data.classes.some(function (c) { return c.name === name; })) {
        S.toast('已经有叫「' + name + '」的类别。', 'bad');
        return;
      }
      saveClasses(data.classes.map(function (c) { return { id: c.id, name: c.name }; }).concat([{ id: 'c' + Date.now(), name: name }]));
    }
    draft.addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    return el('div', null, [
      el('div', null, data.classes.map(function (c, i) {
        var input = el('input', { class: 'form-control', type: 'text', value: c.name, 'aria-label': '类别名称 ' + c.name });
        input.addEventListener('change', function () {
          var name = input.value.trim();
          if (!name) { input.value = c.name; S.toast('类别名称不能空。', 'bad'); return; }
          if (name === c.name) return;
          if (data.classes.some(function (other) { return other.id !== c.id && other.name === name; })) {
            input.value = c.name;
            S.toast('已经有叫「' + name + '」的类别。', 'bad');
            return;
          }
          saveClasses(data.classes.map(function (item) {
            return { id: item.id, name: item.id === c.id ? name : item.name };
          }));
        });
        return el('div', { class: 'class-row' }, [
          el('span', { class: 'swatch', style: 'background:' + D.catColor(i) }),
          input,
          el('span', { class: 'n', text: classCount(c.id) + ' 段' }),
          el('button', {
            class: 'btn btn-danger btn-sm', type: 'button',
            onclick: function () {
              var n = classCount(c.id);
              S.confirm({
                title: '删除类别 ' + c.name + '？',
                body: n ? '这一类在片子上有 ' + n + ' 段，会一起删掉。' : '这一类还没有标注段。',
                danger: true,
                ok: '删除',
                onOk: function () {
                  saveClasses(data.classes.filter(function (item) { return item.id !== c.id; }).map(function (item) {
                    return { id: item.id, name: item.name };
                  }));
                }
              });
            }
          }, '删除')
        ]);
      })),
      el('div', { class: 'row sp-mt-3' }, [
        draft,
        el('button', { class: 'btn btn-primary btn-sm', type: 'button', onclick: add }, '增加')
      ])
    ]);
  }

  global.OD.classEditor = classEditor;
  global.OD.views = global.OD.views || {};
  global.OD.views.data = { render: render, inspector: inspector };
})(window);
