/* 结果：轮次表、单轮详情（契约、划分、逐类、混淆、错误段）、最近三次对比。
   错误段可以直接点进标注工作台并跳到那一帧。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;
  var selected = null;

  function runs() { return global.OD.data.runs; }
  function current() {
    var list = runs();
    for (var i = 0; i < list.length; i++) if (list[i].name === selected) return list[i];
    return list[0];
  }
  function byName(name) {
    var list = runs();
    for (var i = 0; i < list.length; i++) if (list[i].name === name) return list[i];
    return null;
  }

  function render() {
    global.OD.charts.reset();
    var list = runs();
    if (!list.length) {
      return el('div', { class: 'view is-on' }, el('div', { class: 'scroll' },
        el('section', { class: 'card' }, el('div', { class: 'empty' }, [
          icon('flask', 28),
          el('strong', { text: '还没有结果' }),
          el('p', { text: '训练页跑完一轮，这里就会有 best_lstm.pt。能不能用，拿到工位上看。' })
        ]))));
    }
    var run = current();

    var host = el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '结果' }),
        el('p', { text: '每一轮就是 runs/<时间戳> 一个目录，里面的 best_lstm.pt 直接给工位用。分数写在下面，不决定这一轮能不能拿走。' })
      ]),
      listCard(list),
      detailCard(run),
      compareCard(list)
    ]));
    requestAnimationFrame(function () { paintMatrix(run); });
    return host;
  }

  /* ---------- 轮次表 ---------- */

  function listCard(list) {
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: '模型' }),
        el('span', { class: 'end' }, list.some(function (r) { return r.sample; }) ? S.sampleTag('模型为示例') : null)
      ]),
      el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [
          el('th', { text: '模型' }), el('th', { class: 'opt', text: '时间' }),
          el('th', { class: 'n', text: '宏平均 F1' }), el('th', { class: 'n', text: '最低召回' }),
          el('th', { class: 'n', text: '空闲误报' }),           el('th', { class: 'n opt', text: '最佳轮数' }),
          el('th', { text: '' })
        ])),
        el('tbody', null, list.map(function (run) {
          var m = run.metrics;
          return el('tr', { 'aria-current': run.name === current().name ? 'true' : 'false' }, [
            el('td', { class: 'mono', text: run.display_name || ('未命名 · ' + run.name) }),
            el('td', { class: 'mono sp-dim opt', text: run.at }),
            el('td', { class: 'n mono', text: D.fmtNum(m.macro_f1, 3) }),
            el('td', { class: 'n mono', text: D.fmtNum(m.min_recall, 3) }),
            el('td', { class: 'n mono', text: D.fmtNum(m.idle_false_alarm, 3) }),
            el('td', { class: 'n mono opt', text: '第 ' + run.best_epoch + ' / ' + run.epochs }),
            el('td', null, el('button', {
              class: 'btn btn-ghost btn-sm', type: 'button',
              onclick: function () { selected = run.name; S.render(); }
            }, '查看'))
          ]);
        }))
      ]))
    ]);
  }

  /* ---------- 单轮详情 ---------- */

  /* 没有窗口的片子是这一页最该被看见的事：它的标注一段都装不下一个窗口，
     再多轮次也永远学不到它。这句按当前工程实际算，不写死。 */
  function splitNote(run) {
    if (run.sample) {
      return '这是示例轮次，划分用的是随原型附带的示例片子，不是工程里真实的素材。';
    }
    var empty = global.OD.data.clips.filter(function (clip) {
      return S.clipIssues(clip).windows === 0;
    });
    if (empty.length) {
      return '注意：' + empty.map(function (c) { return c.file; }).join('、') + ' 目前一段都不足 ' + D.fmtNum(run.window, 0) +
        ' 帧，一个训练窗口都产不出来。上面的划分是这一轮实际用到的片子，没能覆盖它们。';
    }
    if (run.same_source) {
      return '验证段和训练段来自同一批片子。这个分数看不出新视频会不会被切坏。';
    }
    return '验证集和训练集没有共用片子。这个分数更接近换一条视频的表现。一段有没有被切碎，用下面的新视频再看一次。';
  }

  /* ---------- 拿这一轮看一条新视频 ---------- */

  var checkToken = 0;
  var checkModel = '';
  var projectCheck = null;

  function modelChoices() {
    return (global.OD.data.runs || []).filter(function (item) {
      return !item.sample && (item.files || []).indexOf('best_lstm.pt') >= 0;
    });
  }

  function modelLabel(item, runs) {
    var base = !item.display_name ? '未命名 · ' + item.name : item.display_name;
    var same = runs.filter(function (other) { return other.display_name === item.display_name; });
    if (item.display_name && same.length > 1) base = item.display_name + ' · ' + item.name;
    return item.train_unit === 'timeline' ? base : base + ' · 旧训练方式';
  }

  function clock(sec) {
    var s = Math.max(0, Number(sec) || 0);
    var mm = Math.floor(s / 60);
    var rest = s - mm * 60;
    var text = rest.toFixed(2);
    if (rest < 10) text = '0' + text;
    return (mm < 10 ? '0' : '') + mm + ':' + text;
  }

  function classIndex(result, id) {
    var classes = result.classes || [];
    for (var i = 0; i < classes.length; i++) if (classes[i].id === id) return i;
    return 0;
  }

  function checkCard(run) {
    var choices = modelChoices();
    if (!run.sample && !choices.some(function (item) { return item.name === run.name; })) choices = [run].concat(choices);
    if (!checkModel || !choices.some(function (item) { return item.name === checkModel; })) checkModel = run.name;
    function active() {
      for (var i = 0; i < choices.length; i++) if (choices[i].name === checkModel) return choices[i];
      return run;
    }
    var stage = el('div', { class: 'check-stage' });
    var listHost = el('div');
    var note = el('p', { class: 'hint' });
    function paintNote(extra) {
      var current = active();
      var title = modelLabel(current, choices);
      note.textContent = extra || ('用「' + title + '」看一条新视频。识别按这一轮 ' + D.fmtNum(current.window_seconds, 2) + ' 秒的窗口。视频只放在这个模型下面，不进训练素材。');
    }
    paintNote();
    var select = el('select', { class: 'form-select', 'aria-label': '选择验证模型' });
    if (!choices.length) select.appendChild(el('option', { value: '', text: '还没有训好的模型' }));
    choices.forEach(function (item) {
      select.appendChild(el('option', { value: item.name, text: modelLabel(item, choices) }));
    });
    select.value = active().name;
    select.addEventListener('change', function () {
      if (S.state.job) { select.value = active().name; S.toast('先停掉正在跑的任务，再换模型。', 'warn'); return; }
      var previousName = active().name;
      checkModel = select.value;
      if (previousName === checkModel) return;
      var title = modelLabel(active(), choices);
      paintNote('已换成「' + title + '」。正在把当前视频带到这个模型下面，带完就可以点验证。');
      showActions(true);
      verify.disabled = true;
      loadCheck().then(function (hasVideo) {
        if (!stage.isConnected) return;
        if (hasVideo) {
          showActions(true);
          paintNote('这条视频已经在「' + title + '」下面。点「验证」用这个模型重新识别。');
          return;
        }
        return carryVideo(previousName, active().name).then(function (carried) {
          if (!stage.isConnected) return;
          if (!carried) {
            showActions(false);
            paintNote('「' + title + '」下面还没有视频。先上传一条，再点验证。');
            return;
          }
          return loadCheck().then(function () {
            showActions(true);
            paintNote('视频已换到「' + title + '」。点「验证」重新识别。');
          });
        });
      }).catch(function (err) {
        showActions(!!player);
        S.toast('换模型没完成：' + (err && err.message ? err.message : err), 'bad');
      });
    });
    var player = null;
    var input = el('input', {
      type: 'file', accept: 'video/mp4,video/webm,video/quicktime,.mp4,.mov,.avi,.mkv,.webm,.m4v', class: 'sr',
      onchange: function () {
        var file = input.files && input.files[0];
        input.value = '';
        if (file) acceptVideo(file);
      }
    });
    var upload = el('button', {
      class: 'btn btn-ghost', type: 'button',
      onclick: function () { input.click(); }
    }, [icon('upload', 14), '上传视频']);
    var verify = el('button', {
      class: 'btn btn-primary', type: 'button',
      onclick: function () { startRecognition(); }
    }, '验证');
    var clearBtn = el('button', {
      class: 'btn btn-ghost', type: 'button',
      onclick: function () { clearVideo(); }
    }, '清除');
    verify.hidden = true;
    clearBtn.hidden = true;

    var clipPick = el('select', { class: 'form-select', id: 'check-project-clip', 'aria-label': '工程里的片子' },
      (global.OD.data.clips || []).map(function (clip) {
        return el('option', { value: clip.id, text: clip.file || clip.id });
      }));

    if (global.OD.api.available && S.state.projectId && !run.sample) loadCheck();

    var progressHost = el('div', { class: 'check-progress' });
    progressHost.hidden = true;
    var card = el('section', { class: 'card', id: 'check-player' }, [
      el('header', null, el('h2', { text: '拿模型看新视频' })),
      el('label', { class: 'field' }, [
        el('span', { class: 'form-label', text: '验证模型' }),
        select
      ]),
      el('div', { class: 'row sp-mt-3' }, [
        clipPick,
        el('button', {
          class: 'btn btn-primary', type: 'button',
          onclick: function () { verifyProjectClip(); }
        }, '用这条片子验证')
      ]),
      el('div', { class: 'check-actions sp-mt-3' }, [upload, verify, clearBtn]),
      progressHost,
      input,
      note,
      stage,
      listHost
    ]);
    return card;

    function loadCheck() {
      var token = ++checkToken;
      var runName = active().name;
      return global.OD.api.checkStatus(S.state.projectId, runName).then(function (body) {
        if (token !== checkToken || !stage.isConnected) return false;
        if (!(projectCheck && projectCheck.run === runName && projectCheck.result)) {
          var saved = body && body.saved_clip;
          var videoAt = body && typeof body.checked_at === 'number' ? body.checked_at : 0;
          var clipAt = saved && typeof saved.checked_at === 'number' ? saved.checked_at : 0;
          var hasVideo = !!(body && body.media && body.media.file);
          var videoWins = hasVideo && (!body.ready || videoAt > clipAt);
          if (saved && saved.project_clip && clipAt > 0 && !videoWins) {
            projectCheck = { run: runName, result: saved };
          }
        }
        if (projectCheck && projectCheck.run === runName && projectCheck.result) {
          player = paintCheck(stage, listHost, active(), projectCheck.result, null);
          note.textContent = projectCheck.result.role || '';
          if (clipPick && projectCheck.result.clip_id) clipPick.value = projectCheck.result.clip_id;
          watchProgress();
          return true;
        }
        var hasVideo = !!(body && body.media && body.media.file);
        showActions(hasVideo);
        if (hasVideo) {
          player = paintCheck(stage, listHost, active(), body, player);
          note.textContent = body.ready
            ? '视频在上面。点「验证」会用「' + modelLabel(active(), choices) + '」重新识别。'
            : '视频已经放好。点「验证」才开始识别。';
        } else {
          D.clear(stage);
          D.clear(listHost);
          player = null;
        }
        watchProgress();
        return hasVideo;
      });
    }

    function carryVideo(fromName, toName) {
      var api = global.OD.api;
      if (!fromName || !toName || fromName === toName) return Promise.resolve(false);
      return api.checkStatus(S.state.projectId, fromName).then(function (body) {
        var fileName = body && body.media && body.media.file;
        if (!fileName) return false;
        return fetch(api.checkFileUrl(S.state.projectId, fromName, fileName)).then(function (response) {
          if (!response.ok) throw new Error('读不到上一条验证视频');
          return response.blob();
        }).then(function (blob) {
          var file = new File([blob], fileName, { type: blob.type || 'video/mp4' });
          return api.uploadCheck(S.state.projectId, toName, file).then(function () { return true; });
        });
      });
    }

    function paintProgress(done, total, label) {
      var tot = total > 0 ? total : 1;
      var pct = Math.max(0, Math.min(100, Math.round((done / tot) * 100)));
      progressHost.hidden = false;
      D.clear(progressHost);
      progressHost.appendChild(el('div', { class: 'check-progress-label' }, [
        el('strong', { text: label || '验证' }),
        el('span', { class: 'mono', text: (done || 0) + ' / ' + tot + ' 帧 · ' + pct + '%' })
      ]));
      progressHost.appendChild(el('div', { class: 'check-progress-track', role: 'progressbar', 'aria-valuenow': String(pct), 'aria-valuemin': '0', 'aria-valuemax': '100' },
        el('div', { class: 'check-progress-fill', style: 'width:' + pct + '%' })));
    }

    var progressMark = -1;
    var progressStill = 0;

    function watchProgress() {
      var api = global.OD.api;
      if (!api.available || !S.state.projectId) return;
      api.progress(S.state.projectId).then(function (p) {
        if (!progressHost.isConnected) return;
        var checking = p && (p.status === 'checking' || p.status === 'extracting');
        var c = (p && p.check) || {};
        var live = S.state.job && S.state.job.kind === 'check';
        if (checking && c.total) {
          paintProgress(c.done || 0, c.total, p.status === 'extracting' ? '抽特征' : '验证');
          if (!live && c.done === progressMark) progressStill += 1;
          else progressStill = 0;
          progressMark = c.done;
          if (!live && progressStill >= 8) {
            progressHost.hidden = true;
            return;
          }
          setTimeout(watchProgress, 1000);
          return;
        }
        if (!live) progressHost.hidden = true;
      }).catch(function () {});
    }

    function showActions(hasVideo) {
      upload.lastChild.textContent = hasVideo ? '换一条视频' : '上传视频';
      verify.hidden = !hasVideo;
      clearBtn.hidden = !hasVideo;
      upload.disabled = false;
      verify.disabled = false;
      clearBtn.disabled = false;
    }

    function acceptVideo(file) {
      var api = global.OD.api;
      if (!api.available || !S.state.projectId) {
        S.toast('离线原型没有后端，视频不会写盘。', 'warn');
        return;
      }
      if (active().sample) { S.toast('示例轮次没有权重，不能拿来验证。', 'warn'); return; }
      if (S.state.job) { S.toast('已经有一个任务在跑了，先停掉它。', 'warn'); return; }
      var lower = String(file.name || '').toLowerCase();
      if (!/\.(mp4|mov|avi|mkv|webm|m4v)$/.test(lower)) {
        S.toast('请上传视频。这一轮要看 ' + D.fmtNum(active().window_seconds, 2) + ' 秒，一张图凑不满窗口。', 'warn');
        return;
      }
      player = paintCheck(stage, listHost, active(), {
        ready: false,
        media: { file: file.name, fps: 0, frame_count: 0 },
        localUrl: URL.createObjectURL(file)
      }, player);
      showActions(true);
      card.scrollIntoView({ block: 'nearest' });
      note.textContent = '正在放入视频。放完后点「验证」。';
      api.uploadCheck(S.state.projectId, active().name, file).then(function () {
        showActions(true);
        note.textContent = '视频已放好。点上面的「验证」。';
        S.toast('视频已放好，点「验证」。', 'ok');
      }).catch(function (err) {
        S.toast('视频没放进去：' + err.message, 'bad');
      });
    }

    function clearVideo() {
      var api = global.OD.api;
      if (!api.available || !S.state.projectId) return;
      if (S.state.job) { S.toast('先停掉正在跑的任务，再清除。', 'warn'); return; }
      S.confirm({
        title: '清除这条验证视频？',
        body: '播放窗口和识别出来的段会一起去掉。这一轮的权重还在。',
        danger: true,
        ok: '清除',
        onOk: function () {
          api.deleteCheck(S.state.projectId, active().name).then(function () {
            S.render();
          }).catch(function (err) {
            S.toast('清不掉：' + err.message, 'bad');
          });
        }
      });
    }

    function verifyProjectClip() {
      var api = global.OD.api;
      var projectId = S.state.projectId;
      var clipPick = document.getElementById('check-project-clip');
      var clipId = clipPick ? clipPick.value : '';
      if (!api.available || !projectId || !clipId) return;
      if (S.state.job) { S.toast('已经有一个任务在跑了，先停掉它。', 'warn'); return; }
      paintProgress(0, 1, '验证');
      note.textContent = '正在用工程里的片子验证。不改标注，也不复制视频。';
      S.startJob({
        kind: 'check', label: '验证', total: 1, unit: '帧', period: 1000,
        onTick: function (done, total) { paintProgress(done, total || 1, '验证'); },
        live: {
          get: function () {
            return api.progress(projectId).then(function (p) {
              var c = (p && p.check) || {};
              return { done: c.done || 0, total: c.total || 0, status: p && p.status };
            });
          },
          run: function () { return api.checkClip(projectId, active().name, clipId); }
        },
        onDone: function (result) {
          if (result && result.project_clip) projectCheck = { run: active().name, result: result };
          S.render();
        }
      });
    }

    function startRecognition() {
      var api = global.OD.api;
      var projectId = S.state.projectId;
      if (!api.available || !projectId) return;
      if (S.state.job) { S.toast('已经有一个任务在跑了，先停掉它。', 'warn'); return; }
      verify.disabled = true;
      paintProgress(0, 1, '验证');
      note.textContent = '正在验证。进度在上面，画面可以播放，也可以按住时间条拖动。';
      S.startJob({
        kind: 'check', label: '验证', total: 1, unit: '帧', period: 1000,
        onTick: function (done, total) { paintProgress(done, total || 1, '验证'); },
        live: {
          get: function () {
            return api.progress(projectId).then(function (p) {
              var c = (p && p.check) || {};
              return { done: c.done || 0, total: c.total || 0, status: p && p.status };
            });
          },
          run: function () { return api.runCheck(projectId, active().name); },
          onFail: function () { verify.disabled = false; }
        },
        onDone: function (result) {
          verify.disabled = false;
          if (result && result.stopped) {
            S.toast('验证已停止：' + (result.error || ''), 'warn');
            return;
          }
          S.render();
        }
      });
    }
  }

  function paintCheck(stage, listHost, run, result, previous) {
    var media = result.media || {};
    var count = media.frame_count || 1;
    var fps = media.fps || 25;
    var segments = result.segments || [];
    var warming = result.warming;
    var video = previous && previous.video;
    var sameVideo = video && previous.file === media.file && !result.localUrl;
    if (!sameVideo) {
      if (previous && previous.localUrl) URL.revokeObjectURL(previous.localUrl);
      video = el('video', { controls: true, playsinline: true, preload: 'auto' });
      video.src = media.project_file
        ? global.OD.api.mediaUrl(S.state.projectId, media.file)
        : (result.localUrl || global.OD.api.checkFileUrl(S.state.projectId, run.name, media.file));
    }
    var label = el('div', { class: 'check-label' }, [
      el('strong', { text: result.ready ? '准备播放' : '可以先看' }),
      el('span', { text: result.ready ? '按住下面的条拖动' : '识别完成后，这里标出每一段' })
    ]);
    var truthLane = result.truth_segments ? el('div', { class: 'check-lane', title: '标注' }) : null;
    if (truthLane) {
      result.truth_segments.forEach(function (segment) {
        truthLane.appendChild(el('div', {
          class: 'check-seg',
          style: 'left:' + (segment.start_frame / count * 100) + '%;width:' + ((segment.end_frame - segment.start_frame) / count * 100) + '%;background:' + D.catColor(classIndex(result, segment.class_id)),
          text: segment.name,
          onclick: function () { seek(segment.start_frame, true); }
        }));
      });
    }
    var lane = el('div', { class: 'check-lane', title: '按住拖动，画面跟到那一帧' });
    var playhead = el('div', { class: 'check-playhead' });
    if (warming && warming.end_frame > warming.start_frame) {
      lane.appendChild(el('div', {
        class: 'check-seg is-warm',
        style: 'left:' + (warming.start_frame / count * 100) + '%;width:' + ((warming.end_frame - warming.start_frame) / count * 100) + '%',
        text: '攒帧中'
      }));
    }
    var gated = segments.some(function (segment) { return !!segment.part_check; });
    var rows = segments.map(function (segment) {
      var color = D.catColor(classIndex(result, segment.class_id));
      lane.appendChild(el('div', {
        class: 'check-seg',
        style: 'left:' + (segment.start_frame / count * 100) + '%;width:' + ((segment.end_frame - segment.start_frame) / count * 100) + '%;background:' + color,
        text: segment.name
      }));
      return el('tr', {
        onclick: function () { seek(segment.start_frame, true); }
      }, [
        el('td', null, el('span', { class: 'badge plain' }, [
          el('i', { class: 'swatch', style: 'background:' + color }),
          segment.name
        ])),
        el('td', { class: 'mono', text: clock(segment.start_sec) + ' · 第 ' + segment.start_frame + ' 帧' }),
        el('td', { class: 'mono', text: clock(segment.end_sec) + ' · 第 ' + segment.end_frame + ' 帧' }),
        el('td', { class: 'mono n', text: D.fmtNum(segment.end_sec - segment.start_sec, 2) + ' 秒' }),
        el('td', { class: 'mono n', text: D.fmtNum(segment.prob, 2) }),
        gated ? el('td', { text: partText(segment) }) : null
      ]);
    });
    lane.appendChild(playhead);
    var nowRow = el('tr', { class: 'is-on' }, [
      el('td'), el('td', { class: 'mono' }), el('td', { class: 'mono' }),
      el('td', { class: 'mono n' }), el('td', { class: 'mono n' }),
      gated ? el('td') : null
    ]);
    var nowTable = el('table', { class: 'tbl check-current' }, [
      el('thead', null, el('tr', null, [
        el('th', { text: '动作' }), el('th', { text: '开始' }), el('th', { text: '结束' }),
        el('th', { class: 'n', text: '时长' }), el('th', { class: 'n', text: '概率' }),
        gated ? el('th', { text: '零件' }) : null
      ])),
      el('tbody', null, nowRow)
    ]);
    var canvas = el('canvas', { class: 'check-overlay' });
    var overlayFrames = [];
    D.clear(stage);
    stage.appendChild(video);
    stage.appendChild(canvas);
    stage.appendChild(label);
    if (truthLane) {
      stage.appendChild(el('div', { class: 'sp-meta', text: '标注' }));
      stage.appendChild(truthLane);
      stage.appendChild(el('div', { class: 'sp-meta', text: '模型切出来的' }));
    }
    if (result.scores) {
      var scores = result.scores;
      stage.appendChild(el('p', { class: 'hint', text:
        (result.role ? result.role + '。' : '') +
        '段级 F1@50 ' + D.fmtNum(scores.f1_50, 3) +
        '，编辑分 ' + D.fmtNum(scores.edit, 1) +
        '，逐帧准确率 ' + D.fmtNum(scores.frame_accuracy, 3) +
        '，空闲误报 ' + D.fmtNum(scores.idle_false_alarm, 3) +
        '。切出 ' + scores.pred_segments + ' 段 / 标注 ' + scores.true_segments + ' 段。'
      }));
    }
    stage.appendChild(lane);
    function takeOverlay(body) {
      var frames = (body && body.frames) || [];
      if (!frames.length || frames.length < overlayFrames.length) return;
      overlayFrames = frames;
      if (typeof onTime === 'function') onTime();
    }
    if (!result.project_clip && global.OD.api.available && S.state.projectId && (result.overlay || result.ready)) {
      var overlayTimer = setInterval(function () {
        if (!canvas.isConnected) { clearInterval(overlayTimer); return; }
        global.OD.api.checkOverlay(S.state.projectId, run.name).then(function (body) {
          takeOverlay(body);
          if (result.overlay && overlayFrames.length) clearInterval(overlayTimer);
        }).catch(function () {});
      }, 3000);
      if (result.overlay) {
        global.OD.api.checkOverlay(S.state.projectId, run.name).then(function (body) {
          if (!canvas.isConnected) return;
          takeOverlay(body);
        }).catch(function () {});
      }
    }
    D.clear(listHost);
    listHost.appendChild(nowTable);
    if (segments.length) {
      listHost.appendChild(el('div', { class: 'tbl-wrap check-list sp-mt-3' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [
          el('th', { text: '动作' }), el('th', { text: '开始' }), el('th', { text: '结束' }),
        el('th', { class: 'n', text: '时长' }), el('th', { class: 'n', text: '概率' }),
        gated ? el('th', { text: '零件' }) : null
      ])),
        el('tbody', null, rows)
      ])));
    }
    listHost.appendChild(el('p', { class: 'help sp-mt-2', text: '按住时间条左右拖，画面停在那一帧，左上角写这一帧属于哪一段、从哪开始到哪结束。点表里的一段，从那段开头播放。' }));

    function frameCount() {
      if (media.frame_count) return media.frame_count;
      var fromVideo = video.duration && fps ? Math.round(video.duration * fps) : 0;
      return fromVideo || 1;
    }

    function frameAt(clientX) {
      var rect = lane.getBoundingClientRect();
      var ratio = rect.width ? (clientX - rect.left) / rect.width : 0;
      ratio = Math.min(1, Math.max(0, ratio));
      var total = frameCount();
      return Math.min(total - 1, Math.max(0, Math.floor(ratio * total)));
    }

    function seek(frame, play) {
      var total = frameCount();
      var rate = media.fps || (video.duration ? (total / video.duration) : fps);
      video.pause();
      var apply = function () {
        video.currentTime = frame / (rate || 25);
        show(frame);
        if (play) video.play().catch(function () {});
      };
      if (video.readyState >= 1) apply();
      else video.addEventListener('loadedmetadata', apply, { once: true });
    }

    function ratioOf(clientX) {
      var rect = lane.getBoundingClientRect();
      var ratio = rect.width ? (clientX - rect.left) / rect.width : 0;
      return Math.min(1, Math.max(0, ratio));
    }

    function scrub(event) {
      var ratio = ratioOf(event.clientX);
      if (!(media.fps > 0) && video.duration) {
        video.pause();
        video.currentTime = ratio * video.duration;
        playhead.style.left = (ratio * 100) + '%';
        return;
      }
      seek(Math.min(frameCount() - 1, Math.floor(ratio * frameCount())), false);
    }

    lane.addEventListener('pointerdown', function (event) {
      lane.setPointerCapture(event.pointerId);
      scrub(event);
    });
    lane.addEventListener('pointermove', function (event) {
      if (lane.hasPointerCapture(event.pointerId)) scrub(event);
    });

    var HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
    var BOX_COLORS = ['#3dd68c', '#5eb1ff', '#f5c542', '#e0574b', '#c084fc', '#fb923c'];

    function colorFor(label) {
      var n = 0;
      for (var i = 0; i < label.length; i++) n = (n + label.charCodeAt(i)) % BOX_COLORS.length;
      return BOX_COLORS[n];
    }

    function layoutOverlay() {
      var vw = video.videoWidth;
      var vh = video.videoHeight;
      if (!vw || !vh) return false;
      var cw = video.clientWidth;
      var ch = video.clientHeight;
      var scale = Math.min(cw / vw, ch / vh);
      var dw = vw * scale;
      var dh = vh * scale;
      canvas.style.left = ((cw - dw) / 2) + 'px';
      canvas.style.top = ((ch - dh) / 2) + 'px';
      canvas.style.width = dw + 'px';
      canvas.style.height = dh + 'px';
      if (canvas.width !== vw) canvas.width = vw;
      if (canvas.height !== vh) canvas.height = vh;
      return true;
    }

    function drawOverlay(frame) {
      if (!layoutOverlay()) return;
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var item = overlayFrames[frame];
      if (!item) return;
      var w = canvas.width;
      var h = canvas.height;
      ctx.lineWidth = Math.max(2, w / 480);
      ctx.font = '600 ' + Math.max(14, Math.round(w / 48)) + 'px sans-serif';
      (item.boxes || []).forEach(function (box) {
        var poly = box.poly || [];
        if (poly.length < 2) return;
        ctx.beginPath();
        poly.forEach(function (point, index) {
          var x = point[0] * w;
          var y = point[1] * h;
          if (index) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = colorFor(box.label || '');
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(box.label || '', poly[0][0] * w, Math.max(16, poly[0][1] * h - 6));
      });
      (item.hands || []).forEach(function (hand, index) {
        if (!hand || hand.length < 21) return;
        ctx.strokeStyle = index ? '#f5c542' : '#7ec8ff';
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = Math.max(2, w / 560);
        HAND_LINKS.forEach(function (link) {
          var a = hand[link[0]];
          var b = hand[link[1]];
          ctx.beginPath();
          ctx.moveTo(a[0] * w, a[1] * h);
          ctx.lineTo(b[0] * w, b[1] * h);
          ctx.stroke();
        });
        hand.forEach(function (point) {
          ctx.beginPath();
          ctx.arc(point[0] * w, point[1] * h, Math.max(3, w / 280), 0, Math.PI * 2);
          ctx.fill();
        });
      });
    }

    function onResize() {
      if (!canvas.isConnected) {
        window.removeEventListener('resize', onResize);
        return;
      }
      onTime();
    }
    window.addEventListener('resize', onResize);

    var shown = -2;

    function partText(segment) {
      if (!segment || !segment.part_check) return '—';
      if (segment.part_check === 'pass') return '完成';
      var missing = segment.missing_parts || [];
      return missing.length ? '失败，缺' + missing.join('、') : '失败';
    }

    function paintNow(name, color, start, end, dur, prob, part) {
      D.clear(nowRow.children[0]);
      nowRow.children[0].appendChild(el('span', { class: 'badge plain' }, [
        color ? el('i', { class: 'swatch', style: 'background:' + color }) : null,
        name
      ]));
      nowRow.children[1].textContent = start || '—';
      nowRow.children[2].textContent = end || '—';
      nowRow.children[3].textContent = dur || '—';
      nowRow.children[4].textContent = prob || '—';
      if (nowRow.children[5]) nowRow.children[5].textContent = part || '—';
    }

    function show(frame) {
      drawOverlay(frame);
      var total = frameCount();
      var current = null;
      segments.forEach(function (segment) {
        if (frame >= segment.start_frame && frame < segment.end_frame) current = segment;
      });
      playhead.style.left = (frame / total * 100) + '%';
      D.clear(label);
      if (!result.ready) {
        label.appendChild(el('strong', { text: '可以先看' }));
        label.appendChild(el('span', { text: '第 ' + frame + ' 帧' }));
        paintNow('可以先看', '', '第 ' + frame + ' 帧', '', '', '');
        return;
      }
      if (!current) {
        var warmStart = warming ? clock(warming.start_sec) + ' · 第 ' + warming.start_frame + ' 帧' : '—';
        var warmEnd = warming ? clock(warming.end_sec) + ' · 第 ' + warming.end_frame + ' 帧' : '—';
        label.appendChild(el('strong', { text: '攒帧中' }));
        label.appendChild(el('span', { text: warmStart + ' – ' + warmEnd }));
        paintNow('攒帧中', '', warmStart, warmEnd, '', '');
        return;
      }
      var start = clock(current.start_sec) + ' · 第 ' + current.start_frame + ' 帧';
      var end = clock(current.end_sec) + ' · 第 ' + current.end_frame + ' 帧';
      var dur = D.fmtNum(current.end_sec - current.start_sec, 2) + ' 秒';
      var prob = D.fmtNum(current.prob, 2);
      label.appendChild(el('strong', { text: current.name }));
      label.appendChild(el('span', { text: clock(current.start_sec) + ' – ' + clock(current.end_sec) }));
      label.appendChild(el('span', { class: 'check-sub', text: '第 ' + current.start_frame + '–' + current.end_frame + ' 帧 · 概率 ' + prob }));
      paintNow(current.name, D.catColor(classIndex(result, current.class_id)), start, end, dur, prob, partText(current));
    }

    function onTime() {
      var rate = media.fps || fps;
      var total = frameCount();
      show(Math.min(total - 1, Math.max(0, Math.floor(video.currentTime * rate + 1e-4))));
    }
    video.addEventListener('timeupdate', onTime);
    video.addEventListener('seeked', onTime);
    video.addEventListener('loadedmetadata', onTime);
    show(0);
    return { video: video, file: media.file, localUrl: result.localUrl || '' };
  }

  function detailCard(run) {
    var m = run.metrics;
    return el('div', null, [
      el('section', { class: 'card' }, [
        el('header', null, [
          el('h2', { text: run.name }),
          el('span', { class: 'end' }, [
            run.sample ? S.sampleTag() : null,
            /* 同源验证必须挂在最显眼的位置：它改变的是"门禁通过"这四个字的含义。 */
            run.same_source ? el('span', { class: 'badge warn', title: '训练集和验证集的段来自同一条片子，分数是乐观的' }, '同源验证') : null,
            el('span', { class: 'badge ok' }, '已保存')
          ])
        ]),
        el('div', { class: 'callout ok sp-mb-4' }, [
          icon('check', 15),
          el('div', { text: 'best_lstm.pt 已经写好，可以拿到工位上试。下面的分数只说明验证集上的表现，不代替现场验证。' })
        ]),
        el('div', { class: 'grid c2' }, [
          el('div', null, [
            el('h3', { class: 'subhead', text: '这一轮的结构' }),
            el('dl', { class: 'keyval' }, [
              el('dt', { text: '网络' }), el('dd', { text: run.network === 'mstcn' ? '因果 MS-TCN（已停用，产线不能用）' : 'LSTM' }),
              el('dt', { text: '特征长度' }), el('dd', { text: run.input_size }),
              el('dt', { text: '窗长' }), el('dd', { text: D.fmtNum(run.window, 0) + ' 帧 = ' + D.fmtNum(run.window_seconds, 2) + ' 秒' }),
              el('dt', { text: '时间基' }), el('dd', { text: D.fmtNum(run.time_base_fps, 2) + ' fps' }),
              el('dt', { text: '步长' }), el('dd', { text: run.stride + ' 帧' }),
              el('dt', { text: '随机种子' }), el('dd', { text: run.seed }),
              el('dt', { text: '类别数' }), el('dd', { text: run.class_ids.length + '（含空闲）' }),
              el('dt', { text: '写成文件' }), el('dd', { class: 'left', text: run.files.join('、') })
            ])
          ]),
          el('div', null, [
            el('h3', { class: 'subhead', text: '验证集划分' }),
            el('div', { class: 'row tight sp-mb-2' }, [
              el('span', { class: 'tag', text: '训练' })
            ].concat(run.split.train.map(function (f) { return el('span', { class: 'badge plain', text: f }); }))),
            el('div', { class: 'row tight sp-mb-3' }, [
              el('span', { class: 'tag', text: '验证' })
            ].concat(run.split.val.map(function (f) { return el('span', { class: 'badge plain', text: f }); }))),
            el('p', { class: 'sp-meta', text: splitNote(run) })
          ])
        ])
      ]),
      checkCard(run),
      el('section', { class: 'card' }, [
        el('header', null, el('h2', { text: '逐类指标' })),
        el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
          el('thead', null, el('tr', null, [
            el('th', { text: '类别' }), el('th', { class: 'n', text: run.train_unit === 'timeline' ? '段数' : '窗口数' }),
            el('th', { class: 'n', text: '精确率' }), el('th', { class: 'n', text: '召回' }), el('th', { class: 'n', text: 'F1' })
          ])),
          el('tbody', null, m.per_class.map(function (row) {
            var weak = row.recall < global.OD.data.constants.GATE.min_recall;
            return el('tr', null, [
              el('td', { text: row.name }),
              el('td', { class: 'n mono', text: row.support }),
              el('td', { class: 'n mono', text: D.fmtNum(row.precision, 3) }),
              el('td', { class: 'n mono' + (weak ? ' sp-weak' : ''), text: D.fmtNum(row.recall, 3) }),
              el('td', { class: 'n mono', text: D.fmtNum(row.f1, 3) })
            ]);
          }))
        ])),
        el('p', { class: 'help sp-mt-3', text: run.train_unit === 'timeline'
          ? '这里的数量是段数。召回低于 ' + global.OD.data.constants.GATE.min_recall + ' 的格子标红。'
          : '召回低于 ' + global.OD.data.constants.GATE.min_recall + ' 的格子标红，它决定了"最低召回"这一关。' })
      ]),
      el('section', { class: 'card' }, [
        el('header', null, [
          el('h2', { text: '混淆矩阵' }),
          el('span', { class: 'end' }, el('span', { class: 'unit sp-meta-xs', text: '行是真实、列是判成的' }))
        ]),
        el('div', { class: 'chart tall', id: 'runs-matrix' })
      ]),
      mistakesCard(run)
    ]);
  }

  function paintMatrix(run) {
    var node = document.getElementById('runs-matrix');
    if (!node || !document.body.contains(node)) return;
    global.OD.charts.mount(node, global.OD.charts.matrix(node, {
      names: run.metrics.per_class.map(function (r) { return r.name; }),
      values: run.metrics.confusion
    }));
  }

  function mistakesCard(run) {
    if (!run.mistakes.length) {
      return el('section', { class: 'card' }, [
        el('header', null, el('h2', { text: '判错的段' })),
        el('div', { class: 'callout ok' }, [icon('check', 15), el('div', { text: run.train_unit === 'timeline' ? '验证集里没有整段被判错。' : '验证集里没有整段判错的窗口。' })])
      ]);
    }
    /* 训练台的结果文件里只有"真实 / 判成"，没有置信度。整列都是空的话，表下面直说。 */
    var hasConfidence = run.mistakes.some(function (m) { return m.confidence !== null; });
    return el('section', { class: 'card' }, [
      el('header', null, el('h2', { text: '判错的段' })),
      el('p', { class: 'hint', text: run.train_unit === 'timeline'
        ? '下面列出验证片子上整段被判成别的类，以及空闲上切出来的动作。点进去看画面。'
        : '整段都被判成别的类，说明这一段和它被判成的类长得太像。点进去看画面，多半是标注边界要挪，或者这两类本来就需要更长的窗口才分得开。' }),
      el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [
          el('th', { text: '片子' }), el('th', { text: '帧区间' }),
          el('th', { text: '真实' }), el('th', { text: '判成' }),
          el('th', { class: 'n opt', text: '置信度' }), el('th', { text: '' })
        ])),
        el('tbody', null, run.mistakes.map(function (miss) {
          var clip = null;
          global.OD.data.clips.forEach(function (c) { if (c.file === miss.clip) clip = c; });
          var openable = !!(clip && clip.hasMedia);
          return el('tr', null, [
            el('td', { class: 'mono', text: miss.clip }),
            el('td', { class: 'mono', text: miss.start + ' – ' + miss.end }),
            el('td', null, el('span', { class: 'badge plain', text: miss.truth })),
            el('td', null, el('span', { class: 'badge bad', text: miss.pred })),
            el('td', { class: 'n mono opt', text: D.fmtNum(miss.confidence, 2) }),
            el('td', null, el('button', {
              class: 'btn btn-ghost btn-sm', type: 'button', disabled: !openable,
              title: openable ? '跳到标注工作台的这一帧' : '这条片子的素材不在工程里',
              onclick: function () { S.go('annotate', { clipId: clip.id, frame: miss.start }); }
            }, '看画面'))
          ]);
        }))
      ])),
      hasConfidence ? null : el('p', { class: 'help sp-mt-3', text: '这一轮的结果文件里没有逐段的置信度，所以这一列是空的 —— 不是没显示出来，是训练台没有记。' })
    ]);
  }

  /* ---------- 最近三次对比 ---------- */

  /* 服务端 _project_payload 只把最近三轮里 class_ids 与 input_size 相同的挑出来放进
     comparable_runs（app.py）。换过模型，类别和特征长度就跟着变；拿结构不同的
     两轮并排，会算出一个看着像进步的假差值 —— 所以这里照服务端同一条规则筛。 */
  function compareKey(run) {
    return JSON.stringify([run.class_ids || [], run.input_size]);
  }

  /* 结构摘要，用来说明某一轮为什么没被并排。 */
  function shapeOf(run) {
    return run.input_size + ' 维 / ' + (run.class_ids || []).length + ' 类';
  }

  function compareCard(list) {
    var recent = list.slice(0, 3);
    var baseKey = compareKey(recent[0]);
    var kept = recent.filter(function (run) { return compareKey(run) === baseKey; });
    var dropped = recent.filter(function (run) { return compareKey(run) !== baseKey; });
    var keys = [
      { k: 'macro_f1', label: '宏平均 F1', gate: global.OD.data.constants.GATE.macro_f1, higher: true },
      { k: 'min_recall', label: '最低召回', gate: global.OD.data.constants.GATE.min_recall, higher: true },
      { k: 'idle_false_alarm', label: '空闲误报', gate: global.OD.data.constants.GATE.idle_false_alarm, higher: false }
    ];
    return el('section', { class: 'card' }, [
      el('header', null, [
        el('h2', { text: dropped.length ? '可比的轮次' : '最近三次' }),
        el('span', { class: 'end' }, el('span', { class: 'unit sp-meta-xs', text: '箭头是相对上一轮的变化' }))
      ]),
      el('div', { class: 'tbl-wrap' }, el('table', { class: 'tbl' }, [
        el('thead', null, el('tr', null, [el('th', { text: '指标' })].concat(kept.map(function (run) {
          return el('th', { class: 'n', text: run.name });
        })))),
        el('tbody', null, keys.map(function (key) {
          return el('tr', null, [el('td', { text: key.label })].concat(kept.map(function (run, i) {
            var value = run.metrics[key.k];
            var prev = i + 1 < kept.length ? kept[i + 1].metrics[key.k] : null;
            /* 缺数就不算变化：拿 null 参与减法会算出一个假的差值。 */
            var delta = (value === null || prev === null) ? 0 : Math.round((value - prev) * 1000) / 1000;
            var better = key.higher ? delta > 0 : delta < 0;
            return el('td', { class: 'n' }, [
              el('span', { class: 'mono', text: D.fmtNum(value, 3) }),
              delta === 0 ? null : el('span', {
                class: 'delta sp-ml-2 ' + (better ? 'up' : 'down'),
                text: (delta > 0 ? '↑' : '↓') + Math.abs(delta).toFixed(3)
              })
            ]);
          })));
        }).concat([]))
      ])),
      dropped.length
        ? el('div', { class: 'callout warn sp-mt-3' }, [
          icon('alert', 15),
          el('div', null, [
            el('strong', { text: '这几轮没有并排：' }),
            dropped.map(function (run) { return run.name + '（' + shapeOf(run) + '）'; }).join('、') +
              '，和最近一轮（' + shapeOf(recent[0]) + '）的结构不同。特征长度或类别变了，指标之间就不再可比，并排只会得到一个假的差值。'
          ])
        ])
        : (kept.length < 2
          ? el('p', { class: 'help sp-mt-3', text: '目前只有一轮，还没有可比的第二轮。' })
          : null)
    ]);
  }

  function inspector() {
    var run = current();
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '上工位要带走什么' }),
        el('dl', { class: 'keyval' }, [
          el('dt', { text: '权重' }), el('dd', { text: 'best_lstm.pt' }),
          el('dt', { text: '契约' }), el('dd', { text: 'contract.json' }),
          el('dt', { text: '划分' }), el('dd', { text: 'split.json' })
        ]),
        el('p', null,
          'contract.json 里记着特征长度、窗长、步长和两个模型指纹。工位启动时逐项比对，对不上就拒绝加载，避免训练和现场用的不是同一套特征。')
      ]),
      run ? el('div', { class: 'inspect-block' }, [
        el('h3', { text: '这一轮的分数' }),
        el('p', null, '分数来自这一轮的验证段。模型能不能用，拿到工位上看。')
      ]) : null,
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '别只看最好那一轮' }),
        el('p', null,
          '三轮里指标一路涨不代表模型在变好，可能只是验证集恰好简单。同一批片子换个种子重跑，掉得厉害就说明这一轮不能信。')
      ])
    ]);
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.runs = { render: render, inspector: inspector };
})(window);
