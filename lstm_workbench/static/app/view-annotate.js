/* 标注工作台：多泳道时间轴 + 真实片子。
   泳道按类别分，段可以拖、可以改边、可以删；拖动时实时显示和最短窗长的关系并吸附
   （窗长按秒定义，帧数由这条片子自己的帧率换算，不再是一个写死的帧数）。
   在线时读的是工程里真实的素材与标注，改完点"保存标注"整体覆盖写 annotations.json。
   离线时只带一条示例素材，编辑照常但不会落盘。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;
  var C = global.OD.data.constants;

  var refs = {};
  var drag = null;
  var drawNew = null;
  var scrub = null;

  /* 最短段长按秒算：这条片子自己的帧率 × 窗口秒数。跟训练用的是同一个秒数，
     所以这里不再会出现「界面全标红、训练却全合格」那种两把尺子。 */
  function W() { return S.windowFrames(clip()); }
  var raf = null;
  var fpsEff = 6.28;
  var videoBroken = false;
  /* 上一次写盘（或刚载入）时的标注快照。data 被换掉就重新起算，所以按对象身份认。 */
  var savedFor = null;
  var savedJson = '';
  var seenContent = {};
  var completionConfirmations = {};
  var saveInFlight = null;

  var pickedRun = '';

  function trainedRuns() {
    return (global.OD.data.runs || []).filter(function (run) {
      return !run.sample && (run.files || []).indexOf('best_lstm.pt') >= 0;
    });
  }

  function paintPrelabelProgress(done, total) {
    if (!refs.preProgress) return;
    var tot = total > 0 ? total : 1;
    var pct = Math.max(0, Math.min(100, Math.round((done / tot) * 100)));
    refs.preProgress.hidden = false;
    D.clear(refs.preProgress);
    refs.preProgress.appendChild(el('div', { class: 'check-progress-label' }, [
      el('strong', { text: '抽取并标注' }),
      el('span', { class: 'mono', text: (done || 0) + ' / ' + tot + ' 帧 · ' + pct + '%' })
    ]));
    refs.preProgress.appendChild(el('div', { class: 'check-progress-track' },
      el('div', { class: 'check-progress-fill', style: 'width:' + pct + '%' })));
  }

  function applyDraft(list) {
    var c = clip();
    c.segments = (list || []).map(function (item) {
      var segment = { id: item.id, class_id: item.class_id, start: item.start, end: item.end };
      if (item.missing_parts && item.missing_parts.length) segment.missing_parts = item.missing_parts.slice();
      return segment;
    });
    S.state.segSel = null;
    repaint();
    S.toast('已画上 ' + c.segments.length + ' 段，还没保存。', 'ok');
  }

  function runPrelabel(runName, clipId) {
    var api = global.OD.api;
    var projectId = S.state.projectId;
    paintPrelabelProgress(0, 1);
    if (refs.preNote) refs.preNote.textContent = '正在处理当前这条片子。特征不是最新的会先抽，抽完再画段。';
    S.startJob({
      kind: 'prelabel', label: '预标', total: 1, unit: '帧', period: 1000,
      onTick: function (done, total) { paintPrelabelProgress(done, total || 1); },
      live: {
        get: function () {
          return api.progress(projectId).then(function (p) {
            var f = (p && p.features) || {};
            return { done: f.done || 0, total: f.total || 1, status: p && p.status };
          });
        },
        run: function () { return api.prelabel(projectId, runName, clipId); },
        onFail: function () {
          if (refs.preProgress) refs.preProgress.hidden = true;
        }
      },
      onDone: function (result) {
        if (refs.preProgress) refs.preProgress.hidden = true;
        if (result && result.stopped) {
          S.toast('预标已停止：' + (result.error || ''), 'warn');
          return;
        }
        applyDraft((result && result.segments) || []);
      },
      onStop: function () {
        if (refs.preProgress) refs.preProgress.hidden = true;
      }
    });
  }

  function startPrelabel() {
    if (!canSave()) {
      S.toast('离线不能用模型标注。', 'warn');
      return;
    }
    if (!pickedRun) {
      S.toast('先选择一轮已经训好的模型。', 'warn');
      return;
    }
    if (S.state.job) {
      S.toast('已经有一个任务在跑了，先停掉它。', 'warn');
      return;
    }
    var c = clip();
    var go = function () { runPrelabel(pickedRun, c.id); };
    if (c.segments.length) {
      S.confirm({
        title: '替换这条片子上的标注？',
        body: '会清掉当前这条的动作段，改成模型画出的段。不训练的区间还留着。点保存之前不会写盘。',
        danger: true,
        ok: '替换',
        onOk: go
      });
      return;
    }
    go();
  }

  function runLabel(run, runs) {
    var base = !run.display_name ? '未命名 · ' + run.name : run.display_name;
    var same = runs.filter(function (other) { return other.display_name === run.display_name; });
    if (run.display_name && same.length > 1) base = run.display_name + ' · ' + run.name;
    return run.train_unit === 'timeline' ? base : base + ' · 旧训练方式';
  }

  function prelabelBlock() {
    var runs = trainedRuns();
    var select = el('select', { class: 'form-select', 'aria-label': '选择模型' });
    select.appendChild(el('option', { value: '', text: runs.length ? '选择模型' : '还没有训好的模型' }));
    runs.forEach(function (run) {
      select.appendChild(el('option', { value: run.name, text: runLabel(run, runs) }));
    });
    if (pickedRun && !runs.some(function (run) { return run.name === pickedRun; })) pickedRun = '';
    select.value = pickedRun;
    var button = el('button', {
      class: 'btn btn-primary btn-sm', type: 'button', disabled: !pickedRun,
      onclick: startPrelabel
    }, '标注');
    select.addEventListener('change', function () {
      pickedRun = select.value;
      button.disabled = !pickedRun;
    });
    refs.preNote = el('p', {
      class: 'sp-meta sp-mt-2',
      text: '用选中的模型给当前这条片子画段。特征不是最新的会先只抽这一条。画上之后还要点保存标注。以前没起名的，显示成未命名加上训练时间。'
    });
    refs.preProgress = el('div', { class: 'check-progress sp-mt-3' });
    refs.preProgress.hidden = true;
    return el('div', { class: 'inspect-block' }, [
      el('h3', { text: '用模型标注' }),
      el('div', { class: 'row' }, [select, button]),
      refs.preNote,
      refs.preProgress
    ]);
  }

  function clip() { return S.clipById(S.state.clipId); }
  function segs() { return clip().segments; }
  function cuts() {
    var c = clip();
    if (!c.excluded) c.excluded = [];
    return c.excluded;
  }
  function allSpans() { return segs().concat(cuts()); }
  function findSeg(id) {
    var list = allSpans();
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return null;
  }
  function classOf(id) {
    var list = global.OD.data.classes;
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return { id: id, name: '未命名' };
  }
  function classIndex(id) {
    var list = global.OD.data.classes;
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return i;
    return 0;
  }

  function overlaps() {
    var list = allSpans().slice().sort(function (a, b) { return a.start - b.start; });
    var out = [];
    for (var i = 1; i < list.length; i++) if (list[i].start < list[i - 1].end) out.push([list[i - 1], list[i]]);
    return out;
  }

  /* ---------- 保存 ---------- */

  function canSave() { return !!(global.OD.api.available && S.state.projectId); }

  /* 只传标注字段，整条审核状态随每条片子往返。完成确认动作只在保存请求中附加。 */
  function payloadOf() {
    var data = global.OD.data;
    return {
      classes: data.classes.map(function (c) { return { id: c.id, name: c.name }; }),
      clips: data.clips.map(function (c) {
        return {
          id: c.id, file: c.file, frame_count: c.frame_count,
          fps: c.fps, width: c.width, height: c.height,
          annotation_complete: c.annotation_complete === true,
          segments: c.segments.map(function (s) {
            return { id: s.id, class_id: s.class_id, start: s.start, end: s.end };
          }),
          excluded: (c.excluded || []).map(function (s) {
            return { id: s.id, start: s.start, end: s.end };
          })
        };
      })
    };
  }

  function isDirty() { return JSON.stringify(payloadOf()) !== savedJson; }

  function contentKey(c) {
    return JSON.stringify({
      file: c.file, frame_count: c.frame_count, fps: c.fps, width: c.width, height: c.height,
      segments: c.segments.map(function (s) { return [s.class_id, s.start, s.end]; }).sort(),
      excluded: (c.excluded || []).map(function (s) { return [s.start, s.end]; }).sort()
    });
  }

  function updateCompletion(c) {
    var key = contentKey(c);
    if (seenContent[c.id] !== undefined && seenContent[c.id] !== key) {
      c.annotation_complete = false;
      delete completionConfirmations[c.id];
    }
    seenContent[c.id] = key;
  }

  function paintSaveState() {
    if (!refs.saveBtn || !refs.saveHint) return;
    if (!canSave()) {
      refs.saveBtn.disabled = true;
      refs.saveHint.textContent = '离线不写盘';
      refs.saveHint.title = '离线原型没有后端，标注改动只留在页面上';
      return;
    }
    var dirty = isDirty();
    refs.saveBtn.disabled = !!saveInFlight || !dirty;
    refs.saveHint.textContent = saveInFlight ? '正在保存；此后新改动仍需再次保存' : (dirty ? '有改动还没写回' : '与磁盘一致');
    refs.saveHint.title = dirty
      ? '保存会把整个 annotations.json 覆盖写一遍，所有片子一起'
      : '当前页面上的标注与 annotations.json 一致';
  }

  /* 保存是整体覆盖写：接口没有增量语义，改一条也要把全部片子一起交上去。 */
  function saveNow() {
    if (saveInFlight) return saveInFlight;
    if (!canSave()) {
      S.toast('离线原型没有后端，标注不会写盘。起训练台之后从这个地址打开才能保存。', 'warn');
      return;
    }
    var payload = payloadOf();
    var sentJson = JSON.stringify(payload);
    var data = global.OD.data, projectId = S.state.projectId, clipId = S.state.clipId;
    function currentContext() { return global.OD.data === data && S.state.projectId === projectId; }
    payload.completion_confirmations = Object.keys(completionConfirmations).filter(function (id) {
      return payload.clips.some(function (c) { return c.id === id && c.annotation_complete; });
    });
    refs.saveBtn.disabled = true;
    refs.saveHint.textContent = '正在写 annotations.json…';
    saveInFlight = global.OD.api.saveAnnotations(projectId, payload, clipId).then(function (result) {
      if (!currentContext()) return false;
      savedJson = sentJson;
      payload.completion_confirmations.forEach(function (id) {
        var sent = payload.clips.find(function (c) { return c.id === id; });
        var latest = data.clips.find(function (c) { return c.id === id; });
        if (sent && latest && contentKey(sent) === contentKey(latest)) delete completionConfirmations[id];
      });
      S.toast('标注已写回 annotations.json', 'ok');
      if (result && result.preparation_error) S.toast('标注已保存，特征准备未排队：' + result.preparation_error, 'warn');
      /* 保存后重读门禁与准备状态；特征缓存依据视频与模型，动作边界变化不重抽。 */
      if (JSON.stringify(payloadOf()) !== sentJson) {
        S.toast('本次提交已保存；保存期间的新改动已保留，请再保存一次。', 'warn');
        return true;
      }
      return global.OD.live.reload({shouldApply: function () {
        return currentContext() && JSON.stringify(payloadOf()) === sentJson;
      }}).catch(function (err) {
        if (!currentContext()) return;
        S.toast('保存成功，但重新读取工程失败：' + err.message, 'bad');
      });
    }).catch(function (err) {
      if (!currentContext()) return false;
      S.toast('保存标注失败：' + err.message, 'bad');
      return false;
    }).finally(function () {
      saveInFlight = null;
      if (currentContext()) paintSaveState();
    });
    return saveInFlight;
  }

  function beforeLeave() {
    if (saveInFlight || (savedFor === global.OD.data && isDirty())) {
      S.toast(saveInFlight ? '正在保存标注，请保存完成后再离开。' : '有未保存的动作标注，请先保存后再离开。', 'warn');
      return false;
    }
    return true;
  }

  /* ---------- 视图 ---------- */

  function render() {
    var c = clip();
    refs = {}; drag = null; videoBroken = false;
    fpsEff = c.fps;
    /* 只有整份数据被重新读过（写盘后 reload 会换一个新对象）才重新起算基线。
       切片子、重绘都不动它，否则改到一半切走再切回来，"有改动"就被抹掉了。 */
    if (savedFor !== global.OD.data) {
      savedFor = global.OD.data;
      savedJson = JSON.stringify(payloadOf());
      seenContent = {};
      completionConfirmations = {};
      global.OD.data.clips.forEach(function (item) { seenContent[item.id] = contentKey(item); });
    }
    /* 从结果页点"看画面"过来时带一个起始帧，用完就清掉，免得每次重绘都跳回去。 */
    var startFrame = S.state.seekFrame || 0;
    S.state.seekFrame = 0;

    /* 在线时素材走训练台的直链（支持 Range，能拖进度）；离线示例用随原型附带的那一条。 */
    var canWrite = global.OD.api.available && S.state.projectId;
    var mediaSrc = canWrite
      ? global.OD.api.mediaUrl(S.state.projectId, c.file)
      : 'assets/clip-1.mp4';
    refs.video = el('video', {
      src: mediaSrc, preload: 'metadata', playsinline: true,
      'aria-label': '片子 ' + c.file
    });
    refs.video.addEventListener('loadedmetadata', function () {
      if (refs.video.duration) fpsEff = c.frame_count / refs.video.duration;
      seek(frame());
    });
    refs.video.addEventListener('error', function () {
      videoBroken = true;
      if (refs.stageHint) refs.stageHint.hidden = false;
      if (refs.video) refs.video.hidden = true;
    });

    refs.video.dataset.frame = String(Math.max(0, Math.min(c.frame_count, startFrame)));
    refs.stageHint = el('div', { class: 'empty stage-hint', hidden: true }, canWrite ? [
      icon('film', 28),
      el('strong', { text: '训练台没把这条片子的素材给出来' }),
      el('p', { text: '工程目录里 media/' + c.file + ' 可能已经不在了。时间轴和标注编辑照常可用，改完照样能保存。' })
    ] : [
      icon('film', 28),
      el('strong', { text: '这条片子的视频没有随原型一起打开' }),
      el('p', { text: '离线原型只带了 assets/clip-1.mp4 一条示例素材，其余片子没有画面。时间轴和标注编辑照常可用。' })
    ]);

    refs.stage = el('div', { class: 'stage-cell' }, [refs.video, refs.stageHint, transport()]);
    refs.side = el('aside', { class: 'side', 'aria-label': '类别与段' });
    refs.ruler = el('div', { class: 'tl-ruler', title: '按住拖动可以定位到任意帧' });
    refs.labels = el('div', { class: 'tl-labels' });
    refs.canvas = el('div', { class: 'tl-canvas' });
    refs.tlScroll = el('div', {
      class: 'tl-scroll', tabindex: '0', role: 'application',
      'aria-label': '标注时间轴。回车选中画面当前帧上的段，方向键微调，Delete 删除'
    });

    refs.tlScroll.appendChild(refs.canvas);
    refs.tlScroll.addEventListener('scroll', function () {
      refs.ruler.style.transform = 'translateX(' + (-refs.tlScroll.scrollLeft) + 'px)';
    });
    refs.tlScroll.addEventListener('keydown', onKey);
    /* 刻度条整条都是定位区：它上面没有段、没有把手，是最不容易误触的地方。 */
    refs.ruler.addEventListener('mousedown', function (e) { startScrub(e, refs.ruler); });
    refs.tlScroll.addEventListener('wheel', function (e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZoom(S.state.zoom + (e.deltaY < 0 ? 1 : -1));
    }, { passive: false });

    refs.timeline = el('div', { class: 'timeline-cell' }, [
      el('div', { class: 'tl-head' }, [el('div', { class: 'pad' }), refs.ruler]),
      el('div', { class: 'tl-body' }, [refs.labels, refs.tlScroll])
    ]);

    var root = el('div', { class: 'view is-on annotate' }, [refs.stage, refs.side, refs.timeline]);
    requestAnimationFrame(function () { seek(startFrame); repaint(); });
    return root;
  }

  function transport() {
    var c = clip();
    refs.frameLabel = el('span', { class: 'tc' });
    refs.timeLabel = el('span', { class: 'tc' });
    refs.playBtn = el('button', { class: 'btn-icon', type: 'button', 'aria-label': '播放或暂停', onclick: togglePlay }, icon('play', 15));
    refs.zoomLabel = el('span', { class: 'tc zoom-label' });

    var zoom = el('input', {
      class: 'form-range', type: 'range', min: '1', max: '20', value: String(S.state.zoom),
      'aria-label': '时间轴缩放',
      oninput: function () { setZoom(Number(zoom.value)); }
    });
    refs.zoom = zoom;

    refs.saveHint = el('span', { class: 'tc sp-dim' });
    refs.saveBtn = el('button', {
      class: 'btn btn-primary btn-sm', type: 'button', onclick: saveNow
    }, [icon('check', 13), '保存标注']);

    return el('div', { class: 'transport' }, [
      el('button', { class: 'btn-icon', type: 'button', 'aria-label': '后退 10 帧', onclick: function () { seek(frame() - 10); } }, icon('rewind', 15)),
      el('button', { class: 'btn-icon', type: 'button', 'aria-label': '上一帧', onclick: function () { seek(frame() - 1); } }, icon('chev-left', 15)),
      refs.playBtn,
      el('button', { class: 'btn-icon', type: 'button', 'aria-label': '下一帧', onclick: function () { seek(frame() + 1); } }, icon('chev-right', 15)),
      el('button', { class: 'btn-icon', type: 'button', 'aria-label': '前进 10 帧', onclick: function () { seek(frame() + 10); } }, icon('forward', 15)),
      refs.frameLabel,
      refs.timeLabel,
      el('span', { class: 'spread' }),
      el('label', { class: 'row tight zoom-field' }, ['缩放', zoom]),
      refs.zoomLabel,
      el('span', { class: 'tc sp-dim hint-key', text: 'Ctrl+滚轮 缩放 · 回车 选段 · 方向键 微调 · Delete 删除' }),
      refs.saveHint,
      refs.saveBtn
    ]);
  }

  /* ---------- 播放与定位 ---------- */

  function frame() { return Number(refs.video.dataset.frame || 0); }

  function seek(f) {
    var c = clip();
    f = Math.max(0, Math.min(c.frame_count, Math.round(f)));
    refs.video.dataset.frame = String(f);
    if (!videoBroken) {
      try { refs.video.currentTime = f / fpsEff; } catch (e) { /* 元数据还没到，等 loadedmetadata */ }
    }
    if (refs.frameLabel) refs.frameLabel.innerHTML = '帧 <b>' + f + '</b> / ' + c.frame_count;
    if (refs.timeLabel) refs.timeLabel.textContent = D.fmtTimecode(f, c.fps);
    movePlayhead();
  }

  /* 红线分两截画：刻度条里那截是拖动定位的抓手，泳道里那截还是纯装饰。
     两截都跟着同一个帧号走，看上去就是一条线。 */
  function movePlayhead() {
    var x = frame() * S.state.zoom;
    var hidden = x > refs.canvasWidth;
    if (refs.playhead) {
      refs.playhead.style.left = x + 'px';
      refs.playhead.style.display = hidden ? 'none' : '';
    }
    if (refs.rulerTick) {
      refs.rulerTick.style.left = x + 'px';
      refs.rulerTick.style.display = hidden ? 'none' : '';
    }
  }

  function togglePlay() {
    if (videoBroken) { S.toast('视频没加载上，只有时间轴可用。', 'warn'); return; }
    if (refs.video.paused) {
      refs.video.play().catch(function () { S.toast('浏览器拦下了自动播放，请再点一次。', 'warn'); });
      loop();
    } else {
      refs.video.pause();
      if (raf) { cancelAnimationFrame(raf); raf = null; }
    }
    D.clear(refs.playBtn);
    refs.playBtn.appendChild(icon(refs.video.paused ? 'play' : 'pause', 15));
  }

  function loop() {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(function step() {
      if (refs.video.paused) { raf = null; return; }
      var f = Math.round(refs.video.currentTime * fpsEff);
      refs.video.dataset.frame = String(Math.min(f, clip().frame_count));
      seek(Number(refs.video.dataset.frame));
      raf = requestAnimationFrame(step);
    });
  }

  function setZoom(z) {
    S.state.zoom = Math.max(1, Math.min(20, z));
    if (refs.zoom) refs.zoom.value = String(S.state.zoom);
    repaint();
  }

  /* ---------- 重绘 ---------- */

  function repaint() {
    if (!refs.canvas) return;
    var c = clip(), zoom = S.state.zoom;
    updateCompletion(c);
    var width = Math.max(1, Math.round(c.frame_count * zoom));
    refs.canvasWidth = width;
    refs.canvas.style.width = width + 'px';
    refs.ruler.style.width = width + 'px';
    if (refs.zoomLabel) refs.zoomLabel.textContent = zoom + ' px/帧';

    /* 刻度：间隔随缩放走，保证两刻度之间至少 60px。 */
    D.clear(refs.ruler);
    var stepFrames = Math.max(1, Math.ceil(60 / zoom / 25) * 25);
    for (var f = 0; f <= c.frame_count; f += stepFrames) {
      refs.ruler.appendChild(el('div', { class: 'tick', style: 'left:' + (f * zoom) + 'px' },
        el('span', { text: String(f) })));
    }
    refs.rulerTick = el('div', { class: 'tl-play-tick' });
    refs.ruler.appendChild(refs.rulerTick);

    D.clear(refs.labels);
    D.clear(refs.canvas);

    global.OD.data.classes.forEach(function (cls, i) {
      refs.labels.appendChild(el('div', { class: 'tl-lane-label', onclick: function () { armClass(cls.id); } }, [
        el('span', { class: 'swatch', style: 'background:' + D.catColor(i) }),
        el('span', { text: cls.name })
      ]));
      var lane = el('div', {
        class: 'tl-lane' + (S.state.paintClass === cls.id ? ' is-active' : ''),
        'data-class': cls.id
      });
      lane.addEventListener('mousedown', function (e) { startDraw(e, lane, cls.id); });
      refs.canvas.appendChild(lane);

      segs().filter(function (s) { return s.class_id === cls.id; }).forEach(function (s) {
        lane.appendChild(segNode(s, i));
      });
    });

    refs.labels.appendChild(el('div', { class: 'tl-lane-label' }, [
      el('span', { class: 'swatch is-cut' }),
      el('span', { text: '不训练' })
    ]));
    var cutLane = el('div', { class: 'tl-lane tl-cut' });
    cutLane.addEventListener('mousedown', function (e) { startDraw(e, cutLane, null); });
    refs.canvas.appendChild(cutLane);
    cuts().forEach(function (s) { cutLane.appendChild(segNode(s, -1)); });

    refs.playhead = el('div', { class: 'tl-play' });
    /* 线本身只有 1px，抓不住。给它一条透明的宽把手，光标也换成左右箭头。 */
    refs.playhead.appendChild(el('div', {
      class: 'tl-play-grab', title: '拖动红线定位到任意帧，也可以按住上面的刻度条拖'
    }));
    refs.canvas.appendChild(refs.playhead);
    refs.playhead.addEventListener('mousedown', function (e) { startScrub(e, refs.canvas); });
    movePlayhead();

    paintSide();
    paintSaveState();
  }

  function segNode(s, colorIndex) {
    var zoom = S.state.zoom;
    var len = s.end - s.start;
    var picked = S.state.segSel === s.id;
    var isCut = colorIndex < 0;
    var name = isCut ? '不训练' : classOf(s.class_id).name;
    var node = el('div', {
      class: 'tl-seg' + (isCut ? ' is-cut' : '') + (picked ? ' is-sel' : '') + (!isCut && len < W() ? ' is-short' : ''),
      style: 'left:' + (s.start * zoom) + 'px;width:' + Math.max(6, len * zoom) + 'px' + (isCut ? '' : ';--seg-color:' + D.catColor(colorIndex)),
      'data-id': s.id,
      title: name + (s.missing_parts && s.missing_parts.length ? ' · 缺' + s.missing_parts.join('、') : '') + ' · ' + s.start + '–' + s.end + ' 帧（' + len + ' 帧）',
      role: 'button', tabindex: '-1',
      'aria-label': name + ' 第 ' + s.start + ' 到 ' + s.end + ' 帧' + (picked ? '，已选中' : '')
    }, [
      el('span', { text: name }),
      el('div', { class: 'h l', 'data-h': 'l' }),
      el('div', { class: 'h r', 'data-h': 'r' })
    ]);
    node.addEventListener('mousedown', startDrag);
    return node;
  }

  /* ---------- 拖动 ---------- */

  function startDrag(e) {
    var node = e.currentTarget;
    var s = findSeg(node.dataset.id);
    if (!s) return;
    e.preventDefault();
    S.state.segSel = s.id;
    var handle = e.target.dataset ? e.target.dataset.h : null;
    drag = { id: s.id, mode: handle || 'move', x0: e.clientX, start: s.start, end: s.end, node: node, moved: false };
    node.classList.add('is-drag');
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', endDrag);
    paintSide();
    showMinGuide(s.start);
  }

  function onDrag(e) {
    if (!drag) return;
    var zoom = S.state.zoom;
    var delta = Math.round((e.clientX - drag.x0) / zoom);
    if (delta !== 0) drag.moved = true;
    var s = findSeg(drag.id);
    if (!s) return;
    var limit = clip().frame_count;
    var start = drag.start, end = drag.end;

    if (drag.mode === 'move') {
      var len = drag.end - drag.start;
      start = Math.max(0, Math.min(limit - len, drag.start + delta));
      end = start + len;
    } else if (drag.mode === 'l') {
      start = Math.max(0, Math.min(drag.end - 1, drag.start + delta));
      end = drag.end;
      if (Math.abs((end - start) - W()) <= 2) start = end - W();
    } else {
      end = Math.min(limit, Math.max(drag.start + 1, drag.end + delta));
      start = drag.start;
      if (Math.abs((end - start) - W()) <= 2) end = start + W();
    }

    s.start = Math.round(start);
    s.end = Math.round(end);
    drag.node.style.left = (s.start * zoom) + 'px';
    drag.node.style.width = Math.max(6, (s.end - s.start) * zoom) + 'px';
    drag.node.classList.toggle('is-short', (s.end - s.start) < W());
    showMinGuide(s.start);
  }

  function endDrag() {
    if (drag) {
      var moved = findSeg(drag.id);
      if (moved && overlapsSpan(moved)) {
        moved.start = drag.start;
        moved.end = drag.end;
        S.toast('和另一段重叠了，已退回。', 'warn');
      }
    }
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', endDrag);
    if (drag && drag.node) drag.node.classList.remove('is-drag');
    drag = null;
    hideMinGuide();
    repaint();
  }

  /* ---------- 拖动定位（红线 / 刻度条） ---------- */

  /* 指针位置 → 帧号。canvas 和 ruler 都随横向滚动一起平移，所以各自的 left 就是内容起点，
     用哪个都行 —— 传进来是为了在错的那一个上也能算对。 */
  function frameAt(clientX, host) {
    return Math.round((clientX - host.getBoundingClientRect().left) / S.state.zoom);
  }

  function startScrub(e, host) {
    if (e.button !== 0) return;
    e.preventDefault();
    /* 正在播就先停住：一边播一边被拖着走，看到的位置是两股力在打架。松手再接着播。 */
    scrub = { host: host, wasPlaying: !videoBroken && !refs.video.paused };
    if (scrub.wasPlaying) {
      refs.video.pause();
      if (raf) { cancelAnimationFrame(raf); raf = null; }
      D.clear(refs.playBtn);
      refs.playBtn.appendChild(icon('play', 15));
    }
    document.body.classList.add('is-scrub');
    markScrub(true);
    seek(frameAt(e.clientX, host));
    document.addEventListener('mousemove', onScrub);
    document.addEventListener('mouseup', endScrub);
  }

  function onScrub(e) {
    if (!scrub) return;
    var f = frameAt(e.clientX, scrub.host);
    seek(f);
    /* 拖到视口外面就把时间轴跟着滚，不然长片子上拖不了多远。 */
    var view = refs.tlScroll;
    var x = f * S.state.zoom;
    var margin = 24;
    if (x < view.scrollLeft + margin) view.scrollLeft = Math.max(0, x - margin);
    else if (x > view.scrollLeft + view.clientWidth - margin) view.scrollLeft = x - view.clientWidth + margin;
  }

  function endScrub() {
    document.removeEventListener('mousemove', onScrub);
    document.removeEventListener('mouseup', endScrub);
    document.body.classList.remove('is-scrub');
    markScrub(false);
    var resume = scrub && scrub.wasPlaying;
    scrub = null;
    if (resume && !videoBroken) {
      refs.video.play().catch(function () { /* 浏览器拦下自动播放就停在这儿，不硬来 */ });
      loop();
      D.clear(refs.playBtn);
      refs.playBtn.appendChild(icon('pause', 15));
    }
  }

  /* 抓住的时候两截红线一起变色，让人知道现在拖的是它。 */
  function markScrub(on) {
    if (refs.playhead) refs.playhead.classList.toggle('is-scrub', on);
    if (refs.rulerTick) refs.rulerTick.classList.toggle('is-scrub', on);
  }

  function showMinGuide(startFrame) {
    hideMinGuide();
    var g = el('div', {
      class: 'tl-min',
      style: 'left:' + (startFrame * S.state.zoom) + 'px;width:' + (W() * S.state.zoom) + 'px',
      title: '最短窗长 ' + W() + ' 帧'
    });
    refs.canvas.appendChild(g);
    refs.minGuide = g;
  }

  function hideMinGuide() {
    if (refs.minGuide && refs.minGuide.parentNode) refs.minGuide.parentNode.removeChild(refs.minGuide);
    refs.minGuide = null;
  }

  /* ---------- 增删与键盘 ---------- */

  function overlapsSpan(s) {
    return allSpans().some(function (o) { return o.id !== s.id && s.start < o.end && o.start < s.end; });
  }

  function armClass(classId) {
    S.state.paintClass = classId;
    repaint();
    S.toast('在时间轴上「' + classOf(classId).name + '」那一行，按住从开始拖到结束。', 'ok');
  }

  function startDraw(e, lane, classId) {
    if (e.button !== 0 || e.target !== lane) return;
    e.preventDefault();
    var at = Math.max(0, Math.min(clip().frame_count, frameAt(e.clientX, refs.canvas)));
    var s = { id: (classId ? 'n' : 'x') + Date.now(), start: at, end: Math.min(clip().frame_count, at + 1) };
    if (s.end <= s.start) { S.toast('片尾放不下了。', 'warn'); return; }
    if (classId) {
      s.class_id = classId;
      segs().push(s);
    } else {
      cuts().push(s);
    }
    S.state.segSel = s.id;
    S.state.paintClass = classId;
    drawNew = { id: s.id, anchor: at, moved: false };
    document.addEventListener('mousemove', onDraw);
    document.addEventListener('mouseup', endDraw);
    repaint();
  }

  function onDraw(e) {
    if (!drawNew) return;
    var s = findSeg(drawNew.id);
    if (!s) return;
    var at = Math.max(0, Math.min(clip().frame_count, frameAt(e.clientX, refs.canvas)));
    if (at !== drawNew.anchor) drawNew.moved = true;
    s.start = Math.min(drawNew.anchor, at);
    s.end = Math.max(s.start + 1, Math.max(drawNew.anchor, at));
    var node = refs.canvas.querySelector('[data-id="' + s.id + '"]');
    if (!node) return;
    var zoom = S.state.zoom;
    node.style.left = (s.start * zoom) + 'px';
    node.style.width = Math.max(6, (s.end - s.start) * zoom) + 'px';
  }

  function endDraw() {
    document.removeEventListener('mousemove', onDraw);
    document.removeEventListener('mouseup', endDraw);
    var pending = drawNew;
    drawNew = null;
    if (!pending) return;
    var s = findSeg(pending.id);
    if (!s) return;
    if (!pending.moved) s.end = Math.min(clip().frame_count, s.start + W());
    var overlap = overlapsSpan(s);
    if (overlap || s.end - s.start < 1) {
      var list = segs();
      for (var i = 0; i < list.length; i++) if (list[i].id === s.id) { list.splice(i, 1); break; }
      S.state.segSel = null;
      repaint();
      S.toast(overlap ? '和已有的段重叠了，这一段没留下。' : '这段太短，没留下。', 'warn');
      return;
    }
    repaint();
    S.emit();
    S.toast((s.class_id ? classOf(s.class_id).name : '不训练') + ' 第 ' + s.start + '–' + s.end + ' 帧。左右边缘还可以再拖。', 'ok');
  }

  function addSegment(classId) {
    var at = frame();
    var s = { id: 'n' + Date.now(), class_id: classId, start: at, end: Math.min(at + W(), clip().frame_count) };
    /* 和已有段撞上就往后挪，直到让出位置。 */
    var guard = 0;
    while (guard++ < 200 && segs().some(function (o) { return s.start < o.end && o.start < s.end; })) {
      s.start = s.end; s.end = Math.min(s.start + W(), clip().frame_count);
      if (s.end - s.start < 1) break;
    }
    if (s.end - s.start < 1) { S.toast('片尾放不下了，先把时间轴挪到前面一点。', 'warn'); return; }
    segs().push(s);
    S.state.segSel = s.id;
    repaint();
    S.emit();
    S.toast('在第 ' + s.start + ' 帧加了一段' + classOf(classId).name + '，长 ' + (s.end - s.start) + ' 帧。', 'ok');
  }

  function deleteSegment(id) {
    var c = clip();
    var lists = [c.segments, c.excluded || []];
    for (var n = 0; n < lists.length; n++) {
      for (var i = 0; i < lists[n].length; i++) {
        if (lists[n][i].id !== id) continue;
        var removed = lists[n][i];
        lists[n].splice(i, 1);
        S.state.segSel = null;
        repaint();
        S.emit();
        var name = removed.class_id ? classOf(removed.class_id).name : '不训练';
        S.toast('删掉了 ' + name + ' 第 ' + removed.start + '–' + removed.end + ' 帧。', 'warn');
        return;
      }
    }
  }

  /* 键盘选段：回车在"画面当前帧盖住的段"之间循环。
     没有这一步，选段就只能靠鼠标拖，窄屏和纯键盘都用不了。 */
  function cycleSelection() {
    var at = frame();
    var hits = segs().filter(function (s) { return s.start <= at && at < s.end; })
      .sort(function (a, b) { return a.start - b.start; });
    if (!hits.length) {
      S.toast('第 ' + at + ' 帧上没有段。用方向键把画面挪到某一段里面再按回车。', 'warn');
      return;
    }
    var i = -1;
    for (var k = 0; k < hits.length; k++) if (hits[k].id === S.state.segSel) i = k;
    var next = hits[(i + 1) % hits.length];
    S.state.segSel = next.id;
    repaint();
    S.toast('选中了' + classOf(next.class_id).name + ' 第 ' + next.start + '–' + next.end + ' 帧。' +
      (hits.length > 1 ? '这一帧上还有 ' + (hits.length - 1) + ' 段，再按回车换下一段。' : '方向键微调，Delete 删除。'), 'ok');
  }

  function onKey(e) {
    var s = S.state.segSel ? findSeg(S.state.segSel) : null;
    var step = e.shiftKey ? 10 : 1;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); cycleSelection(); return; }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      var dir = e.key === 'ArrowRight' ? 1 : -1;
      if (s) {
        var len = s.end - s.start;
        var start = Math.max(0, Math.min(clip().frame_count - len, s.start + dir * step));
        s.start = start; s.end = start + len;
        repaint();
      } else {
        seek(frame() + dir * step);
      }
      return;
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && s) { e.preventDefault(); deleteSegment(s.id); return; }
    if (e.key === 'Escape') { S.state.segSel = null; repaint(); return; }
    if (e.key === 'Home') { e.preventDefault(); seek(0); return; }
    if (e.key === 'End') { e.preventDefault(); seek(clip().frame_count); return; }
    if (e.key === ',') { seek(frame() - 1); }
    if (e.key === '.') { seek(frame() + 1); }
  }

  /* ---------- 右侧面板 ---------- */

  function paintSide() {
    if (!refs.side) return;
    var c = clip();
    D.clear(refs.side);

    var completionCheck = el('input', {
      type: 'checkbox', checked: c.annotation_complete === true,
      'aria-label': '已检查整条视频，未涂区间确认为空闲',
      onchange: function () {
        c.annotation_complete = completionCheck.checked;
        if (completionCheck.checked) completionConfirmations[c.id] = true;
        else delete completionConfirmations[c.id];
        paintSaveState();
      }
    });
    refs.side.appendChild(el('div', { class: 'inspect-block' }, [
      el('h3', { text: '整条标注完成确认' }),
      el('label', { class: 'row tight' }, [completionCheck, el('span', { text: '已检查整条视频，未涂区间确认为空闲' })]),
      el('p', { class: 'sp-meta sp-mt-2', text: '勾选后点击“保存标注”才生效。训练和验证视频都需要确认；修改动作段或不训练区间会撤销确认。后台特征准备可提前进行。' })
    ]));

    refs.side.appendChild(prelabelBlock());
    refs.side.appendChild(global.OD.preparation.render());

    refs.side.appendChild(el('div', { class: 'inspect-block' }, [
      el('h3', { text: '类别' }),
      el('p', { class: 'sp-meta sp-mb-3', text: '增加、改名或删除。改名会留下原来的段；删除会去掉这一类在所有片子上的标注。' }),
      global.OD.classEditor()
    ]));

    refs.side.appendChild(el('div', { class: 'inspect-block' }, [
      el('h3', { text: '在时间轴上画一段' }),
      el('p', { class: 'sp-meta sp-mb-3', text: '先点一个类别，再在下面同名的那一行按住，从动作开始拖到动作结束。最底下一行是「不训练」：拖过的画面不进训练，也不会变成空闲。' }),
      el('div', null, global.OD.data.classes.map(function (cls, i) {
        var armed = S.state.paintClass === cls.id;
        return el('button', {
          class: 'step pick' + (armed ? ' is-on' : ''), type: 'button',
          onclick: function () { armClass(cls.id); }
        }, [
          el('span', { class: 'swatch', style: 'background:' + D.catColor(i) }),
          el('span', { class: 'name', text: cls.name }),
          el('span', { class: 'sub', text: armed ? '正在画' : '点选' })
        ]);
      }))
    ]));

    var sel = S.state.segSel ? findSeg(S.state.segSel) : null;
    refs.side.appendChild(el('div', { class: 'inspect-block' }, [
      el('h3', { text: sel ? '选中的段' : '没有选中段' }),
      sel ? segmentEditor(sel) : el('p', { class: 'sp-meta' },
        '点时间轴上任意一段来改它的起止帧。纯键盘也能用：光标停在时间轴上按回车选中画面当前帧上的段，方向键微调 1 帧，按住 Shift 是 10 帧。')
    ]));

    var issues = S.clipIssues(c);
    var ov = overlaps();
    refs.side.appendChild(el('div', { class: 'inspect-block' }, [
      el('h3', { text: '这条片子的检查' }),
      el('dl', { class: 'keyval sp-mb-3' }, [
        el('dt', { text: '标注段' }), el('dd', { text: c.segments.length + ' 段' }),
        el('dt', { text: '可训练窗口' }), el('dd', { text: issues.windows + ' 个' }),
        el('dt', { text: '短于 ' + S.windowSeconds() + ' 秒' }), el('dd', { text: issues.short.length + ' 段' }),
        el('dt', { text: '片尾未标注' }), el('dd', { text: issues.tail + ' 帧' })
      ]),
      issues.windows === 0 ? el('div', { class: 'callout bad' }, [
        icon('alert', 15),
        el('div', null, [el('strong', { text: '现在产不出训练样本。 ' }), '每一段都短于 ' + W() + ' 帧。把同一个动作的相邻段连起来，或者重新标一段长的。'])
      ]) : null,
      ov.length ? el('div', { class: 'callout bad sp-mt-3' }, [
        icon('alert', 15),
        el('div', { text: '有 ' + ov.length + ' 处重叠：第 ' + ov[0][0].start + '–' + ov[0][1].end + ' 帧。重叠会让窗口归属不清，训练直接挡住。' })
      ]) : null,
      issues.narrow.length ? el('div', { class: 'callout warn sp-mt-3' }, [
        icon('info', 15),
        el('div', { text: issues.narrow.length + ' 处窄缝（≤3 帧）。同一个动作被切成两段时，把中间补上或把两段合并。' })
      ]) : null
    ]));
  }

  function segmentEditor(s) {
    var len = s.end - s.start;
    var startIn = el('input', { class: 'form-control mono', type: 'number', value: s.start, min: 0, max: clip().frame_count, 'aria-label': '起始帧' });
    var endIn = el('input', { class: 'form-control mono', type: 'number', value: s.end, min: 1, max: clip().frame_count, 'aria-label': '结束帧' });

    function commit() {
      var a = Math.max(0, Math.min(clip().frame_count - 1, Number(startIn.value) || 0));
      var b = Math.max(a + 1, Math.min(clip().frame_count, Number(endIn.value) || a + 1));
      s.start = Math.round(a); s.end = Math.round(b);
      repaint();
    }
    startIn.addEventListener('change', commit);
    endIn.addEventListener('change', commit);

    var issues = S.clipIssues(clip());
    return el('div', null, [
      el('div', { class: 'row tight sp-mb-3' }, [
        el('span', { class: 'swatch', style: 'background:' + D.catColor(classIndex(s.class_id)) }),
        el('strong', { class: 'sp-fs-13', text: s.class_id ? classOf(s.class_id).name : '不训练' })
      ]),
      el('div', { class: 'seg-form' }, [
        el('div', { class: 'field' }, [el('label', { class: 'form-label', text: '起始帧' }), startIn]),
        el('div', { class: 'field' }, [el('label', { class: 'form-label', text: '结束帧' }), endIn])
      ]),
      el('dl', { class: 'keyval sp-mt-3' }, [
        el('dt', { text: '段长' }), el('dd', { class: len < W() ? 'sp-weak' : '', text: len + ' 帧 / ' + D.fmtDuration(len, clip().fps) }),
        el('dt', { text: '产出窗口' }), el('dd', { text: (len >= W() ? Math.floor((len - W()) / C.STRIDE) + 1 : 0) + ' 个' }),
        el('dt', { text: '最短窗长' }), el('dd', { text: W() + ' 帧（' + S.windowSeconds() + ' 秒）' })
      ]),
      len < W() ? el('div', { class: 'help bad sp-mt-2' },
        '短了 ' + (W() - len) + ' 帧（这条片子 ' + clip().fps.toFixed(2) + 'fps，最短要 ' + S.windowSeconds() + ' 秒）。' +
        '拖右边把它拉到 ' + W() + ' 帧以上，或者把相邻的同一类段连起来。') : null,
      s.missing_parts && s.missing_parts.length ? el('div', { class: 'help bad sp-mt-2', text: '缺' + s.missing_parts.join('、') + '。这段还在时间轴上，你看着留或删。' }) : null,
      el('div', { class: 'row sp-mt-3' }, [
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { seek(s.start); } }, '跳到开头'),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', onclick: function () { seek(s.end); } }, '跳到结尾'),
        el('span', { class: 'spread' }),
        el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: function () { deleteSegment(s.id); } }, [icon('trash', 13), '删除'])
      ])
    ]);
  }

  global.OD.views = global.OD.views || {};
  global.addEventListener('beforeunload', function (event) {
    if (saveInFlight || (savedFor === global.OD.data && isDirty())) {
      event.preventDefault(); event.returnValue = '';
    }
  });
  global.OD.views.annotate = { render: render, beforeLeave: beforeLeave, inspector: function () { return null; } };
})(window);
