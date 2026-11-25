/* 专业四点 OBB 标注画布：支持选择编辑、水平框转 OBB、3 点旋转框、Meta SAM 交互点选/拉框与视觉示例框选。
   坐标始终归一化到 [0, 1]；旋转与 3 点垂直计算在原图像素比例空间完成。 */
(function (global) {
  'use strict';

  var COLORS = [
    '#3b82f6', '#a855f7', '#f59e0b', '#14b8a6', '#ec4899', '#84cc16',
    '#06b6d4', '#f97316', '#8b5cf6', '#10b981', '#e11d48', '#6366f1'
  ];

  function copy(v) { return JSON.parse(JSON.stringify(v)); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function center(p) {
    return [
      p.reduce(function (a, v) { return a + v[0]; }, 0) / 4,
      p.reduce(function (a, v) { return a + v[1]; }, 0) / 4
    ];
  }
  function rectangle(a, b) {
    return [
      [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
      [Math.max(a[0], b[0]), Math.min(a[1], b[1])],
      [Math.max(a[0], b[0]), Math.max(a[1], b[1])],
      [Math.min(a[0], b[0]), Math.max(a[1], b[1])]
    ];
  }
  function valid(p) {
    if (!Array.isArray(p) || p.length !== 4 || p.some(function (v) {
      return !Array.isArray(v) || v.length !== 2 || v.some(function (n) {
        return !Number.isFinite(n) || n < 0 || n > 1;
      });
    })) return false;
    var sign = 0, area = 0;
    for (var i = 0; i < 4; i++) {
      var a = p[i], b = p[(i + 1) % 4], c = p[(i + 2) % 4];
      var cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (Math.abs(cross) < 1e-9) return false;
      if (sign && Math.sign(cross) !== sign) return false;
      sign = Math.sign(cross);
      area += a[0] * b[1] - a[1] * b[0];
    }
    return Math.abs(area) > 1e-7;
  }
  function contains(p, q) {
    var inside = false;
    for (var i = 0, j = p.length - 1; i < p.length; j = i++) {
      if (((p[i][1] > q[1]) !== (p[j][1] > q[1])) &&
          (q[0] < (p[j][0] - p[i][0]) * (q[1] - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) {
        inside = !inside;
      }
    }
    return inside;
  }
  function translate(p, dx, dy) {
    dx = clamp(dx, -Math.min.apply(null, p.map(function (v) { return v[0]; })), 1 - Math.max.apply(null, p.map(function (v) { return v[0]; })));
    dy = clamp(dy, -Math.min.apply(null, p.map(function (v) { return v[1]; })), 1 - Math.max.apply(null, p.map(function (v) { return v[1]; })));
    return p.map(function (v) { return [v[0] + dx, v[1] + dy]; });
  }
  function rotate(p, r, aspect) {
    var c = center(p), a = aspect || 1;
    return p.map(function (v) {
      var x = (v[0] - c[0]) * a, y = v[1] - c[1];
      return [c[0] + (x * Math.cos(r) - y * Math.sin(r)) / a, c[1] + x * Math.sin(r) + y * Math.cos(r)];
    });
  }
  function scale(p, factor) {
    var c = center(p);
    return p.map(function (v) { return [c[0] + (v[0] - c[0]) * factor, c[1] + (v[1] - c[1]) * factor]; });
  }
  function fitInsideUnit(pts) {
    if (valid(pts)) return pts;
    var c = center(pts);
    var cx = clamp(c[0], 0.002, 0.998), cy = clamp(c[1], 0.002, 0.998);
    var alpha = 1;
    pts.forEach(function (v) {
      var dx = v[0] - cx, dy = v[1] - cy;
      if (v[0] < 1e-5 && dx < -1e-12) alpha = Math.min(alpha, (cx - 1e-5) / Math.abs(dx));
      if (v[0] > 1 - 1e-5 && dx > 1e-12) alpha = Math.min(alpha, (1 - 1e-5 - cx) / Math.abs(dx));
      if (v[1] < 1e-5 && dy < -1e-12) alpha = Math.min(alpha, (cy - 1e-5) / Math.abs(dy));
      if (v[1] > 1 - 1e-5 && dy > 1e-12) alpha = Math.min(alpha, (1 - 1e-5 - cy) / Math.abs(dy));
    });
    alpha = clamp(alpha * 0.999, 0, 1);
    var shrunk = pts.map(function (v) { return [clamp(cx + (v[0] - cx) * alpha, 0, 1), clamp(cy + (v[1] - cy) * alpha, 0, 1)]; });
    return valid(shrunk) ? shrunk : null;
  }
  function fromThreePoints(p1, p2, p3, aspect) {
    var a = aspect || 1;
    var ux = (p2[0] - p1[0]) * a, uy = p2[1] - p1[1];
    var len = Math.hypot(ux, uy);
    if (len < 1e-4) return null;
    var nx = -uy / len, ny = ux / len;
    var wx = (p3[0] - p2[0]) * a, wy = p3[1] - p2[1];
    var dist = wx * nx + wy * ny;
    if (Math.abs(dist) < 1e-4) return null;
    var ox = (nx * dist) / a, oy = ny * dist;
    var quad = [
      [p1[0], p1[1]],
      [p2[0], p2[1]],
      [p2[0] + ox, p2[1] + oy],
      [p1[0] + ox, p1[1] + oy]
    ];
    return fitInsideUnit(quad);
  }
  function bboxOf(points) {
    var xs = points.map(function (v) { return v[0]; });
    var ys = points.map(function (v) { return v[1]; });
    return [Math.min.apply(null, xs), Math.min.apply(null, ys), Math.max.apply(null, xs), Math.max.apply(null, ys)];
  }
  function colorFor(classId) {
    return COLORS[Math.abs(Number(classId) || 0) % COLORS.length];
  }

  var geometry = {
    rectangle: rectangle,
    valid: valid,
    contains: contains,
    translate: translate,
    rotate: rotate,
    scale: scale,
    center: center,
    fromThreePoints: fromThreePoints,
    fitInsideUnit: fitInsideUnit,
    bboxOf: bboxOf,
    colorFor: colorFor
  };

  function Editor(host, options) {
    this.options = options || {};
    this.host = host;
    this.boxes = copy(options.boxes || []);
    this.classes = copy(options.classes || []);
    this.selected = -1;
    this.classId = Number(options.classId || 0);
    this.mode = options.mode || 'select'; // select | draw(hbb) | obb3 | smart | exemplar
    this.prevMode = 'select';
    this.zoom = 1;
    this.undoStack = [];
    this.redoStack = [];
    this.drag = null;
    this.obb3Points = [];
    this.cursor = null;
    this.spaceDown = false;
    this.pan = null;
    this.smartPoints = [];
    this.smartLabels = [];
    this.smartBbox = null;
    this.smartPreview = null; // {points, polygon, score, source}
    this.smartBusy = false;
    this.ready = false;

    var self = this;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'obb-canvas';
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('aria-label', '零件旋转框画布');
    this.canvas.setAttribute('role', 'application');
    host.appendChild(this.canvas);

    this.image = new Image();
    this.image.onload = function () {
      self.canvas.width = self.image.naturalWidth;
      self.canvas.height = self.image.naturalHeight;
      self.ready = true;
      self.paint();
    };
    this.image.onerror = function () {
      if (self.options.onError) self.options.onError('图片读取失败，请检查素材是否仍存在');
    };
    this.image.src = options.src;

    this.canvas.addEventListener('pointerdown', function (e) { self.down(e); });
    this.canvas.addEventListener('pointermove', function (e) { self.move(e); });
    this.canvas.addEventListener('pointerup', function (e) { self.up(e); });
    this.canvas.addEventListener('pointercancel', function () { self.cancel(); });
    this.canvas.addEventListener('pointerleave', function () { self.cursor = null; self.paint(); });
    this.canvas.addEventListener('contextmenu', function (e) {
      if (self.mode === 'smart' || self.mode === 'obb3') e.preventDefault();
    });
    this.canvas.addEventListener('keydown', function (e) { self.key(e); });
    this.canvas.addEventListener('keyup', function (e) {
      if (e.code === 'Space') { self.spaceDown = false; self._updateCursor(); }
    });
    this.canvas.addEventListener('wheel', function (e) {
      if (!self.ready) return;
      e.preventDefault();
      var factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      self.zoomAround(self.zoom * factor, e.clientX, e.clientY);
    }, { passive: false });
    this._updateCursor();
  }

  Editor.prototype._scrollHost = function () {
    return this.host && this.host.parentElement ? this.host.parentElement : null;
  };
  Editor.prototype._aspect = function () {
    return (this.canvas.width && this.canvas.height) ? (this.canvas.width / this.canvas.height) : 1;
  };
  Editor.prototype._updateCursor = function () {
    if (this.pan) { this.canvas.style.cursor = 'grabbing'; return; }
    if (this.spaceDown) { this.canvas.style.cursor = 'grab'; return; }
    if (this.mode === 'draw' || this.mode === 'hbb' || this.mode === 'obb3' || this.mode === 'smart' || this.mode === 'exemplar') {
      this.canvas.style.cursor = 'crosshair';
    } else {
      this.canvas.style.cursor = 'default';
    }
  };
  Editor.prototype.point = function (e) {
    var r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return [0, 0];
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };
  Editor.prototype.distance = function (a, b) {
    var r = this.canvas.getBoundingClientRect();
    return Math.hypot((a[0] - b[0]) * r.width, (a[1] - b[1]) * r.height);
  };
  Editor.prototype.rotationHandle = function (p) {
    var c = center(p), m = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2], r = this.canvas.getBoundingClientRect();
    var x = (m[0] - c[0]) * r.width, y = (m[1] - c[1]) * r.height, len = Math.hypot(x, y) || 1;
    return [m[0] + x / len * 28 / (r.width || 1), m[1] + y / len * 28 / (r.height || 1)];
  };

  Editor.prototype.down = function (e) {
    if (!this.ready) return;
    this.canvas.focus();

    // 中键或空格+左键：平移画布
    if (e.button === 1 || (e.button === 0 && this.spaceDown)) {
      var sc = this._scrollHost();
      if (sc) {
        e.preventDefault();
        this.canvas.setPointerCapture(e.pointerId);
        this.pan = { x: e.clientX, y: e.clientY, left: sc.scrollLeft, top: sc.scrollTop };
        this._updateCursor();
      }
      return;
    }

    var p = this.point(e);
    this.cursor = p;

    // 3 点旋转框：右键撤销上一个点
    if (this.mode === 'obb3') {
      e.preventDefault();
      if (e.button === 2) {
        this.obb3Points.pop();
        this.paint();
        return;
      }
      if (e.button !== 0) return;
      if (!this.classes.length) {
        if (this.options.onError) this.options.onError('请先添加至少一个零件类别');
        return;
      }
      if (this.obb3Points.length < 2) {
        this.obb3Points.push(p);
        this.paint();
        return;
      }
      var quad = fromThreePoints(this.obb3Points[0], this.obb3Points[1], p, this._aspect());
      if (quad) {
        this.commit(function () {
          this.boxes.push({ class_id: this.classId, points: quad, source: 'manual' });
          this.selected = this.boxes.length - 1;
        });
      } else if (this.options.onError) {
        this.options.onError('三点框面积过小或三点共线，请重新点击');
      }
      this.obb3Points = [];
      this.paint();
      return;
    }

    // 智能点选/拉粗框模式（SAM 3 / SAM 2.1 / MobileSAM）
    if (this.mode === 'smart') {
      if (e.button !== 0 && e.button !== 2) return;
      e.preventDefault();
      if (!this.classes.length) {
        if (this.options.onError) this.options.onError('请先添加至少一个零件类别');
        return;
      }
      if (e.button === 2) {
        // 右键负样本背景点
        this.smartPoints.push(p);
        this.smartLabels.push(0);
        this.paint();
        this._fireSmartPrompt();
        return;
      }
      this.canvas.setPointerCapture(e.pointerId);
      this.drag = { kind: 'smart', start: p, startClient: [e.clientX, e.clientY], current: p };
      return;
    }

    // 视觉示例框选模式（YOLOE / SAM 3 Visual Exemplar）
    if (this.mode === 'exemplar') {
      if (e.button !== 0) return;
      e.preventDefault();
      this.canvas.setPointerCapture(e.pointerId);
      this.drag = { kind: 'exemplar', start: p, current: p };
      this.paint();
      return;
    }

    if (e.button !== 0) return;
    e.preventDefault();
    this.canvas.setPointerCapture(e.pointerId);
    var before = copy(this.boxes), i, j, kind = '';
    var isDraw = (this.mode === 'draw' || this.mode === 'hbb');

    if (!isDraw && this.selected >= 0 && this.boxes[this.selected]) {
      var points = this.boxes[this.selected].points;
      for (j = 0; j < 4; j++) {
        if (this.distance(points[j], p) < 10) { kind = 'vertex'; break; }
      }
      if (!kind && this.distance(this.rotationHandle(points), p) < 13) kind = 'rotate';
    }
    if (kind) {
      this.drag = { kind: kind, vertex: j, start: p, before: before, index: this.selected };
      return;
    }
    if (!isDraw) {
      for (i = this.boxes.length - 1; i >= 0; i--) {
        if (contains(this.boxes[i].points, p)) {
          this.selected = i;
          this.drag = { kind: 'move', start: p, before: before, index: i };
          this.changedSelection();
          this.paint();
          return;
        }
      }
    }
    if (isDraw) {
      if (!this.classes.length) {
        if (this.options.onError) this.options.onError('请先添加至少一个零件类别');
        return;
      }
      this.boxes.push({ class_id: this.classId, points: rectangle(p, p), source: 'manual' });
      this.selected = this.boxes.length - 1;
      this.drag = { kind: 'draw', start: p, before: before, index: this.selected };
    } else {
      this.selected = -1;
    }
    this.changedSelection();
    this.paint();
  };

  Editor.prototype.move = function (e) {
    if (this.pan) {
      var sc = this._scrollHost();
      if (sc) {
        sc.scrollLeft = this.pan.left - (e.clientX - this.pan.x);
        sc.scrollTop = this.pan.top - (e.clientY - this.pan.y);
      }
      return;
    }
    var p = this.point(e);
    this.cursor = p;

    if (!this.drag) {
      if (this.mode === 'draw' || this.mode === 'hbb' || this.mode === 'obb3' || this.mode === 'smart' || this.mode === 'exemplar') {
        this.paint();
      }
      return;
    }
    var d = this.drag;
    if (d.kind === 'smart' || d.kind === 'exemplar') {
      d.current = p;
      this.paint();
      return;
    }
    var old = d.before[d.index], next;
    if (d.kind === 'draw') {
      this.boxes[d.index].points = rectangle(d.start, p);
    } else if (d.kind === 'move') {
      this.boxes[d.index].points = translate(old.points, p[0] - d.start[0], p[1] - d.start[1]);
    } else if (d.kind === 'vertex') {
      next = copy(old.points);
      next[d.vertex] = p;
      if (valid(next)) this.boxes[d.index].points = next;
    } else if (d.kind === 'rotate') {
      var c = center(old.points), a = this._aspect();
      var angle = Math.atan2(p[1] - c[1], (p[0] - c[0]) * a) - Math.atan2(d.start[1] - c[1], (d.start[0] - c[0]) * a);
      next = rotate(old.points, angle, a);
      if (valid(next)) this.boxes[d.index].points = next;
    }
    this.paint();
  };

  Editor.prototype.up = function (e) {
    if (this.pan) {
      this.pan = null;
      this._updateCursor();
      return;
    }
    if (!this.drag) return;
    var d = this.drag;
    this.drag = null;

    if (d.kind === 'smart') {
      var movedPx = e ? Math.hypot(e.clientX - d.startClient[0], e.clientY - d.startClient[1]) : 0;
      if (movedPx > 8) {
        this.smartBbox = [
          Math.min(d.start[0], d.current[0]),
          Math.min(d.start[1], d.current[1]),
          Math.max(d.start[0], d.current[0]),
          Math.max(d.start[1], d.current[1])
        ];
      } else {
        this.smartPoints.push(d.start);
        this.smartLabels.push(1);
      }
      this.paint();
      this._fireSmartPrompt();
      return;
    }

    if (d.kind === 'exemplar') {
      var box = [
        Math.min(d.start[0], d.current[0]),
        Math.min(d.start[1], d.current[1]),
        Math.max(d.start[0], d.current[0]),
        Math.max(d.start[1], d.current[1])
      ];
      this.setMode(this.prevMode || 'select');
      if ((box[2] - box[0]) > 0.005 && (box[3] - box[1]) > 0.005) {
        if (this.options.onExemplarPicked) this.options.onExemplarPicked(box);
      }
      this.paint();
      return;
    }

    var before = d.before;
    if (!valid(this.boxes[d.index].points)) {
      this.boxes = before;
      this.selected = -1;
    } else if (JSON.stringify(before) !== JSON.stringify(this.boxes)) {
      this.undoStack.push(before);
      this.redoStack = [];
      this.notify();
    }
    this.changedSelection();
    this.paint();
  };

  Editor.prototype.cancel = function () {
    this.pan = null;
    this.obb3Points = [];
    if (this.drag) {
      if (this.drag.before) this.boxes = this.drag.before;
      this.drag = null;
      this.selected = -1;
    }
    this._updateCursor();
    this.paint();
  };

  Editor.prototype._fireSmartPrompt = function () {
    if (this.options.onSmartPrompt) {
      this.options.onSmartPrompt({
        points: copy(this.smartPoints),
        labels: copy(this.smartLabels),
        bbox: this.smartBbox ? copy(this.smartBbox) : null
      });
    }
  };
  Editor.prototype.setSmartBusy = function (busy) {
    this.smartBusy = !!busy;
    this.paint();
  };
  Editor.prototype.setSmartPreview = function (preview) {
    this.smartPreview = preview ? copy(preview) : null;
    this.paint();
  };
  Editor.prototype.clearSmart = function () {
    this.smartPoints = [];
    this.smartLabels = [];
    this.smartBbox = null;
    this.smartPreview = null;
    this.smartBusy = false;
    this.paint();
  };
  Editor.prototype.acceptSmartPreview = function () {
    if (!this.smartPreview || !valid(this.smartPreview.points)) return false;
    var preview = this.smartPreview;
    this.commit(function () {
      var box = { class_id: this.classId, points: copy(preview.points) };
      if (preview.source) box.source = preview.source;
      if (typeof preview.score === 'number' && Number.isFinite(preview.score)) box.confidence = preview.score;
      this.boxes.push(box);
      this.selected = this.boxes.length - 1;
    });
    this.clearSmart();
    return true;
  };

  Editor.prototype.commit = function (change) {
    var before = copy(this.boxes);
    change.call(this);
    if (JSON.stringify(before) !== JSON.stringify(this.boxes)) {
      this.undoStack.push(before);
      this.redoStack = [];
      this.notify();
      this.changedSelection();
      this.paint();
    }
  };
  Editor.prototype.addBoxes = function (incoming, replace) {
    if (!Array.isArray(incoming) || !incoming.length) return;
    this.commit(function () {
      var validBoxes = incoming.filter(function (b) { return b && valid(b.points); }).map(copy);
      if (replace) this.boxes = validBoxes;
      else this.boxes = this.boxes.concat(validBoxes);
      this.selected = this.boxes.length ? this.boxes.length - 1 : -1;
    });
  };
  Editor.prototype.setClasses = function (classes) {
    this.classes = copy(classes || []);
    if (this.classId >= this.classes.length) this.classId = 0;
    this.paint();
  };
  Editor.prototype.notify = function () {
    if (this.options.onChange) this.options.onChange(copy(this.boxes));
  };
  Editor.prototype.changedSelection = function () {
    if (this.selected >= 0 && this.boxes[this.selected]) this.classId = this.boxes[this.selected].class_id;
    if (this.options.onSelect) this.options.onSelect(this.selected, this.selected >= 0 && this.boxes[this.selected] ? copy(this.boxes[this.selected]) : null);
  };
  Editor.prototype.undo = function () {
    if (!this.undoStack.length) return;
    this.redoStack.push(copy(this.boxes));
    this.boxes = this.undoStack.pop();
    this.selected = -1;
    this.notify();
    this.changedSelection();
    this.paint();
  };
  Editor.prototype.redo = function () {
    if (!this.redoStack.length) return;
    this.undoStack.push(copy(this.boxes));
    this.boxes = this.redoStack.pop();
    this.selected = -1;
    this.notify();
    this.changedSelection();
    this.paint();
  };
  Editor.prototype.remove = function () {
    if (this.selected < 0) return;
    this.commit(function () {
      this.boxes.splice(this.selected, 1);
      this.selected = -1;
    });
  };
  Editor.prototype.duplicate = function () {
    if (this.selected < 0) return;
    this.commit(function () {
      var box = copy(this.boxes[this.selected]);
      box.points = translate(box.points, 0.02, 0.02);
      this.boxes.push(box);
      this.selected = this.boxes.length - 1;
    });
  };
  Editor.prototype.setClass = function (id) {
    this.classId = Number(id);
    if (this.selected >= 0 && this.boxes[this.selected]) {
      this.commit(function () { this.boxes[this.selected].class_id = this.classId; });
    }
  };
  Editor.prototype.rotateBy = function (degrees) {
    if (this.selected < 0) return;
    this.commit(function () {
      var p = rotate(this.boxes[this.selected].points, degrees * Math.PI / 180, this._aspect());
      if (valid(p)) this.boxes[this.selected].points = p;
    });
  };
  Editor.prototype.scaleBy = function (factor) {
    if (this.selected < 0) return;
    this.commit(function () {
      var p = scale(this.boxes[this.selected].points, factor);
      if (valid(p)) this.boxes[this.selected].points = p;
    });
  };
  Editor.prototype.setZoom = function (zoom) {
    this.zoom = clamp(zoom, 0.25, 6);
    this.canvas.style.width = (this.zoom * 100) + '%';
    this.canvas.style.maxWidth = 'none';
    if (this.options.onZoom) this.options.onZoom(this.zoom);
    this.paint();
  };
  Editor.prototype.zoomAround = function (nextZoom, clientX, clientY) {
    var sc = this._scrollHost();
    var before = this.canvas.getBoundingClientRect();
    var relX = before.width ? (clientX - before.left) / before.width : 0.5;
    var relY = before.height ? (clientY - before.top) / before.height : 0.5;
    this.setZoom(nextZoom);
    if (sc) {
      var after = this.canvas.getBoundingClientRect();
      var scRect = sc.getBoundingClientRect();
      sc.scrollLeft += relX * after.width - (clientX - scRect.left) + (after.left - scRect.left);
      sc.scrollTop += relY * after.height - (clientY - scRect.top) + (after.top - scRect.top);
    }
  };
  Editor.prototype.setMode = function (mode) {
    var normalized = mode === 'hbb' ? 'draw' : mode;
    if (normalized === 'exemplar' && this.mode !== 'exemplar') this.prevMode = this.mode;
    this.mode = normalized;
    this.obb3Points = [];
    if (normalized !== 'smart') this.clearSmart();
    this._updateCursor();
    if (this.options.onModeChange) this.options.onModeChange(normalized);
    this.paint();
  };

  Editor.prototype.key = function (e) {
    var ctrl = e.ctrlKey || e.metaKey;
    var k = (e.key || '').toLowerCase();
    if (e.code === 'Space' && !ctrl) {
      // 空格在有智能预览时不冲突；由外层快捷键决定“审核并下一张”或长按平移
      return;
    }
    if (ctrl && k === 's') { e.preventDefault(); if (this.options.onSave) this.options.onSave(); return; }
    if (ctrl && k === 'z') { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); return; }
    if (ctrl && k === 'y') { e.preventDefault(); this.redo(); return; }
    if (ctrl && k === 'd') { e.preventDefault(); this.duplicate(); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); this.remove(); return; }
    if (e.key === 'Enter' && this.mode === 'smart' && this.smartPreview) {
      e.preventDefault();
      this.acceptSmartPreview();
      return;
    }
    if (e.key === 'Escape') {
      if (this.mode === 'smart' && (this.smartPoints.length || this.smartBbox || this.smartPreview)) {
        this.clearSmart();
        return;
      }
      if (this.obb3Points.length) {
        this.obb3Points = [];
        this.paint();
        return;
      }
      this.cancel();
      this.setMode('select');
      this.selected = -1;
      this.paint();
      return;
    }
    if (k === 'n' || k === 'w') this.setMode('draw');
    else if (k === 'v') this.setMode('select');
    else if (k === 'r') this.setMode('obb3');
    else if (k === 's' && !ctrl) this.setMode('smart');
    else if (e.key === '[') this.rotateBy(-5);
    else if (e.key === ']') this.rotateBy(5);
  };

  Editor.prototype.paint = function () {
    if (!this.ready) return;
    var ctx = this.canvas.getContext('2d');
    var w = this.canvas.width, h = this.canvas.height;
    var r = this.canvas.getBoundingClientRect();
    var unit = w / (r.width || w);
    var self = this;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(this.image, 0, 0, w, h);

    // 1. 已标注 OBB 框与角标
    this.boxes.forEach(function (box, index) {
      var p = box.points, selected = index === self.selected;
      var col = colorFor(box.class_id);
      ctx.strokeStyle = selected ? '#ffffff' : col;
      ctx.fillStyle = col + (selected ? '33' : '22');
      ctx.lineWidth = (selected ? 2.6 : 1.7) * unit;
      ctx.beginPath();
      p.forEach(function (v, i) { i ? ctx.lineTo(v[0] * w, v[1] * h) : ctx.moveTo(v[0] * w, v[1] * h); });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      var srcTag = box.source && box.source !== 'manual' ? (' · ' + box.source.toUpperCase()) : '';
      var confTag = typeof box.confidence === 'number' ? (' ' + Math.round(box.confidence * 100) + '%') : '';
      var label = (index + 1) + '. ' + (self.classes[box.class_id] || box.class_id) + srcTag + confTag;
      ctx.font = '600 ' + (12 * unit) + 'px sans-serif';
      var tw = ctx.measureText(label).width;
      var lx = clamp(p[0][0] * w, 0, Math.max(0, w - tw - 10 * unit));
      var ly = clamp(p[0][1] * h - 20 * unit, 0, Math.max(0, h - 20 * unit));
      ctx.fillStyle = selected ? 'rgba(15,23,42,.92)' : 'rgba(15,23,42,.78)';
      ctx.fillRect(lx, ly, tw + 10 * unit, 20 * unit);
      ctx.fillStyle = col;
      ctx.fillRect(lx, ly, 3 * unit, 20 * unit);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, lx + 6 * unit, ly + 14.5 * unit);

      if (selected) {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5 * unit;
        p.forEach(function (v) {
          ctx.fillRect(v[0] * w - 4.5 * unit, v[1] * h - 4.5 * unit, 9 * unit, 9 * unit);
          ctx.strokeRect(v[0] * w - 4.5 * unit, v[1] * h - 4.5 * unit, 9 * unit, 9 * unit);
        });
        var rh = self.rotationHandle(p), m = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
        ctx.strokeStyle = '#ffffff';
        ctx.beginPath(); ctx.moveTo(m[0] * w, m[1] * h); ctx.lineTo(rh[0] * w, rh[1] * h); ctx.stroke();
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(rh[0] * w, rh[1] * h, 5.5 * unit, 0, Math.PI * 2); ctx.fill();
        ctx.stroke();
      }
    });

    // 2. 3 点旋转框绘制中预览
    if (this.mode === 'obb3' && this.obb3Points.length) {
      var col3 = colorFor(this.classId);
      ctx.save();
      ctx.strokeStyle = col3;
      ctx.lineWidth = 2 * unit;
      ctx.setLineDash([6 * unit, 4 * unit]);
      if (this.obb3Points.length === 1 && this.cursor) {
        ctx.beginPath();
        ctx.moveTo(this.obb3Points[0][0] * w, this.obb3Points[0][1] * h);
        ctx.lineTo(this.cursor[0] * w, this.cursor[1] * h);
        ctx.stroke();
      } else if (this.obb3Points.length === 2 && this.cursor) {
        var previewQuad = fromThreePoints(this.obb3Points[0], this.obb3Points[1], this.cursor, this._aspect());
        if (previewQuad) {
          ctx.fillStyle = col3 + '28';
          ctx.beginPath();
          previewQuad.forEach(function (v, i) { i ? ctx.lineTo(v[0] * w, v[1] * h) : ctx.moveTo(v[0] * w, v[1] * h); });
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      this.obb3Points.forEach(function (pt) {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(pt[0] * w, pt[1] * h, 5 * unit, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      });
      ctx.restore();
    }

    // 3. Meta SAM 智能分割掩码 + 拟合 OBB 虚线预览 + 正负提示点 + 粗略框
    if (this.mode === 'smart') {
      ctx.save();
      if (this.smartPreview) {
        var poly = this.smartPreview.polygon || [];
        if (poly.length >= 3) {
          ctx.fillStyle = 'rgba(20, 184, 166, 0.32)';
          ctx.strokeStyle = '#2dd4bf';
          ctx.lineWidth = 1.6 * unit;
          ctx.beginPath();
          poly.forEach(function (v, i) { i ? ctx.lineTo(v[0] * w, v[1] * h) : ctx.moveTo(v[0] * w, v[1] * h); });
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
        if (valid(this.smartPreview.points)) {
          ctx.strokeStyle = '#facc15';
          ctx.lineWidth = 2.6 * unit;
          ctx.setLineDash([8 * unit, 5 * unit]);
          ctx.beginPath();
          this.smartPreview.points.forEach(function (v, i) { i ? ctx.lineTo(v[0] * w, v[1] * h) : ctx.moveTo(v[0] * w, v[1] * h); });
          ctx.closePath();
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      var activeBbox = (this.drag && this.drag.kind === 'smart')
        ? [Math.min(this.drag.start[0], this.drag.current[0]), Math.min(this.drag.start[1], this.drag.current[1]),
           Math.max(this.drag.start[0], this.drag.current[0]), Math.max(this.drag.start[1], this.drag.current[1])]
        : this.smartBbox;
      if (activeBbox) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2 * unit;
        ctx.setLineDash([5 * unit, 4 * unit]);
        ctx.strokeRect(activeBbox[0] * w, activeBbox[1] * h, (activeBbox[2] - activeBbox[0]) * w, (activeBbox[3] - activeBbox[1]) * h);
        ctx.setLineDash([]);
      }
      this.smartPoints.forEach(function (pt, idx) {
        var pos = self.smartLabels[idx] !== 0;
        ctx.fillStyle = pos ? '#22c55e' : '#ef4444';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2 * unit;
        ctx.beginPath();
        ctx.arc(pt[0] * w, pt[1] * h, 6 * unit, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
      ctx.restore();
    }

    // 4. 视觉示例框拖拽预览
    if (this.drag && this.drag.kind === 'exemplar') {
      var ex = [
        Math.min(this.drag.start[0], this.drag.current[0]),
        Math.min(this.drag.start[1], this.drag.current[1]),
        Math.max(this.drag.start[0], this.drag.current[0]),
        Math.max(this.drag.start[1], this.drag.current[1])
      ];
      ctx.save();
      ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5 * unit;
      ctx.setLineDash([6 * unit, 4 * unit]);
      ctx.fillRect(ex[0] * w, ex[1] * h, (ex[2] - ex[0]) * w, (ex[3] - ex[1]) * h);
      ctx.strokeRect(ex[0] * w, ex[1] * h, (ex[2] - ex[0]) * w, (ex[3] - ex[1]) * h);
      ctx.restore();
    }

    // 5. 全画布十字辅助准星
    if (this.cursor && (this.mode === 'draw' || this.mode === 'hbb' || this.mode === 'obb3' || this.mode === 'smart' || this.mode === 'exemplar')) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.58)';
      ctx.lineWidth = 1 * unit;
      ctx.setLineDash([4 * unit, 4 * unit]);
      ctx.beginPath();
      ctx.moveTo(this.cursor[0] * w, 0); ctx.lineTo(this.cursor[0] * w, h);
      ctx.moveTo(0, this.cursor[1] * h); ctx.lineTo(w, this.cursor[1] * h);
      ctx.stroke();
      ctx.restore();
    }
  };

  var exported = { geometry: geometry, Editor: Editor };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  else { global.OD = global.OD || {}; global.OD.obb = exported; }
})(typeof window !== 'undefined' ? window : globalThis);
