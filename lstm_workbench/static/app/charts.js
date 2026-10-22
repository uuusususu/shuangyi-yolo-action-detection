/* ECharts 主题与图表工厂。六张图全部走 ECharts，不再手绘 canvas。
   配色在加载时从 tokens.css 读出来 —— 图表和界面共用同一份取值，不另立一套。
   ECharts 没加载上时不留空白框，直接在图位上写清楚缺什么。 */
(function (global) {
  'use strict';

  /* 令牌在 CSS 里，按名字取。取不到就退回一份同值的兜底，
     保证样式表没加载上时图还是画得出来。 */
  function tok(name, fallback) {
    try {
      var v = global.getComputedStyle(document.documentElement).getPropertyValue(name);
      return (v && v.trim()) || fallback;
    } catch (e) { return fallback; }
  }

  var C = {
    ink: tok('--ink', '#e9ecef'),
    ink2: tok('--ink-2', '#c2c9d1'),
    muted: tok('--muted', '#8b959f'),
    line: tok('--line-2', '#3a424b'),
    grid: tok('--surface-3', '#252b31'),
    ok: tok('--ok', '#34c08b'),
    warn: tok('--warn', '#e0a33a'),
    bad: tok('--bad', '#e0574b'),
    busy: tok('--busy', '#38bdf8'),
    accent: tok('--accent', '#2563eb'),
    surface: tok('--surface-3', '#252b31'),
    card: tok('--surface', '#16191d'),
    onCat: tok('--on-cat', '#0b0d0f'),
    heat0: tok('--heat-0', '#151a1f'),
    heat1: tok('--heat-1', '#1f5c46'),
    cats: [
      tok('--cat-1', '#5b9cf8'), tok('--cat-2', '#c084fc'), tok('--cat-3', '#f0a04b'),
      tok('--cat-4', '#4fd1c5'), tok('--cat-5', '#f472b6'), tok('--cat-6', '#a3e635')
    ]
  };

  var FS = 11;
  var FS_SMALL = 10;

  var instances = [];
  var resizeBound = false;

  /* 系统设了"减弱动效"，图也别再逐帧长出来。 */
  function reduced() {
    try {
      return !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) { return false; }
  }

  /* 图位被压到 360px 以下时收轴、缩字号，不然刻度和格内数字会叠在一起。 */
  function narrow(node) { return !!(node && node.clientWidth && node.clientWidth < 360); }

  var TEXT = { fontFamily: '"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif', color: C.muted };

  /* 上边留 30px 给门禁那条线的文字，左右随容器收。 */
  function grid(small, extra) {
    return merge({ left: small ? 34 : 46, right: small ? 8 : 14, top: 30, bottom: small ? 32 : 36 }, extra || {});
  }

  function tooltipBase() {
    return {
      backgroundColor: C.surface, borderColor: C.line, borderWidth: 1,
      textStyle: { color: C.ink, fontSize: 12 },
      axisPointer: { lineStyle: { color: C.line } }
    };
  }

  function xBase(small) {
    return {
      type: 'category', boundaryGap: false,
      axisLine: { lineStyle: { color: C.line } },
      axisTick: { show: false },
      /* hideOverlap 让 ECharts 自己抽稀刻度，窄容器下不会糊成一条。 */
      axisLabel: { color: C.muted, fontSize: small ? FS_SMALL : FS, hideOverlap: true }
    };
  }

  function yBase(small) {
    return {
      type: 'value',
      splitLine: { lineStyle: { color: C.grid } },
      axisLabel: { color: C.muted, fontSize: small ? FS_SMALL : FS }
    };
  }

  function reset() {
    instances.forEach(function (c) { try { c.dispose(); } catch (e) { /* 已经随 DOM 一起没了 */ } });
    instances = [];
  }

  /* ECharts 缺席时不留空框：写清楚缺什么、放哪儿。 */
  function fallback(node) {
    var D = global.OD && global.OD.dom;
    if (!D || node.firstChild) return;
    node.appendChild(D.el('div', { class: 'chart-off' }, [
      D.icon('alert', 20),
      D.el('strong', { text: '图表库没有加载上' }),
      D.el('span', { text: '把 vendor/echarts.min.js 放在 index.html 旁边，六张图就会显示。' })
    ]));
  }

  function mount(node, option) {
    if (!node) return null;
    if (!global.echarts) { fallback(node); return null; }
    var chart = global.echarts.init(node);
    chart.setOption(option);
    instances.push(chart);
    if (!resizeBound) {
      resizeBound = true;
      global.addEventListener('resize', function () {
        instances.forEach(function (c) { try { c.resize(); } catch (e) { /* 忽略 */ } });
      });
    }
    return chart;
  }

  function merge(base, extra) {
    var out = {};
    Object.keys(base).forEach(function (k) { out[k] = base[k]; });
    Object.keys(extra).forEach(function (k) { out[k] = extra[k]; });
    return out;
  }

  function fmtTip(value) { return (Math.round(value * 1000) / 1000).toFixed(3); }

  /* tooltip 里补一句离门禁还有多远：光看曲线看不出来。 */
  function thresholdNote(spec, value) {
    if (spec.threshold === undefined) return '';
    var pass = spec.side === 'max' ? value <= spec.threshold : value >= spec.threshold;
    return '<br/><span style="color:' + C.muted + '">' + (spec.thresholdLabel || '门禁 ' + spec.threshold) +
      ' · 这一轮' + (pass ? '达标' : '没达标') + '</span>';
  }

  /* 门禁线：标签贴在自己的底色上，压在网格线上也读得清。 */
  function markLine(spec) {
    if (spec.threshold === undefined) return undefined;
    return {
      silent: true, symbol: 'none',
      lineStyle: { color: C.ink, type: 'dashed', width: 1 },
      label: {
        formatter: spec.thresholdLabel || ('门禁 ' + spec.threshold),
        color: C.ink2, fontSize: FS_SMALL, position: 'insideEndTop',
        backgroundColor: C.card, padding: [1, 4], borderRadius: 2
      },
      data: [{ yAxis: spec.threshold }]
    };
  }

  /* 单条折线：用于 F1、召回、空闲误报。threshold 画一条门禁线。 */
  function line(node, spec) {
    var small = narrow(node);
    var color = spec.color || C.busy;
    var series = [{
      type: 'line', name: spec.name, data: spec.data,
      smooth: true, showSymbol: false,
      lineStyle: { width: 2, color: color },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{ offset: 0, color: hexA(color, .22) }, { offset: 1, color: hexA(color, 0) }]
        }
      },
      markLine: markLine(spec),
      markPoint: spec.best ? {
        symbol: 'pin', symbolSize: 26, itemStyle: { color: C.ok },
        label: { color: C.onCat, fontSize: FS_SMALL, formatter: String(spec.best) },
        data: [{ coord: [spec.best - 1, spec.data[spec.best - 1]], name: '最佳轮次' }]
      } : undefined
    }];
    return {
      animation: !reduced(),
      textStyle: TEXT,
      grid: grid(small),
      tooltip: merge(tooltipBase(), {
        valueFormatter: fmtTip,
        formatter: function (params) {
          var p = params[0];
          return '第 ' + p.axisValue + ' 轮<br/>' + p.marker + spec.name + ' ' + fmtTip(p.value) +
            thresholdNote(spec, p.value);
        }
      }),
      xAxis: merge(xBase(small), { data: spec.x, name: '轮次', nameTextStyle: { color: C.muted, fontSize: FS }, nameGap: 6, nameLocation: 'middle' }),
      yAxis: merge(yBase(small), { min: spec.min, max: spec.max }),
      series: series
    };
  }

  /* 损失曲线：训练与验证两条，同轴对比。 */
  function loss(node, spec) {
    var small = narrow(node);
    return {
      animation: !reduced(),
      textStyle: TEXT,
      grid: grid(small),
      legend: { data: ['训练损失', '验证损失'], textStyle: { color: C.muted, fontSize: FS }, top: 0, right: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 3 },
      tooltip: merge(tooltipBase(), {
        valueFormatter: fmtTip,
        formatter: function (params) {
          return '第 ' + params[0].axisValue + ' 轮<br/>' + params.map(function (p) {
            return p.marker + p.seriesName + ' ' + fmtTip(p.value);
          }).join('<br/>');
        }
      }),
      xAxis: merge(xBase(small), { data: spec.x, name: '轮次', nameTextStyle: { color: C.muted, fontSize: FS }, nameGap: 6, nameLocation: 'middle' }),
      yAxis: yBase(small),
      series: [
        { type: 'line', name: '训练损失', data: spec.train, smooth: true, showSymbol: false, lineStyle: { width: 2, color: C.busy } },
        { type: 'line', name: '验证损失', data: spec.val, smooth: true, showSymbol: false, lineStyle: { width: 2, color: C.warn, type: 'dashed' } }
      ]
    };
  }

  /* 各类精确率 / 召回 / F1 分组柱。 */
  function perClass(node, rows) {
    var small = narrow(node);
    var names = rows.map(function (r) { return r.name; });
    return {
      animation: !reduced(),
      textStyle: TEXT,
      grid: grid(small, { bottom: small ? 46 : 42 }),
      legend: { data: ['精确率', '召回', 'F1'], textStyle: { color: C.muted, fontSize: FS }, top: 0, right: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 8 },
      tooltip: merge(tooltipBase(), { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: fmtTip }),
      xAxis: merge(xBase(small), {
        type: 'category', boundaryGap: true, data: names,
        axisLabel: { color: C.muted, fontSize: small ? FS_SMALL : FS, hideOverlap: true, rotate: small ? 30 : 0 }
      }),
      yAxis: merge(yBase(small), { min: 0, max: 1 }),
      series: [
        bar('精确率', rows.map(function (r) { return r.precision; }), C.busy),
        bar('召回', rows.map(function (r) { return r.recall; }), C.ok),
        bar('F1', rows.map(function (r) { return r.f1; }), C.cats[1])
      ]
    };
  }

  function bar(name, data, color) {
    return { type: 'bar', name: name, data: data, barMaxWidth: 18, itemStyle: { color: color, borderRadius: [2, 2, 0, 0] } };
  }

  /* 混淆矩阵热力图：行是真实、列是预测，格子里直接写数量。
     数字带一圈底色描边，落在最亮的那一格上也读得清。 */
  function matrix(node, spec) {
    var small = narrow(node);
    var names = spec.names;
    var data = [];
    var max = 0;
    spec.values.forEach(function (row, y) {
      row.forEach(function (v, x) { data.push([x, y, v]); if (v > max) max = v; });
    });
    return {
      animation: !reduced(),
      textStyle: TEXT,
      grid: { left: small ? 56 : 76, right: 16, top: 10, bottom: small ? 56 : 46 },
      tooltip: merge(tooltipBase(), {
        trigger: 'item',
        formatter: function (p) {
          return '真实 ' + names[p.value[1]] + ' → 判成 ' + names[p.value[0]] + '<br/>' + p.value[2] + ' 个窗口';
        }
      }),
      xAxis: {
        type: 'category', data: names, splitArea: { show: false },
        axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: small ? FS_SMALL : FS, rotate: small ? 32 : 18, hideOverlap: true },
        name: '判成', nameLocation: 'middle', nameGap: small ? 40 : 30, nameTextStyle: { color: C.muted, fontSize: FS }
      },
      yAxis: {
        type: 'category', data: names, splitArea: { show: false },
        axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: small ? FS_SMALL : FS },
        name: '真实', nameLocation: 'middle', nameGap: small ? 44 : 60, nameTextStyle: { color: C.muted, fontSize: FS }
      },
      visualMap: {
        min: 0, max: max || 1, calculable: false, show: false,
        inRange: { color: [C.heat0, C.heat1, C.ok] }
      },
      series: [{
        type: 'heatmap', data: data,
        label: {
          show: true, color: C.ink, fontSize: small ? FS_SMALL : FS, fontFamily: 'Consolas, monospace',
          textBorderColor: C.card, textBorderWidth: 2
        },
        itemStyle: { borderColor: C.card, borderWidth: 2 },
        emphasis: { itemStyle: { borderColor: C.ink, borderWidth: 2 } }
      }]
    };
  }

  function hexA(hex, alpha) {
    if (!hex || hex.charAt(0) !== '#') return hex;
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + [(n >> 16) & 255, (n >> 8) & 255, n & 255].join(',') + ',' + alpha + ')';
  }

  global.OD.charts = {
    colors: C, reset: reset, mount: mount,
    line: line, loss: loss, perClass: perClass, matrix: matrix
  };
})(window);
