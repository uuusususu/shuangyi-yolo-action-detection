/* 极小的 DOM 与格式化工具。全站共用，避免每个视图各写一份。 */
(function (global) {
  'use strict';

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'html') node.innerHTML = value;
        else if (key.indexOf('on') === 0) node.addEventListener(key.slice(2), value);
        else if (key === 'dataset') Object.keys(value).forEach(function (d) { node.dataset[d] = value[d]; });
        else node.setAttribute(key, value === true ? '' : value);
      });
    }
    append(node, children);
    return node;
  }

  function append(node, children) {
    if (children === null || children === undefined || children === false) return;
    if (Array.isArray(children)) { children.forEach(function (c) { append(node, c); }); return; }
    node.appendChild(children instanceof Node ? children : document.createTextNode(String(children)));
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

  function qs(selector, root) { return (root || document).querySelector(selector); }
  function qsa(selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); }

  /* 图标：内联 sprite，不用图标字体（file:// 下 @font-face 会被 CORS 拦掉）。
     href 与 xlink:href 都写：老一点的 WebView 只认后者。
     宽高写死在元素上，CSS 没加载上也不会撑成一大块。 */
  var XLINK = 'http://www.w3.org/1999/xlink';

  function icon(name, size) {
    var px = String(size || 16);
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'ico');
    svg.setAttribute('width', px);
    svg.setAttribute('height', px);
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    var use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#i-' + name);
    use.setAttributeNS(XLINK, 'xlink:href', '#i-' + name);
    svg.appendChild(use);
    return svg;
  }

  /* ---- 格式化 ---- */

  function fmtNum(value, digits) {
    if (value === null || value === undefined || value === '' || isNaN(value)) return '—';
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function fmtInt(value) {
    if (value === null || value === undefined || value === '' || isNaN(value)) return '—';
    return String(Math.round(Number(value)));
  }

  function fmtTimecode(frame, fps) {
    var f = Math.max(0, Math.round(frame));
    if (!fps) return '帧 ' + f;
    var total = f / fps;
    var mm = Math.floor(total / 60);
    var ss = Math.floor(total % 60);
    var cs = Math.floor((total - Math.floor(total)) * 100);
    return pad(mm) + ':' + pad(ss) + '.' + pad(cs);
  }

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  function fmtDuration(frames, fps) {
    if (!fps) return fmtInt(frames) + ' 帧';
    var sec = frames / fps;
    return sec >= 60 ? (sec / 60).toFixed(1) + ' 分' : sec.toFixed(1) + ' 秒';
  }

  function shortHash(hash, head) {
    if (!hash) return '—';
    var n = head || 10;
    return hash.length > n * 2 ? hash.slice(0, n) + '…' + hash.slice(-4) : hash;
  }

  /* 大小未知就是未知。真实接口不返回文件大小，这里返回 0 B 会把"不知道"说成"空的"。 */
  function fmtBytes(bytes) {
    if (bytes === null || bytes === undefined || bytes === '' || isNaN(bytes)) return '—';
    if (!bytes) return '0 B';
    var units = ['B', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(1024));
    i = Math.min(i, units.length - 1);
    return (bytes / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + units[i];
  }

  /* 类别色按索引取，超出 6 个循环。 */
  function catColor(index) { return 'var(--cat-' + ((index % 6) + 1) + ')'; }

  global.OD = global.OD || {};
  global.OD.dom = {
    el: el, append: append, clear: clear, qs: qs, qsa: qsa, icon: icon,
    fmtNum: fmtNum, fmtInt: fmtInt, fmtTimecode: fmtTimecode, fmtDuration: fmtDuration,
    shortHash: shortHash, fmtBytes: fmtBytes, catColor: catColor
  };
})(window);
