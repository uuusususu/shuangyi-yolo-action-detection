/* 入口：DOM 就绪后挂载外壳。所有视图通过 OD.views 注册，这里不关心具体有哪些。
   挂载交给 live.boot()：页面由训练台托管时它先取真实工程再挂，双击打开时直接挂示例数据。
   两条路都会把外壳画出来，boot() 自己吞掉失败，所以这里不需要兜底。 */
(function (global) {
  'use strict';

  function boot() {
    if (!global.OD || !global.OD.live) return;
    global.OD.live.boot();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})(window);
