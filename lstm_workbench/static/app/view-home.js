/* 工程列表。离线时只展示唯一那份示例工程；在线时列出训练台里全部工程，点一张卡就切过去。 */
(function (global) {
  'use strict';

  var D = global.OD.dom, el = D.el, icon = D.icon;
  var S = global.OD.shell;

  function render() {
    var live = global.OD.live.isLive();
    var names = global.OD.live.projects();
    var cards;

    if (live && names.length) {
      var current = global.OD.live.current();
      cards = names.map(function (name) {
        return name === current ? projectCard(global.OD.data, name) : otherProjectCard(name);
      });
    } else {
      cards = [projectCard(global.OD.data, global.OD.data.project.id)];
    }
    cards.push(newProjectCard());

    return el('div', { class: 'view is-on' }, el('div', { class: 'scroll' }, [
      el('div', { class: 'page-head' }, [
        el('h1', { text: '全部工程' }),
        el('p', { text: '一个工程就是一条产线的训练批次：片子、标注、特征缓存和训练轮次都收在它自己的目录里。' })
      ]),
      offlineCallout(),
      el('div', { class: 'grid c3' }, cards)
    ]));
  }

  /* 服务没起来时说清楚现在看到的是什么、要点哪里才能连上，别让人以为数据丢了。 */
  function offlineCallout() {
    if (global.OD.live.isLive()) return null;
    var err = global.OD.live.error();
    return el('div', { class: 'callout warn sp-mb-4' }, [
      icon('info', 16),
      el('div', null, [
        el('strong', { text: '现在看到的是离线示例。 ' }),
        err
          ? '没能连上训练台（' + err + '）。'
          : '这个页面是双击打开的，浏览器不允许它访问本机接口。',
        el('div', { class: 'sp-mt-2 mono sp-fs-12', text: 'python -m lstm_workbench.app' }),
        el('div', { class: 'sp-mt-1', text: '起好之后从它打开的地址（http://127.0.0.1:8771/next/）进这个页面，读写才会真的落到工程目录。' })
      ])
    ]);
  }

  function openProject(name) {
    if (name === global.OD.live.current()) { S.go('model'); return; }
    S.toast('正在读取工程 ' + name + '…');
    global.OD.live.loadProject(name).then(function () {
      S.toast('已切到 ' + name, 'ok');
      S.render();
    }).catch(function (err) {
      S.toast('打不开 ' + name + '：' + err.message, 'bad');
    });
  }

  function statRows(data) {
    var clips = data.clips;
    var segments = clips.reduce(function (n, c) { return n + c.segments.length; }, 0);
    return [
      el('dt', { text: '片子' }), el('dd', { text: clips.length + ' 条' }),
      el('dt', { text: '类别' }), el('dd', { text: data.classes.length + ' 个' }),
      el('dt', { text: '标注段' }), el('dd', { text: segments + ' 段' }),
      el('dt', { text: '最近训练' }), el('dd', { text: data.runs.length ? data.runs[0].at : '还没有' })
    ];
  }

  function projectCard(data, name) {
    var issues = S.allGateItems();
    /* 整张卡就是一个控件：不再另外放一个"打开"按钮，两处入口就是两套状态。 */
    function open() { S.go('model'); }
    return el('article', {
      class: 'card is-click', role: 'button', tabindex: '0',
      onclick: open,
      onkeydown: function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      }
    }, [
      el('header', null, [
        el('h2', { text: name }),
        el('span', { class: 'end' }, [
          global.OD.live.isLive() ? el('span', { class: 'badge ok', text: '当前' }) : S.sampleTag('示例工程名')
        ])
      ]),
      el('dl', { class: 'keyval' }, statRows(data)),
      el('div', { class: 'row sp-mt-4' }, [
        el('span', { class: 'badge ' + (issues.some(function (i) { return i.kind === 'bad'; }) ? 'bad' : 'plain') },
          issues.length ? issues.length + ' 条门禁待处理' : '门禁没有待处理项'),
        el('span', { class: 'spread' }),
        el('span', { class: 'btn btn-ghost btn-sm', text: '打开' })
      ])
    ]);
  }

  /* 没打开过的工程不猜它的片子数：那要再拉一次接口，这里只给名字和一个切换入口。 */
  function otherProjectCard(name) {
    function open() { openProject(name); }
    return el('article', {
      class: 'card is-click', role: 'button', tabindex: '0',
      onclick: open,
      onkeydown: function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      }
    }, [
      el('header', null, [
        el('h2', { text: name }),
        el('span', { class: 'end' }, el('span', { class: 'badge plain', text: '未打开' }))
      ]),
      el('p', { class: 'help', text: '点一下切到这个工程，界面按它的 settings、标注与轮次重画。' }),
      el('div', { class: 'row sp-mt-4' }, [
        el('span', { class: 'spread' }),
        el('span', { class: 'btn btn-ghost btn-sm', text: '切过去' })
      ])
    ]);
  }

  function newProjectCard() {
    var api = global.OD.api;
    var nameInput = el('input', { class: 'form-control', placeholder: '例如 包装线-02', 'aria-label': '工程名' });
    var hint = el('div', { class: 'help' });
    var button = el('button', {
      class: 'btn btn-primary', type: 'button',
      onclick: function () {
        var name = nameInput.value.trim();
        hint.className = 'help';
        if (!name) { hint.className = 'help bad'; hint.textContent = '工程名不能为空。'; return; }
        /* 名字同时是目录名，服务端也会拦，这里先拦一次省一个来回。 */
        if (/[\\/]/.test(name) || name === '.' || name === '..') {
          hint.className = 'help bad';
          hint.textContent = '工程名里不能有 / \\ 或单独的 . 与 ..，它同时是目录名。';
          return;
        }
        if (!api.available) {
          hint.className = 'help bad';
          hint.textContent = '离线原型没有后端，建不了工程。起训练台之后才能真的建目录。';
          return;
        }
        button.disabled = true;
        api.createProject(name).then(function () {
          hint.className = 'help';
          nameInput.value = '';
          /* 先重新拉一遍工程名单，新的那张卡才会出现在这一页上；
             新工程是空的，切过去正好从"模型与工位"开始走一遍流程。 */
          return global.OD.live.refreshProjects().then(function () {
            S.toast('工程 ' + name + ' 已建立', 'ok');
            return openProject(name);
          });
        }).catch(function (err) {
          hint.className = 'help bad';
          hint.textContent = err.message;
        }).then(function () { button.disabled = false; });
      }
    }, '新建工程');

    return el('article', { class: 'card' }, [
      el('header', null, el('h2', { text: '新建工程' })),
      el('div', { class: 'field' }, [
        el('label', { class: 'form-label', text: '工程名' }),
        nameInput,
        hint
      ]),
      el('div', { class: 'row sp-mt-4' }, button)
    ]);
  }

  function inspector() {
    return el('div', null, [
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '这个页面连着谁' }),
        el('p', null,
          global.OD.live.isLive()
            ? '数据来自本机训练台（127.0.0.1:8771），页面上的写操作会真的落到工程目录：上传素材、保存标注、选模型记指纹、抽特征、训练、停止。'
            : '离线示例：没有后端，所有写操作都不会落盘，训练与结果页里的数字是随原型附带的示例。')
      ]),
      el('div', { class: 'inspect-block' }, [
        el('h3', { text: '建议路线' }),
        el('p', null,
          '模型 → 数据与标注 → 特征 → 训练 → 结果。左侧每一步的状态点会告诉你哪一步还没准备好。')
      ])
    ]);
  }

  global.OD.views = global.OD.views || {};
  global.OD.views.home = { render: render, inspector: inspector };
})(window);
