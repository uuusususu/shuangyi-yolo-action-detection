/* 运行：node lstm_workbench/tests/test_obb_editor.js */
'use strict';
const assert = require('node:assert/strict');
const G = require('../static/app/obb-editor.js').geometry;
const rect = G.rectangle([.2,.3],[.6,.7]);
assert.equal(G.valid(rect), true);
assert.equal(G.contains(rect,[.4,.5]), true);
assert.equal(G.contains(rect,[.9,.9]), false);
assert.equal(G.valid([[0,0],[1,1],[0,1],[1,0]]), false);
assert.equal(G.valid([[0,0],[0,0],[0,0],[0,0]]), false);
assert.equal(G.valid([[0,0],[1,0],[1,Infinity],[0,1]]), false);
const moved = G.translate(rect,.8,.8);
assert.ok(moved.every(p=>p.every(n=>n>=0&&n<=1)));
assert.ok(Math.abs((moved[1][0]-moved[0][0])-.4)<1e-9);
const rotated = G.rotate(rect,Math.PI/2,2);
assert.equal(G.valid(rotated),true);
assert.ok(Math.abs(rotated[1][1]-rotated[0][1]-.8)<1e-9,'非正方形图片旋转使用像素比例');
const scaled = G.scale(rect, .5);
assert.ok(Math.abs((scaled[1][0]-scaled[0][0])-.2)<1e-9);
assert.ok(Math.abs(G.center(rect)[0]-.4)<1e-9);
assert.ok(Math.abs(G.center(rect)[1]-.5)<1e-9);
/* 3 点旋转框几何：斜边 + 非正方形比例 + 贴边自动安全回缩 + 共线兜底 */
const obb3 = G.fromThreePoints([0.2, 0.3], [0.6, 0.4], [0.55, 0.65], 1.5);
assert.ok(obb3 && G.valid(obb3), '3 点旋转框应生成合法凸四边形');
const obb3Edge = G.fromThreePoints([0.01, 0.05], [0.15, 0.01], [0.02, 0.3], 1.6);
assert.ok(obb3Edge && G.valid(obb3Edge), '贴边 3 点旋转框应自动约束在 [0,1] 内');
assert.equal(G.fromThreePoints([0.2, 0.2], [0.5, 0.5], [0.8, 0.8], 1), null, '三点共线应返回 null');
console.log('OBB 几何测试通过');

/* 旧草稿不能冒用磁盘的新修订；HTTP 失败必须保留草稿。 */
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
async function staleDraftCannotOverwrite() {
  const original = {id:'i',name:'一号图',source:'a.png',source_group:'batch-a',width:100,height:100,revision:1,status:'pending',boxes:[]};
  const stale = {revision:0,status:'pending',source_group:'batch-a',boxes:[{class_id:0,points:rect}]};
  const saved = new Map([['parts-draft:p:i',JSON.stringify(stale)]]);
  const payloads=[];
  const localStorage={getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v),removeItem:k=>saved.delete(k)};
  const fakeNode=()=>({children:[],isConnected:false,appendChild(v){this.children.push(v);},setAttribute(){}});
  const api={
    parts:async()=>({revision:1,classes:['零件'],items:[structuredClone(original)],versions:[]}),
    yolo:async()=>({runs:[]}),models:async()=>({yolo:{files:[]}}),
    yoloValidations:async()=>({evaluations:[]}),
    partsSave:async(pid,id,payload)=>{payloads.push(payload);throw new Error('素材修订冲突');}
  };
  const window={OD:{dom:{el:fakeNode,clear(){}},shell:{state:{projectId:'p',view:'pdata'},render(){},toast(){}},api,live:{isLive:()=>true},views:{}},addEventListener(){}};
  const context=vm.createContext({window,localStorage,setTimeout,clearTimeout,console});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app/view-parts.js'),'utf8'),context);
  window.OD.parts.renderData();
  await new Promise(resolve=>setImmediate(resolve));
  window.OD.parts.renderData();
  assert.equal(window.OD.parts.beforeNavigate(),false);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(payloads.length,1);
  assert.equal(payloads[0].revision,0,'必须使用草稿原始修订，不能用磁盘的新修订');
  assert.ok(saved.has('parts-draft:p:i'),'失败保留草稿');
  assert.deepEqual(original.boxes,[],'不会修改磁盘真值');
  console.log('OBB 旧草稿冲突保护测试通过');
}
staleDraftCannotOverwrite().catch(error=>{console.error(error);process.exitCode=1;});
