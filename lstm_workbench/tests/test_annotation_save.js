/* 运行：node lstm_workbench/tests/test_annotation_save.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/app/view-annotate.js'), 'utf8');
const section = source.slice(source.indexOf('function canSave()'), source.indexOf('/* ---------- 视图 ---------- */'));
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  const data = {classes: [{id:'pack', name:'包装'}], clips:[{id:'a', file:'a.mp4', frame_count:100, fps:10, width:10, height:10, annotation_complete:false, segments:[], excluded:[]}]};
  const calls = [], toasts = [];
  const OD = {data, api:{available:true, saveAnnotations(project, payload) {
    let resolve, reject;
    const promise = new Promise((yes,no) => {resolve=yes; reject=no;});
    calls.push({project,payload,resolve,reject}); return promise;
  }},live:{reload:async () => {throw new Error('不应重载含有新编辑的数据');}}};
  const S = {state:{projectId:'p',clipId:'a'},toast:(...v)=>toasts.push(v),render(){}};
  const context = vm.createContext({global:{OD},S,refs:{saveBtn:{},saveHint:{}},savedFor:data,savedJson:'',seenContent:{},completionConfirmations:{},saveInFlight:null,console});
  vm.runInContext(section + ';savedJson=JSON.stringify(payloadOf());', context);
  return {OD,S,context,calls,toasts,save:()=>vm.runInContext('saveNow()',context),dirty:()=>vm.runInContext('isDirty()',context)};
}
async function run() {
  {
    const f=fixture();f.OD.data.clips[0].segments.push({id:'s1',class_id:'pack',start:0,end:10});
    const first=f.save();f.OD.data.clips[0].segments.push({id:'s2',class_id:'pack',start:20,end:30});
    assert.equal(vm.runInContext('beforeLeave()',f.context),false,'保存过程中不得离开丢稿');
    f.save();assert.equal(f.calls.length,1,'重复保存不得并发写入');f.calls[0].resolve({});await first;await tick();
    assert.equal(f.OD.data.clips[0].segments.length,2,'保存期间新编辑必须保留');assert.equal(f.dirty(),true,'新编辑仍需保存');
    f.OD.live.reload=async()=>{};const second=f.save();assert.equal(f.calls.length,2);assert.equal(f.calls[1].payload.clips[0].segments.length,2);f.calls[1].resolve({});await second;
    assert.equal(f.dirty(),false);
    assert.equal(vm.runInContext('beforeLeave()',f.context),true);
  }
  {
    const f=fixture();f.OD.data.clips[0].annotation_complete=true;f.context.completionConfirmations.a=true;
    const request=f.save();f.calls[0].reject(new Error('磁盘写入失败'));await request;
    assert.equal(f.dirty(),true);assert.equal(f.context.completionConfirmations.a,true,'失败后明确确认动作必须保留');
    const next=f.save();assert.deepEqual(Array.from(f.calls[1].payload.completion_confirmations),['a']);f.calls[1].resolve({});await next;
  }
  {
    const f=fixture();f.OD.data.clips[0].segments.push({id:'s',class_id:'pack',start:0,end:10});const request=f.save();
    const baseline=f.context.savedJson;const q={classes:[],clips:[]};f.OD.data=q;f.S.state.projectId='q';f.calls[0].resolve({});await request;
    assert.equal(f.OD.data,q);assert.equal(f.context.savedJson,baseline,'旧工程回包不能改新工程保存基线');
  }
  {
    const f=fixture();f.OD.data.clips[0].annotation_complete=true;f.context.completionConfirmations.a=true;
    const request=f.save();f.OD.data.clips[0].segments.push({id:'after',class_id:'pack',start:1,end:3});
    f.OD.data.clips[0].annotation_complete=false;delete f.context.completionConfirmations.a;
    f.calls[0].resolve({});await request;
    assert.equal(f.OD.data.clips[0].annotation_complete,false,'旧保存成功不能重新确认之后编辑过的标注');
    assert.equal(f.dirty(),true);
  }
  {
    const live=fs.readFileSync(path.join(__dirname,'../static/app/live.js'),'utf8');
    const start=live.indexOf('var loadSerial = 0;'),end=live.indexOf('/* 建了工程',start);
    let resolveProject,applied=0,allow=true;
    const c=vm.createContext({api:{project:()=>new Promise(r=>{resolveProject=r;}),run:async()=>({})},arr:v=>Array.isArray(v)?v:[],RUN_DETAIL_LIMIT:5,projectId:'p',isLive:()=>true,apply:()=>applied++,global:{OD:{shell:{render(){}}}}});
    vm.runInContext(live.slice(start,end),c);c.guard=()=>allow;
    const pending=vm.runInContext('reload({shouldApply:guard})',c);allow=false;resolveProject({runs:[]});await pending;
    assert.equal(applied,0,'读取门禁期间的新编辑应阻止旧载荷覆盖');
    const next=vm.runInContext('reload()',c);resolveProject({runs:[]});await next;assert.equal(applied,1,'普通重读仍应生效');
  }
  {
    const src=fs.readFileSync(path.join(__dirname,'../static/app/view-data.js'),'utf8');
    const fn=src.slice(src.indexOf('function saveClasses('),src.indexOf('function classEditor('));
    const f=fixture();let resolveSave,reloaded=false,reloadOptions;
    f.OD.api.saveAnnotations=()=>new Promise(r=>{resolveSave=r;});
    f.OD.live.reload=async options=>{reloadOptions=options;if(!options||options.shouldApply())reloaded=true;};
    const c=vm.createContext({global:{OD:f.OD},S:f.S,classesSaveInFlight:null});vm.runInContext(fn,c);
    const request=vm.runInContext("saveClasses([{id:'pack',name:'新名称'}])",c);
    f.OD.data.clips[0].segments.push({id:'new',class_id:'pack',start:1,end:3});
    resolveSave({});await request;await tick();assert.equal(reloaded,false,'保存类别的旧回包不能重载新标注');
    assert.deepEqual(JSON.parse(JSON.stringify(f.OD.data.classes)),[{id:'pack',name:'新名称'}],'跳过重载时本地类别表也要换成已保存的新表，否则下次保存会把旧表写回');
    assert.ok(f.OD.data.clips[0].segments.some(s=>s.id==='new'),'保存期间新画的段要保留');
  }
  {
    const src=fs.readFileSync(path.join(__dirname,'../static/app/view-data.js'),'utf8');
    const fn=src.slice(src.indexOf('function saveClasses('),src.indexOf('function classEditor('));
    const f=fixture();let reloadOptions;
    f.OD.api.saveAnnotations=async()=>({});
    f.OD.live.reload=async options=>{reloadOptions=options;};
    const c=vm.createContext({global:{OD:f.OD},S:f.S,classesSaveInFlight:null});vm.runInContext(fn,c);
    await vm.runInContext("saveClasses([{id:'pack',name:'新名称'}])",c);await tick();
    assert.equal(typeof reloadOptions?.shouldApply,'function','期间没改过时照常按门禁整份重载');
  }
  console.log('LSTM 保存并发、失败重试、跨工程与重载守卫测试通过');
}
run().catch(error=>{console.error(error);process.exitCode=1;});
