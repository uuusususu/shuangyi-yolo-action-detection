/* 运行 node lstm_workbench/tests/test_parts_flow.js；真实页面模块的延迟响应与发布闭环。 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../static/app/view-parts.js'),'utf8');
const tick=()=>new Promise(r=>setImmediate(r));
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
function snapshot(group='a'){return {revision:1,classes:['零件'],items:[{id:'one',name:'one.png',source:'x',source_group:group,revision:0,status:'unlabeled',boxes:[],width:10,height:10}],versions:[]};}
function harness(api={}){
 const nodes=[];
 function el(tag,attrs,children){const n={tag,attrs:attrs||{},children:[],value:attrs&&attrs.value||'',isConnected:false,appendChild(c){this.children.push(c);},setAttribute(){}};n.text=(attrs&&attrs.text)||'';const add=c=>{if(Array.isArray(c))c.forEach(add);else if(c!==null&&c!==undefined)n.children.push(c);};add(children);nodes.push(n);return n;}
 const store=new Map(),base={parts:async()=>snapshot(),yolo:async()=>({runs:[]}),models:async()=>({yolo:{files:[]}}),yoloValidations:async()=>({evaluations:[]})};
 const window={OD:{api:Object.assign(base,api),dom:{el,clear(){}},live:{isLive:()=>true},views:{},shell:{state:{projectId:'p',view:'pdata'},render(){},renderJob(){},toast(){}}},addEventListener(){},confirm:()=>true};
 const context=vm.createContext({window,localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},setTimeout,clearTimeout,console});
 vm.runInContext(source.replace('global.OD.parts={','global.test={state:state,management:management,assignSplits:assignSplits,publishCard:publishCard};global.OD.parts={'),context);
 const button=text=>{const n=nodes.slice().reverse().find(n=>n.tag==='button'&&(n.children.includes(text)||n.text===text));assert.ok(n,text);return n;};
 return {window,nodes,button,context,test:window.test,async boot(){await window.OD.parts.prefetch();window.OD.parts.renderData();}};
}
async function testLateImportABA(){
 const late=deferred();const h=harness({partsImport:()=>late.promise});await h.boot();
 h.nodes.find(n=>n.attrs.id==='parts-source').value='source';h.button('导入素材').attrs.onclick();
 h.window.OD.shell.state.projectId='q';h.window.OD.parts.reset();await h.window.OD.parts.prefetch();
 h.window.OD.shell.state.projectId='p';h.window.OD.parts.reset();await h.window.OD.parts.prefetch();
 late.resolve({...snapshot(),classes:['过期类别']});await tick();await tick();
 assert.deepEqual(Array.from(h.test.state.data.classes),['零件'],'ABA 旧导入不得覆盖当前工程');
}
async function testRemovedGroupIsNotPublished(){
 let sent;const h=harness({partsPublish:async(_p,p)=>{sent=p;return {id:'v'};}});await h.boot();
 h.test.state.splits={a:'train',b:'val',deleted:'test'};h.test.state.data.items.push({...h.test.state.data.items[0],id:'two',source_group:'b'});
 h.test.assignSplits();h.test.publishCard();h.button('发布固定数据集版本').attrs.onclick();await tick();await tick();
 assert.deepEqual(Object.keys(sent.splits).sort(),['a','b']);assert.equal(sent.splits.b,'val');
}
async function testPublishRefreshesTrainingVersions(){
 let refreshed=0;const h=harness({partsPublish:async()=>({id:'v'})});h.window.OD.yolo={refresh:async()=>{refreshed++;}};
 await h.boot();h.button('发布固定数据集版本').attrs.onclick();await tick();await tick();
 assert.equal(refreshed,1,'发布必须刷新训练页的数据版本');
}
async function testLateManagementResponses(){
 for(const [method,label] of [['partsClasses','保存类别'],['partsQuality','检查数据质量']]){
   const late=deferred(),h=harness({[method]:()=>late.promise});await h.boot();
   h.button(label).attrs.onclick();await tick();
   h.window.OD.shell.state.projectId='q';h.window.OD.parts.reset();await h.window.OD.parts.prefetch();
   late.resolve(method==='partsQuality'?{ok:false,issues:[{message:'旧工程错误'}]}:{...snapshot(),classes:['旧工程类别']});await tick();await tick();
   assert.deepEqual(Array.from(h.test.state.data.classes),['零件'],label+' 不得污染新工程');
   assert.equal(h.test.state.quality,null);
 }
}
async function testLateLoadABA(){
 const late=deferred();let first=true;const h=harness({parts:()=>{if(first){first=false;return late.promise;}return Promise.resolve(snapshot('new'));}});
 const pending=h.window.OD.parts.prefetch();
 h.window.OD.shell.state.projectId='q';h.window.OD.parts.reset();await h.window.OD.parts.prefetch();
 h.window.OD.shell.state.projectId='p';h.window.OD.parts.reset();await h.window.OD.parts.prefetch();
 late.resolve(snapshot('old'));await pending;
 assert.equal(h.test.state.data.items[0].source_group,'new');
}
async function testTrainingCacheSeesPublishedVersion(){
 let published=false,requests=0;
 const h=harness({yolo:async()=>{requests++;return {dataset:{exists:false},runs:[],versions:published?[{id:'new-version',classes:['零件']}]:[]};},yoloEnv:async()=>({ok:true}),partsPublish:async()=>{published=true;return {id:'new-version'};}});
 h.window.OD.dom.icon=()=>null;
 h.window.OD.dom.fmtNum=String;
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app/view-yolo.js'),'utf8'),h.context);
 h.window.OD.yolo.prefetch();await tick();await tick();await h.boot();
 h.button('发布固定数据集版本').attrs.onclick();await tick();await tick();
 h.window.OD.views.ptrain.render();
 assert.ok(h.nodes.some(n=>n.tag==='option'&&n.attrs.value==='new-version'),'发布后训练选项应立即包含新版本');
 assert.equal(h.nodes.slice().reverse().find(n=>n.attrs.id==='y-start').attrs.disabled,false);
 const device=h.nodes.slice().reverse().find(n=>n.attrs.id==='y-device');
 assert.equal(device.attrs.value,'cpu','没有 GPU 时默认使用 CPU');
 device.attrs.oninput({target:{value:'0'}});
 await h.window.OD.yolo.refresh('p');h.window.OD.views.ptrain.render();
 assert.equal(h.nodes.slice().reverse().find(n=>n.attrs.id==='y-device').attrs.value,'0','刷新保留用户手填的设备');
 assert.ok(requests<8,'刷新不能进入请求循环');
}
async function testSmartAutoInitUnavailableOpensPanel(){
 let activatedWith=null;
 const h=harness({
   partsImageUrl:(_p,id)=>'/img/'+id,
   smartStatus:async()=>({state:'idle',training_locked:false,available_models:[]}),
   smartActivate:async(_p,body)=>{activatedWith=body;return {state:'unavailable',training_locked:false,show_model_panel:true,reason:'缺少 sam3.pt',available_models:[{name:'sam2.1_b.pt',downloadable:true,local:false}]};}
 });
 h.window.OD.obb={Editor:function(){this.zoom=1;this.selected=-1;this.setMode=function(){};this.setClass=function(){};this.paint=function(){};}};
 h.window.OD.shell.state.view='palabel';
 await h.boot();
 h.window.OD.views.palabel.render();
 await tick();await tick();await tick();
 assert.equal(activatedWith.model,'sam3.pt','进入标注页且无训练时应默认尝试拉起 sam3.pt');
 assert.equal(h.test.state.smartPanelOpen,true,'SAM 3 不可用时应自动展开可用标注模型面板');
}
async function testSmartTrainingLockedBlocksActivateAndUnloadsOnLeave(){
 let activateCount=0,unloadCount=0;
 const h=harness({
   partsImageUrl:(_p,id)=>'/img/'+id,
   smartStatus:async()=>({state:'blocked_by_training',training_locked:true,reason:'训练中禁用智能标注',available_models:[]}),
   smartActivate:async()=>{activateCount++;return {};},
   smartUnload:async()=>{unloadCount++;return {state:'idle'};}
 });
 h.window.OD.obb={Editor:function(){this.zoom=1;this.selected=-1;this.setMode=function(){};this.setClass=function(){};this.paint=function(){};}};
 h.window.OD.shell.state.view='palabel';
 await h.boot();
 h.window.OD.views.palabel.render();
 await tick();await tick();
 assert.equal(activateCount,0,'训练进行中绝不能尝试激活智能标注模型');
 h.test.state.smart={state:'ready',training_locked:false};
 assert.equal(h.window.OD.views.palabel.beforeLeave(),true);
 assert.equal(unloadCount,1,'离开标注页时应自动卸载释放显存');
}
async function testSmartManualSelectLoadAndStop(){
 let activated=null,unloaded=0;
 const models=[
   {name:'sam2.1_b.pt',label:'Meta SAM 2.1 Base',family:'sam2',kind:'sam_family',capabilities:['interactive','video'],local:true,bytes:1024,active:false},
   {name:'sam2.1_t.pt',label:'Meta SAM 2.1 Tiny',family:'sam2',kind:'sam_family',capabilities:['interactive','video'],local:true,bytes:512,active:true}
 ];
 const h=harness({
   partsImageUrl:(_p,id)=>'/img/'+id,
   smartStatus:async()=>({state:'ready',training_locked:false,active_model:'sam2.1_t.pt',active_slot:'interactive',sam_dir:'models/sam',available_models:models}),
   smartActivate:async(_p,body)=>{activated=body;return {state:'ready',training_locked:false,active_model:body.model,active_slot:'interactive',sam_dir:'models/sam',available_models:models.map(m=>({...m,active:m.name===body.model}))};},
   smartUnload:async()=>{unloaded++;return {state:'idle',training_locked:false,active_model:'',reason_code:'stopped',sam_dir:'models/sam',available_models:models.map(m=>({...m,active:false}))};}
 });
 h.window.OD.obb={Editor:function(){this.zoom=1;this.selected=-1;this.setMode=function(){};this.setClass=function(){};this.paint=function(){};}};
 h.window.OD.shell.state.view='palabel';
 await h.boot();
 h.window.OD.views.palabel.render();
 await tick();await tick();
 h.nodes.length=0;
 h.window.OD.views.palabel.render();
 const picker=h.nodes.find(n=>n.tag==='select'&&n.attrs['aria-label']==='选择 models/sam 文件夹中的模型');
 assert.ok(picker,'顶部应直接呈现本地文件夹模型选择器');
 picker.attrs.onchange({target:{value:'sam2.1_b.pt'}});
 h.nodes.length=0;
 h.window.OD.views.palabel.render();
 h.button('切换加载').attrs.onclick();
 await tick();await tick();
 assert.equal(activated.model,'sam2.1_b.pt','应按用户选择加载对应模型');
 h.nodes.length=0;
 h.window.OD.views.palabel.render();
 h.button('停止模型').attrs.onclick();
 await tick();await tick();
 assert.equal(unloaded,1,'点击停止模型应立即调用卸载释放显存');
 assert.equal(h.test.state.smart.state,'idle');
}
(async()=>{for(const test of [testLateImportABA,testLateManagementResponses,testLateLoadABA,testRemovedGroupIsNotPublished,testPublishRefreshesTrainingVersions,testTrainingCacheSeesPublishedVersion,testSmartAutoInitUnavailableOpensPanel,testSmartTrainingLockedBlocksActivateAndUnloadsOnLeave,testSmartManualSelectLoadAndStop]){await test();console.log(test.name+' passed');}})().catch(e=>{console.error(e);process.exitCode=1;});
