'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const handlers={},callbacks=[],calls=[],nodes={englishButton:{},chineseButton:{},automaticExplanation:{remove(){this.removed=true;}}};
const context=vm.createContext({console,URLSearchParams,performance,setTimeout,clearTimeout,setInterval(){},
 localStorage:{getItem(){return null}},sessionStorage:{getItem(){return null}},
 location:{pathname:'/kitchen/',search:''},window:{addEventListener(){}},
 document:{hidden:false,getElementById(id){return nodes[id]||null},addEventListener(name,fn){handlers[name]=fn}},
 IntersectionObserver:class{constructor(cb){callbacks.push(cb)}observe(){}disconnect(){}},calls});
let source=fs.readFileSync('study_v3/web/app.js','utf8');source=source.slice(0,source.lastIndexOf('(async()=>{'));
vm.runInContext(source,context);const run=s=>vm.runInContext(s,context);
(async()=>{
 run(`api=async(path,payload)=>{calls.push({path,payload});return {ok:true}};
 view={instance_id:'i1',run_id:'r2',can_ask:true,stage:'task2',automatic_explanations:[{id:'e1',turn:5,body:'I need <egg>.',displayed:false}]};`);
 assert.match(run('automaticPanel()'),/I need &lt;egg&gt;/);
 assert.match(run('automaticPanel()'),/Turn 5/);
 run('replay={};');assert.equal(run('automaticPanel()'),'');run('replay=null');
 run('observeAutomatic()');callbacks.at(-1)([{isIntersecting:true,intersectionRatio:.4}]);assert.equal(calls.length,0);
 callbacks.at(-1)([{isIntersecting:true,intersectionRatio:.8}]);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(calls.length,1);assert.equal(calls[0].payload.explanation_id,'e1');
 run('observeAutomatic()');assert.equal(callbacks.length,1);
 run(`view.automatic_explanations=[{id:'e2',turn:10,body:'new',displayed:false}];observeAutomatic()`);
 run(`view.run_id='r3';view.can_ask=false`);callbacks.at(-1)([{isIntersecting:true,intersectionRatio:1}]);
 assert.equal(calls.length,1);assert.equal(run('automaticPanel()'),'');
 run('clearChat()');assert.equal(run('view.automatic_explanations.length'),0);assert(nodes.automaticExplanation.removed);
 run(`view.can_ask=true;view.automatic_explanations=[{id:'e3',turn:15,body:'Read this reason.',requires_confirmation:true,confirmed:false}];`);
 assert.equal(run('canMove()'),false);assert.match(run('automaticDialog()'),/Confirm and continue/);
 let prevented=false;handlers.keydown({key:' ',repeat:true,preventDefault(){prevented=true}});assert(prevented);
 console.log('Automatic explanation frontend: visibility threshold, idempotent exposure, replay and Task 3 isolation passed.');
})().catch(e=>{console.error(e);process.exit(1)});
