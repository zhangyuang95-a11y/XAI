/* Independent controlled-network race and keyboard checks. No browser automation. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const elements=Object.fromEntries(['englishButton','chineseButton','currentButton','askButton','questionInput'].map(id=>[id,{id,disabled:false,value:'',focus(){},classList:{toggle(){}}}]));
const actionButton={dataset:{action:'wait'},disabled:false};
const exampleButton={dataset:{example:'help'},disabled:false};
const handlers={};const calls=[];const errors=[];
const context=vm.createContext({console,URLSearchParams,performance,crypto:require('node:crypto').webcrypto,setTimeout,clearTimeout,setInterval(){},
 localStorage:{getItem(){return null},setItem(){}},sessionStorage:{getItem(){return null},setItem(){},removeItem(){}},
 location:{pathname:'/pong/',search:''},window:{addEventListener(name,fn){handlers['window:'+name]=fn}},
 document:{hidden:false,activeElement:null,getElementById(id){return elements[id]||null},addEventListener(name,fn){handlers[name]=fn},querySelector(){return null},querySelectorAll(selector){return selector==='[data-action]'?[actionButton]:selector==='[data-example]'?[exampleButton]:[]}},
 calls,errors,actionButton,exampleButton,elements});
let source=fs.readFileSync(path.join(__dirname,'../study_v3/web/app.js'),'utf8');
source=source.slice(0,source.lastIndexOf('(async()=>{'));
vm.runInContext(source,context);
const run=code=>vm.runInContext(code,context);
run(`render=async()=>{};report=e=>errors.push(e);api=(path,payload)=>new Promise((resolve,reject)=>calls.push({path,payload,resolve,reject}));
 view={instance_id:'instance',revision:1,stage:'task2',run_id:'run2',can_ask:true,state:{turn:10,terminal:false},actions:['left','right','wait'],task_runs:[{id:'run2',turn:10}],questions:[]};`);
// Presets describe an editable, concrete hypothetical lane; they never move.
assert.match(run(`exampleQuestion('why')`), /this action/);
assert.equal(run(`hypotheticalLane=8;exampleQuestion('position')`),'If I were at lane 8 now, how would you move?');
assert.equal(run(`lang='zh';exampleQuestion('position')`),'如果我现在在第 8 道，你会怎么移动？');
run(`lang='en';view.questionnaire={items:[],comprehension:[{id:'removed',text:'old quiz',options:['one']}]};`);
assert(!run('surveyPage()').includes('old quiz'));
assert(!run('surveyPage()').includes('check_removed'));
const deferredFrame=turn=>({state:{turn,max_turns:90},can_ask:true});
(async()=>{
 // A late historical response must not reopen replay after returning to now.
 const first=run(`loadFrame('run2',2)`);
 assert.equal(run('replayLoading'),true);assert.equal(run('canMove()'),false);
 assert.equal(actionButton.disabled,true);assert.equal(elements.askButton.disabled,true);
 run('returnToCurrentFrame()');
 calls[0].resolve({...deferredFrame(2),can_ask:false});await first;
 assert.equal(run('replay'),null);assert.equal(run('view.can_ask'),true);
 // Select two frames quickly: network order must not decide the displayed frame.
 const older=run(`loadFrame('run2',3)`),newer=run(`loadFrame('run2',7)`);
 calls[2].resolve(deferredFrame(7));await newer;
 calls[1].resolve(deferredFrame(3));await older;
 assert.equal(run('replay.state.turn'),7);
 // Cross-window advancement invalidates an outstanding replay snapshot.
 run('returnToCurrentFrame()');
 const stale=run(`loadFrame('run2',4)`);run('view.revision++');
 calls[3].resolve(deferredFrame(4));await stale;
 assert.equal(run('replay'),null);assert.equal(run('replayLoading'),false);
 assert.equal(run('canMove()'),true);
 // An answer in progress freezes its chosen frame; examples preserve its draft.
 run(`bind();asking=true;draftQuestion='original question';`);
 const count=calls.length;await run(`loadFrame('run2',5)`);exampleButton.onclick();
 assert.equal(calls.length,count);assert.equal(run('draftQuestion'),'original question');
 run('asking=false');
 // Direction changes and OS repeats during an unacknowledged action never queue it.
 const key=(key,repeat=false)=>({key,repeat,preventDefault(){}});
 handlers.keydown(key('a'));handlers.keydown(key('a',true));handlers.keydown(key('d'));
 assert.equal(calls.length,count+1);assert.equal(calls.at(-1).payload.action,'left');
 handlers.keyup(key('d'));
 const next=JSON.parse(run('JSON.stringify(view)'));next.revision++;next.state.turn++;
 calls.at(-1).resolve(next);await new Promise(resolve=>setTimeout(resolve,10));
 assert.equal(calls.length,count+1);assert.equal(run('heldDirection'),null);assert.equal(run('busy'),false);
 // Space advances once per keydown; auto-repeat cannot create another action.
 handlers.keydown(key(' '));handlers.keydown(key(' ',true));
 assert.equal(calls.length,count+2);assert.equal(calls.at(-1).payload.action,'wait');
 next.revision++;next.state.turn++;calls.at(-1).resolve(next);await new Promise(resolve=>setTimeout(resolve,10));
 assert.equal(calls.length,count+2);assert.equal(errors.length,0);
 console.log('Frontend independent review: replay order/cancellation, frame locking, draft preservation and keyboard queue checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
