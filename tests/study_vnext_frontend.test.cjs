const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm'),fs=require('node:fs');
const Queue=require('../ui/model_preview/input_queue.js');
const clone=x=>JSON.parse(JSON.stringify(x));
const tick=()=>new Promise(resolve=>setImmediate(resolve));

async function harness(t,{initialSession=null,initialView=null,failAnswers=false}={}){
  const elements=new Map(),handlers={},calls=[],timers=new Set(),presentations=[];let document;
  const element=(id,tagName='BUTTON',dataset={})=>({id,tagName,dataset,children:[],hidden:false,disabled:false,
    value:id==='domain'?'warehouse':'',checked:false,style:{},listeners:{},textContent:'',
    addEventListener(type,fn){(this.listeners[type]??=[]).push(fn);},
    async fire(type,event={}){for(const fn of this.listeners[type]||[])await fn({target:this,preventDefault(){},...event});},
    append(...nodes){this.children.push(...nodes);},replaceChildren(...nodes){this.children=[...nodes];},
    get childElementCount(){return this.children.length;},focus(){document.activeElement=this;}});
  const $=id=>{if(!elements.has(id))elements.set(id,element(id));return elements.get(id);};
  const buttons=['up','down','left','right','wait','interact'].map(action=>element('action-'+action,'BUTTON',{action}));
  document={hidden:false,activeElement:{tagName:'BODY'},getElementById:$,querySelectorAll:()=>buttons,
    createElement:tag=>element('created',tag.toUpperCase()),addEventListener(type,fn){(handlers[type]??=[]).push(fn);}};
  let now=0;const stored=new Map();if(initialSession)stored.set('policylens-study-protocol-20261029.v1',JSON.stringify(initialSession));
  const status={protocol_id:'synthetic',consent_version:'synthetic',questionnaire_fields:['coordination','understanding','predictability','workload'],question_service_mode:'template_only'};
  const server={answers:{questions:[]},view:initialView,demo:{initial_public:{domain:'warehouse'},records:[{public_after:{domain:'warehouse'}}]},failDemo:false,failAnswers};
  const context=vm.createContext({document,window:{addEventListener(){},confirm(){return false;}},location:{pathname:'/warehouse/',assign(path){context.navigated=path;}},
    localStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v),removeItem:k=>stored.delete(k)},
    PreviewInputQueue:Queue,PreviewBoard:class{setState(state,options={}){presentations.push(state);options.onPresent?.();return Promise.resolve();}ready(){return Promise.resolve();}},
    setInterval(){},setTimeout(fn,ms){const timer=setTimeout(()=>{timers.delete(timer);fn();},ms);timers.add(timer);return timer;},
    clearTimeout(timer){clearTimeout(timer);timers.delete(timer);},performance:{now:()=>now},AbortController,
    crypto:require('node:crypto').webcrypto,console,fetch:async(path,options={})=>{
      const payload=options.body?JSON.parse(options.body):null;calls.push({path,payload});
      let result=path.endsWith('/status')?status:path.endsWith('/answers')?server.answers:path.endsWith('/demo')?server.demo:server.view;
      if(server.failDemo&&path.endsWith('/demo'))return {ok:false,status:503,json:async()=>({error:'synthetic demo failure'})};
      if(server.failAnswers&&path.endsWith('/answers'))return {ok:false,status:503,json:async()=>({error:'synthetic answer service unavailable'})};
      return {ok:true,status:200,json:async()=>clone(result)};
    }});
  vm.runInContext(fs.readFileSync('study_vnext/web/app.js','utf8'),context);
  const settle=async()=>{for(let i=0;i<5;i++)await tick();};await settle();
  t.after(()=>{for(const timer of timers)clearTimeout(timer);});
  return {$,calls,server,buttons,context,settle,presentations,
    assign(value){server.view=clone(value);context.assigned=value;vm.runInContext("session={id:'synthetic',token:'synthetic-token'};view=assigned;render();",context);},
    eval:code=>vm.runInContext(code,context),
    async key(code){for(const fn of handlers.keydown||[])fn({code,repeat:false,preventDefault(){}});await settle();},
    advance(ms){now+=ms;}};
}
function view(stage='enrolled',questions=false){return {instance:{stage,domain:'warehouse',binding_sha256:'synthetic-binding'},
  runs:[],current:null,study:{phase:stage,preparation:{controls:false,demonstration:false}},explanation_questions_allowed:questions};}

test('demo cannot be acknowledged or generated before neutral controls practice',async t=>{
  const h=await harness(t);h.assign(view());
  assert(h.$('demo').disabled);assert(h.$('demo-finish').hidden);assert(h.$('practice-finish').disabled);
  for(const code of ['KeyW','KeyA','KeyS','KeyD','Space'])await h.key(code);
  assert.equal(h.$('practice-finish').disabled,false);
  const prepared=view();prepared.study.preparation.controls=true;h.assign(prepared);
  assert.equal(h.$('demo').disabled,false);assert(h.$('demo-finish').disabled);
});

test('failed demo does not register playback or permit acknowledgement',async t=>{
  const h=await harness(t),v=view();v.study.preparation.controls=true;h.assign(v);h.server.failDemo=true;
  await h.$('demo').fire('click');
  assert(!h.calls.some(call=>call.path.endsWith('/preparation')));assert(h.$('demo-finish').disabled);
});

test('saved Task2 answers restore, explicit reading blocks actions, Task3 removes answers',async t=>{
  const h=await harness(t),v=view('task_2',true);v.current={public:{domain:'warehouse',task:2,terminal:false,turn:1,max_turns:2,score:{task_score:0}}};h.assign(v);
  h.server.answers={questions:[{target_task:1,result_turn:1,answer:'saved committed answer'}]};
  await h.eval('restoreAnswers()');assert.equal(h.$('answers').childElementCount,1);
  assert.equal(h.$('read-answers').hidden,false);await h.$('read-answers').fire('click');
  assert(h.buttons.every(b=>b.disabled));assert.equal(h.eval('readingAnswers'),true);
  v.instance.stage='task_3';v.explanation_questions_allowed=false;h.assign(v);
  assert(h.$('qa').hidden);assert.equal(h.$('answers').childElementCount,0);assert.equal(h.eval('readingAnswers'),false);
});

test('question entry and explicit answer reading are separate timing proxies',async t=>{
  const h=await harness(t),v=view('task_2',true);v.current={public:{domain:'warehouse',task:2,terminal:false,turn:1,max_turns:2,score:{task_score:0}}};h.assign(v);
  h.$('question').focus();h.advance(1000);h.eval('sampleTime()');
  assert.equal(h.eval('timingSeconds.question_entry'),1);assert.equal(h.eval('timingSeconds.reading||0'),0);
  h.eval('readingAnswers=true');h.advance(1000);h.eval('sampleTime()');assert.equal(h.eval('timingSeconds.reading'),1);
});

test('stale replay slider cannot replace the board after returning to live',async t=>{
  const h=await harness(t),v=view('task_2',true);h.assign(v);
  h.eval("replayFrames=[{turn:0,public:{domain:'warehouse',task:1}}];replaying=true;");
  await h.$('live').fire('click');const before=h.presentations.length;
  await h.$('replay-turn').fire('input');assert.equal(h.presentations.length,before);
  assert.equal(h.eval('replayFrames.length'),0);assert(h.$('replay-turn').disabled);
});

test('retry cannot send a real action during replay, reading or another request',async t=>{
  const h=await harness(t);h.assign(view('task_2',true));
  h.eval('queue.retry=()=>{globalThis.syntheticRetries=(globalThis.syntheticRetries||0)+1;};');
  for(const mode of ['replaying','readingAnswers','busy']){
    h.eval(mode+'=true;locks()');await h.$('retry').fire('click');assert.equal(h.eval('globalThis.syntheticRetries||0'),0);
    h.eval(mode+'=false;');
  }
  await h.$('retry').fire('click');assert.equal(h.eval('globalThis.syntheticRetries'),1);
});

test('failed answer restoration retains a valid session and offers independent retry',async t=>{
  const h=await harness(t,{initialSession:{id:'synthetic',token:'synthetic-token'},initialView:view('task_2',true),failAnswers:true});
  assert.equal(h.eval('session.id'),'synthetic');assert(h.$('status').textContent.includes('当前会话仍保留'));
  h.server.failAnswers=false;await h.$('restore-answers').fire('click');assert.equal(h.eval('session.id'),'synthetic');
});

test('pending timing events retry even after terminal has cleared timingTask',async t=>{
  const h=await harness(t);h.assign(view('complete'));
  h.eval("timingTask=null;session.events=[{event_id:'saved-terminal-event',task:3,kind:'active',seconds:2}];");
  await h.eval('flushTime()');assert(h.calls.some(c=>c.path.endsWith('/event')&&c.payload.event_id==='saved-terminal-event'));
  assert.equal(h.eval('session.events.length'),0);
});

test('another researcher test preserves records and needs confirmation if unfinished',async t=>{
  const h=await harness(t);h.assign(view('task_2',true));
  assert.equal(h.$('new-test').hidden,false);await h.$('new-test').fire('click');
  assert.equal(h.eval('session.id'),'synthetic');
  const finished=view('complete');finished.study.phase='finished';h.assign(finished);
  assert.equal(h.$('new-test').hidden,false);await h.$('new-test').fire('click');
  assert.equal(h.eval('session'),null);assert.equal(h.context.navigated,'/study/');
  assert(!h.calls.some(call=>/delete|reset|enroll/.test(call.path)));
});

test('explicitly confirmed unfinished researcher test returns to entry without deleting history',async t=>{
  const h=await harness(t);h.assign(view('task_2',true));
  h.eval('window.confirm=()=>true;');await h.$('new-test').fire('click');
  assert.equal(h.eval('session'),null);assert.equal(h.context.navigated,'/study/');
  assert(!h.calls.some(call=>/delete|reset|enroll/.test(call.path)));
});

test('another test cannot discard an unconfirmed action',async t=>{
  const h=await harness(t),finished=view('complete');finished.study.phase='finished';h.assign(finished);
  h.eval("session.pending={command_id:'unconfirmed'};locks();");
  assert(h.$('new-test').disabled);await h.$('new-test').fire('click');
  assert.equal(h.eval('session.pending.command_id'),'unconfirmed');assert(!h.context.navigated);
});
