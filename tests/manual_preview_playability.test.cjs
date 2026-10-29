const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm'),fs=require('node:fs'),{performance}=require('node:perf_hooks');
const Queue=require('../ui/model_preview/input_queue.js');
const Sequence=require('../ui/model_preview/task_sequence.js');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const clone=value=>JSON.parse(JSON.stringify(value));

function backend({maxTurns=2,pongX=1,kitchenInteraction=false}={}){
  const sessions=new Map(),starts=[],actions=[],pending=[];let id=0;
  const server={sessions,starts,actions,pending,deferActions:false,failAction:null};
  const response=(value,status=200)=>({ok:status<400,status,json:async()=>clone(value)});
  const publicView=session=>({state:clone(session.state),binding:clone(session.binding),last_action:session.last||null});
  server.fetch=async(path,options={})=>{
    const token=options.headers?.Authorization?.slice(7),payload=options.body&&JSON.parse(options.body);
    if(path.endsWith('models'))return response({models:Object.fromEntries(['warehouse','pong','kitchen'].map(domain=>[domain,{version:'test-frozen',seeds:[domain==='warehouse'?40786:domain==='pong'?880101:1000]}]))});
    if(path.endsWith('start')){
      starts.push(clone(payload));const token=`preview-session-${++id}`;
      const binding={domain:payload.domain,variant:payload.variant,actor_sha256:`frozen-${payload.domain}-${payload.variant}`,programs:[]};
      const state={domain:payload.domain,task:payload.task,turn:0,max_turns:maxTurns,terminal:false,
        binding_identity_sha256:`identity-${payload.domain}-${payload.variant}`,score:{task_score:0},
        orders:[],human:{x:payload.domain==='pong'?pongX:1,holding:null},ai:{x:6},lanes:9,balls:[],
        deliveries:0,collision_events:0,shutdown_events:0,interaction:{available:kitchenInteraction}};
      const session={state,binding};sessions.set(token,session);return response({...publicView(session),session_token:token});
    }
    const session=sessions.get(token);if(!session)return response({error:'preview_session_unavailable'},409);
    if(path.endsWith('session'))return response(publicView(session));
    if(path.endsWith('action')){
      actions.push(clone(payload));
      if(server.failAction)return response({error:server.failAction},409);
      const perform=()=>{
        assert.equal(payload.expected_turn,session.state.turn);
        const action=payload.human_action;
        if(session.state.domain==='pong'&&((action==='left'&&session.state.human.x===0)||(action==='right'&&session.state.human.x===8)))return response({error:'illegal carried human command'},409);
        if(session.state.domain==='kitchen'&&action==='interact'&&!session.state.interaction.available)return response({error:'illegal carried human command'},409);
        if(action==='left')session.state.human.x=Math.max(0,session.state.human.x-1);
        if(action==='right')session.state.human.x=Math.min(8,session.state.human.x+1);
        session.state.turn++;session.state.terminal=session.state.turn>=maxTurns;
        session.state.score.task_score=session.state.turn*10;
        session.last={human:action,ai:'wait',two_NN:action===null,ai_action_overrides:0};
        return response(publicView(session));
      };
      return server.deferActions?new Promise(resolve=>pending.push(()=>resolve(perform()))):perform();
    }
    throw Error('unexpected endpoint '+path);
  };
  return server;
}

async function harness(t,{domain='warehouse',server=backend(),memory=new Map(),storageThrows=false,deferMotion=false}={}){
  const elements=new Map(),handlers={},windowHandlers={},timers=new Set(),statusHistory=[],motions=[];
  let document;
  function element(id,dataset={}){
    const listeners={};let text='';
    const e={id,dataset,tagName:id==='board'?'CANVAS':'BUTTON',disabled:false,hidden:false,
      value:({domain,variant:'regularized',task:'1',seed:domain==='warehouse'?'40786':domain==='pong'?'880101':'1000'})[id]||'',
      checked:id==='sequence',innerHTML:'',addEventListener(type,fn){(listeners[type]??=[]).push(fn);},
      async fire(type,event={}){for(const fn of listeners[type]||[])await fn({target:e,preventDefault(){},...event});},
      focus(){document.activeElement=e;},click(){return e.disabled?undefined:e.fire('click');}};
    Object.defineProperty(e,'textContent',{get:()=>text,set:value=>{text=String(value);if(id==='status')statusHistory.push(text);}});
    return e;
  }
  const $=id=>{if(!elements.has(id))elements.set(id,element(id));return elements.get(id);};
  const buttons=['up','left','wait','right','down','interact'].map(action=>element('action-'+action,{action}));
  document={hidden:false,getElementById:$,activeElement:{tagName:'BODY'},
    querySelectorAll:()=>buttons,querySelector:selector=>buttons.find(b=>selector.includes(`"${b.dataset.action}"`)),
    addEventListener(type,fn){(handlers[type]??=[]).push(fn);},createElement:tag=>element('created-'+tag)};
  const later=(fn,ms)=>{const timer=setTimeout(()=>{timers.delete(timer);fn();},ms);timers.add(timer);return timer;};
  const context=vm.createContext({document,window:{addEventListener(type,fn){windowHandlers[type]=fn;}},
    location:{pathname:`/${domain}/`},localStorage:{getItem:key=>{if(storageThrows)throw Error('storage denied');return memory.get(key)??null;},
      setItem:(key,value)=>{if(storageThrows)throw Error('storage denied');memory.set(key,value);},removeItem:key=>{if(storageThrows)throw Error('storage denied');memory.delete(key);}},
    PreviewBoard:class{setState(state,options){options.onPresent?.();return deferMotion&&options.animate?new Promise(resolve=>motions.push(resolve)):Promise.resolve();}ready(){return Promise.resolve();}},
    PreviewInputQueue:Queue,PreviewTaskSequence:Sequence,AbortController,performance,crypto:require('node:crypto').webcrypto,
    setTimeout:later,clearTimeout:timer=>{clearTimeout(timer);timers.delete(timer);},
    requestAnimationFrame:fn=>setImmediate(()=>fn(performance.now())),fetch:server.fetch,console});
  t.after(()=>{for(const timer of timers)clearTimeout(timer);});
  vm.runInContext(fs.readFileSync('ui/model_preview/app.js','utf8'),context);
  const settle=async()=>{for(let i=0;i<5;i++)await tick();};
  await settle();
  const emit=async(type,event)=>{for(const fn of handlers[type]||[])fn({preventDefault(){},repeat:false,...event});await settle();};
  const key=async(code,key='Process')=>{await emit('keydown',{code,key});await emit('keyup',{code,key});};
  const burst=events=>{for(const [code,key='Process'] of events)for(const type of ['keydown','keyup'])for(const fn of handlers[type]||[])fn({code,key,repeat:false,preventDefault(){}});};
  return {$,document,server,memory,statusHistory,settle,key,burst,emit,buttons,motions,
    start:async()=>{await $('selection').fire('submit');await settle();},
    click:async id=>{await $(id).click();await settle();},
    progress:()=>JSON.parse(vm.runInContext('JSON.stringify(course?.toJSON()??null)',context)),
    queue:()=>vm.runInContext('({pending:queue.items.length,failed:!!queue.failedItem,current:!!queue.current})',context)};
}

test('actual app completes Task 1 -> 2 -> 3 and retains the same frozen model',async t=>{
  const h=await harness(t);await h.start();
  for(let task=1;task<=3;task++){
    assert(h.$('turn').textContent.startsWith(`Task ${task}`));assert(h.$('next-task').hidden);
    await h.key('KeyW');await h.key('Space');
    assert.equal(h.progress().runs[task-1].terminal,true);
    if(task<3){assert.equal(h.$('next-task').hidden,false);assert.equal(h.$('next-task').disabled,false);await h.click('next-task');}
  }
  assert.equal(h.$('progress-title').textContent,'三个任务已全部结束');assert(h.$('next-task').hidden);
  assert.deepEqual(h.server.starts.map(s=>s.task),[1,2,3]);
  assert(h.server.starts.every(s=>s.domain==='warehouse'&&s.variant==='regularized'&&s.seed===40786));
  assert.equal(new Set(h.progress().runs.map(r=>r.binding_identity_sha256)).size,1);
});

test('actual app restores an in-progress second task after refresh',async t=>{
  const server=backend(),memory=new Map(),first=await harness(t,{server,memory});await first.start();
  await first.key('Space');await first.key('Space');await first.click('next-task');await first.key('KeyW');
  const restored=await harness(t,{server,memory});
  assert(restored.$('turn').textContent.includes('Task 2 · 步数 1'));assert.equal(restored.document.activeElement,restored.$('board'));
  assert.equal(restored.progress().runs.length,2);assert.equal(server.starts.length,2);
  await restored.key('Space');await restored.click('next-task');await restored.key('Space');await restored.key('Space');
  assert.equal(restored.$('progress-title').textContent,'三个任务已全部结束');
});

test('physical KeyW and Space work under Chinese IME and start transfers focus to canvas',async t=>{
  const h=await harness(t);h.document.activeElement={tagName:'SELECT'};await h.start();
  assert.equal(h.document.activeElement,h.$('board'));
  await h.key('KeyW','Process');await h.key('Space','Process');
  assert.deepEqual(h.server.actions.map(a=>a.human_action),['up','wait']);
});

test('Pong boundary and queued newly-invalid move do not deadlock subsequent Space',async t=>{
  const server=backend({maxTurns:10,pongX:1});server.deferActions=true;
  const h=await harness(t,{domain:'pong',server});await h.start();
  h.burst([['KeyA'],['KeyA'],['Space']]);await h.settle();assert.equal(server.actions.length,1);
  server.pending.shift()();await h.settle();
  assert.deepEqual(server.actions.map(a=>a.human_action),['left','wait']);
  assert(h.statusHistory.some(s=>s.includes('这一步在当前局面不可用')));
  server.pending.shift()();await h.settle();
  assert.equal(h.queue().failed,false);assert.equal(h.queue().pending,0);
  await h.key('KeyA');assert.equal(server.actions.length,2);
  await h.key('Space');assert.equal(server.actions.length,3);server.pending.shift()();await h.settle();
  assert.equal(h.queue().failed,false);assert(h.$('turn').textContent.includes('步数 3'));
});

test('unavailable Kitchen E and denied browser storage still allow WASD and Space',async t=>{
  const h=await harness(t,{domain:'kitchen',server:backend({maxTurns:10}),storageThrows:true});await h.start();
  await h.key('KeyE');assert.equal(h.server.actions.length,0);assert.equal(h.queue().failed,false);
  await h.key('KeyW');await h.key('Space');
  assert.deepEqual(h.server.actions.map(a=>a.human_action),['up','wait']);assert.equal(h.queue().failed,false);
});

test('failed command preserves the visible error and blocks new input until retry',async t=>{
  const server=backend({maxTurns:10}),h=await harness(t,{server});await h.start();server.failAction='preview_session_unavailable';
  await h.key('KeyW');const error=h.$('status').textContent;
  assert(error.includes('preview_session_unavailable'));assert.equal(h.queue().failed,true);
  await h.key('Space');await h.key('KeyD');
  assert.equal(server.actions.length,1);assert.equal(h.$('status').textContent,error);assert.equal(h.$('connect').hidden,false);
  const original=clone(server.actions[0]);server.failAction=null;await h.click('connect');
  assert.deepEqual(server.actions[1],original);assert.equal(h.queue().failed,false);
  await h.key('Space');assert.equal(server.actions.at(-1).human_action,'wait');
});

test('old in-flight failure cannot attach its command to a newly started session',async t=>{
  const server=backend({maxTurns:10}),original=server.fetch;let rejectOld,once=true;
  server.fetch=async(path,options)=>{
    if(path.endsWith('action')&&once){once=false;return new Promise(resolve=>{rejectOld=()=>resolve({ok:false,status:409,json:async()=>({error:'preview_session_unavailable'})});});}
    return original(path,options);
  };
  const h=await harness(t,{server});await h.start();await h.key('KeyW');
  const restarting=h.start();await h.settle();rejectOld();await restarting;
  assert.equal(server.starts.length,2);assert.equal(h.queue().failed,false);
  await h.key('Space');assert.equal(server.actions.length,1);assert.equal(server.actions[0].human_action,'wait');
  assert.equal(h.progress().runs[0].turn,1);
});


test('older motion completion cannot enable Next before the terminal frame finishes',async t=>{
  const h=await harness(t,{deferMotion:true});await h.start();
  await h.key('KeyW');await h.key('Space');
  assert.equal(h.$('next-task').hidden,false);assert.equal(h.$('next-task').disabled,true);
  h.motions.shift()();await h.settle();assert.equal(h.$('next-task').disabled,true);
  h.motions.shift()();await h.settle();assert.equal(h.$('next-task').disabled,false);
});
