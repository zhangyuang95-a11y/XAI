const {test}=require('node:test');
const assert=require('node:assert/strict');
const Queue=require('../ui/model_preview/input_queue.js');
const tick=()=>new Promise(r=>setImmediate(r));

test('burst: one in flight, eight pending, FIFO with explicit ninth rejection',async()=>{
  let resolve,full=0,active=0,max=0;const done=[];
  const q=new Queue({full:()=>full++,send:async x=>{max=Math.max(max,++active);await new Promise(r=>resolve=r);done.push(x.action);active--;}});
  assert(q.enqueue(0));for(let i=1;i<=8;i++)assert(q.enqueue(i));assert(!q.enqueue(9));assert.equal(full,1);
  for(let i=0;i<=8;i++){resolve();await tick();}
  assert.deepEqual(done,[0,1,2,3,4,5,6,7,8]);assert.equal(max,1);
});
test('hold queues at most one repeat; keyup cancels only unsent repeats',async()=>{
  let release;const done=[];const q=new Queue({send:async x=>{await new Promise(r=>release=r);done.push(x.action);}});
  q.enqueue('left');for(let i=0;i<10;i++)q.enqueue('left',true);q.enqueue('right');
  assert.equal(q.items.filter(x=>x.held).length,1);q.release();
  release();await tick();release();await tick();assert.deepEqual(done,['left','right']);
});
test('failed command retries exact same identity and payload before subsequent input',async()=>{
  let first=true;const calls=[];const q=new Queue({send:async item=>{
    item.payload??={command_id:'fixed-id',expected_turn:0};calls.push(item.payload);
    if(first){first=false;throw Error('lost acknowledgement');}
  }});
  q.enqueue('left');q.enqueue('right');await tick();assert(q.failedItem);assert.equal(calls.length,1);
  q.retry();await tick();assert.equal(calls.length,3);assert.strictEqual(calls[0],calls[1]);
});
test('pause, model change, new game or terminal clears pending; sent frame can finish',async()=>{
  let release;const done=[];const q=new Queue({send:async item=>{await new Promise(r=>release=r);done.push(item.action);}});
  q.enqueue('sent');q.enqueue('pending');q.enqueue('held',true);assert.equal(q.clear(),2);
  const idle=q.idle();release();await idle;await tick();assert.deepEqual(done,['sent']);assert.equal(q.items.length,0);
});

test('actual app rejects extra inputs until the step completes and retries only the original command',async()=>{
  const vm=require('node:vm'),fs=require('node:fs'),{performance}=require('node:perf_hooks');
  const elements=new Map(),handlers={},buttons=['up','left','wait','right','down','interact'].map(action=>element('action-'+action,{action}));
  function element(id,dataset={}){return {id,dataset,disabled:false,hidden:false,value:({domain:'warehouse',variant:'raw',task:'1',seed:'40786'})[id]||'',textContent:'',addEventListener(name,fn){this[name]=fn;}};}
  const $=id=>{if(!elements.has(id))elements.set(id,element(id));return elements.get(id);};
  let turn=0,pending=[],actionCalls=[],failNext=true;
  const state=()=>({state:{domain:'warehouse',task:1,turn,max_turns:120,terminal:false,score:{task_score:0},orders:[],deliveries:0,collision_events:0,shutdown_events:0},binding:{variant:'raw',actor_sha256:'fixed'},last_action:null});
  const document={getElementById:$,activeElement:{tagName:'BODY'},querySelectorAll:()=>buttons,querySelector:s=>buttons.find(b=>s.includes('"'+b.dataset.action+'"')),addEventListener:(n,fn)=>handlers[n]=fn};
  const context=vm.createContext({document,window:{addEventListener(){}},location:{pathname:'/warehouse/'},localStorage:{getItem:()=>null,setItem(){}},PreviewBoard:class{setState(s,o){o.onPresent?.();}},PreviewInputQueue:Queue,AbortController,performance,crypto:require('node:crypto').webcrypto,setTimeout,clearTimeout,requestAnimationFrame:fn=>setImmediate(()=>fn(performance.now())),console,
    fetch:async(path,options)=>{
      if(path.endsWith('models'))return {ok:true,json:async()=>({models:{warehouse:{seeds:[40786],version:'test'}}})};
      if(path.endsWith('start'))return {ok:true,json:async()=>({...state(),session_token:'test-token'})};
      if(path.endsWith('action')){
        const payload=JSON.parse(options.body);actionCalls.push(payload);
        if(failNext){failNext=false;throw Error('ack lost');}
        return new Promise(resolve=>pending.push(()=>{turn++;resolve({ok:true,json:async()=>state()});}));
      }
      throw Error('unexpected path '+path);
    }});
  vm.runInContext(fs.readFileSync('ui/model_preview/app.js','utf8'),context);await tick();
  await $('selection').submit({preventDefault(){}});
  const key=(name,type='keydown')=>handlers[type]({key:name,repeat:false,preventDefault(){}});
  const started=performance.now();key('d');key('d','keyup');key('a');key('a','keyup');key('w');key('w','keyup');
  assert(performance.now()-started<50);assert($('input-status').textContent.includes('这一步正在执行'));
  assert(buttons.every(b=>b.disabled));await tick();
  assert.equal(actionCalls.length,2);assert.deepEqual(actionCalls[0],actionCalls[1]);
  pending.shift()();await tick();
  assert.equal(turn,1);assert.deepEqual(actionCalls.slice(1).map(p=>p.human_action),['right']);
  assert.deepEqual(actionCalls.slice(1).map(p=>p.expected_turn),[0]);
  key('s');await new Promise(r=>setTimeout(r,420));key('s','keyup');
  assert.equal(actionCalls.length,3);pending.shift()();await tick();assert.equal(turn,2);
  assert($('input-status').textContent.includes('可以进行下一步'));
  key('a');key('a','keyup');key('d');key('d','keyup');$('pause').click();pending.shift()();await tick();
  assert.equal(turn,3);assert.equal(pending.length,0);assert($('input-status').textContent.includes('可以进行下一步'));
});
