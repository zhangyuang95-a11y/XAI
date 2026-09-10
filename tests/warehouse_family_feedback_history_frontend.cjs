// Run the complete new app with a small DOM/transport stand-in. No browser,
// HTTP connection, service, model, physics, or participant records are used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const web = path.resolve(__dirname, '../ui/warehouse_family_feedback_research');
const script = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
const counts = {app_instances:0, synthetic_gets:0, synthetic_posts:0,
  actual_HTTP_requests:0, environment_steps:0, NN_calls:0, service_starts:0};
const tick = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {let resolve, reject; const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const response = data => ({ok:true,status:200,json:async()=>data});
const dataName = name => name.replace(/-([a-z])/g,(_,c)=>c.toUpperCase());

function view({run='run1',frame=20,version=20,stage='task1',explain=false}={}) {
  return {session_id:'session1',run_id:run,version,history_count:frame+1,
    flow:{mode:'study',stage,participant_id:'qa_transport'},horizon:120,
    release:{model_ready:true,study_ready:true,explanation_ready:true},study_allowed:true,
    allowed_kinds:['action','end','next','start','select_run','question','answer_seen'],
    explain_allowed:explain,state:{frame,agents:[]},answers:[],runs:[]};
}
function history(run='run1',frame=20) {return {run_id:run,frames:Array.from({length:frame+1},(_,i)=>({state:{frame:i,agents:[]}}))};}

async function harness(initial=view()) {
  counts.app_instances++;
  const elements=[],ids=new Map(),windowHandlers=new Map(),intervals=[];
  let document;
  class Element {
    constructor(tag='div') {
      this.tagName=tag.toUpperCase();this.dataset={};this.attributes={};this.handlers=new Map();
      this.value='';this.textContent='';this.disabled=false;this.isConnected=true;this.children=[];
      this.clientWidth=this.clientHeight=0; // Canvas paint is irrelevant to dispatch and does not execute.
      const names=new Set();this.classList={toggle:(key,on)=>on?names.add(key):names.delete(key),contains:key=>names.has(key)};
      this.parentElement={classList:this.classList};
    }
    setAttribute(k,v){this.attributes[k]=String(v);}
    addEventListener(k,fn){if(!this.handlers.has(k))this.handlers.set(k,[]);this.handlers.get(k).push(fn);}
    dispatch(k,values={}){for(const fn of this.handlers.get(k)||[])fn({target:this,preventDefault(){},...values});}
    focus(){document.activeElement=this;}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=nodes;}
    querySelectorAll(){return [];}
  }
  for(const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
    const el=new Element(match[1]);elements.push(el);
    for(const a of match[2].matchAll(/([a-z][a-z0-9-]*)="([^"]*)"/gi)) {
      const [_,key,value]=a;el.setAttribute(key,value);
      if(key==='id'){el.id=value;ids.set(value,el);}
      if(key==='value')el.value=value;
      if(key.startsWith('data-'))el.dataset[dataName(key.slice(5))]=value;
    }
  }
  const agreement=new Element();
  document={body:new Element('body'),documentElement:new Element('html'),visibilityState:'visible',
    getElementById:id=>{assert.ok(ids.has(id),`Known real DOM id: ${id}`);return ids.get(id);},
    querySelector:s=>{assert.equal(s,'.agreement');return agreement;},
    querySelectorAll:s=>{const m=/^\[data-([a-z0-9-]+)\]$/.exec(s);assert.ok(m,`Known selector: ${s}`);return elements.filter(e=>dataName(m[1]) in e.dataset);},
    createElement:tag=>new Element(tag)};
  document.activeElement=document.body;
  const store=new Map();let current=structuredClone(initial);const requests=[],posts=[],timers=new Set();
  const window={scrollX:0,scrollY:0,devicePixelRatio:1,scrollTo(){},addEventListener(k,fn){if(!windowHandlers.has(k))windowHandlers.set(k,[]);windowHandlers.get(k).push(fn);}};
  const context={console,module:{exports:{}},document,window,crypto:{randomUUID:()=>`synthetic-op-${posts.length+1}`},
    localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},
    AbortController,requestAnimationFrame:fn=>fn(),setInterval:fn=>{intervals.push(fn);},
    setTimeout:(fn,ms)=>{const id=setTimeout(fn,ms);timers.add(id);return id;},clearTimeout:id=>{clearTimeout(id);timers.delete(id);},
    fetch:async(url,options)=>{
      if(url==='/api/history'){counts.synthetic_gets++;const d=deferred();requests.push(d);return d.promise;}
      if(url==='/api/view'){counts.synthetic_gets++;return response(structuredClone(current));}
      assert.equal(url,'/api/command');counts.synthetic_posts++;
      const body=JSON.parse(options.body);posts.push(body);
      current={...current,version:current.version+1};
      if(body.kind==='action')current={...current,history_count:current.history_count+1,state:{...current.state,frame:current.state.frame+1}};
      return response(structuredClone(current));
    }};
  vm.runInNewContext(script,context,{filename:path.join(web,'app.js')});await tick();
  return {document,ids,requests,posts,api:context.module.exports,
    click(id){const e=ids.get(id);if(!e.disabled)e.dispatch('click');},
    force(id,event='click',values={}){ids.get(id).dispatch(event,values);},
    action(action){elements.find(e=>e.dataset.action===action).dispatch('click');},
    key(key){for(const fn of windowHandlers.get('keydown')||[])fn({key,target:ids.get('warehouseCanvas'),preventDefault(){},repeat:false});},
    async refresh(next){current=structuredClone(next);intervals[0]();await tick();},
    cleanup(){for(const id of timers)clearTimeout(id);},
    disabledActions(){return elements.filter(e=>e.dataset.action).every(e=>e.disabled);},
  };
}

test('pending history immediately blocks keyboard and all command buttons; replay and live retain their rules',async t=>{
  const h=await harness();t.after(()=>h.cleanup());h.click('previousFrame');
  assert.equal(h.requests.length,1);assert.equal(h.document.body.dataset.historyLoading,'true');
  assert.equal(h.document.body.dataset.replay,'false');assert.ok(h.disabledActions());
  for(const id of ['endButton','nextButton','startButton','retryButton'])assert.equal(h.ids.get(id).disabled,true,id);
  assert.match(h.ids.get('statusText').textContent,/正在加载历史/);
  h.key(' ');h.key('ArrowUp');h.action('WAIT');
  // Force callbacks even for disabled elements to check dispatch-level protection.
  for(const id of ['endButton','nextButton','startButton','retryButton'])h.force(id);
  h.force('runSelect','change',{target:{value:'other'}});
  assert.equal(h.posts.length,0);
  h.click('languageButton');assert.match(h.ids.get('statusText').textContent,/Loading history/);
  h.requests[0].resolve(response(history()));await tick();
  assert.equal(h.document.body.dataset.historyLoading,'false');assert.equal(h.document.body.dataset.replay,'true');
  h.key(' ');h.action('WAIT');assert.equal(h.posts.length,0);
  h.click('liveButton');h.key(' ');await tick();
  assert.equal(h.posts.length,1);assert.equal(h.posts[0].action,'WAIT');assert.equal(h.document.body.dataset.frame,'21');
});

for(const failure of ['network','wrong_run','missing_frame'])test(`${failure} releases only history lock and permits retry/action`,async t=>{
  const h=await harness();t.after(()=>h.cleanup());h.click('previousFrame');
  if(failure==='network')h.requests[0].reject(new Error('synthetic network failure'));
  else h.requests[0].resolve(response(failure==='wrong_run'?history('wrong'):{run_id:'run1',frames:[]}));
  await tick();assert.equal(h.document.body.dataset.historyLoading,'false');assert.equal(h.document.body.dataset.replay,'false');
  assert.match(h.ids.get('statusText').textContent,/历史加载失败/);
  h.click('languageButton');assert.match(h.ids.get('statusText').textContent,/History could not be loaded/);
  h.click('previousFrame');assert.equal(h.requests.length,2);h.requests[1].resolve(response(history()));await tick();
  assert.equal(h.document.body.dataset.replay,'true');h.click('liveButton');h.action('WAIT');await tick();
  assert.equal(h.posts.length,1);
});

for(const stale of ['success','failure'])test(`stale ${stale} cannot unlock or overwrite newer history`,async t=>{
  const h=await harness();t.after(()=>h.cleanup());h.click('previousFrame');
  await h.refresh(view({run:'run2',frame:30,version:31,stage:'task2'}));
  assert.equal(h.document.body.dataset.historyLoading,'false');assert.match(h.ids.get('statusText').textContent,/旧历史请求已取消/);
  h.click('previousFrame');assert.equal(h.requests.length,2);assert.equal(h.document.body.dataset.historyLoading,'true');
  if(stale==='success')h.requests[0].resolve(response(history()));else h.requests[0].reject(new Error('late failure'));
  await tick();assert.equal(h.document.body.dataset.historyLoading,'true');assert.equal(h.document.body.dataset.frame,'30');
  h.key(' ');h.force('nextButton');assert.equal(h.posts.length,0);
  h.requests[1].resolve(response(history('run2',30)));await tick();
  assert.equal(h.document.body.dataset.historyLoading,'false');assert.equal(h.document.body.dataset.replay,'true');
  assert.equal(h.ids.get('historySlider').value,'29');assert.equal(h.document.body.dataset.explanationAllowed,'false');
});

test('same-run version change cancels pending read; late response cannot restore an old frame',async t=>{
  const h=await harness();t.after(()=>h.cleanup());h.click('previousFrame');
  await h.refresh(view({version:21}));assert.equal(h.document.body.dataset.historyLoading,'false');
  h.key(' ');await tick();assert.equal(h.posts.length,1);
  h.requests[0].resolve(response(history()));await tick();
  assert.equal(h.document.body.dataset.replay,'false');assert.equal(h.document.body.dataset.frame,'21');
});

test('existing A/B and Task2 answer visibility remains gated',async t=>{
  const h=await harness();t.after(()=>h.cleanup());const a={...view({explain:true}),answers:[{id:'answer1',run_id:'run1',text:'synthetic answer'}]};
  assert.equal(h.api.visibleAnswers(a).length,1);
  assert.equal(h.api.visibleAnswers({...a,explain_allowed:false}).length,0);
  assert.equal(h.api.visibleAnswers({...a,flow:{mode:'study',stage:'task2'}}).length,0);
  assert.equal(h.api.visibleAnswers({...a,study_version_mismatch:true}).length,0);
  assert.equal(h.api.FRONTEND_VERSION,'warehouse-family-feedback-research.v2');
});

process.on('exit',()=>console.log('FEEDBACK_HISTORY_FRONTEND_SCOPE='+JSON.stringify(counts)));
