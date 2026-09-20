'use strict';
// Controlled service recovery and form preservation checks; no production data.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
let source=fs.readFileSync(path.join(__dirname,'../study_v3/web/app.js'),'utf8');
source=source.slice(0,source.lastIndexOf('(async()=>{languageUI();'));
function fixture(preview=false){
 let now=0,nextTimer=1;const timers=new Map(),calls=[],handlers={},stored=[];
 const ids=['englishButton','chineseButton','app','startForm','startButton','entryStatus','entryStatusMessage','entryRetryButton','participantInput','recoveryInput','groupInput','consentInput',...(preview?['adminInput']:[])];
 const elements=Object.fromEntries(ids.map(id=>[id,{id,disabled:false,hidden:false,value:'',checked:false,textContent:'',classList:{toggle(){}}}]));
 Object.assign(elements.participantInput,{value:'anonymous-draft'});elements.recoveryInput.value='private-recovery-fixture';elements.groupInput.value='B';elements.consentInput.checked=true;if(preview)elements.adminInput.value='private-key-fixture';
 const context=vm.createContext({console,URLSearchParams,AbortController,crypto:require('node:crypto').webcrypto,performance:{now:()=>now},
  setTimeout(fn,delay){const id=nextTimer++;timers.set(id,{fn,at:now+delay});return id},clearTimeout(id){timers.delete(id)},setInterval(){},
  localStorage:{getItem(){return null},setItem(...args){stored.push(args)}},sessionStorage:{getItem(){return null},setItem(...args){stored.push(args)},removeItem(){}},
  location:{pathname:'/warehouse/',search:preview?'?preview=1':''},window:{addEventListener(n,f){handlers[n]=f}},
  document:{hidden:false,activeElement:null,getElementById:id=>elements[id]||null,addEventListener(n,f){handlers[n]=f},querySelector(){return null},querySelectorAll(){return []},createElement(){return {content:{}}}},
  fetch(url,options){return new Promise((resolve,reject)=>{calls.push({url,options,resolve,reject});options.signal?.addEventListener('abort',()=>reject(new Error('aborted')));});},elements});
 vm.runInContext(source,context);const run=s=>vm.runInContext(s,context);
 run(`languageUI=()=>{};report=()=>{};release={study_ready:false};entryCheckedAt=performance.now();`);
 return {run,elements,calls,handlers,timers,stored,advance(ms){now+=ms},fire(){const due=[...timers].filter(([,t])=>t.at<=now);for(const[id,t]of due){timers.delete(id);t.fn();}},resolve(i,ready){calls[i].resolve({ok:true,json:async()=>({study_ready:ready})})}};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 const f=fixture();f.run('bind()');assert.equal(f.elements.startButton.disabled,true);assert.match(f.elements.entryStatusMessage.textContent,/temporarily unavailable/);assert.equal(f.elements.consentInput.checked,true);
 const draft=JSON.stringify(f.run('captureEntryDraft()'));
 // Checking consent does not falsely bypass service readiness; a late recovery does unlock entry.
 f.advance(3000);const recovering=f.run('checkEntryReadiness()');assert.equal(f.calls.length,1);assert.equal(f.elements.entryRetryButton.disabled,true);
 await f.run('checkEntryReadiness()');assert.equal(f.calls.length,1);assert.equal(JSON.stringify(f.run('captureEntryDraft()')),draft);
 f.resolve(0,true);await recovering;assert.equal(f.elements.startButton.disabled,false);assert.equal(f.elements.entryStatus.hidden,true);assert.equal(f.timers.size,0);assert.equal(JSON.stringify(f.run('captureEntryDraft()')),draft);
 assert.equal(f.calls[0].url,'/api/release');assert.equal(f.calls[0].options.body,undefined);
 // Refresh/language rendering retains all current form fields without storing them.
 f.run(`reconcileChildren=()=>{for(const id of ['participantInput','recoveryInput','groupInput'])elements[id].value='';elements.consentInput.checked=false;};`);
 await f.run('render()');assert.equal(JSON.stringify(f.run('captureEntryDraft()')),draft);assert.deepEqual(f.stored,[]);
 // Failed checks and a hung request keep entry closed, recover later, and never spin.
 const g=fixture();g.run('entryFailed=true;bind()');assert.match(g.elements.entryStatusMessage.textContent,/could not reach/);
 g.advance(14999);g.fire();assert.equal(g.calls.length,0);g.advance(1);g.fire();assert.equal(g.calls.length,1);
 g.advance(12000);g.fire();await tick();assert.equal(g.run('entryChecking'),false);assert.equal(g.elements.startButton.disabled,true);assert.equal(g.calls[0].options.signal.aborted,true);
 g.advance(2999);g.fire();assert.equal(g.calls.length,1);g.advance(1);g.fire();assert.equal(g.calls.length,2);
 g.resolve(1,false);await tick();await g.run('checkEntryReadiness()');assert.equal(g.calls.length,2);
 g.run('refresh=async()=>{};document.hidden=true;');g.handlers.visibilitychange();g.advance(60000);g.fire();assert.equal(g.calls.length,2);assert.equal(g.timers.size,0);
 g.run('document.hidden=false');g.handlers.visibilitychange();g.fire();assert.equal(g.calls.length,3);g.resolve(2,true);await tick();assert.equal(g.elements.startButton.disabled,false);
 // A session readiness race returns to the same form, rather than losing entered fields.
 const h=fixture();h.run(`release={study_ready:true};api=async()=>{const e=new Error();e.code='study_not_ready';throw e;};bind();`);
 const prior=JSON.stringify(h.run('captureEntryDraft()'));await h.elements.startForm.onsubmit({preventDefault(){}});assert.equal(h.elements.startButton.disabled,true);assert.equal(JSON.stringify(h.run('captureEntryDraft()')),prior);assert.equal(h.calls.length,0);h.fire();assert.equal(h.calls.length,1);h.resolve(0,true);await tick();assert.equal(h.elements.startButton.disabled,false);
 // Preview remains usable, does not poll, and never persists its private key.
 const p=fixture(true);p.run('bind()');assert.equal(p.elements.startButton.disabled,false);assert.equal(p.timers.size,0);
 const privateDraft=JSON.stringify(p.run('captureEntryDraft()'));p.run(`reconcileChildren=()=>{for(const id of ['participantInput','recoveryInput','groupInput','adminInput'])elements[id].value='';elements.consentInput.checked=false;};`);await p.run('render()');assert.equal(JSON.stringify(p.run('captureEntryDraft()')),privateDraft);assert.deepEqual(p.stored,[]);
 // No poll, render, or gameplay change once an existing task is active.
 const a=fixture();a.advance(3000);const outstanding=a.run('checkEntryReadiness()');a.run(`view={instance_id:'active',stage:'task1',state:{turn:7}};delete elements.startForm;delete elements.startButton;delete elements.entryStatus;delete elements.entryStatusMessage;delete elements.entryRetryButton;`);a.resolve(0,true);await outstanding;assert.equal(a.run('view.state.turn'),7);assert.equal(a.timers.size,0);await a.run('checkEntryReadiness()');assert.equal(a.calls.length,1);
 console.log('Entry recovery checks passed: automatic/manual retry, timeout/rate limit, visibility, unchanged drafts, preview privacy, session race, and active-task isolation.');
})().catch(error=>{console.error(error);process.exitCode=1});
