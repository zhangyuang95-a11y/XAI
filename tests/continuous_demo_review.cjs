'use strict';
// Controlled animation/network race checks; this is not a browser acceptance.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const storage=new Map(),pending=[],commands=[],drawn=[],handlers={};
const elements=Object.fromEntries(['englishButton','chineseButton','demoBoard','studyCanvas','demoPlayButton','demoPauseButton','demoSkipButton','demoCaption','demoProgress'].map(id=>[id,{id,disabled:false,textContent:'',dataset:{},classList:{toggle(){}}}]));
const context=vm.createContext({console,URLSearchParams,performance,crypto:require('node:crypto').webcrypto,
 setTimeout(fn){queueMicrotask(fn);return 1},clearTimeout(){},setInterval(){},
 localStorage:{getItem(){return null},setItem(){}},sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
 location:{pathname:'/pong/',search:''},window:{addEventListener(n,f){handlers[n]=f}},
 document:{hidden:false,activeElement:null,getElementById:id=>elements[id]||null,addEventListener(n,f){handlers[n]=f},querySelector(){return null},querySelectorAll(){return []},createElement(){return {content:{}}}},pending,commands,drawn,elements});
let source=fs.readFileSync(path.join(__dirname,'../study_v3/web/app.js'),'utf8');source=source.slice(0,source.lastIndexOf('(async()=>{languageUI();'));vm.runInContext(source,context);
const run=s=>vm.runInContext(s,context);
run(`reconcileChildren=()=>{};command=async kind=>commands.push(kind);boardRenderer={canvas:elements.studyCanvas,setState(s){drawn.push(s.turn);return new Promise(resolve=>pending.push(resolve));}};
 view={instance_id:'demo-instance',release_id:'fixture',stage:'demo',questions:[],state:null,demo:{index:0,frames:Array.from({length:4},(_,turn)=>({domain:'pong',turn,max_turns:3,score:{task_score:0},balls:[]})),captions:[{index:0,en:'Start',zh:'开始'},{index:2,en:'Next caption',zh:'接着看'}]}};`);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 const playing=run('playDemo()');assert.deepEqual(drawn,[1]);assert.equal(run('demoPlaying'),true);
 run('pauseDemo()');pending.shift()();await playing;await tick();
 assert.deepEqual(drawn,[1]);assert.equal(commands.length,0);assert.equal(run('demoFrame()'),1);
 // Resume after pause at the saved frame without replaying or using task time.
 const resumed=run('playDemo()');assert.deepEqual(drawn,[1,2]);pending.shift()();await tick();
 assert.deepEqual(drawn,[1,2,3]);pending.shift()();await resumed;
 assert.deepEqual(commands,['demo_finish']);assert.equal(run('view.state'),null);
 assert(!run('demoPage()').includes('demoNextButton'));assert(!run('demoPage()').includes('Play this step'));
 run(`view.questions=[{question:'OLDER QUESTION',result:{answer:'OLDER ANSWER'}},{question:'LATEST QUESTION',result:{answer:'LATEST ANSWER'},target_turn:4}];view.can_ask=true;`);
 assert(!run('questionPanel()').includes('OLDER ANSWER'));assert(run('questionPanel()').includes('LATEST ANSWER'));
 // Leaving the tab during the next playback blocks any further frames.
 run(`saveDemoFrame(0);view.demo.index=0;view.can_ask=false;boardRenderer.stop=()=>{};`);
 const hidden=run('playDemo()');run('document.hidden=true;pauseDemo()');pending.shift()();await hidden;await tick();
 assert.equal(drawn.at(-1),1);assert.deepEqual(commands,['demo_finish']);
 assert.equal(run(`scoreFeedback({events:[{type:'waste',score_delta:-10},{type:'score_delta',score_delta:-10}]})`).match(/-10/g).length,1);
 console.log('Continuous demo pause/resume/end/visibility and latest-answer checks passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
