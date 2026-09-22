'use strict';
// Execute the production UI with a controlled network, including pause races.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const handlers={},calls=[],notices=[],renders=[],elements={};
for(const id of ['englishButton','chineseButton','tutorialStartButton','tutorialPauseButton','tutorialRetryButton','tutorialSkipButton'])elements[id]={id,disabled:false,classList:{toggle(){}}};
const actionButton={dataset:{action:'wait'},disabled:false};
const context=vm.createContext({console,URLSearchParams,performance,crypto:require('node:crypto').webcrypto,setTimeout,clearTimeout,setInterval(){},
 sessionStorage:{getItem(){return null},setItem(){},removeItem(){}},location:{pathname:'/kitchen/',search:''},
 window:{addEventListener(name,fn){handlers['window:'+name]=fn}},document:{hidden:false,activeElement:null,documentElement:{},getElementById:id=>elements[id]||null,addEventListener(name,fn){handlers[name]=fn},querySelector(){return null},querySelectorAll(selector){return selector==='[data-action]'?[actionButton]:[]}},calls,notices,renders,elements});
let source=fs.readFileSync(path.join(__dirname,'../study_v3/web/app.js'),'utf8');source=source.slice(0,source.lastIndexOf('(async()=>{languageUI();'));vm.runInContext(source,context);
const run=code=>vm.runInContext(code,context),tick=()=>new Promise(resolve=>setImmediate(resolve));
run(`render=async options=>renders.push(options);toast=message=>notices.push(message);report=e=>{throw e};api=(path,payload)=>new Promise(resolve=>calls.push({path,payload,resolve}));
 view={instance_id:'practice',revision:1,stage:'demo',mode:'preview',group:'A',can_ask:false,actions:['up','down','left','right','wait'],task_runs:[],public_help:[{id:'controls',en:'WASD and E operate the front station.',zh:'WASD 与 E 操作前方工位。'},{id:'menu',en:'Five dishes; deadlines 100, 140, 240, 280, 360.',zh:'五道菜，截止回合 100、140、240、280、360。'},{id:'score',en:'Serve +30; the action uses one turn.',zh:'上菜 +30，动作占用一回合。'}],rule_metadata:{orders_per_task:5,score:{served:30,step:-1,single_component_discard:-3,combined_dish_discard:-10}},tutorial:{index:0,version:'practice-v1',playing:false,completed:false,progress:{},goal:{en:'Move, face a counter, and wait.',zh:'移动、面向工位并等待。'},highlights:['prep'],state:{domain:'kitchen',turn:0,max_turns:360,score:{task_score:0},orders:[{recipe:'egg_tomato',deadline:100,status:'pending'}],interaction:{available:false,label_en:'Face a station first.',label_zh:'先面向工位。'}}}};view.state=view.tutorial.state;`);
const key=(key,repeat=false)=>({key,repeat,preventDefault(){}});
const response=(mutate=()=>{})=>{const result=JSON.parse(run('JSON.stringify(view)'));result.revision++;mutate(result);return result;};
(async()=>{
 const html=run('demoPage()');assert(html.includes('tutorialStartButton'));assert(html.includes('tutorialRetryButton'));assert(!html.includes('demoPlayButton'));assert(!html.includes('demoNextButton'));assert(!html.includes('first protein'));assert(!html.includes('combine'));
 handlers.keydown(key('d'));assert.equal(calls.length,0,'Paused practice cannot move.');
 const started=run(`tutorialCommand('start')`);assert.equal(calls[0].path,'/api/study/tutorial');assert.equal(calls[0].payload.command,'start');assert(calls[0].payload.command_id);
 calls[0].resolve(response(v=>v.tutorial.playing=true));await started;assert.equal(run('canMove()'),true);
 // An unavailable E is recorded by the sandbox and returns feedback without
 // spending a turn. Formal-task invalid interactions retain their local guard.
 handlers.keydown(key('e'));assert.equal(calls.length,2);assert.equal(calls[1].payload.action,'interact');assert.equal(calls[1].path,'/api/study/tutorial');
 calls[1].resolve(response(v=>v.tutorial.feedback={en:'Face a station first.',zh:'先面向工位。'}));await tick();assert.equal(run('view.state.turn'),0);assert(run('demoPage()').includes('Face a station first.'));
 handlers.keydown(key('d'));handlers.keydown(key('d',true));assert.equal(calls.length,3);assert.equal(calls[2].payload.command,'action');assert.equal(calls[2].payload.action,'right');assert.equal(calls[2].path,'/api/study/tutorial');assert(!('run_id' in calls[2].payload));
 // Pausing during the network request stops held input immediately, then
 // persists the pause after the authoritative action has been acknowledged.
 run('pauseTutorial()');assert.equal(run('heldDirection'),null);assert.equal(run('canMove()'),false);
 calls[2].resolve(response(v=>{v.state.turn++;v.tutorial.state=v.state;}));await tick();assert.equal(renders.at(-1).animate,true);
 assert.equal(calls.length,4);assert.equal(calls[3].payload.command,'pause');
 calls[3].resolve(response(v=>v.tutorial.playing=false));await tick();assert.equal(run('view.tutorial.playing'),false);assert.equal(calls.length,4);
 const retry=run(`tutorialCommand('retry')`);assert.equal(calls[4].payload.command,'retry');calls[4].resolve(response(v=>{v.tutorial.playing=true;v.tutorial.state.turn=0;}));await retry;
 // Automatic segment changes release the held key; no queued action can run
 // against the independently seeded next practice scene.
 handlers.keydown(key('a'));assert.equal(calls.length,6);
 calls[5].resolve(response(v=>{v.tutorial.index=1;v.tutorial.state.turn=0;}));await tick();assert.equal(run('heldDirection'),null);assert.equal(calls.length,6);assert.equal(renders.at(-1).animate,false,'Independent scenes must not animate a teleport.');
 run(`heldDirection='right'`);handlers['window:blur']();assert.equal(run('heldDirection'),null);
 run(`document.activeElement={tagName:'TEXTAREA'}`);handlers.keydown(key(' '));assert.equal(calls.length,6);run('document.activeElement=null');
 // Public help and menu numbers are provided by the saved rules, never +100.
 const menu=run('kitchenMenu(view.state)');assert(menu.includes('Serve on time +30'));assert(menu.includes('Due 100'));assert(!menu.includes('+100'));assert(menu.includes('Menu · 5 dishes'));
 const prep=run(`tutorialPreparation({human:{preparation:{label_en:'Whisk egg',label_zh:'打散鸡蛋',completed:1,required:4,remaining:3,ready:false}}})`);assert(prep.includes('Whisk egg'));assert(prep.includes('1 / 4'));assert(prep.includes('3 more interactions'));
 run(`view.tutorial.index=5;view.tutorial.total_segments=6`);const last=run('demoPage()');assert(last.includes('Step 6 / 6'));assert(!last.includes('class="panel kitchen-menu"'));assert(last.includes('deadlines 100, 140'));assert(last.includes('Serve +30'));
 run(`lang='zh'`);assert(run('demoPage()').includes('截止回合'));assert(run('conciseRules(view.state)').includes('前方工位'));
 run(`view.tutorial.completed=true;view.tutorial.playing=false`);assert(run('demoPage()').includes('开始 Task 1'));assert.equal(run('canMove()'),false);
 assert(!run('demoPage()').includes('tutorialSkipButton'));
 // The demo-specific controls never re-enable Task 2 questions.
 assert.equal(run('view.can_ask'),false);
 run(`view.stage='task1';view.state.terminal=false`);handlers.keydown(key('e'));assert.equal(calls.length,6);assert.equal(notices.at(-1),'先面向工位。');
 console.log('Kitchen tutorial frontend: routing, pause race, held keys, section reset, metadata, public help and completion passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
