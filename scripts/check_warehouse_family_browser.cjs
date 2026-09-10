#!/usr/bin/env node
'use strict';

// Real UI only. The operator supplies an admitted release on a separate
// browser_http_qa.sqlite3 service. This script never opens or edits SQLite.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const VERSION = 'warehouse-family-browser-acceptance.v1';
const CAP = Object.freeze({participants: 4, gameplay_steps: 44, ordinary_questions: 4,
  ordinary_answer_replay_upper_bound: 4, total_environment_upper_bound: 48});
const MATRIX = Object.freeze([
  {width:1365,height:900,language:'zh'}, {width:1365,height:900,language:'en'},
  {width:1280,height:800,language:'zh'}, {width:1280,height:800,language:'en'}]);
const QUESTIONS = Object.freeze({zh:'你刚才为什么这样行动？', en:'Why did you just take that action?'});
const PLAN = Object.freeze({version:VERSION, execute:false, caps:CAP, matrix:MATRIX,
  server_requirement:'Genuine qualified family service on a new, independent browser_http_qa.sqlite3; operator verifies release manifest and QA database before execution.',
  workflow:'Four isolated contexts: register, consent, practice 5 actions, Task1 3 rounds and Task2 3 rounds (one action then end each), 8 predictions plus 3 self-ratings, completion.',
  permissions:'Infer A/B from visible Task1 permission; operator separately verifies hidden A/B × XY/YX allocation using read-only QA records.',
  actions:'All progression uses DOM clicks and physical keyboard events. No fetch/API commands, database writes or frontend state injection.',
  failure:'Exclusive output, durable request/response logs, screenshots and storage state; stop on failure, no automatic restart/resume.',
  limitations:['Does not test model ability or explanation effectiveness.',
    'Does not independently verify the private release hash, SQLite identity, NN internals, or actual answer-replay step count.',
    'No process restart, response-loss injection, HTTP authorization matrix or live counterfactual questions; those have separate checks.',
    'A short returned answer cannot establish a genuinely long-answer rendering case.']});
function requireThat(ok, message) { if (!ok) throw new Error(message); }
function sha(raw) { return crypto.createHash('sha256').update(raw).digest('hex'); }
function browserLaunch(bundledExecutable, exists=fs.existsSync) {
  const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const common={policy:'bundled-then-isolated-chrome.v1',headless:true,
    profile:'new_playwright_temporary_profile',connect_to_existing_browser:false,
    bundled_executable:bundledExecutable};
  if(exists(bundledExecutable))return {...common,channel:null,selected_executable:bundledExecutable};
  if(exists(chrome))return {...common,channel:'chrome',selected_executable:chrome};
  throw new Error('No browser executable: install Playwright Chromium or Google Chrome. Existing browser sessions are never connected.');
}
function plannedBrowser() {
  // Reading the installed path does not start a browser or touch a user profile.
  return browserLaunch(require('playwright').chromium.executablePath());
}
function parseArgs(argv) {
  const args={execute:false,base:'http://127.0.0.1:8009',output:null};
  for(let i=0;i<argv.length;i++) {
    const key=argv[i];
    if(key==='--execute') args.execute=true;
    else if(key==='--base'||key==='--output') {
      requireThat(i+1<argv.length && !argv[i+1].startsWith('--'),`Missing value for ${key}`);
      args[key.slice(2)]=argv[++i];
    } else throw new Error(`Unknown argument: ${key}`);
  }
  const url=new URL(args.base);
  requireThat(url.protocol==='http:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname)
    && !url.username && !url.password && url.pathname==='/' && !url.search && !url.hash,
    'Use a loopback HTTP service origin without credentials, paths or queries');
  args.base=url.origin;
  if(args.execute) requireThat(args.output,'--execute requires a new independent --output directory');
  return args;
}
function validateView(v) {
  const r=v?.release;
  requireThat(v?.study_only===true && r?.status==='local_pilot_technically_verified'
    && r.namespace==='local_pilot' && r.model_ready===true && r.explanation_ready===true
    && r.study_ready===true && r.formal_ready===false && r.test_fixture===false
    && v.verification_only===false && !v.study_version_mismatch,
    'The public view is not an admitted genuine family study');
  requireThat(!Object.hasOwn(v.flow,'condition') && !Object.hasOwn(v.flow,'task_order'),
    'Participant view exposes hidden allocation');
  return v;
}
function physical(v) { return JSON.stringify({run_id:v.run_id,state:v.state,metrics:v.metrics,history_count:v.history_count,ended:v.ended}); }
function outputStore(directory) {
  const output=path.resolve(directory);
  fs.mkdirSync(output,{mode:0o700}); // Refuse every previous success, failure or partial run.
  const log=fs.openSync(path.join(output,'events.jsonl'),'wx',0o600);
  function write(name,value) {
    const filename=path.join(output,name), raw=Buffer.from(JSON.stringify(value,null,2)+'\n');
    const fd=fs.openSync(filename,'wx',0o600);
    try {fs.writeFileSync(fd,raw);fs.fsyncSync(fd);} finally {fs.closeSync(fd);}
    const parent=fs.openSync(path.dirname(filename),'r');try{fs.fsyncSync(parent);}finally{fs.closeSync(parent);}
    return {path:filename,sha256:sha(raw),size:raw.length};
  }
  function record(value) { fs.writeSync(log,JSON.stringify({time:new Date().toISOString(),...value})+'\n');fs.fsyncSync(log); }
  return {output,write,record,close:()=>fs.closeSync(log)};
}

async function execute(args) {
  const store=outputStore(args.output), started=new Date().toISOString();
  const counters={action_requests:0,confirmed_gameplay_steps:0,question_requests:0,confirmed_questions:0};
  const participants=[], contexts=[], shots=[], errors=[];
  let browser, launch=null, browserVersion=null, halted=null, closing=false, serial=0;
  const check=(ok,message)=>{requireThat(!halted,halted);requireThat(ok,message);};
  async function screenshot(t,label) {
    const name=`${String(++serial).padStart(3,'0')}_${t.id}_${label}.png`;
    await t.page.screenshot({path:path.join(store.output,name),fullPage:true});
    const raw=fs.readFileSync(path.join(store.output,name));fs.chmodSync(path.join(store.output,name),0o600);
    shots.push({path:name,sha256:sha(raw),participant:t.id,label});
  }
  async function settled(t) {
    check(true,'');
    await t.page.waitForFunction(()=>['已同步','Synced'].includes(document.getElementById('statusText')?.textContent),null,{timeout:20000});
    validateView(t.view);
  }
  async function command(t,kind,gesture,expectedStatus=200) {
    await settled(t);
    const responsePromise=t.page.waitForResponse(r=>{
      if(new URL(r.url()).pathname!=='/api/command' || r.request().method()!=='POST') return false;
      try{return r.request().postDataJSON()?.kind===kind;}catch{return false;}
    },{timeout:25000});
    const [response]=await Promise.all([responsePromise,gesture()]);
    const value=await response.json();
    check(response.status()===expectedStatus,`${kind}: HTTP ${response.status()} ${JSON.stringify(value)}`);
    if(expectedStatus===200) {
      validateView(value);if(!t.view || value.version>=t.view.version)t.view=value;
      await t.page.waitForFunction(v=>Number(document.body.dataset.version)>=v,value.version,{timeout:20000});
      await settled(t);
    } else {
      check(value.error==='participant_id_taken','Unexpected duplicate-ID rejection');
      await t.page.waitForFunction(()=>['此用户 ID 已登记，请换一个编号。','This ID is already registered. Choose another ID.'].includes(document.getElementById('statusText')?.textContent));
    }
    return value;
  }
  async function language(t,lang) {
    const expected=lang==='zh'?'zh-CN':'en';
    if(await t.page.locator('html').getAttribute('lang')!==expected) await t.page.locator('#languageButton').click();
    check(await t.page.locator('html').getAttribute('lang')===expected,'Language did not change');
  }
  async function checkpoint(t,label) {
    await settled(t);await language(t,t.matrix.language);
    const data=await t.page.evaluate(()=>{
      const box=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};};
      return {stage:document.body.dataset.stage,frame:document.body.dataset.frame,viewport:{width:innerWidth,height:innerHeight},
        scroll:{x:scrollX,y:scrollY},horizontalOverflow:document.documentElement.scrollWidth>innerWidth+1,
        canvas:box('warehouseCanvas'),controls:box('operationPanel'),language:document.documentElement.lang};
    });
    check(!data.horizontalOverflow,'Horizontal page overflow');
    if(['practice','task1','task2'].includes(data.stage)) {
      await t.page.evaluate(()=>window.scrollTo(0,0));
      const visible=await t.page.evaluate(()=>['warehouseCanvas','operationPanel'].every(id=>{
        const r=document.getElementById(id).getBoundingClientRect();return r.width>0&&r.height>0&&r.top>=0&&r.bottom<=innerHeight+1;
      }));
      check(visible,'Entire map and principal controls are not visible on the first screen');
    }
    store.record({event:'ui_checkpoint',participant:t.id,label,...data});
    await screenshot(t,label);
    store.write(`${t.id}_${label}_storage.json`,await t.context.storageState());
  }
  async function permission(t,expected) {
    check(await t.page.locator('body').getAttribute('data-explanation-allowed')===String(expected),'Explanation permission flag differs');
    check(await t.page.locator('#explanationPanel').isVisible()===expected,'Explanation panel visibility differs');
    if(!expected) check(await t.page.locator('#answerList .answer').count()===0,'Hidden stage retains rendered answers');
    check(await t.page.locator('#freeplayTab').isVisible()===false && await t.page.locator('#savedRunsField').isVisible()===false,
      'Research entry exposes freeplay or old-run selection');
  }
  async function action(t,gesture,focusCanvas=false) {
    await settled(t);const before=t.view.state.frame,run=t.view.run_id,n=counters.action_requests;
    const canvas=await t.page.locator('#warehouseCanvas').elementHandle();
    if(focusCanvas)await t.page.locator('#warehouseCanvas').click();
    const beforeDOM=await t.page.evaluate(()=>({x:scrollX,y:scrollY}));
    const value=await command(t,'action',gesture);
    check(value.run_id===run && value.state.frame===before+1,'An input did not advance exactly one frame');
    check(counters.action_requests===n+1,'Held/repeated keyboard event sent multiple commands');
    check(await canvas.evaluate(el=>el===document.getElementById('warehouseCanvas')),'Canvas node was replaced');
    const afterDOM=await t.page.evaluate(()=>({x:scrollX,y:scrollY,focus:document.activeElement?.id}));
    check(beforeDOM.x===afterDOM.x && beforeDOM.y===afterDOM.y,'Action moved the page scroll');
    if(focusCanvas)check(afterDOM.focus==='warehouseCanvas','Keyboard action lost map focus');
    t.steps++;check(!value.ended,'The bounded UI route ended before its declared one-step round; stop without reopening');
  }
  async function history(t) {
    const before=physical(t.view),frame=t.view.state.frame;
    await t.page.locator('#previousFrame').click();
    await t.page.waitForFunction(()=>document.body.dataset.replay==='true');
    check(await t.page.locator('[data-action="WAIT"]').isDisabled(),'Replay left actions enabled');
    check(Number(await t.page.locator('body').getAttribute('data-frame'))===frame,'Replay advanced the live frame');
    await t.page.locator('#liveButton').click();
    await t.page.waitForFunction(()=>document.body.dataset.replay==='false');
    check(physical(t.view)===before,'Replay changed public live state');
  }
  async function refresh(t,label) {
    await settled(t);const before=physical(t.view),id=t.view.session_id;
    await t.page.reload({waitUntil:'domcontentloaded'});await settled(t);
    check(t.view.session_id===id && physical(t.view)===before,'Refresh did not preserve the same session and confirmed state');
    store.record({event:'refresh_preserved',participant:t.id,label,session_id:id,frame:t.view.state?.frame});
  }
  async function ask(t,lang) {
    await settled(t);await language(t,lang);const before=physical(t.view),question=QUESTIONS[lang];
    await t.page.locator('#questionFocus').selectOption('executed');
    await t.page.locator('#questionInput').fill(question);
    check(physical(t.view)===before,'Typing in the question input advanced the game');
    await command(t,'question',()=>t.page.locator('#askButton').click());
    const deadline=Date.now()+120000;
    while(!t.view.answers?.some(a=>a.question===question && a.status==='complete' && a.text)) {
      check(Date.now()<deadline,'Ordinary answer did not finish within two minutes');
      check(!t.view.answers?.some(a=>a.question===question && ['failed','expired'].includes(a.status)),'Ordinary answer failed');
      await t.page.waitForTimeout(100);
    }
    await settled(t);
    const answer=t.page.locator('#answerList .answer').filter({hasText:question});
    await answer.waitFor({state:'visible'});
    await answer.scrollIntoViewIfNeeded();
    const text=await answer.innerText();check(text.includes(question) && text.length>question.length+15,'Answer is not fully rendered');
    const layout=await answer.evaluate(el=>({height:el.getBoundingClientRect().height,viewport:innerHeight,textLength:el.innerText.length}));
    store.record({event:'ordinary_answer_read',participant:t.id,question,layout,frame:t.view.state.frame,
      long_answer_case_exercised:layout.height>layout.viewport});
    await screenshot(t,`answer_${lang}`);
    check(physical(t.view)===before,'Ordinary question or reading changed live state');
    await t.page.evaluate(()=>window.scrollTo(0,0));
  }
  try {
    const {chromium}=require('playwright');
    launch=browserLaunch(chromium.executablePath());
    store.write('plan.json',{...PLAN,execute:true,base:args.base,started,browser_launch:launch,
      browser_version:null,browser_version_scope:'Recorded after actual launch in events and result',
      script_sha256:sha(fs.readFileSync(__filename))});
    // launch() creates a fresh temporary profile; never use persistentContext,
    // userDataDir or connectOverCDP to borrow the user's running Chrome session.
    browser=await chromium.launch({headless:true,...(launch.channel?{channel:launch.channel}:{})});
    browserVersion=browser.version();
    store.record({event:'browser_launched',browser_launch:launch,browser_version:browserVersion});
    for(let i=0;i<4;i++) {
      const context=await browser.newContext({viewport:{width:MATRIX[i].width,height:MATRIX[i].height}});
      const id=`qa_browser_${crypto.randomBytes(5).toString('hex')}_${i+1}`;
      const t={id,context,matrix:MATRIX[i],view:null,steps:0,condition:null};contexts.push(t);
      await context.route('**/api/command',async route=>{
        try {
          requireThat(!halted,'Run is stopped');const body=route.request().postDataJSON();
          if(body.kind==='action') {requireThat(counters.action_requests<CAP.gameplay_steps,'Game-step request cap exceeded');counters.action_requests++;}
          if(body.kind==='question') {
            requireThat(counters.question_requests<CAP.ordinary_questions && body.focus==='executed'
              && Object.values(QUESTIONS).includes(body.question),'Question is outside the ordinary-question budget');
            counters.question_requests++;
          }
          store.record({event:'ui_command_request',participant:id,body,counters:{...counters}});
          await route.continue();
        } catch(error) {halted=error.message;store.record({event:'request_rejected_by_local_budget',participant:id,error:error.message});await route.abort();}
      });
      const page=t.page=await context.newPage();page.setDefaultTimeout(20000);
      page.on('pageerror',error=>{errors.push({participant:id,kind:'pageerror',message:error.message});halted=error.message;});
      page.on('console',msg=>{if(msg.type()==='error')errors.push({participant:id,kind:'console',message:msg.text()});});
      page.on('requestfailed',request=>{if(!closing)errors.push({participant:id,kind:'requestfailed',url:request.url(),error:request.failure()});});
      t.responses=new Set();
      page.on('response',response=>{
        const job=(async()=>{
          const pathname=new URL(response.url()).pathname;if(!pathname.startsWith('/api/'))return;
          const raw=await response.text();let value;try{value=JSON.parse(raw);}catch{throw new Error('Non-JSON public API response');}
          const request=response.request();let body=null;try{body=request.postDataJSON();}catch{}
          store.record({event:'public_response',participant:id,url:response.url(),status:response.status(),body,raw});
          if(response.status()===200 && value?.session_id && Number.isInteger(value.version)) {
            validateView(value);if(!t.view || value.version>=t.view.version)t.view=value;
            if(body?.kind==='action')counters.confirmed_gameplay_steps++;
            if(body?.kind==='question')counters.confirmed_questions++;
          }
        })().catch(error=>{if(!closing){halted=error.message;errors.push({participant:id,kind:'response_record',message:error.message});}});
        t.responses.add(job);job.finally(()=>t.responses.delete(job));
      });
      await page.goto(args.base,{waitUntil:'domcontentloaded'});await settled(t);
      check(t.view.flow.mode==='enrollment' && t.view.flow.stage==='registration','A browser context reused an enrolled session');
      await checkpoint(t,'registration');
      if(i===1) {
        await page.locator('#participantInput').fill(contexts[0].id);
        await command(t,'start',()=>page.locator('#startButton').click(),409);
        check(await page.locator('#participantInput').inputValue()===contexts[0].id,'Duplicate-ID error erased typed input');
      }
      await page.locator('#participantInput').fill(id);
      // A corrected ID can be submitted after the expected duplicate error.
      if(i===1)await page.reload({waitUntil:'domcontentloaded'});
      await settled(t);await page.locator('#participantInput').fill(id);
      await command(t,'start',()=>page.locator('#startButton').click());
      check(t.view.flow.stage==='consent','Registration skipped separate consent');
      await checkpoint(t,'consent');
      check(!await page.locator('#consentInput').isChecked(),'Consent was preselected');
      await page.locator('#consentInput').check();
      await command(t,'next',()=>page.locator('#startButton').click());
      check(t.view.flow.stage==='practice','Consent did not start practice');
      await permission(t,false);await checkpoint(t,'practice_before');
      await action(t,async()=>{
        await page.keyboard.down('ArrowUp');
        for(let n=0;n<3;n++){await page.waitForTimeout(120);await page.keyboard.down('ArrowUp');}
        await page.keyboard.up('ArrowUp');
      },true);
      await action(t,()=>page.keyboard.press('a'),true);
      await action(t,()=>page.keyboard.press('Space'),true);
      await action(t,()=>page.locator('[data-action="RIGHT"]').click());
      await page.locator('[data-action="WAIT"]').focus();
      await action(t,()=>page.keyboard.press('Space'));
      await page.locator('#warehouseCanvas').focus();await page.keyboard.press('Tab');
      check(await page.evaluate(()=>document.activeElement!==document.body),'Tab has no visible focus target');
      await history(t);await refresh(t,'practice');await checkpoint(t,'practice_after');
      await command(t,'end',()=>page.locator('#endButton').click());
      await command(t,'next',()=>page.locator('#nextButton').click());
      for(const stage of ['task1','task2']) {
        check(t.view.flow.stage===stage,`Expected ${stage}`);
        if(stage==='task1')t.condition=t.view.explain_allowed?'A':'B';
        await permission(t,stage==='task1'&&t.condition==='A');await checkpoint(t,stage);
        for(let round=1;round<=3;round++) {
          check(t.view.flow.stage===stage && t.view.flow.round_index===round,'UI round order differs');
          await action(t,()=>page.locator('[data-action="WAIT"]').click());
          if(stage==='task1'&&round===1) {
            await history(t);
            if(t.condition==='A')await ask(t,'zh');
          }
          await command(t,'end',()=>page.locator('#endButton').click());
          if(stage==='task1'&&round===1&&t.condition==='A')await ask(t,'en');
          await command(t,'next',()=>page.locator('#nextButton').click());
          if(stage==='task1'&&round===3)await permission(t,false);
        }
      }
      check(t.view.flow.stage==='questionnaire','Six rounds did not reach questionnaire');
      await permission(t,false);await checkpoint(t,'questionnaire');
      const fields=page.locator('#questionnaireFields [data-q-id]');
      check(await fields.count()===11,'Questionnaire is not 8 predictions plus 3 ratings');
      for(const prefix of ['prediction_next_action_','prediction_wait_three_'])
        check(await page.locator(`[data-q-id^="${prefix}"]`).count()===4,'Prediction group is incomplete');
      for(const id of ['cooperation','predictability','difficulty'])check(await page.locator(`[data-q-id="${id}"]`).count()===1,'Self-rating is missing');
      for(const prefix of ['prediction_next_action_','prediction_wait_three_']) {
        await page.locator(`[data-preview-id^="${prefix}"]`).first().click();
        check((await page.locator('#frameNote').innerText()).length>0,'Prediction preview has no frame binding');
        await screenshot(t,`preview_${prefix}`);
      }
      const first=fields.first(),choice=await first.locator('option').nth(1).getAttribute('value');
      const firstId=await first.getAttribute('data-q-id');await first.selectOption(choice);
      await command(t,'questionnaire',()=>page.locator('#saveQuestionnaire').click());
      await refresh(t,'questionnaire_draft');
      check(await page.locator(`[data-q-id="${firstId}"]`).inputValue()===choice,'Saved questionnaire draft was lost');
      for(let j=0;j<await fields.count();j++) {
        const field=fields.nth(j);await field.selectOption(await field.locator('option').nth(1).getAttribute('value'));
      }
      await command(t,'questionnaire',()=>page.locator('#submitQuestionnaire').click());
      check(t.view.flow.stage==='completed','Questionnaire submission did not complete');
      await permission(t,false);await checkpoint(t,'completed');await refresh(t,'completed');
      check(await page.locator('#completedPanel').isVisible() && !await page.locator('#operationPanel').isVisible(),'Completed page still exposes gameplay');
      participants.push({id,session_id:t.view.session_id,observed_condition:t.condition,matrix:t.matrix,
        gameplay_steps:t.steps,stage:t.view.flow.stage,hidden_allocation_requires_operator_read_only_check:true});
    }
    for(const t of contexts)await Promise.all([...t.responses]);
    check(participants.filter(p=>p.observed_condition==='A').length===2 && participants.filter(p=>p.observed_condition==='B').length===2,'Visible A/B allocation is not 2+2');
    check(counters.confirmed_gameplay_steps===44 && counters.action_requests===44 && counters.confirmed_questions===4 && counters.question_requests===4,'Confirmed UI totals differ from bounded plan');
    store.write('report.json',{version:VERSION,status:errors.length?'completed_requires_error_review':'passed_real_browser_ui',started,ended:new Date().toISOString(),
      participants,counters,shots,errors,browser_launch:launch,browser_version:browserVersion,
      qualification_granted:false,actual_answer_environment_steps:'not_observable_in_public_browser_responses',
      environment_steps_upper_bound:48,limitations:PLAN.limitations});
  } catch(error) {
    halted=error.message;store.record({event:'failure',message:error.message,stack:error.stack,counters});
    for(const t of contexts) {
      try{await screenshot(t,'failure');}catch{}
      try{store.write(`${t.id}_failure_storage.json`,await t.context.storageState());}catch{}
    }
    store.write('failure.json',{version:VERSION,started,error:error.message,stack:error.stack,counters,participants,shots,errors,
      browser_launch:launch,browser_version:browserVersion,
      contexts:contexts.map(t=>({id:t.id,matrix:t.matrix,observed_condition:t.condition,confirmed_view:t.view})),
      retry_allowed:false,qualification_granted:false,scope:'Actual operations before failure may already be committed; do not restart this run.'});
    throw error;
  } finally {
    closing=true;if(browser)await browser.close();for(const t of contexts)await Promise.all([...t.responses]);store.close();
  }
  return {output:store.output,...counters};
}

module.exports={VERSION,CAP,MATRIX,PLAN,parseArgs,validateView,physical,browserLaunch,plannedBrowser};
if(require.main===module) {
  (async()=>{const args=parseArgs(process.argv.slice(2));console.log(JSON.stringify(args.execute?await execute(args):
    {...PLAN,browser_launch:plannedBrowser(),browser_version:null},null,2));})()
    .catch(error=>{console.error(error.stack||error.message);process.exitCode=1;});
}
