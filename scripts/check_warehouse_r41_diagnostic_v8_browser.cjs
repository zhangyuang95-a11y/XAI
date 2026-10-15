#!/usr/bin/env node
'use strict';

/*
 * Real Playwright acceptance for the r4.1-diagnostic v8 participant UI.
 * It uses DOM gestures only for state changes.  SQLite is opened read-only by
 * a small Python subprocess after the browser flow to authenticate the hidden
 * A/B × XY/YX allocation and stored neural-action authority.
 */

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const https = require('node:https');
const {spawnSync} = require('node:child_process');

const VERSION = 'warehouse-r41-diagnostic-v8-playwright-acceptance.v1';
const MATRIX = Object.freeze([
  {width:1365,height:900,language:'zh'},
  {width:1365,height:900,language:'en'},
  {width:1280,height:800,language:'zh'},
  {width:1280,height:800,language:'en'},
]);
const HEX = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z][A-Za-z0-9_-]{2,31}$/;
const COMMAND_PATH = '/api/study/command';
const sha = raw => crypto.createHash('sha256').update(raw).digest('hex');
const requireThat = (ok,message) => { if(!ok) throw new Error(message); };

function parseArgs(argv) {
  const result={execute:false,allowSyntheticFixture:false,base:null,database:null,
    caCertificate:null,expectedPackageSha256:null,expectedManifestSha256:null,
    expectedActorSha256:null,output:null};
  for(let index=0;index<argv.length;index++) {
    const key=argv[index];
    if(key==='--execute') result.execute=true;
    else if(key==='--allow-synthetic-fixture') result.allowSyntheticFixture=true;
    else if(['--base','--database','--ca-certificate','--expected-package-sha256',
      '--expected-manifest-sha256','--expected-actor-sha256','--output'].includes(key)) {
      requireThat(index+1<argv.length && !argv[index+1].startsWith('--'),`Missing value for ${key}`);
      result[key.slice(2).replace(/-([a-z])/g,(_,letter)=>letter.toUpperCase())]=argv[++index];
    } else throw new Error(`Unknown argument: ${key}`);
  }
  for(const key of ['base','database','caCertificate','expectedPackageSha256',
    'expectedManifestSha256','expectedActorSha256','output']) requireThat(result[key],`Missing --${key.replace(/[A-Z]/g,x=>'-'+x.toLowerCase())}`);
  const url=new URL(result.base);
  requireThat(url.protocol==='https:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname)
    && !url.username && !url.password && url.pathname==='/' && !url.search && !url.hash,
    'Use a credential-free loopback HTTPS origin');
  result.base=url.origin;
  for(const key of ['expectedPackageSha256','expectedManifestSha256','expectedActorSha256'])
    requireThat(HEX.test(result[key]),`${key} must be an exact lowercase SHA-256`);
  result.database=path.resolve(result.database);result.caCertificate=path.resolve(result.caCertificate);
  result.output=path.resolve(result.output);
  requireThat(fs.existsSync(result.database) && fs.statSync(result.database).isFile()
    && path.basename(result.database).endsWith('_diagnostic_v8_browser_qa.sqlite3'),
    'Use a separate *_diagnostic_v8_browser_qa.sqlite3 database');
  requireThat(fs.existsSync(result.caCertificate) && fs.statSync(result.caCertificate).isFile(),
    'CA certificate is missing');
  requireThat(result.execute,'Plan only. Add --execute after reviewing all arguments.');
  return result;
}

function writeNew(directory,name,value) {
  const file=path.join(directory,name),raw=Buffer.from(JSON.stringify(value,null,2)+'\n');
  fs.writeFileSync(file,raw,{flag:'wx',mode:0o600});
  return {path:file,sha256:sha(raw),size:raw.length};
}

async function tlsProbe(args) {
  const endpoint=new URL('/health',args.base),ca=fs.readFileSync(args.caCertificate);
  return await new Promise((resolve,reject)=>{
    const request=https.get(endpoint,{ca,rejectUnauthorized:true,minVersion:'TLSv1.2'},response=>{
      const chunks=[];let size=0;const socket=response.socket;
      const protocol=socket?.getProtocol(),peer=socket?.getPeerCertificate();
      response.on('data',chunk=>{size+=chunk.length;if(size>65536)request.destroy(new Error('HTTPS health response is too large'));else chunks.push(chunk);});
      response.on('end',()=>{
        try {
          requireThat(response.statusCode===200,'Authenticated HTTPS health probe failed');
          requireThat(response.headers['cache-control']==='no-store','HTTPS health response is cacheable');
          const value=JSON.parse(Buffer.concat(chunks).toString('utf8'));
          requireThat(value.status==='ok'&&value.release_version==='r4.1-diagnostic'
            && value.pilot_class==='internal_diagnostic'&&value.formal_ready===false
            && value.formal_sample_eligible===false&&value.data_persistent===false,
          'HTTPS health identity differs from diagnostic v8 expectations');
          requireThat(peer&&peer.raw,'HTTPS peer certificate is unavailable');
          resolve({protocol,peer_certificate_sha256:sha(peer.raw),health:value});
        } catch(error) {reject(error);}
      });
    });
    request.setTimeout(15000,()=>request.destroy(new Error('Authenticated HTTPS health probe timed out')));
    request.on('error',reject);
  });
}

function publicOnly(value) {
  const forbidden=new Set(['probabilities','logits','decision','policy_actions','proposed_actions',
    'snapshot','actor_sha256','package_sha256','manifest_sha256','fingerprint','seed']);
  if(Array.isArray(value)) for(const child of value) publicOnly(child);
  else if(value && typeof value==='object') {
    for(const key of Object.keys(value)) requireThat(!forbidden.has(key),'Participant response leaked '+key);
    for(const child of Object.values(value)) publicOnly(child);
  }
}

function validateView(view) {
  publicOnly(view);const release=view?.release || {};
  requireThat(release.release_version==='r4.1-diagnostic'
    && release.pilot_class==='internal_diagnostic'
    && release.model_ready===true && release.explanation_ready===true && release.study_ready===true,
    'Browser reached the wrong release');
  requireThat(release.formal_ready===false && release.formal_sample_eligible===false
    && release.data_persistent===false,'Diagnostic UI overstates eligibility or persistence');
  requireThat(view?.enrollment?.mode==='internal_diagnostic'
    && view.enrollment.formal_sample_eligible===false,'Wrong diagnostic enrollment projection');
  requireThat(!Object.hasOwn(view.flow || {},'condition') && !Object.hasOwn(view.flow || {},'task_order'),
    'Participant UI response exposes allocation');
  return view;
}

function pythonAudit(args,participantIds,empty=false) {
  const program=String.raw`
import json,sqlite3,sys
db_path=sys.argv[1]; package=sys.argv[2]; manifest=sys.argv[3]; actor=sys.argv[4]
ids=json.loads(sys.stdin.read()); db=sqlite3.connect('file:'+db_path+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
meta={r['key']:json.loads(r['value']) for r in db.execute('select key,value from metadata')}
assert meta.get('service_family')=='warehouse_alignment_online_r2' and meta.get('namespace')=='online_diagnostic'
contexts=[v for k,v in meta.items() if k.startswith('service_context:')]
assert len(contexts)==1; ctx=contexts[0]
assert ctx.get('version')=='warehouse-r41-diagnostic-online-release.v8'
assert ctx.get('package_sha256')==package and ctx.get('manifest_sha256')==manifest and ctx.get('actor_sha256')==actor
if ${empty?'True':'False'}:
  assert all(db.execute('select count(*) from '+t).fetchone()[0]==0 for t in ('sessions','runs','frames','questions','operations','events','blocks'))
rows=[]
for participant in ids:
  s=db.execute('select * from sessions where participant_id=?',(participant,)).fetchone()
  if s is None: continue
  runs=[dict(r) for r in db.execute('select rowid,* from runs where session_id=? order by rowid',(s['id'],))]
  authority=True
  for run in runs:
    metrics=json.loads(run['metrics']); authority=authority and metrics.get('overrides')==0
    for frame in db.execute('select internal from frames where run_id=? and frame>0',(run['id'],)):
      x=json.loads(frame[0]);policy=x.get('decision',{}).get('policy_actions',{}).get('robot_2');submitted=x.get('submitted_actions',{}).get('robot_2')
      authority=authority and policy is not None and policy==submitted
  questions=[dict(q) for q in db.execute('select stage,status,shown,answer,evidence_detail from questions where session_id=? order by rowid',(s['id'],))]
  rows.append({'participant_id':participant,'condition':s['condition'],'task_order':s['task_order'],'position':s['position'],'stage':s['stage'],
    'run_scene_indices':[json.loads(r['provenance'])['scene_index'] for r in runs],
    'run_stages':[[r['stage'],r['round_index']] for r in runs],
    'run_count':len(runs),'questions':questions,'action_authority':bool(authority)})
print(json.dumps({'context_sha256':__import__('hashlib').sha256(json.dumps(ctx,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'participants':rows},sort_keys=True))
`;
  const python=process.env.PYTHON || 'python';
  const completed=spawnSync(python,['-c',program,args.database,args.expectedPackageSha256,
    args.expectedManifestSha256,args.expectedActorSha256],{input:JSON.stringify(participantIds),encoding:'utf8'});
  requireThat(completed.status===0,`Read-only database audit failed: ${completed.stderr || completed.stdout}`);
  return JSON.parse(completed.stdout);
}

function launchOptions(chromium) {
  const bundled=chromium.executablePath();
  if(bundled && fs.existsSync(bundled)) return {headless:true,executablePath:bundled,selected:bundled};
  const chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  requireThat(fs.existsSync(chrome),'No isolated Playwright Chromium or Google Chrome installation');
  return {headless:true,channel:'chrome',selected:chrome};
}

function physical(view) { return JSON.stringify({run_id:view?.run_id,state:view?.state,
  metrics:view?.metrics,ended:view?.ended,history_count:view?.history_count,
  stage:view?.flow?.stage,round:view?.flow?.round_index}); }

async function execute(args) {
  const tls=await tlsProbe(args);
  fs.mkdirSync(args.output,{recursive:false,mode:0o700});
  const eventFd=fs.openSync(path.join(args.output,'events.jsonl'),'wx',0o600);
  const record=value=>{fs.writeSync(eventFd,JSON.stringify({time:new Date().toISOString(),...value})+'\n');fs.fsyncSync(eventFd);};
  const initialAudit=pythonAudit(args,[],true);
  let browser=null;const participants=[],screenshots=[],motionReports=[],errors=[];let frontendTiming=null;
  const counters={commands:0,actions:0,questions:0};let serial=0;
  const plan={version:VERSION,base:args.base,database:args.database,
    ca_certificate_sha256:sha(fs.readFileSync(args.caCertificate)),
    expected_package_sha256:args.expectedPackageSha256,
    expected_manifest_sha256:args.expectedManifestSha256,
    expected_actor_sha256:args.expectedActorSha256,matrix:MATRIX,
    synthetic_fixture:args.allowSyntheticFixture,
    release_acceptance_eligible:!args.allowSyntheticFixture,
    source_sha256:sha(fs.readFileSync(__filename)),initial_context_sha256:initialAudit.context_sha256,
    authenticated_tls_probe:tls};
  writeNew(args.output,'plan.json',plan);
  const shot=async(t,label)=>{const name=`${String(++serial).padStart(3,'0')}_${t.id}_${label}.png`;
    await t.page.screenshot({path:path.join(args.output,name),fullPage:true});fs.chmodSync(path.join(args.output,name),0o600);
    const row={participant:t.id,label,file:name,sha256:sha(fs.readFileSync(path.join(args.output,name)))};screenshots.push(row);return row;};
  const apiView=async t=>{const response=await t.context.request.get(args.base+'/api/view');
    requireThat(response.status()===200,'GET /api/view failed');return validateView(await response.json());};
  const settled=async t=>{await t.page.waitForFunction(()=>['已同步','Synced'].includes(document.getElementById('statusText')?.textContent),null,{timeout:30000});
    await t.page.waitForFunction(()=>document.body.dataset.version!==undefined);};
  const command=async(t,kind,gesture)=>{await settled(t);const responsePromise=t.page.waitForResponse(response=>{
      if(new URL(response.url()).pathname!==COMMAND_PATH || response.request().method()!=='POST') return false;
      try{return response.request().postDataJSON()?.kind===kind;}catch{return false;}
    },{timeout:30000});
    const [response]=await Promise.all([responsePromise,gesture()]);const value=await response.json();
    requireThat(response.status()===200,`${kind} failed: HTTP ${response.status()} ${JSON.stringify(value)}`);
    validateView(value);counters.commands++;if(kind==='action')counters.actions++;if(kind==='question')counters.questions++;
    t.view=value;record({event:'command',participant:t.id,kind,version:value.version,stage:value.flow.stage,frame:value.state?.frame});
    await t.page.waitForFunction(version=>Number(document.body.dataset.version)>=version,value.version,{timeout:30000});await settled(t);return value;};
  const geometry=async page=>page.evaluate(()=>{const box=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};};
    return {canvas:box('warehouseCanvas'),controls:box('operationPanel'),scroll:{x:scrollX,y:scrollY},
      viewport:{width:innerWidth,height:innerHeight},documentWidth:document.documentElement.scrollWidth,focus:document.activeElement?.id};});
  const assertLayout=async(t,before=null)=>{if(!before)await t.page.evaluate(()=>window.scrollTo(0,0));const current=await geometry(t.page);requireThat(current.documentWidth<=current.viewport.width+1,'Horizontal overflow');
    record({event:'layout',participant:t.id,stage:await t.page.locator('body').getAttribute('data-stage'),geometry:current});
    if(!before && ['task1','task2'].includes(await t.page.locator('body').getAttribute('data-stage'))) {
      requireThat(current.canvas.y>=0 && current.canvas.bottom<=current.viewport.height+1 && current.controls.bottom<=current.viewport.height+1,
        'Map or primary controls are below the first screen');
    }
    if(before) {for(const area of ['canvas','controls']) for(const key of ['x','y','width','height'])
      requireThat(Math.abs(current[area][key]-before[area][key])<1,`Confirmed action moved ${area}.${key}`);
      requireThat(current.scroll.x===before.scroll.x && current.scroll.y===before.scroll.y,'Confirmed action moved page scroll');}
    return current;};
  const animatedCommand=async(t,kind,gesture)=>{await settled(t);if(kind==='action')await t.page.locator('#warehouseCanvas').evaluate(node=>node.focus({preventScroll:true}));const before=await apiView(t),beforeGeometry=await geometry(t.page);
    const responsePromise=t.page.waitForResponse(response=>{if(new URL(response.url()).pathname!==COMMAND_PATH)return false;
      try{return response.request().postDataJSON()?.kind===kind;}catch{return false;}},{timeout:30000});
    await gesture();const response=await responsePromise;const after=validateView(await response.json());requireThat(response.status()===200,`${kind} failed`);
    counters.commands++;if(kind==='action')counters.actions++;
    const positions=view=>Object.fromEntries((view.state?.agents||[]).map(a=>[a.id,a.position]));const old=positions(before),next=positions(after);
    const moved=Object.keys(next).some(id=>JSON.stringify(old[id])!==JSON.stringify(next[id]));let samples=[],elapsed=0;
    if(moved) {
      await t.page.waitForFunction(()=>document.getElementById('warehouseCanvas')?.dataset.animationRunning==='true',null,{timeout:5000});
      const observed=await t.page.evaluate(()=>new Promise((resolve,reject)=>{const canvas=document.getElementById('warehouseCanvas'),start=performance.now(),rows=[];
        const timer=setTimeout(()=>reject(new Error('animation did not finish')),3000);function tick(){rows.push({time:performance.now(),progress:Number(canvas.dataset.animationProgress),
          robot_1:JSON.parse(canvas.dataset.robot1Position),robot_2:JSON.parse(canvas.dataset.robot2Position)});
          if(canvas.dataset.animationRunning!=='true'){clearTimeout(timer);resolve({rows,elapsed:performance.now()-start});return;}requestAnimationFrame(tick);}requestAnimationFrame(tick);}));
      samples=observed.rows;elapsed=observed.elapsed;
      const interior=samples.filter(row=>row.progress>0 && row.progress<1);requireThat(interior.length>=2,'No continuous intermediate animation frames');
      requireThat(samples.every((row,index)=>index===0 || row.progress+1e-9>=samples[index-1].progress),'Animation progress is not monotone');
      requireThat(elapsed>=250,'380ms transition ended too quickly to be continuous');
      const hasIntermediate=interior.some(row=>Object.keys(next).some(id=>{const value=row[id];return value && JSON.stringify(value)!==JSON.stringify(old[id]) && JSON.stringify(value)!==JSON.stringify(next[id]);}));
      requireThat(hasIntermediate,'Animation jumped between endpoint coordinates');
    }
    t.view=after;await t.page.waitForFunction(version=>Number(document.body.dataset.version)>=version,after.version,{timeout:30000});await settled(t);
    const finalGeometry=await assertLayout(t,beforeGeometry);if(kind==='action')requireThat(finalGeometry.focus==='warehouseCanvas','Keyboard action lost canvas focus');
    const report={participant:t.id,kind,moved,elapsed_ms:elapsed,sample_count:samples.length,
      interior_samples:samples.filter(row=>row.progress>0&&row.progress<1).length,before_frame:before.state?.frame,after_frame:after.state?.frame};
    if(moved)motionReports.push(report);record({event:'animation',...report});return {after,moved};};
  const refresh=async(t,label)=>{await settled(t);const before=physical(await apiView(t));await t.page.reload({waitUntil:'domcontentloaded'});await settled(t);
    t.view=await apiView(t);requireThat(physical(t.view)===before,'Refresh changed confirmed state');record({event:'refresh',participant:t.id,label});};
  const ask=async(t,question,focus)=>{const replay=await t.page.locator('body').getAttribute('data-replay')==='true';const frame=Number(replay?await t.page.locator('#historySlider').inputValue():await t.page.locator('body').getAttribute('data-frame')),run=t.view.run_id,before=physical(await apiView(t));
    await t.page.locator('#questionFocus').selectOption(focus);await t.page.locator('#questionInput').fill(question);
    const value=await command(t,'question',()=>t.page.locator('#askButton').click());
    const candidates=value.answers.filter(a=>a.question===question&&a.frame===frame);const answerId=candidates[candidates.length-1]?.id;requireThat(answerId,'Question response has no frame-bound identity');
    const deadline=Date.now()+60000;let answer;
    while(Date.now()<deadline) {const responsePromise=t.page.waitForResponse(r=>new URL(r.url()).pathname==='/api/view',{timeout:10000});
      await t.page.evaluate(()=>window.dispatchEvent(new Event('online')));await responsePromise;await t.page.waitForTimeout(30);
      const view=await apiView(t);answer=view.answers.find(a=>a.id===answerId);if(answer && !['pending','running'].includes(answer.status))break;}
    requireThat(answer?.status==='complete'&&answer.text.trim(),'Answer did not complete');requireThat(answer.run_id===run&&answer.frame===frame,'Answer rebound to a new frame/run');
    const article=t.page.locator('#answerList .answer').filter({hasText:question}).last();await article.waitFor({state:'visible'});
    const rendered=await article.innerText();requireThat(rendered.includes(answer.text),'Full answer text is not rendered');
    const details=article.locator('details');if(await details.count()) {await details.locator('summary').click();const fit=await details.locator('pre').evaluate(node=>({scrollWidth:node.scrollWidth,clientWidth:node.clientWidth,scrollHeight:node.scrollHeight,clientHeight:node.clientHeight,text:node.textContent}));
      requireThat(fit.text.length===answer.evidence_detail.length && fit.scrollWidth<=fit.clientWidth+1,'Evidence detail is clipped horizontally');}
    requireThat(physical(await apiView(t))===before,'Question/read changed physical state');await shot(t,`answer_${t.matrix.language}_${frame}`);
    return {id:answerId,run_id:run,frame,language:t.matrix.language,text_length:answer.text.length,evidence_length:answer.evidence_detail.length};};
  const endRound=async t=>{if(!t.view.ended)await command(t,'end',()=>t.page.locator('#endButton').click());requireThat(t.view.ended,'Round did not end');};
  const nextRound=async t=>command(t,'next',()=>t.page.locator('#nextButton').click());
  try {
    const {chromium}=require('playwright');const launch=launchOptions(chromium);const selected=launch.selected;delete launch.selected;
    browser=await chromium.launch(launch);record({event:'browser_launched',version:browser.version(),executable:selected});
    const token=crypto.randomBytes(4).toString('hex');
    for(let index=0;index<MATRIX.length;index++) {
      const matrix=MATRIX[index],context=await browser.newContext({viewport:{width:matrix.width,height:matrix.height},ignoreHTTPSErrors:true});
      const id=`qa_ui_v8_${token}_${index+1}`;requireThat(ID.test(id),'Generated participant ID is invalid');
      const t={id,matrix,context,page:await context.newPage(),view:null,answers:[]};participants.push(t);t.page.setDefaultTimeout(30000);
      t.page.on('pageerror',error=>errors.push({participant:id,kind:'pageerror',message:error.message}));
      t.page.on('console',message=>{if(message.type()==='error')errors.push({participant:id,kind:'console',message:message.text()});});
      await t.page.goto(args.base,{waitUntil:'domcontentloaded'});await settled(t);t.view=await apiView(t);
      if(frontendTiming===null) {const asset=await context.request.get(args.base+'/assets/app.js');requireThat(asset.status()===200,'Frontend JavaScript asset is unavailable');
        const source=await asset.text(),match=source.match(/const\s+MOTION_DURATION_MS\s*=\s*(\d+)\s*;/);
        requireThat(match&&Number(match[1])===380,'Frozen frontend does not declare a 380ms confirmed transition');
        frontendTiming={declared_motion_duration_ms:Number(match[1]),app_js_sha256:sha(Buffer.from(source))};}
      requireThat(t.view.flow.stage==='registration','New browser did not enter registration');
      if(matrix.language==='en')await t.page.locator('#languageButton').click();
      requireThat(await t.page.locator('html').getAttribute('lang')===(matrix.language==='zh'?'zh-CN':'en'),'Language switch failed');
      await t.page.locator('#participantInput').fill(id);await t.page.locator('#consentInput').check();
      await command(t,'start',()=>t.page.locator('#startButton').click());requireThat(t.view.flow.stage==='instructions','Registration skipped instructions');
      requireThat(await t.page.locator('[data-question]').count()===6,'Six quick questions are not present');
      await shot(t,'instructions');
      if(index===0) {
        let observed=false;
        while(!t.view.tutorial.complete && !observed) {const result=await animatedCommand(t,'tutorial_advance',()=>t.page.locator('#tutorialNextButton').click());observed=result.moved;}
        requireThat(observed,'Tutorial never produced a continuous move');
        await t.page.locator('#tutorialPlayButton').click();await t.page.waitForFunction(()=>document.body.dataset.tutorialPlaying==='true');
        await t.page.waitForTimeout(80);await t.page.locator('#tutorialPlayButton').click();await t.page.waitForFunction(()=>document.body.dataset.tutorialPlaying==='false');
      } else if(index===1) {
        await command(t,'tutorial_advance',()=>t.page.locator('#tutorialNextButton').click());
        await command(t,'tutorial_restart',()=>t.page.locator('#tutorialRestartButton').click());await refresh(t,'tutorial');
      }
      await command(t,'begin_task1',()=>t.page.locator('#beginTask1Button').click());requireThat(t.view.flow.stage==='task1','Tutorial did not enter Task 1');
      await assertLayout(t);await shot(t,'task1_start');
    }
    const allocation=pythonAudit(args,participants.map(t=>t.id));requireThat(allocation.participants.length===4,'Database missed browser participants');
    requireThat(new Set(allocation.participants.map(x=>x.condition+'/'+x.task_order)).size===4,'Browser participants do not cover four allocation cells');
    for(const t of participants) {const row=allocation.participants.find(x=>x.participant_id===t.id);t.condition=row.condition;t.taskOrder=row.task_order;
      const visible=await t.page.locator('#explanationPanel').isVisible();requireThat(visible===(t.condition==='A'),'Task 1 explanation visibility differs from allocation');}
    const aParticipants=participants.filter(t=>t.condition==='A'),bParticipants=participants.filter(t=>t.condition==='B');
    const primary=aParticipants[0];if(primary) {
      const question=primary.matrix.language==='zh'?'机器人2当前在朝哪个任务前进？':'Which task is Robot 2 moving toward?';
      const frame=primary.view.state.frame,run=primary.view.run_id;await primary.page.locator('#questionFocus').selectOption('next');await primary.page.locator('#questionInput').fill(question);
      const pending=await command(primary,'question',()=>primary.page.locator('#askButton').click());requireThat(pending.answers.some(a=>a.question===question&&a.frame===frame),'Live question was not saved');
      let moved=false;for(const key of ['ArrowRight','ArrowUp','ArrowLeft','ArrowDown']) {const result=await animatedCommand(primary,'action',()=>primary.page.keyboard.press(key));moved=result.moved;if(moved)break;}
      requireThat(moved,'No formal Task 1 action produced continuous motion');
      const answer=await ask(primary,primary.matrix.language==='zh'?'我的上一步动作影响了机器人2吗？':'Did my last action affect Robot 2?','executed');primary.answers.push(answer);
      const original=await apiView(primary);const first=pending.answers.filter(a=>a.question===question&&a.frame===frame).pop();
      const deadline=Date.now()+60000;let completed;
      while(Date.now()<deadline) {const v=await apiView(primary);completed=v.answers.find(a=>a.id===first.id);if(completed&&!['pending','running'].includes(completed.status))break;await primary.page.waitForTimeout(50);}
      requireThat(completed?.status==='complete'&&completed.run_id===run&&completed.frame===frame,'In-flight answer lost original frame while play continued');
      requireThat(original.run_id===run,'Live question changed run');
    }
    for(const t of participants) {
      if(t!==primary) {let moved=false;for(const key of ['ArrowRight','ArrowUp','ArrowLeft','ArrowDown']) {const result=await animatedCommand(t,'action',()=>t.page.keyboard.press(key));moved=result.moved;if(moved)break;}requireThat(moved,'Formal action did not animate');}
      if(t.condition==='A') {
        await t.page.locator('#previousFrame').click();await t.page.waitForFunction(()=>document.body.dataset.replay==='true');
        requireThat(await t.page.locator('[data-action="WAIT"]').isDisabled(),'Replay left actions enabled');
        const q=t.matrix.language==='zh'?'机器人2当前在朝哪个任务前进？':'Which task is Robot 2 moving toward?';t.answers.push(await ask(t,q,'next'));
        await t.page.locator('#liveButton').click();await t.page.waitForFunction(()=>document.body.dataset.replay==='false');
      } else requireThat(!(await t.page.locator('#explanationPanel').isVisible()),'B group sees explanation panel');
      await refresh(t,'task1');await endRound(t);
      if(t.condition==='A') {const q=t.matrix.language==='zh'?'本局结束前，机器人2刚才为什么这样行动？':'Before this round ended, why did Robot 2 take that action?';t.answers.push(await ask(t,q,'executed'));}
      await nextRound(t);await endRound(t);await nextRound(t);await endRound(t);const oldRun=t.view.run_id;await nextRound(t);
      requireThat(t.view.flow.stage==='task2'&&t.view.answers.length===0&&!t.view.explain_allowed,'Task 2 exposed old explanations');
      requireThat(!(await t.page.locator('#explanationPanel').isVisible())&&await t.page.locator('#answerList .answer').count()===0,'Task 2 retained answer DOM');
      await shot(t,'task2');
      await endRound(t);await nextRound(t);await endRound(t);await nextRound(t);await endRound(t);await nextRound(t);
      requireThat(t.view.flow.stage==='questionnaire','Six rounds did not enter questionnaire');
      const fields=t.page.locator('#questionnaireFields [data-q-id]');requireThat(await fields.count()===11,'Questionnaire is not 8+3');
      if(t===participants[0]) {for(let i=0;i<4;i++)await fields.nth(i).selectOption({index:1});await command(t,'questionnaire',()=>t.page.locator('#saveQuestionnaire').click());await refresh(t,'questionnaire_draft');}
      for(let i=0;i<await fields.count();i++)if(await fields.nth(i).inputValue()==='')await fields.nth(i).selectOption({index:1});
      await command(t,'questionnaire',()=>t.page.locator('#submitQuestionnaire').click());requireThat(t.view.flow.stage==='completed','Questionnaire did not complete');await shot(t,'completed');
    }
    const audit=pythonAudit(args,participants.map(t=>t.id));requireThat(new Set(audit.participants.map(x=>x.condition+'/'+x.task_order)).size===4,'Final allocation matrix differs');
    for(const row of audit.participants) {const expected=row.task_order==='XY'?[1,2,3,4,5,6]:[4,5,6,1,2,3];
      requireThat(row.stage==='completed'&&row.run_count===6&&JSON.stringify(row.run_scene_indices)===JSON.stringify(expected),'Saved six-round scene order differs');
      requireThat(row.action_authority,'Stored Actor action was not submitted unchanged');
      requireThat(row.questions.every(q=>q.stage==='task1'&&q.status==='complete'&&q.shown&&q.answer&&q.evidence_detail!==null),'Stored answer is incomplete or outside Task 1');
      requireThat((row.condition==='A')===(row.questions.length>0),'Questions crossed A/B allocation');}
    requireThat(motionReports.some(x=>x.kind==='tutorial_advance'&&x.moved)&&motionReports.some(x=>x.kind==='action'&&x.moved),'Tutorial and formal 380ms motion were not both observed');
    requireThat(errors.length===0,'Browser emitted page or console errors: '+JSON.stringify(errors));
    const report={version:VERSION,status:args.allowSyntheticFixture?'passed_synthetic_browser_harness_validation':'passed_r41_diagnostic_v8_browser_acceptance',
      release_acceptance_eligible:!args.allowSyntheticFixture,synthetic_fixture:args.allowSyntheticFixture,
      matrix:MATRIX,participants:audit.participants.map(x=>({participant_id:x.participant_id,condition:x.condition,task_order:x.task_order})),
      authenticated_tls_probe:tls,frontend_timing:frontendTiming,motion_reports:motionReports,screenshots,counters,errors,checks:['authenticated_loopback_TLS','declared_380ms_and_observed_intermediate_frames','four_isolated_contexts','A_XY_A_YX_B_XY_B_YX','zh_en','tutorial_play_pause_restart_skip_refresh',
        'task1_live_historical_post_round_questions','B_hidden','task2_old_and_new_explanations_hidden','380ms_intermediate_tutorial_and_task_motion',
        'stable_canvas_controls_scroll_focus','refresh_restore','six_rounds','questionnaire_draft_refresh','NN_action_authority']};
    writeNew(args.output,'report.json',report);console.log(JSON.stringify({status:report.status,output:args.output,cells:report.participants.map(x=>[x.condition,x.task_order]),motions:motionReports.length,screenshots:screenshots.length}));
    return report;
  } catch(error) {
    writeNew(args.output,'failure.json',{version:VERSION,error:String(error.stack||error),counters,motionReports,screenshots,errors,release_acceptance_eligible:false});throw error;
  } finally {
    if(browser)await browser.close();fs.closeSync(eventFd);
  }
}

let args;
try {args=parseArgs(process.argv.slice(2));}
catch(error) {console.error(error.message);process.exitCode=2;}
if(args)execute(args).catch(error=>{console.error(error.stack||error);process.exitCode=1;});
