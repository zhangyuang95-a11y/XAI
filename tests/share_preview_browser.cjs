'use strict';
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const base=process.env.STUDY_SMOKE_URL||'http://127.0.0.1:9131';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try{
  const context=await browser.newContext(),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const saved={};
  for(const domain of ['pong','warehouse','kitchen']){
   await page.goto(base+'/try/'+domain);
   if(domain==='warehouse')await page.locator('#demoLanguage').selectOption('zh');
   await page.locator(`[data-game="${domain}"]`).click();
   await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy&&!animationPending);
   const v=await page.evaluate(()=>({stage:view.stage,mode:view.mode,group:view.group,id:view.instance_id,turn:view.state.turn,runs:view.task_runs}));
   if(domain==='warehouse')assert.equal(await page.evaluate(()=>lang),'zh');
   assert.equal(v.stage,'task2');assert.equal(v.mode,'preview');assert.equal(v.group,'A');
   assert(v.runs.every(r=>r.task===2));assert.equal(await page.locator('#questionInput').count(),0);
   await page.keyboard.press('Escape');assert.equal(await page.locator('#automaticDialog').evaluate(e=>e.open),true);
   assert.equal(v.turn,0);assert.equal(await page.locator('#understoodExplanationButton').count(),0);
   assert.equal(await page.locator('#confirmExplanationButton').count(),0);
   await page.locator('#requestExplanationButton').click();
   await page.waitForFunction(()=>!busy&&pendingAutomatic()?.requested);
   assert(await page.locator('#automaticReason').innerText());
   await page.reload();await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy);
   assert.equal(await page.locator('#requestExplanationButton').count(),0);
   await page.locator('#confirmExplanationButton').click();await page.waitForFunction(()=>!busy&&!pendingAutomatic());
   assert.equal(await page.evaluate(()=>view.state.turn),v.turn);
   saved[domain]=v.id;
   console.log(domain,'Task 2 guided question at turn zero, answer persistence and confirmation verified');
   if(domain==='kitchen'){
    assert.equal(await page.getByRole('heading',{name:'Menu · 4 dishes',exact:true}).count(),1);
    await page.evaluate(async()=>{for(let i=0;i<5;i++)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});});
    await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy);
    assert.equal(await page.locator('#understoodExplanationButton').count(),1);
    assert.equal(await page.locator('#confirmExplanationButton').count(),0);
    await page.locator('#understoodExplanationButton').click();await page.waitForFunction(()=>!busy&&!pendingAutomatic());
    assert.equal(await page.evaluate(()=>view.automatic_explanations.at(-1).displayed),false);
    await page.evaluate(async()=>{for(let i=0;i<5;i++)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});});
    await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy);
    await page.locator('#requestExplanationButton').click();await page.waitForFunction(()=>!busy&&pendingAutomatic()?.requested);
    await page.locator('#confirmExplanationButton').click();await page.waitForFunction(()=>!busy&&!pendingAutomatic());
    assert.equal(await page.evaluate(()=>view.state.turn),10);
    console.log('Kitchen later-node skip and optional explanation paths verified');
   }
  }
  await page.goto(base+'/try/pong');await page.locator('[data-game="pong"]').click();
  await page.waitForURL(base+'/pong/');await page.waitForFunction(()=>typeof view!=='undefined'&&!!view?.instance_id&&!busy);
  assert.equal(await page.evaluate(()=>view.instance_id),saved.pong);
  const stranger=await browser.newContext();
  const denied=await stranger.request.get(base+'/api/study/view?instance_id='+saved.pong);assert.equal(denied.status(),401);
  for(const route of ['/api/study/admin/export','/api/prolific/admin/payments'])assert.equal((await context.request.get(base+route)).status(),404);
  for(const route of ['/api/study/session','/api/prolific/enrol','/api/study/ask','/api/study/next'])assert.equal((await context.request.post(base+route,{data:{}})).status(),403);
  assert.equal((await context.request.post(base+'/api/try/start',{headers:{Origin:'https://other.example'},data:{domain:'pong'}})).status(),403);
  // Complete one disposable game and verify that no Task 3 / research survey follows.
  await page.evaluate(async()=>{
   for(let i=0;i<200&&view.run_status!=='completed';i++){
    const card=pendingAutomatic();
    if(card)await command('confirm_explanation',{explanation_id:card.id,...(card.guided?{choice:card.requested?'explanation':'understood'}:{})});
    else await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});
   }
  });
  await page.getByRole('heading',{name:'Task 2 demo complete',exact:true}).waitFor({state:'visible'});
  assert.equal(await page.locator('#nextButton').count(),0);
  assert.deepEqual(errors,[]);
  console.log('Same-browser game switching, session isolation, restricted endpoints and demo completion verified');
  await stranger.close();await context.close();
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
