'use strict';
const {chromium}=require('playwright'),assert=require('node:assert/strict');
const base=process.env.STUDY_SMOKE_URL||'http://127.0.0.1:9155';
(async()=>{const b=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
try{for(const [domain,label] of [['warehouse','Warehouse / 体验仓库'],['pong','Pong / 体验接球'],['kitchen','Kitchen / 体验厨房']]){
 const p=await b.newPage({viewport:{width:1440,height:1080}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.goto(base+'/try/'+domain);await p.getByRole('button',{name:'Try '+label,exact:true}).click();
 await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);
 await p.evaluate(async()=>{while(!pendingAutomatic()&&!view.state.terminal){
  if(pendingUnderstanding())await command('understanding_rating',{rating_id:pendingUnderstanding().id,rating:3});
  else await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});
 }});
 assert(await p.locator('#aiQuestionButton').evaluate(e=>e.classList.contains('question-guide-target')));
 assert.equal(await p.locator('dialog').count(),0);
 const turn=await p.evaluate(()=>view.state.turn);if(domain==='warehouse'){
  assert((await p.evaluate(()=>pendingAutomatic().trigger_types)).includes('collision_risk'));
  assert(!(await p.evaluate(()=>view.state.events||[])).some(e=>e.type==='collision'));
 }
 await p.locator('#aiQuestionButton').click();await p.waitForFunction(()=>!!questionGuide()?.requested&&!busy);
 await p.locator('#guidedAnswer').waitFor();assert.equal(await p.evaluate(()=>view.state.turn),turn);
 assert(await p.locator('#chatPanel').evaluate(e=>!!e.closest('aside')));
 await p.screenshot({path:'output/sidebar-checks/'+domain+'-first-answer.png',fullPage:true});
 await p.locator('#confirmExplanationButton').click();await p.waitForFunction(()=>!pendingAutomatic()&&!busy);
 await p.evaluate(async()=>{for(let i=0;i<200&&!pendingAutomatic()&&!view.state.terminal;i++){
  if(pendingUnderstanding())await command('understanding_rating',{rating_id:pendingUnderstanding().id,rating:3});
  else await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});
 }});
 if(await p.evaluate(()=>!!pendingAutomatic())){
  if(await p.evaluate(()=>!!pendingUnderstanding()))await p.evaluate(()=>command('understanding_rating',{rating_id:pendingUnderstanding().id,rating:3}));
  await p.locator('#automaticDialog').waitFor();assert(await p.locator('#automaticDialog').evaluate(e=>!!e.closest('aside')));
  const board=await p.locator('#studyCanvas').boundingBox(),panel=await p.locator('#automaticDialog').boundingBox();assert(panel.x>=board.x+board.width);
  const t=await p.evaluate(()=>view.state.turn);await p.keyboard.press('Space');await p.keyboard.press('d');assert.equal(await p.evaluate(()=>view.state.turn),t);
  await p.screenshot({path:'output/sidebar-checks/'+domain+'-choice.png',fullPage:true});
  await p.locator('#requestExplanationButton').click();await p.waitForFunction(()=>!!pendingAutomatic()?.requested&&!busy);
  assert((await p.locator('#automaticReason').innerText()).length>10);
  await p.locator('#confirmExplanationButton').click();await p.waitForFunction(()=>!pendingAutomatic()&&!busy);
 }
 assert.deepEqual(errors,[]);console.log(domain,'question-mark onboarding, early trigger and inline choices passed');await p.close();
}}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
