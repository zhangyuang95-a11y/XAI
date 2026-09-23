'use strict';
const {chromium}=require('playwright');const assert=require('node:assert/strict');
const base=process.env.STUDY_SMOKE_URL||'http://127.0.0.1:9155';
(async()=>{const b=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
try{const p=await b.newPage(),errors=[];p.on('pageerror',e=>errors.push(e.message));
await p.goto(base+'/try/pong');await p.getByRole('button',{name:'Try Pong / 体验接球',exact:true}).click();
await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);
for(const checkpoint of [20,40,60,80,100]){
 await p.evaluate(async()=>{for(let i=0;i<200&&!pendingUnderstanding();i++){
  const card=pendingAutomatic();
  if(card){if(card.onboarding&&!card.requested)await command('request_explanation',{explanation_id:card.id,question_id:'why'});
   else await command('confirm_explanation',{explanation_id:card.id,choice:card.requested?'explanation':'understood'});
  }else if(!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'});
  else throw Error('Reached end without required rating');
 }});
 await p.waitForFunction(()=>!!document.querySelector('#understandingDialog')&&!busy);
 assert.equal(await p.evaluate(()=>pendingUnderstanding().checkpoint),checkpoint);
 const turn=await p.evaluate(()=>view.state.turn);assert.equal(turn,checkpoint*90/100);
 assert.equal(await p.locator('#understandingForm input:checked').count(),0);assert(await p.locator('#submitUnderstanding').isDisabled());
 await p.keyboard.press('Escape');await p.keyboard.press('Space');await p.keyboard.press('a');
 assert.equal(await p.evaluate(()=>view.state.turn),turn);assert(await p.locator('#understandingDialog').isVisible());assert.equal(await p.locator('dialog[open]').count(),0);
 if(checkpoint===20){
  await p.reload();await p.waitForFunction(()=>!!document.querySelector('#understandingDialog')&&!busy);
  assert.equal(await p.evaluate(()=>view.state.turn),turn);
  await p.screenshot({path:'output/understanding-ratings/pong-en.png'});
  await p.evaluate(()=>{lang='zh';return command('language',{language:'zh'})});
  assert((await p.locator('#understandingTitle').innerText()).includes('为什么'));
  await p.setViewportSize({width:390,height:844});
  await p.screenshot({path:'output/understanding-ratings/pong-zh-mobile.png'});
  assert(await p.locator('#understandingDialog').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
  await p.setViewportSize({width:1280,height:900});
  await p.evaluate(()=>{lang='en';return command('language',{language:'en'})});
 }
 await p.locator('#understandingForm input[value="4"]').check();await p.locator('#submitUnderstanding').click();
 await p.waitForFunction(()=>!busy&&!pendingUnderstanding());assert.equal(await p.evaluate(()=>view.state.turn),turn);
 console.log('Rating checkpoint',checkpoint,'%, persistence, required selection and continuation passed');
}
await p.getByRole('heading',{name:'Task 2 demo complete',exact:true}).waitFor();
assert.deepEqual(errors,[]);
console.log('Final rating precedes completion; no browser errors');
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
