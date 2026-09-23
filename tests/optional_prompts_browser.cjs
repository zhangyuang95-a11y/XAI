'use strict';
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{const b=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
try{for(const [domain,label] of [['warehouse','Warehouse / 体验仓库'],['pong','Pong / 体验接球'],['kitchen','Kitchen / 体验厨房']]){
const p=await b.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));
await p.goto('http://127.0.0.1:9155/try/'+domain);await p.getByRole('button',{name:'Try '+label,exact:true}).click();
await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);
assert(await p.evaluate(()=>view.optional_prompts));
await p.evaluate(async()=>{while(!pendingAutomatic()&&!pendingUnderstanding()&&!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})});
await p.evaluate(async()=>{const g=questionGuide();if(g?.four_step){await command('request_explanation',{explanation_id:g.id,question_id:'why'});await command('confirm_explanation',{explanation_id:g.id,choice:'explanation'});while(!pendingAutomatic()&&!pendingUnderstanding()&&!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})}});
let turn=await p.evaluate(()=>view.state.turn);
if(await p.evaluate(()=>!!pendingAutomatic()&&!pendingUnderstanding())){
assert(await p.locator('[data-action="wait"]').isEnabled());
await p.locator('[data-action="wait"]').click();await p.waitForFunction(t=>view.state.turn===t+1&&!busy,turn);
assert((await p.evaluate(()=>view.automatic_explanations)).some(c=>c.skipped&&!c.confirmed));
}
await p.evaluate(async()=>{while(!pendingUnderstanding()&&!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})});
assert(await p.locator('#understandingDialog').isVisible());turn=await p.evaluate(()=>view.state.turn);
assert(await p.locator('[data-action="wait"]').isDisabled());
assert.equal(await p.locator('[data-skip-prompts]').count(),0);
await p.locator('#studyCanvas').focus();await p.keyboard.press('Space');assert.equal(await p.evaluate(()=>view.state.turn),turn);
await p.reload();await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);assert.equal(await p.evaluate(()=>view.state.turn),turn);
assert(await p.locator('#understandingDialog').isVisible());assert(await p.locator('#submitUnderstanding').isDisabled());
await p.locator('#understandingForm input[value="3"]').check();await p.locator('#submitUnderstanding').click();await p.waitForFunction(()=>!pendingUnderstanding()&&!busy);
assert(await p.locator('[data-action="wait"]').isEnabled());
await p.locator('[data-action="wait"]').click();await p.waitForFunction(t=>view.state.turn===t+1&&!busy,turn);
assert.deepEqual(errors,[]);console.log(domain,'optional explanation and mandatory rating verified, including refresh and resume');await p.close();
}}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
