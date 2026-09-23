'use strict';
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{const b=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
try{for(const [domain,label] of [['warehouse','Warehouse / 体验仓库'],['pong','Pong / 体验接球'],['kitchen','Kitchen / 体验厨房']]){
const p=await b.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));
await p.goto('http://127.0.0.1:9155/try/'+domain);await p.getByRole('button',{name:'Try '+label,exact:true}).click();
await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);
assert(await p.evaluate(()=>view.optional_prompts));
await p.evaluate(async()=>{while(!pendingAutomatic()&&!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})});
let turn=await p.evaluate(()=>view.state.turn);
assert(await p.locator('[data-action="wait"]').isEnabled());
await p.locator('[data-action="wait"]').click();await p.waitForFunction(t=>view.state.turn===t+1&&!busy,turn);
assert((await p.evaluate(()=>view.automatic_explanations)).some(c=>c.skipped&&!c.confirmed));
await p.evaluate(async()=>{while(!pendingUnderstanding()&&!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})});
assert(await p.locator('#understandingDialog').isVisible());turn=await p.evaluate(()=>view.state.turn);
await p.locator('#studyCanvas').focus();await p.keyboard.press('Space');await p.waitForFunction(t=>view.state.turn===t+1&&!busy,turn);
assert.equal(await p.locator('#understandingDialog').count(),0);
await p.reload();await p.waitForFunction(()=>typeof view!=='undefined'&&!!view?.state&&!busy);assert.equal(await p.evaluate(()=>view.state.turn),turn+1);
if(domain==='pong'){
 await p.evaluate(async()=>{while(!view.state.terminal)await command('action',{run_id:view.run_id,turn:view.state.turn,action:'wait'})});
 assert(await p.locator('#understandingDialog').isVisible());
 await p.locator('[data-skip-prompts]').first().click();await p.waitForFunction(()=>!pendingUnderstanding()&&!busy);
 await p.getByRole('heading',{name:'Task 2 demo complete',exact:true}).waitFor();
}
assert.deepEqual(errors,[]);console.log(domain,'unconfirmed prompt: button and keyboard movement, refresh and skip verified');await p.close();
}}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
