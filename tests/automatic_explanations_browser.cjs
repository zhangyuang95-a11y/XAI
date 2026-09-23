'use strict';
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base=process.env.STUDY_SMOKE_URL||'http://127.0.0.1:9130';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 fs.mkdirSync('output/automatic-preview',{recursive:true});
 try{
  for(const domain of ['kitchen','pong','warehouse']){
   const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(base+'/auto-preview/'+domain);
   await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy&&!animationPending);
   const before=await page.evaluate(()=>({turn:view.state.turn,cards:view.automatic_explanations.length,id:pendingAutomatic().id}));
   if(domain==='pong'){
    for(const viewport of [{width:1440,height:1000},{width:800,height:825},{width:390,height:844}]){
     await page.setViewportSize(viewport);
     await page.reload();await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy&&!animationPending);
     const bounds=await page.evaluate(()=>{
      const rect=id=>{const r=document.querySelector(id).getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right}};
      return {dialog:rect('#automaticDialog'),court:rect('#studyCanvas'),dock:rect('#pongExplanationDock'),replay:rect('.replay-panel'),height:innerHeight,width:innerWidth};
     });
     assert(bounds.dialog.top>bounds.court.bottom,'Pong explanation stays below court');
     assert(bounds.dialog.bottom<=bounds.dock.bottom,'Dock reserves full explanation height');
     assert(bounds.dialog.bottom<bounds.replay.top,'Replay is not covered');
     assert(bounds.dialog.left>=0&&bounds.dialog.right<=bounds.width,'Bubble fits viewport width');
     assert(bounds.dialog.top>=0&&bounds.dialog.bottom<=bounds.height,'Confirmation is visible');
     await page.screenshot({path:`output/automatic-preview/pong-docked-${viewport.width}.png`});
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.reload();await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy&&!animationPending);
   }
   assert.match(await page.locator('#automaticDialog').innerText(),/game is paused/i);
   assert.equal(await page.locator('[data-action]').first().isDisabled(),true);
   await page.keyboard.press('ArrowRight');await page.keyboard.press('Escape');
   assert.equal(await page.evaluate(()=>view.state.turn),before.turn);
   assert.equal(await page.locator('#automaticDialog').evaluate(el=>el.open),true);
   // Client controls and direct server actions both require an explicit confirmation.
   const denied=await page.evaluate(async()=>{
    const r=await fetch('/api/study/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({instance_id:view.instance_id,revision:view.revision,command_id:crypto.randomUUID(),run_id:view.run_id,turn:view.state.turn,action:'wait'})});
    return {status:r.status,...await r.json()};
   });
   assert.equal(denied.status,409);assert.equal(denied.error,'explanation_confirmation_required');
   await page.screenshot({path:`output/automatic-preview/${domain}-confirm.png`});
   await page.reload();await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open&&!busy);
   assert.equal(await page.evaluate(()=>pendingAutomatic().id),before.id);
   // Verify a failed confirmation leaves the modal locked and lets the user retry.
   if(domain==='kitchen'){
    await page.route('**/api/study/confirm_explanation',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"server_error"}'}),{times:1});
    await page.locator('#confirmExplanationButton').click();await page.waitForFunction(()=>!busy);
    assert.equal(await page.locator('#automaticDialog').evaluate(el=>el.open),true);
   }
   await page.locator('#confirmExplanationButton').click();await page.waitForFunction(()=>!pendingAutomatic()&&!busy);
   assert.equal(await page.locator('#automaticDialog').count(),0);
   assert.equal(await page.evaluate(()=>view.state.turn),before.turn);
   await page.reload();await page.waitForFunction(()=>!!view?.state&&!busy);
   assert.equal(await page.locator('#automaticDialog').count(),0);
   const repetitions=domain==='kitchen'?5:1;
   for(let i=0;i<repetitions;i++){
    await page.locator('[data-action="wait"]').click();await page.waitForFunction(()=>!busy&&!animationPending);
   }
   assert.equal(await page.evaluate(()=>view.state.turn),before.turn+repetitions);
   if(domain==='kitchen'){
    await page.waitForFunction(()=>document.querySelector('#automaticDialog')?.open);
    assert.equal(await page.evaluate(()=>pendingAutomatic().turn),before.turn+5);
    const bounds=await page.locator('#automaticDialog').boundingBox();assert(bounds.x>=0&&bounds.y>=0&&bounds.y+bounds.height<=1000);
   }
   assert.deepEqual(errors,[]);console.log(domain,'modal visible, Escape/movement blocked, API gate enforced, refresh persistent, explicit confirmation unlocks');
   await page.close();
  }
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
