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
   const page=await browser.newPage({viewport:{width:1440,height:1050}});const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(base+'/auto-preview/'+domain);
   await page.locator('#automaticExplanation').waitFor();
   await page.waitForFunction(()=>view?.can_ask&&!busy&&!animationPending);
   const before=await page.evaluate(()=>({turn:view.state.turn,cards:view.automatic_explanations.length,id:view.automatic_explanations.at(-1).id}));
   await page.waitForFunction(()=>automaticAcknowledged.has(latestAutomatic().id)||latestAutomatic().displayed);
   assert.match(await page.locator('#automaticExplanation').innerText(),/task 2/i);
   await page.evaluate(()=>window.scrollTo(0,500));
   const rect=await page.locator('#automaticExplanation').boundingBox();
   assert(rect.y>=76&&rect.y<120,'Explanation stays visible below the header while playing');
   await page.evaluate(()=>window.scrollTo(0,0));
   await page.screenshot({path:`output/automatic-preview/${domain}.png`,fullPage:true});
   await page.reload();await page.locator('#automaticExplanation').waitFor();
   const after=await page.evaluate(()=>({turn:view.state.turn,cards:view.automatic_explanations.length,id:view.automatic_explanations.at(-1).id}));
   assert.deepEqual(after,before);
   await page.locator('#chineseButton').click();await page.waitForFunction(()=>lang==='zh'&&!busy);
   assert((await page.locator('#automaticExplanation').innerText()).includes('AI 队友解释'));
   await page.locator('#englishButton').click();await page.waitForFunction(()=>lang==='en'&&!busy);
   await page.locator('#replaySlider').evaluate(el=>{el.value='0';el.dispatchEvent(new Event('change',{bubbles:true}));});
   await page.waitForFunction(()=>!!replay&&!replayLoading);
   assert.equal(await page.locator('#automaticExplanation').count(),0);
   await page.locator('#currentButton').click();await page.locator('#automaticExplanation').waitFor();
   if(domain==='kitchen'){
    for(let i=0;i<5;i++){
     await page.locator('[data-action="wait"]').click();
     await page.waitForFunction(()=>!busy&&!animationPending);
    }
    assert.equal(await page.evaluate(()=>view.automatic_explanations.at(-1).turn),before.turn+5);
   }
   assert.deepEqual(errors,[]);console.log(domain, 'visible, exposure acknowledged, refresh deduplicated, bilingual, replay isolated');
   await page.close();
  }
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
