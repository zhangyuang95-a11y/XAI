'use strict';
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
try{for(const [domain,name,bounds] of [['warehouse','Warehouse / 体验仓库','0 to 100'],['pong','Pong / 体验接球','45 to 65'],['kitchen','Kitchen / 体验厨房','60 to 120']]){
 const p=await browser.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.goto('http://127.0.0.1:9155/try/'+domain);await p.getByRole('button',{name:'Try '+name,exact:true}).click();
 const panel=p.locator('.reward-notice');await panel.waitFor();let text=await panel.innerText();
 assert(text.includes('£2.80')&&text.includes('£3.40')&&text.includes(bounds)&&text.includes('unpaid'));
 await p.screenshot({path:'output/performance-rewards/'+domain+'.png',fullPage:true});
 await p.reload();await panel.waitFor();assert((await panel.innerText()).includes(bounds));
 await p.evaluate(()=>{lang='zh';return command('language',{language:'zh'})});assert((await panel.innerText()).includes('此体验不支付报酬'));
 await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 assert.deepEqual(errors,[]);console.log(domain+' reward rules, reload, language and mobile passed');await p.close();
}}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
