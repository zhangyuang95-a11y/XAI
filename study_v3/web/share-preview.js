'use strict';
const selected=location.pathname.split('/').filter(Boolean)[1];
if(selected)document.querySelector(`[data-game="${CSS.escape(selected)}"]`)?.focus();
for(const button of document.querySelectorAll('[data-game]'))button.addEventListener('click',async()=>{
 const buttons=[...document.querySelectorAll('[data-game]')],status=document.getElementById('previewStatus');
 buttons.forEach(b=>b.disabled=true);status.textContent='Preparing your Task 2 preview… / 正在准备体验…';
 try{
  const r=await fetch('/api/try/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:button.dataset.game,language:document.getElementById('demoLanguage').value,restart:new URLSearchParams(location.search).get('restart')==='1'})});
  const result=await r.json();if(!r.ok)throw new Error(result.error);
  location.assign(result.url);
 }catch(e){status.textContent='Could not start the demo. Please try again. / 暂时无法启动，请重试。';buttons.forEach(b=>b.disabled=false);}
});
