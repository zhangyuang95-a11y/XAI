'use strict';
const $=id=>document.getElementById(id), escapeText=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=null,view=null,busy=false,playing=false,token='',board=new PreviewBoard($('board'));
let bindingShown='',demoTimer=null,heldTimer=null,heldKey=null,exporting=false;
const feedbackSamples=[];
const savedKey='policylens-manual-preview-20261024.v1';
const requested=location.pathname.split('/')[1];
if(['warehouse','pong','kitchen'].includes(requested))$('domain').value=requested;
function message(text){$('status').textContent=text;}
function locks(){
  $('start').disabled=busy||!models;
  const d=view?.state.domain,terminal=!view||view.state.terminal;
  for(const b of document.querySelectorAll('[data-action]')){const a=b.dataset.action;b.disabled=busy||playing||terminal||(d==='pong'&&!['left','right','wait'].includes(a))||(d!=='kitchen'&&a==='interact');}
  $('nn-step').disabled=busy||terminal||playing;$('play').disabled=busy||terminal||playing;$('pause').disabled=!playing&&!queue.current&&!queue.items.length;$('export').disabled=busy||exporting||!view;
}
async function api(path,payload,sessionToken=token){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),['models','start'].includes(path)?90000:15000);
  try{
    const response=await fetch('/manual-api/'+path,{signal:controller.signal,method:payload?'POST':'GET',headers:{...(payload?{'Content-Type':'application/json'}:{}),...(sessionToken?{'Authorization':'Bearer '+sessionToken}:{})},...(payload?{body:JSON.stringify(payload)}:{})});
    const result=await response.json();if(!response.ok){const e=new Error(result.error||'请求失败');e.permanent=response.status>=400&&response.status<500;throw e;}return result;
  }finally{clearTimeout(timer);}
}
function save(pending=null){localStorage.setItem(savedKey,JSON.stringify({token,domain:view.state.domain,pending}));}
function inputStatus(){
  $('input-status').textContent=queue.failedItem?'原命令结果尚未确认，后续输入已暂停。':`${queue.current?'等待服务器确认 · ':''}已排队 ${queue.items.length} / 8`;
  if(feedbackSamples.length){const sorted=[...feedbackSamples].sort((a,b)=>a-b);$('input-status').textContent+=` · 输入反馈 P95 ${Math.ceil(sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.95))])}ms（${sorted.length}次）`;}
  $('retry').hidden=!queue.failedItem;locks();
}
function stopPending(){playing=false;clearTimeout(demoTimer);clearTimeout(heldTimer);heldKey=null;return queue.clear();}
function scheduleDemo(started){
  clearTimeout(demoTimer);
  if(playing&&!view.state.terminal)demoTimer=setTimeout(()=>queue.enqueue(null),Math.max(0,300-(performance.now()-started)));
}
const queue=new PreviewInputQueue({changed:inputStatus,full:()=>message('队列已满（8项）；本次输入未接受，请稍后再按。'),failed:e=>{playing=false;clearTimeout(demoTimer);message('操作尚未确认：'+e.message+'。重试将复用原命令编号。');},send:async item=>{
  if(!view||(view.state.terminal&&!item.payload)){stopPending();return;}
  const started=performance.now();
  item.payload??={command_id:crypto.randomUUID(),expected_turn:view.state.turn,human_action:item.action};
  save(item.payload);
  let result;
  for(let attempt=0;attempt<3;attempt++){
    try{result=await api('action',item.payload);break;}
    catch(e){if(e.permanent||attempt===2)throw e;message('连接中断，正在重试同一命令…');}
  }
  view=result;save();render(true);
  if(!view.state.terminal)message(playing?'双 NN 正在运行；可随时暂停。':'服务器已确认操作。蓝色角色由你控制，橙色角色由冻结 NN 控制。');
  scheduleDemo(started);
}});
function action(human,held=false){
  const began=performance.now();
  if(busy||playing||!view||view.state.terminal)return false;
  const accepted=queue.enqueue(human,held);
  if(accepted){$('input-status').dataset.feedbackAt=String(performance.now());
    if(!held){message(`已接受 ${human??'双 NN'}，按顺序执行。`);
      requestAnimationFrame(()=>{feedbackSamples.push(performance.now()-began);if(feedbackSamples.length>60)feedbackSamples.shift();inputStatus();});}}
  return accepted;
}
function choices(){const d=$('domain').value;$('seed').innerHTML=(models.models[d].seeds||[]).map((s,i)=>`<option value="${s}">场景 ${i+1} · ${s}</option>`).join('');}
function render(animate=false){
  if(!view)return;const s=view.state,d=s.domain;$('workspace').hidden=false;
  $('model-label').textContent=`${d.toUpperCase()} · ${models.models[d].version} · ${view.binding.variant}`;
  $('turn').textContent=`步数 ${s.turn} / ${s.max_turns}`;$('score').textContent=`得分 ${Number(s.score?.task_score??0).toFixed(1)}`;
  $('instructions').textContent=d==='pong'?'A / D 或左右方向键移动，空格等待。蓝色挡板由你控制。':d==='kitchen'?'WASD 或方向键移动和转向，E 操作正前方工位，空格等待。先准备原料，再向中央传递口交接。':'WASD 或方向键移动，空格等待。到取货格自动取货，到配送格自动配送；紫色闪电为充电站。';
  $('metrics').textContent=d==='warehouse'?`配送 ${s.deliveries} · 碰撞 ${s.collision_events} · 停机 ${s.shutdown_events}`:d==='kitchen'?`完成订单 ${s.metrics?.completed_orders??s.orders.filter(o=>o.status==='completed').length} / 5 · 你持有 ${s.human.holding?.ingredient||s.human.holding?.stage||'空'}`:`当前球数 ${s.balls?.length??0} · 蓝色挡板位置 ${s.human.x} · 橙色挡板位置 ${s.ai.x}`;
  $('orders').innerHTML=(s.orders||[]).map(o=>`<div class="order"><b>${escapeText(o.id)}</b> · ${escapeText(o.status)} ${d==='warehouse'?`<br>取货 (${escapeText(o.pickup)}) → 配送 (${escapeText(o.dropoff)})`:`<br>${escapeText(o.recipe)} · 截止 ${escapeText(o.deadline)} 步`}</div>`).join('');
  const last=view.last_action;$('last-action').textContent=last?`${last.two_NN?'双 NN 演示':'真人接管'} · 角色1：${last.human} · AI：${last.ai} · AI动作覆盖：${last.ai_action_overrides}`:'当前为第0帧。你可以直接操作，或先播放双 NN。';
  const bindingKey=JSON.stringify(view.binding);if(bindingKey!==bindingShown){$('binding').textContent=JSON.stringify(view.binding,null,2);bindingShown=bindingKey;}
  board.setState(s,{animate,language:'zh'});locks();
  if(s.terminal){const cancelled=stopPending();message(`本局已结束，已清除 ${cancelled} 项待执行输入。可以下载记录，或开始新一局。`);locks();}
}
async function start(event){
  event?.preventDefault();if(busy)return;const cancelled=stopPending();busy=true;locks();message(`正在装载绑定模型与场景…${cancelled?'已清除 '+cancelled+' 项待执行输入。':''}`);
  await queue.idle();
  try{const result=await api('start',{domain:$('domain').value,variant:$('variant').value,task:Number($('task').value),seed:Number($('seed').value)});token=result.session_token;delete result.session_token;view=result;save();render();message('模型已绑定。直接操作蓝色角色，或播放双 NN。');}
  catch(e){message('无法开始：'+e.message);}finally{busy=false;locks();}
}
$('selection').addEventListener('submit',start);$('domain').addEventListener('change',choices);
for(const b of document.querySelectorAll('[data-action]'))b.addEventListener('click',()=>action(b.dataset.action));
$('nn-step').addEventListener('click',()=>action(null));
$('play').addEventListener('click',async()=>{stopPending();playing=true;locks();await queue.idle();clearTimeout(demoTimer);if(playing)queue.enqueue(null);});
$('pause').addEventListener('click',()=>{const n=stopPending();locks();message(`已暂停；已发出的至多一步会完成，清除 ${n} 项待执行输入。`);});
$('retry').addEventListener('click',()=>queue.retry());
for(const id of ['domain','variant','task','seed'])$(id).addEventListener('change',()=>{const n=stopPending();locks();message(`选择已改变，待执行输入已清除（${n}项）；点击“开始新一局”绑定选择。`);});
const keys={w:'up',arrowup:'up',s:'down',arrowdown:'down',a:'left',arrowleft:'left',d:'right',arrowright:'right',' ':'wait',e:'interact'};
function release(){heldKey=null;clearTimeout(heldTimer);queue.release();}
function repeatHeld(){
  if(!heldKey)return;const a=keys[heldKey],button=document.querySelector(`[data-action="${a}"]`);
  if(button&&!button.disabled)action(a,true);heldTimer=setTimeout(repeatHeld,150);
}
document.addEventListener('keydown',event=>{
  if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName))return;
  const key=event.key.toLowerCase(),a=keys[key];if(!a)return;event.preventDefault();if(event.repeat||playing)return;
  const button=document.querySelector(`[data-action="${a}"]`);if(button&&!button.disabled){release();action(a);heldKey=key;heldTimer=setTimeout(repeatHeld,250);}
});
document.addEventListener('keyup',event=>{if(event.key.toLowerCase()===heldKey){event.preventDefault();release();}});
window.addEventListener('blur',release);
$('export').addEventListener('click',async()=>{if(exporting||!view)return;exporting=true;const exportToken=token,domain=view.state.domain,variant=view.binding.variant;locks();message('正在逐帧核对快照回放；可继续操作。');try{const data=await api('export',null,exportToken);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`policylens-${domain}-${variant}-manual.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);message('记录已下载；快照 NN 与物理回放核对通过。');}catch(e){message('导出失败：'+e.message);}finally{exporting=false;locks();}});
const waking=setTimeout(()=>message('免费服务可能正在唤醒，请稍候；首次连接会比后续操作慢。'),3000);
(async()=>{try{models=await api('models');choices();const saved=JSON.parse(localStorage.getItem(savedKey)||'null');if(saved&&(!requested||saved.domain===requested)){token=saved.token;try{view=await api('session');$('domain').value=view.state.domain;$('variant').value=view.binding.variant;$('task').value=String(view.state.task);choices();render();message('已恢复当前试玩会话。');if(saved.pending){queue.items.push({action:saved.pending.human_action,held:false,payload:saved.pending});queue.pump();}}catch{token='';localStorage.removeItem(savedKey);message('旧试玩会话已失效，请开始新一局。');}}else message('三个模型已核验。请选择模型和场景，开始新一局。');}catch(e){message('模型不可用：'+e.message);}finally{clearTimeout(waking);locks();}})();
