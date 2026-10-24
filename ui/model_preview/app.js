'use strict';
const $=id=>document.getElementById(id), escapeText=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=null,view=null,busy=false,playing=false,token='',board=new StudyBoard($('board'));
const savedKey='policylens-manual-preview-20261024.v1';
const requested=location.pathname.split('/')[1];
if(['warehouse','pong','kitchen'].includes(requested))$('domain').value=requested;
function message(text){$('status').textContent=text;}
function locks(){
  $('start').disabled=busy||!models;
  const d=view?.state.domain,terminal=!view||view.state.terminal;
  for(const b of document.querySelectorAll('[data-action]')){const a=b.dataset.action;b.disabled=busy||playing||terminal||(d==='pong'&&!['left','right','wait'].includes(a))||(d!=='kitchen'&&a==='interact');}
  $('nn-step').disabled=busy||terminal||playing;$('play').disabled=busy||terminal||playing;$('pause').disabled=!playing;$('export').disabled=busy||!view;
}
async function api(path,payload){
  const response=await fetch('/manual-api/'+path,{method:payload?'POST':'GET',headers:{...(payload?{'Content-Type':'application/json'}:{}),...(token?{'Authorization':'Bearer '+token}:{})},...(payload?{body:JSON.stringify(payload)}:{})});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'请求失败');return result;
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
  $('binding').textContent=JSON.stringify(view.binding,null,2);
  board.setState(s,{animate,language:'zh'});locks();
  if(s.terminal){playing=false;message('本局已结束。可以下载记录，或开始新一局。');locks();}
}
async function start(event){
  event?.preventDefault();if(busy)return;playing=false;busy=true;locks();message('正在装载绑定模型与场景…');
  try{const result=await api('start',{domain:$('domain').value,variant:$('variant').value,task:Number($('task').value),seed:Number($('seed').value)});token=result.session_token;delete result.session_token;view=result;localStorage.setItem(savedKey,JSON.stringify({token,domain:view.state.domain}));render();message('模型已绑定。直接操作蓝色角色，或播放双 NN。');}
  catch(e){message('无法开始：'+e.message);}finally{busy=false;locks();}
}
async function action(human){
  if(busy||!view||view.state.terminal)return;busy=true;locks();
  try{view=await api('action',{command_id:crypto.randomUUID(),expected_turn:view.state.turn,human_action:human});render(true);if(!view.state.terminal)message(playing?'双 NN 正在运行；点击暂停后即可接管蓝色角色。':'你控制蓝色角色，橙色 AI 始终由绑定 NN 决定动作。');}
  catch(e){playing=false;message('操作未提交：'+e.message);try{view=await api('session');render();}catch{}}
  finally{busy=false;locks();}
}
$('selection').addEventListener('submit',start);$('domain').addEventListener('change',choices);
for(const b of document.querySelectorAll('[data-action]'))b.addEventListener('click',()=>action(b.dataset.action));
$('nn-step').addEventListener('click',()=>action(null));
$('play').addEventListener('click',()=>{playing=true;locks();action(null);});
$('pause').addEventListener('click',()=>{playing=false;locks();message('已暂停。下一次按键将由你控制蓝色角色。');});
setInterval(()=>{if(playing&&!busy)action(null);},650);
document.addEventListener('keydown',event=>{if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName))return;const key=event.key.toLowerCase(),a=({w:'up',arrowup:'up',s:'down',arrowdown:'down',a:'left',arrowleft:'left',d:'right',arrowright:'right',' ':'wait',e:'interact'})[key];if(!a)return;event.preventDefault();if(playing)return;const button=document.querySelector(`[data-action="${a}"]`);if(button&&!button.disabled)action(a);});
$('export').addEventListener('click',async()=>{if(busy)return;playing=false;busy=true;locks();message('正在逐帧核对本局 NN 与物理回放…');try{const data=await api('export');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`policylens-${view.state.domain}-${view.binding.variant}-manual.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);message('记录已下载；NN 与物理回放核对通过。');}catch(e){message('导出失败：'+e.message);}finally{busy=false;locks();}});
(async()=>{try{models=await api('models');choices();const saved=JSON.parse(localStorage.getItem(savedKey)||'null');if(saved&&(!requested||saved.domain===requested)){token=saved.token;try{view=await api('session');$('domain').value=view.state.domain;$('variant').value=view.binding.variant;$('task').value=String(view.state.task);choices();render();message('已恢复当前试玩会话。');}catch{token='';localStorage.removeItem(savedKey);message('旧试玩会话已失效，请开始新一局。');}}else message('三个模型已核验。请选择模型和场景，开始新一局。');}catch(e){message('模型不可用：'+e.message);}finally{locks();}})();
