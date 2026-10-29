'use strict';
const $=id=>document.getElementById(id), escapeText=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=null,view=null,course=null,busy=false,playing=false,token='',board=new PreviewBoard($('board'));
let bindingShown='',ordersShown='',demoTimer=null,heldTimer=null,heldKey=null,exporting=false,terminalAnimating=false,renderRevision=0;
const feedbackSamples=[],savedKey='policylens-manual-preview-20261024.v1';
const requested=location.pathname.split('/')[1];
if(['warehouse','pong','kitchen'].includes(requested))$('domain').value=requested;
function message(text){$('status').textContent=text;}
function readSaved(){try{return JSON.parse(localStorage.getItem(savedKey)||'null');}catch{return null;}}
function save(pending=null){try{if(view)localStorage.setItem(savedKey,JSON.stringify({token,domain:view.state.domain,pending,course:course?.toJSON()??null}));}catch{/* Play stays usable if browser storage is unavailable. */}}
function forget(){try{localStorage.removeItem(savedKey);}catch{}}
function focusBoard(){$('board').focus?.({preventScroll:true});}
function humanLegal(action){
  const s=view?.state;if(!s||s.terminal)return false;
  if(s.domain==='pong')return action==='wait'||action==='left'&&s.human.x>0||action==='right'&&s.human.x<s.lanes-1;
  if(action==='interact')return s.domain==='kitchen'&&s.interaction?.available===true;
  return ['up','down','left','right','wait'].includes(action);
}
function locks(){
  $('start').disabled=busy||!models;
  const d=view?.state.domain,terminal=!view||view.state.terminal,failed=!!queue.failedItem;
  for(const b of document.querySelectorAll('[data-action]')){const a=b.dataset.action;b.disabled=busy||playing||failed||terminal||!humanLegal(a);}
  $('nn-step').disabled=busy||terminal||playing||failed||!!course;$('play').disabled=busy||terminal||playing||failed||!!course;
  $('pause').disabled=!playing&&!queue.current&&!queue.items.length;$('export').disabled=busy||exporting||!view;
  $('next-task').disabled=busy||terminalAnimating||!!queue.current||failed;
  $('task-picker').hidden=$('sequence').checked;$('demo-controls').hidden=!!course||$('sequence').checked;
}
async function api(path,payload,sessionToken=token){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),['models','start','session','export'].includes(path)?90000:15000);
  try{
    const response=await fetch('/manual-api/'+path,{signal:controller.signal,method:payload?'POST':'GET',headers:{...(payload?{'Content-Type':'application/json'}:{}),...(sessionToken?{'Authorization':'Bearer '+sessionToken}:{})},...(payload?{body:JSON.stringify(payload)}:{})});
    const result=await response.json();if(!response.ok){const e=new Error(result.error||'请求失败');e.code=result.error;e.permanent=response.status>=400&&response.status<500;throw e;}return result;
  }finally{clearTimeout(timer);}
}
function inputStatus(){
  $('input-status').textContent=queue.failedItem?'操作未确认，键盘输入已暂停；请重新连接或重新开始任务。':`${queue.current?'正在确认 / 播放操作 · ':''}已排队 ${queue.items.length} / 8`;
  if(feedbackSamples.length){const sorted=[...feedbackSamples].sort((a,b)=>a-b);$('input-status').textContent+=` · 输入反馈 P95 ${Math.ceil(sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.95))])}ms（${sorted.length}次）`;}
  $('retry').hidden=!queue.failedItem;locks();
}
function stopPending(){playing=false;clearTimeout(demoTimer);release();return queue.clear();}
function scheduleDemo(started){clearTimeout(demoTimer);if(playing&&!view.state.terminal)demoTimer=setTimeout(()=>queue.enqueue(null),Math.max(0,380-(performance.now()-started)));}
const queue=new PreviewInputQueue({changed:inputStatus,full:()=>message('队列已满（8项）；本次输入未接受，请稍后再按。'),failed:e=>{
  playing=false;clearTimeout(demoTimer);release();$('connect').hidden=false;
  message(`操作未确认：${e.message}。请点击“重新连接”；旧会话失效时可点击“重新开始三个任务”。`);
},send:async item=>{
  if(!view||(view.state.terminal&&!item.payload)){stopPending();return;}
  if(!item.payload&&item.action!==null&&!humanLegal(item.action)){message('这一步在当前局面不可用，未推进回合；请换一个方向或按空格。');return;}
  const started=performance.now();
  item.payload??={command_id:crypto.randomUUID(),expected_turn:view.state.turn,human_action:item.action};save(item.payload);
  let result;
  for(let attempt=0;attempt<3;attempt++){try{result=await api('action',item.payload);break;}catch(e){if(e.code==='illegal carried human command'){save();message('当前动作不可用，未推进回合；可继续使用 WASD／空格。');return;}if(e.permanent||attempt===2)throw e;message('连接中断，正在重试同一命令…');}}
  if(course)course.accept(result,token);view=result;save();render(true);
  if(!view.state.terminal)message(playing?'双 NN 正在运行；可随时暂停。':`Task ${view.state.task} 进行中：WASD 移动，空格等待。橙色队友由冻结 NN 控制。`);
  await board.ready?.();scheduleDemo(started);
}});
function action(human,held=false){
  const began=performance.now();if(busy||playing||queue.failedItem||!view||view.state.terminal)return false;
  if(human!==null&&!humanLegal(human)){if(!held)message('当前方向或工位不可用；请换一个方向或按空格等待。');return false;}
  const accepted=queue.enqueue(human,held);
  if(accepted){$('input-status').dataset.feedbackAt=String(performance.now());if(!held){message(`已接受 ${human??'双 NN'}，按顺序执行。`);requestAnimationFrame(()=>{feedbackSamples.push(performance.now()-began);if(feedbackSamples.length>60)feedbackSamples.shift();inputStatus();});}}
  return accepted;
}
function choices(){const d=$('domain').value;$('seed').innerHTML=(models.models[d].seeds||[]).map((s,i)=>`<option value="${s}">场景 ${i+1} · ${s}</option>`).join('');}
function progress(){
  $('task-flow').hidden=!course;if(!course){$('next-task').hidden=true;return;}
  const completed=course.runs.filter(r=>r.terminal).length;
  $('progress-title').textContent=course.finished?'三个任务已全部结束':`三任务进度 · 已完成 ${completed} / 3`;
  $('task-progress').innerHTML=[1,2,3].map(t=>{const r=course.runs.find(x=>x.task===t);return `<li${view?.state.task===t?' aria-current="step"':''}>Task ${t} · ${r?.terminal?'已结束':r?'进行中':'待开始'}${r?.terminal?` · 得分 ${Number(r.score).toFixed(1)}`:''}</li>`;}).join('');
  const next=course.nextTask;$('next-task').hidden=!next||!view?.state.terminal;
  if(next)$('next-task').textContent=`进入 Task ${next}`;
  $('task-result').textContent=course.finished?'你已完成 Task 1、Task 2、Task 3，可下载三个任务的记录或切换游戏再玩。':view?.state.terminal?`Task ${view.state.task} 已结束。点击“进入 Task ${next}”继续。`:'每次按键推进一步。当前任务结束后会显示下一任务按钮。';
}
function render(animate=false){
  if(!view)return;const snapshot=view,s=snapshot.state,revision=++renderRevision;terminalAnimating=!!s.terminal;
  const shown=board.setState(s,{animate,language:'zh',onPresent:()=>present(snapshot)});
  Promise.resolve(shown).then(()=>{if(revision===renderRevision){terminalAnimating=false;locks();}});locks();
  if(s.terminal){const cancelled=stopPending();message(course?.finished?'三个任务已全部结束。可下载任务记录。':`Task ${s.task} 已结束${course?'，请点击下一任务按钮':''}。已清除 ${cancelled} 项待执行输入。`);locks();}
}
function present(snapshot){
  const s=snapshot.state,d=s.domain;$('workspace').hidden=false;
  $('model-label').textContent=`${d.toUpperCase()} · ${models.models[d].version} · ${snapshot.binding.variant}`;
  $('turn').textContent=`Task ${s.task} · 步数 ${s.turn} / ${s.max_turns}`;$('score').textContent=`得分 ${Number(s.score?.task_score??0).toFixed(1)}`;
  $('instructions').textContent=d==='pong'?'A / D 或左右方向键移动，空格等待。蓝色挡板由你控制。':d==='kitchen'?'WASD 或方向键移动和转向，E 操作正前方工位，空格等待。先准备原料，再向中央传递口交接。':'WASD 或方向键移动，空格等待。到取货格自动取货，到配送格自动配送；紫色闪电为充电站。';
  $('metrics').textContent=d==='warehouse'?`配送 ${s.deliveries} · 碰撞 ${s.collision_events} · 停机 ${s.shutdown_events}`:d==='kitchen'?`完成订单 ${s.metrics?.completed_orders??s.orders.filter(o=>o.status==='completed').length} / 5 · 你持有 ${s.human.holding?.ingredient||s.human.holding?.stage||'空'}`:`当前球数 ${s.balls?.length??0} · 蓝色挡板位置 ${s.human.x} · 橙色挡板位置 ${s.ai.x}`;
  const orderKey=JSON.stringify([d,s.orders]);if(orderKey!==ordersShown){ordersShown=orderKey;$('orders').innerHTML=(s.orders||[]).map(o=>`<div class="order"><b>${escapeText(o.id)}</b> · ${escapeText(o.status)} ${d==='warehouse'?`<br>取货 (${escapeText(o.pickup)}) → 配送 (${escapeText(o.dropoff)})`:`<br>${escapeText(o.recipe)} · 截止 ${escapeText(o.deadline)} 步`}</div>`).join('');}
  const last=snapshot.last_action;$('last-action').textContent=last?`${last.two_NN?'双 NN 演示':'真人接管'} · 角色1：${last.human} · AI：${last.ai} · AI动作覆盖：${last.ai_action_overrides}`:'WASD 移动，空格等待。点击棋盘可重新获得键盘控制。';
  const bindingKey=JSON.stringify(snapshot.binding);if(bindingKey!==bindingShown){$('binding').textContent=JSON.stringify(snapshot.binding,null,2);bindingShown=bindingKey;}progress();
}
async function beginTask(task,sequence){
  if(busy)return;const cancelled=stopPending();busy=true;locks();message(`正在连接并准备 Task ${task}…${cancelled?'已清除待执行输入。':''}`);await queue.idle();
  // An old in-flight failure must never be retried against the new task token.
  stopPending();
  try{
    const selection=sequence?{domain:sequence.domain,variant:sequence.variant,seed:sequence.seed}:{domain:$('domain').value,variant:$('variant').value,seed:Number($('seed').value)};
    const result=await api('start',{...selection,task});const nextToken=result.session_token;delete result.session_token;
    if(sequence)sequence.accept(result,nextToken);
    token=nextToken;course=sequence;view=result;$('connect').hidden=true;save();render();focusBoard();
    message(`Task ${task} 已开始。直接按 WASD 移动、空格等待${selection.domain==='kitchen'?'、E 操作工位':''}。`);
  }catch(e){$('connect').hidden=false;message('无法开始：'+e.message+'。请重新连接后再试。');}finally{busy=false;$('start').textContent=course?'重新开始三个任务':$('sequence').checked?'开始 Task 1':'开始单局';locks();}
}
function start(event){event?.preventDefault();if(!models||busy)return;const sequence=$('sequence').checked?new PreviewTaskSequence({domain:$('domain').value,variant:$('variant').value,seed:Number($('seed').value)}):null;return beginTask(sequence?1:Number($('task').value),sequence);}
$('selection').addEventListener('submit',start);$('domain').addEventListener('change',choices);
$('next-task').addEventListener('click',()=>{if(course?.nextTask&&view.state.terminal&&!terminalAnimating&&!queue.current)return beginTask(course.nextTask,course);});
$('board').addEventListener('pointerdown',focusBoard);
for(const b of document.querySelectorAll('[data-action]'))b.addEventListener('click',()=>{action(b.dataset.action);focusBoard();});
$('nn-step').addEventListener('click',()=>action(null));
$('play').addEventListener('click',async()=>{if(course)return;stopPending();playing=true;locks();await queue.idle();clearTimeout(demoTimer);if(playing)queue.enqueue(null);});
$('pause').addEventListener('click',()=>{const n=stopPending();locks();message(`已暂停；已确认动作完整播放，已发出的至多一步会完成。清除 ${n} 项待执行输入。`);});
$('retry').addEventListener('click',()=>{queue.retry();focusBoard();});
for(const id of ['domain','variant','task','seed','sequence'])$(id).addEventListener('change',()=>{const n=stopPending();locks();$('start').textContent=$('sequence').checked?'开始 Task 1':'开始单局';message(`选择已改变，已清除 ${n} 项待执行输入；点击开始按钮应用。`);});
const keys={w:'up',arrowup:'up',s:'down',arrowdown:'down',a:'left',arrowleft:'left',d:'right',arrowright:'right',' ':'wait',e:'interact'};
const codes={KeyW:'up',ArrowUp:'up',KeyS:'down',ArrowDown:'down',KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right',Space:'wait',KeyE:'interact'};
function keyAction(event){return codes[event.code]||keys[String(event.key||'').toLowerCase()];}
function release(){heldKey=null;clearTimeout(heldTimer);queue.release();}
function repeatHeld(){if(!heldKey)return;const button=document.querySelector(`[data-action="${heldKey}"]`);if(button&&!button.disabled)action(heldKey,true);heldTimer=setTimeout(repeatHeld,150);}
document.addEventListener('keydown',event=>{
  if(event.ctrlKey||event.metaKey||event.altKey||/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)||document.activeElement?.isContentEditable)return;
  const a=keyAction(event);if(!a||!view||view.state.terminal)return;event.preventDefault();if(event.repeat||playing||queue.failedItem)return;
  const button=document.querySelector(`[data-action="${a}"]`);if(button&&!button.disabled){release();action(a);heldKey=a;heldTimer=setTimeout(repeatHeld,250);}
});
document.addEventListener('keyup',event=>{if(keyAction(event)===heldKey){event.preventDefault();release();}});
document.addEventListener('focusin',event=>{if(/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)||event.target.isContentEditable)release();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;clearTimeout(demoTimer);release();}});
window.addEventListener('blur',release);
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
$('export').addEventListener('click',async()=>{
  if(exporting||!view)return;exporting=true;const run=course?.toJSON(),exportToken=token,domain=view.state.domain,variant=view.binding.variant;locks();message('正在核对任务回放；可继续操作。');
  try{
    if(run){const tasks=[];for(const r of run.runs){message(`正在核对 Task ${r.task} 回放…`);const data=await api('export',null,r.token);if(JSON.stringify(data.replay.binding)!==JSON.stringify(run.binding))throw Error('模型绑定不一致');tasks.push(data);}download({mode:'three_task_manual_preview',human_study_sample:false,scientific_evaluation:false,binding:run.binding,tasks},`policylens-${domain}-${variant}-three-tasks.json`);}
    else download(await api('export',null,exportToken),`policylens-${domain}-${variant}-manual.json`);
    message('任务记录已下载，NN 与物理回放核对通过。');
  }catch(e){message('导出失败：'+e.message);}finally{exporting=false;locks();}
});
async function connect(){
  if(busy)return;busy=true;locks();message('正在连接服务器…');const waking=setTimeout(()=>message('免费服务正在唤醒，请稍候。连接后即可使用键盘操作。'),3000);
  try{
    models=await api('models');choices();const saved=readSaved();
    if(saved?.token&&(!requested||saved.domain===requested)){
      token=saved.token;
      try{
        const restored=await api('session');course=saved.course?new PreviewTaskSequence(saved.course):null;
        // Old single-task saves remain single-task; their original seed was not persisted.
        if(course)course.accept(restored,token);view=restored;$('domain').value=view.state.domain;$('variant').value=view.binding.variant;$('task').value=String(view.state.task);$('sequence').checked=!!course;choices();if(course)$('seed').value=String(course.seed);render();focusBoard();save(saved.pending);
        if(saved.pending){queue.items.push({action:saved.pending.human_action,held:false,payload:saved.pending});queue.pump();}
        message('已恢复任务。直接使用 WASD／空格操作。');
      }catch(e){
        if(!e.permanent)throw e;
        token='';view=null;course=null;stopPending();forget();$('workspace').hidden=true;message('旧会话已失效。请选择游戏，点击“开始 Task 1”重新开始。');
      }
    }else message('已连接。选择游戏，点击“开始 Task 1”，用 WASD／空格依次完成三个任务。');
    $('connect').hidden=true;
  }catch(e){$('connect').hidden=false;message('连接失败：'+e.message+'。点击“重新连接”重试。');}
  finally{clearTimeout(waking);busy=false;$('start').textContent=course?'重新开始三个任务':$('sequence').checked?'开始 Task 1':'开始单局';locks();}
}
$('connect').addEventListener('click',async()=>{
  if(view&&queue.failedItem){busy=true;locks();try{await api('session');$('connect').hidden=true;queue.retry();message('已重新连接，正在核对原命令。');focusBoard();}catch(e){message('当前会话无法恢复：'+e.message+'。点击“重新开始三个任务”重新开始。');}finally{busy=false;locks();}}else await connect();
});
void connect();
