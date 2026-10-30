'use strict';
const $=id=>document.getElementById(id),board=new PreviewBoard($('board'));
let config=null,view=null,session=null,busy=false,replaying=false,replayFrames=[],held=null,holdTimer=null,motionRevision=0;
let providerWaiting=false,readingAnswers=false,demoViewed=false,lastTiming=performance.now(),timingTask=null,timingSeconds={};
let understandingBank=null,understandingLoading=false,timeFlush=null;
board.duration=120;
const practiced=new Set();let practiceX=0,practiceY=0;
const storageKey='policylens-study-protocol-20261029.v1';
const requested=location.pathname.split('/')[1];if(['warehouse','pong','kitchen'].includes(requested))$('domain').value=requested;
const message=t=>$('status').textContent=t;
function sampleTime(){const now=performance.now(),seconds=Math.min(5,(now-lastTiming)/1000);lastTiming=now;const s=state();if(!s||s.terminal||document.hidden)return;timingTask??=s.task;const kind=providerWaiting?'explanation_wait':replaying?'replay':readingAnswers?'reading':document.activeElement===$('question')?'question_entry':'active';timingSeconds[kind]=(timingSeconds[kind]||0)+seconds;}
async function flushTime(){
 if(timeFlush)return timeFlush;
 timeFlush=(async()=>{sampleTime();if(!session)return;session.events??=[];
 if(timingTask){for(const [kind,seconds]of Object.entries(timingSeconds))if(seconds>0)session.events.push({event_id:crypto.randomUUID(),task:timingTask,kind,seconds});timingSeconds={};}
 timingTask=state()?.terminal?null:state()?.task??null;save();
 for(const event of [...session.events]){try{await api(route('event'),event);session.events=session.events.filter(e=>e.event_id!==event.event_id);save();}catch{break;}}
 })();try{return await timeFlush;}finally{timeFlush=null;}
}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(session));}catch{}}
function focus(){$('board').focus({preventScroll:true});}
async function api(path,payload,administrator=false){
 const headers={...(session?{Authorization:'Bearer '+session.token}:{}),...(payload?{'Content-Type':'application/json'}:{})};
 if(administrator)headers['X-Admin-Token']=$('admin').value;
 const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),90000);
 try{const r=await fetch('/study-api/'+path,{method:payload?'POST':'GET',headers,signal:abort.signal,...(payload?{body:JSON.stringify(payload)}:{})});const value=await r.json();if(!r.ok){const e=Error(value.error||'连接失败');e.permanent=r.status<500;throw e;}return value;}finally{clearTimeout(timer);}
}
setInterval(sampleTime,1000);setInterval(()=>{if(session&&!busy&&!queue.current)void flushTime();},30000);
const route=tail=>session.id+'/'+tail;
function state(){return view?.current?.public;}
function legal(action){const s=state();if(!s||s.terminal)return false;if(s.domain==='pong')return action==='wait'||action==='left'&&s.human.x>0||action==='right'&&s.human.x<s.lanes-1;if(action==='interact')return s.domain==='kitchen'&&s.interaction?.available===true;return ['up','down','left','right','wait'].includes(action);}
function release(){held=null;clearTimeout(holdTimer);queue.release();}
function locks(){
 for(const b of document.querySelectorAll('[data-action]'))b.disabled=busy||replaying||readingAnswers||!!queue.failedItem||!legal(b.dataset.action);
 $('retry').hidden=!queue.failedItem;$('ask').disabled=busy||!!queue.current;
 $('retry').disabled=busy||replaying||readingAnswers||!!queue.current;
 $('next').disabled=busy||!!queue.current;$('demo').disabled=busy||!view?.study?.preparation?.controls;$('demo-finish').disabled=busy||!demoViewed;
 $('practice-finish').disabled=busy||!practiceCodes().every(code=>practiced.has(code));
 $('read-answers').textContent=readingAnswers?'结束阅读，返回操作':'阅读已保存回答';$('read-answers').hidden=!$('answers').childElementCount;
 $('read-answers').disabled=busy||!!queue.current||!!queue.failedItem;
 $('restore-answers').disabled=busy||!!queue.current||!!queue.failedItem;
 $('new-test').disabled=busy||!!queue.current||!!queue.failedItem||!!session?.pending;
 $('replay-turn').disabled=busy||!replaying;
 $('input-feedback').textContent=queue.current?'操作已接受，正在由服务器确认'+(queue.items.length?'；另有 '+queue.items.length+' 个排队输入。':'。'):queue.failedItem?'这次操作尚未确认，请重试原动作。':'';
}
const queue=new PreviewInputQueue({changed:locks,full:()=>message('输入队列已满，本次按键未接受。'),failed:e=>{release();message('动作尚未确认：'+e.message+'。请重试原动作或刷新恢复。');},send:async item=>{
 if(!item.payload&&!legal(item.action)){message('该操作在当前局面不可用，未推进回合。');return;}
 item.payload??={command_id:crypto.randomUUID(),expected_turn:state().turn,human_action:item.action};session.pending=item.payload;save();
 let result;for(let attempt=0;attempt<3;attempt++){try{result=await api(route('action'),item.payload);break;}catch(e){if(e.message==='illegal carried human command'){session.pending=null;save();message('操作不可用，可继续移动或等待。');return;}if(e.permanent||attempt===2)throw e;}}
 view=result;session.pending=null;save();render(true);await board.ready();
}});
function move(action,repeat=false){if(busy||replaying||readingAnswers||queue.failedItem||!legal(action))return;queue.enqueue(action,repeat);}
function render(animate=false){
 $('entry').hidden=!!session;$('workspace').hidden=!session;if(!view)return;
 const stage=view.instance.stage,phase=view.study?.phase,s=state();
 const taskStage=stage.match(/^task_(\d+)(_complete)?$/);
 $('stage').textContent=phase==='finished'?'测试已完成':phase==='questionnaire'?'填写结束问卷':stage==='enrolled'?'先观看冻结双 NN 合作演示':stage==='demo_complete'?'演示已确认':taskStage?'Task '+taskStage[1]+' · '+(taskStage[2]?'已结束':'进行中'):stage;
 $('progress').textContent=(view.runs||[]).map(r=>`Task ${r.task}：${r.status==='complete'?'已结束':'进行中'}`).join(' · ');
 $('practice').hidden=stage!=='enrolled'||view.study?.preparation?.controls===true;
 $('practice-status').textContent='请分别按下：'+practiceCodes().map(code=>(practiced.has(code)?'✓ ':'')+code.replace('Key','')).join('、');
 demoViewed=view.study?.preparation?.demonstration===true;
 $('demo').hidden=stage!=='enrolled';$('demo-finish').hidden=stage!=='enrolled'||!demoViewed;
 $('next').hidden=!['demo_complete','task_1_complete','task_2_complete'].includes(stage);
 const next=(view.runs?.length||0)+1;$('next').textContent='开始 Task '+next;
 $('survey').hidden=phase!=='questionnaire';$('finished').hidden=phase!=='finished';
 $('new-test').hidden=!session;
 if(phase==='questionnaire')void loadUnderstanding();
 $('qa').hidden=!view.explanation_questions_allowed;if($('qa').hidden){$('answers').replaceChildren();readingAnswers=false;}
 const availableAlternatives=view.instance.domain==='pong'?['wait','left','right']:['wait','up','down','left','right',...(view.instance.domain==='kitchen'?['interact']:[])];
 for(const option of Array.from($('alternative').options||[]))option.hidden=!!option.value&&!availableAlternatives.includes(option.value);
 if($('alternative').value&&!availableAlternatives.includes($('alternative').value))$('alternative').value='';
 if(s){const revision=++motionRevision;board.setState(s,{animate,onPresent:()=>{
   if(revision!==motionRevision)return;$('score').textContent=`Task ${s.task} · ${s.turn}/${s.max_turns}步 · 得分 ${Number(s.score?.task_score??0).toFixed(1)}`;
 }});$('instructions').textContent=s.domain==='pong'?'A / D 左右移动，空格等待。':'WASD 移动，空格等待。'+(s.domain==='kitchen'?'E 操作正前方工位。':'');}
 if($('binding').textContent!==view.instance.binding_sha256)$('binding').textContent=view.instance.binding_sha256;
 if(s?.terminal){void flushTime();release();queue.clear();message('本任务已结束，请点击下一任务按钮。');}locks();
}
async function command(action,payload={}){if(busy)return;release();queue.clear();busy=true;locks();message(action==='next-task'?'正在完整核对记录并准备下一任务…':'正在保存并核对记录…');await queue.idle();queue.clear();await flushTime();try{view=await api(route(action),payload);replaying=false;replayFrames=[];render();focus();if(action==='next-task')message('Task '+state().task+' 已开始，请用按键控制你的角色。');}catch(e){message(e.message);}finally{busy=false;locks();}}
$('entry').addEventListener('submit',async event=>{
 event.preventDefault();if(busy||!config)return;busy=true;locks();
 try{const payload={domain:$('domain').value,participant_reference:$('reference').value,consent:$('consent').checked,consent_version:config.consent_version,question_transmission_consent:$('transmission-consent').checked};if($('group').value)payload.group=$('group').value;
 const result=await api('enroll',payload,true);session={id:result.instance.id,token:result.session_token,pending:null};save();view=await api(route('view'));render();message('已绑定模型，请观看双 NN 演示。');}
 catch(e){message('无法开始：'+e.message);}finally{busy=false;locks();}
});
$('demo').addEventListener('click',async()=>{
 if(busy||!view?.study?.preparation?.controls)return;busy=true;demoViewed=false;locks();try{const demo=await api(route('demo'),{});const saved=await api(route('view'));await board.setState(demo.initial_public);for(const r of demo.records){await board.setState(r.public_after,{animate:true});}const prep=saved.study.preparation;view=await api(route('preparation'),{kind:'demonstration',bundle_sha256:prep.demonstration_sha256,final_frame_count:prep.demonstration_frame_count});demoViewed=true;render();message('演示播放完成。');}catch(e){message(e.message);}finally{busy=false;locks();}
});
function practiceCodes(){const domain=view?.instance?.domain||$('domain').value;return domain==='pong'?['KeyA','KeyD','Space']:['KeyW','KeyA','KeyS','KeyD','Space',...(domain==='kitchen'?['KeyE']:[])];}
function practiceKey(code){if(!practiceCodes().includes(code))return;practiced.add(code);practiceX+=code==='KeyA'?-14:code==='KeyD'?14:0;practiceY+=code==='KeyW'?-14:code==='KeyS'?14:0;$('practice-marker').style.transform=`translate(${practiceX}px,${practiceY}px)`;render();}
$('practice-finish').addEventListener('click',()=>command('preparation',{kind:'controls',observed_codes:practiceCodes()}));
$('demo-finish').addEventListener('click',()=>command('demo-complete'));
$('next').addEventListener('click',()=>command('next-task'));
$('retry').addEventListener('click',()=>{if(!busy&&!replaying&&!readingAnswers)queue.retry();});
for(const button of document.querySelectorAll('[data-action]'))button.addEventListener('click',()=>{move(button.dataset.action);focus();});
const keys={KeyW:'up',KeyA:'left',KeyS:'down',KeyD:'right',Space:'wait',KeyE:'interact',ArrowUp:'up',ArrowLeft:'left',ArrowDown:'down',ArrowRight:'right'};
function repeat(){if(!held)return;move(held,true);holdTimer=setTimeout(repeat,150);}
document.addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey||/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)||document.activeElement?.isContentEditable)return;if(view?.instance.stage==='enrolled'&&!view.study?.preparation?.controls){if(practiceCodes().includes(e.code)){e.preventDefault();if(!e.repeat)practiceKey(e.code);}return;}const action=keys[e.code];if(!action)return;e.preventDefault();if(e.repeat)return;release();move(action);held=action;holdTimer=setTimeout(repeat,250);});
document.addEventListener('keyup',e=>{if(keys[e.code]===held)release();});window.addEventListener('blur',release);
document.addEventListener('visibilitychange',()=>{if(document.hidden)release();});$('board').addEventListener('pointerdown',focus);
$('replay').addEventListener('click',async()=>{if(busy||queue.current||queue.failedItem)return;release();queue.clear();busy=true;locks();try{const task=Number($('replay-task').value);const data=await api(route('replay')+'?task='+task);replayFrames=data.frames;replaying=true;$('replay-turn').max=String(replayFrames.length-1);$('replay-turn').value='0';showReplay();}catch(e){message(e.message);}finally{busy=false;locks();}});
function showReplay(){if(!replaying)return;const frame=replayFrames[Number($('replay-turn').value)];if(frame){board.setState(frame.public);$('history-label').textContent=`历史 Task ${frame.public.task} · 回合 ${frame.turn}（不推进游戏）`;}}
$('replay-turn').addEventListener('input',()=>{if(!busy&&replaying)showReplay();});$('live').addEventListener('click',()=>{if(busy)return;replaying=false;replayFrames=[];render();focus();});
function appendAnswer(result){const p=document.createElement('p');p.textContent=`Task ${result.target_task} 第${result.result_turn}步：${result.language_rendering?.answer||result.answer}`;$('answers').append(p);}
async function restoreAnswers(){if(!view?.explanation_questions_allowed)return;const result=await api(route('answers'));if(!view?.explanation_questions_allowed)return;$('answers').replaceChildren();for(const answer of result.questions)appendAnswer(answer);locks();}
$('restore-answers').addEventListener('click',async()=>{if(busy||queue.current||queue.failedItem)return;busy=true;locks();try{await restoreAnswers();}catch(e){message('回答暂未恢复：'+e.message+'。当前会话仍保留，可重试。');}finally{busy=false;locks();}});
$('read-answers').addEventListener('click',()=>{if(busy||queue.current||queue.failedItem)return;sampleTime();release();queue.clear();readingAnswers=!readingAnswers;locks();if(!readingAnswers)focus();});
$('question-form').addEventListener('submit',async event=>{event.preventDefault();if(busy||queue.current||!view.explanation_questions_allowed)return;release();busy=true;locks();
 try{providerWaiting=true;sampleTime();message('正在核对已执行的历史与解释证据…');const payload={question:$('question').value,target_task:Number($('question-task').value),decision_turn:Number($('question-step').value)-1};if($('alternative').value)payload.human_actions=[$('alternative').value];
 const result=await api(route('question'),payload);view=await api(route('view'));render();if(view.explanation_questions_allowed){appendAnswer(result);readingAnswers=true;message('回答已保存；结束阅读后可继续操作。');await api(route('event'),{event_id:crypto.randomUUID(),task:2,kind:'answer_displayed',details:{question_id:result.question_id}});}}
 catch(e){message('问题无法回答：'+e.message);}finally{sampleTime();providerWaiting=false;busy=false;locks();}
});
async function loadUnderstanding(){if(understandingBank||understandingLoading)return;understandingLoading=true;$('submit-survey').disabled=true;try{
 understandingBank=await api(route('understanding'));const saved=view.study?.understanding_answers||{};
 for(const item of understandingBank.questions){const label=document.createElement('label');label.textContent=item.prompt;const select=document.createElement('select');select.id='understanding-'+item.id;select.required=true;
 const empty=document.createElement('option');empty.value='';empty.textContent='请选择';select.append(empty);for(const option of item.options){const node=document.createElement('option');node.value=option.id;node.textContent=option.text;select.append(node);}select.value=saved[item.id]||'';label.append(select);$('understanding-items').append(label);}
 if(view.study?.survey_answers){for(const [key,value]of Object.entries(view.study.survey_answers))$('rating-'+key).value=String(value);$('feedback').value=view.study.survey_feedback||'';}
 }catch(e){understandingBank=null;message('理解题未加载：'+e.message+'。请刷新重试。');}finally{understandingLoading=false;$('submit-survey').disabled=!understandingBank;}}
$('survey').addEventListener('submit',async event=>{event.preventDefault();if(busy||!understandingBank)return;const answers={},understanding={};for(const name of config.questionnaire_fields)answers[name]=Number($('rating-'+name).value);for(const item of understandingBank.questions)understanding[item.id]=$('understanding-'+item.id).value;
 busy=true;locks();try{view=await api(route('understanding'),{answers:understanding,version:understandingBank.version,sha256:understandingBank.sha256});view=await api(route('survey'),{answers,feedback:$('feedback').value});render();}catch(e){message('问卷未完成保存：'+e.message+'。请重试相同答案。');}finally{busy=false;locks();}});
$('new-test').addEventListener('click',async()=>{
 if(busy||queue.current||queue.failedItem||session?.pending||!session||!view)return;
 if(view.study?.phase!=='finished'&&!window.confirm('当前未完成测试不会被删除，但本浏览器将退出该会话。是否保留服务器记录并开始另一测试？'))return;
 release();queue.clear();busy=true;locks();
 try{await flushTime();if(session.events?.length){message('仍有测试记录等待保存，请连接后重试。');return;}
 localStorage.removeItem(storageKey);session=null;view=null;location.assign(location.pathname==='/research/'?'/research/':'/study/');
 }catch{message('无法清除本浏览器的会话凭据，请重试。服务器记录保留。');}finally{busy=false;locks();}
});
(async()=>{busy=true;locks();try{config=await api('status');$('mode').textContent='协议 '+config.protocol_id+' · 正式招募关闭 · 问答模式：'+config.question_service_mode;for(const [name,label]of[['coordination','协作顺畅程度'],['understanding','理解队友行为'],['predictability','能否预测队友'],['workload','操作负担']]){const l=document.createElement('label');l.textContent=label;const input=document.createElement('input');input.type='number';input.min='1';input.max='7';input.required=true;input.id='rating-'+name;l.append(input);$('ratings').append(l);}
 try{session=JSON.parse(localStorage.getItem(storageKey)||'null');}catch{session=null;}if(session){try{view=await api(route('view'));render();}catch(e){message('会话暂未恢复：'+e.message+'。已保留本地凭据，请刷新重试。');return;}
 try{await restoreAnswers();}catch(e){message('回答暂未恢复：'+e.message+'。当前会话仍保留，可点击恢复回答。');}
 focus();void flushTime();if(session.pending){queue.items.push({action:session.pending.human_action,held:false,payload:session.pending});queue.pump();}}
 }catch(e){message('服务不可用：'+e.message);}finally{busy=false;locks();}})();
