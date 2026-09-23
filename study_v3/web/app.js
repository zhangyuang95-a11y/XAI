'use strict';
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// Every page opens in English; the header switch applies during this visit.
let lang='en';
let surveyDraft={},refreshSequence=0;
function surveyKey(v=view){return v?.instance_id?'policylens.survey.v3.'+v.release_id+'.'+v.instance_id:null;}
function saveSurveyDraft(){const form=$('surveyForm'),key=surveyKey();if(form&&key&&view.stage==='questionnaire'){surveyDraft=Object.fromEntries(new FormData(form));try{sessionStorage.setItem(key,JSON.stringify(surveyDraft));}catch{}}}
let timeSince=performance.now(),timeBuckets={active:0,reading:0,replay:0};
function sampleTime(){const now=performance.now(),seconds=Math.min(5,(now-timeSince)/1000);timeSince=now;if(!view?.stage?.startsWith('task')||view.state?.terminal||document.hidden||asking)return;const kind=replay?'replay':($('questionInput')===document.activeElement||document.querySelector('details[open]'))?'reading':'active';timeBuckets[kind]+=seconds;}
function takeTimings(){sampleTime();const items=Object.entries(timeBuckets).filter(([,seconds])=>seconds>0).map(([kind,seconds])=>({kind,seconds}));timeBuckets={active:0,reading:0,replay:0};return items;}
setInterval(sampleTime,1000);
let view=null,release=null,busy=false,asking=false,adminKey='',draftQuestion='',selectedAction='wait',replay=null,demoTimer=null,demoPlaying=false,demoDone=false,participantId='',chatEpoch=0;
let entryTimer=null,entryChecking=false,entryFailed=false,entryCheckedAt=-Infinity;
const ENTRY_RECHECK_MS=15000,ENTRY_MANUAL_COOLDOWN_MS=3000;
let boardRenderer=null,animationPending=false,heldDirection=null,heldTimer=null,pendingRefresh=false,demoGeneration=0,replayRequestSequence=0,replayLoading=false;
let tutorialPauseRequested=false;
const prolificEntry=location.pathname==='/prolific/';
let domain=prolificEntry?null:location.pathname.split('/').filter(Boolean)[0]||null;
const preview=new URLSearchParams(location.search).get('preview')==='1';
const channel='BroadcastChannel' in window?new BroadcastChannel('policylens.study.v3'):null;
const EN={home:'Choose your shared challenge.',homeSub:'Work with an AI teammate. Learn the controls, coordinate your actions, and complete the shared task.',eyebrow:'A study of human–AI teamwork',pilot:'Research pilot',footer:'Small decisions. Better teamwork.',back:'← All tasks',start:'Begin demonstration',consent:'I agree to take part and have read the information above.',participant:'Study ID',participantHint:'Use an anonymous ID: 3–64 letters, numbers, hyphens or underscores.',data:'This pilot records your actions, scores, questions and questionnaire responses for research. Use an anonymous study ID. Task 2 questions may be processed by the configured language service; do not include personal information. You may leave at any time.',ready:'The study is being prepared. Participant entry will open after the question service and persistent storage have been verified.',preview:'Researcher preview',previewKey:'Researcher access key',group:'Group',resume:'Resume your task',demo:'Demo',questionnaire:'Questionnaire',completed:'Completed',you:'You',ai:'AI teammate',turn:'Turn',remaining:'Remaining',score:'Team score',actions:'Controls',wait:'Wait',up:'Up',down:'Down',left:'Left',right:'Right',controls:'Each action advances one move. The scene animates smoothly; nothing advances while you are not playing.',rules:'How this task works',replay:'Review a turn',current:'Return to current turn',events:'What happened',noEvents:'Your next shared action will appear here.',ask:'Ask your AI teammate',questionPlaceholder:'Ask about this turn, a previous decision, or what would happen if…',send:'Ask question',asking:'Checking this turn…',why:'Why are you waiting?',help:'How can I help?',whatif:'What if I wait?',closed:'Explanations are unavailable',baseline:'In Task 1, learn how your teammate behaves by playing together.',transfer:'In Task 3, use what you have learned to work with your teammate. Questions and previous answers are unavailable.',control:'Use the same controls and replay to work with your teammate.',task2Intro:'In Task 2, you can ask about any turn you have already played. Questions do not use game turns.',demoPlay:'Start playback',demoPause:'Pause',demoSkip:'Skip playback and start Task 1',demoComplete:'Demonstration complete',startTask:'Start Task 1',demoNote:'A separate demonstration. These scores do not count toward your tasks.',step:'Step',finishTitle:'Task complete',finishSub:'Your result is saved. Continue when you are ready.',nextTask:'Continue to Task',toSurvey:'Continue to questionnaire',readOnly:'You are viewing a past turn. Return to the current turn before moving.',surveyTitle:'How did the teamwork feel?',surveySub:'Think about all three tasks. There are no right or wrong answers to the ratings.',disagree:'Strongly disagree',agree:'Strongly agree',na:'Not used / N/A',checkTitle:'A few new situations',checkSub:'Choose the answer that best matches what you learned. Answers are not shown during this study.',feedback:'Anything else you would like to tell us? (optional)',submit:'Submit questionnaire',thanks:'Thank you for taking part.',saved:'Your three tasks and questionnaire have been saved.',homeButton:'Return to task selection',task:'Task',holding:'Carrying',empty:'Nothing',battery:'Battery',handoff:'Handoff counter',orders:'Orders',raw:'Raw',chopped:'Prepared',cooked:'Cooked',plated:'Plated',tomato:'Tomato',onion:'Onion',wave:'Wave',teamBall:'Team ball',smallBall:'Small ball',publicInfo:'Shared information',prepare:'Prepare',served:'Served',expired:'Expired',delivered:'Delivered',deadline:'Due',noScore:'Not scored',notMeasured:'Human performance improvement has not yet been measured.',loading:'Preparing your workspace…',close:'Close',retry:'Please try again.'};
const ZH={home:'选择你的协作任务。',homeSub:'与 AI 队友共同完成任务。先熟悉操作，再一步一步做决定，找到你们的配合方式。',eyebrow:'人机协作研究',pilot:'研究预实验',footer:'每个小决定，让配合更顺畅。',back:'← 全部任务',start:'开始观看演示',consent:'我已阅读上述说明，并同意参加。',participant:'研究编号',participantHint:'请使用匿名编号：3–64 位英文字母、数字、连字符或下划线。',data:'本预实验将记录操作、得分、问题和问卷回答，用于研究。请使用匿名编号。Task 2 的问题可能由配置的语言服务处理，请勿输入个人信息。你可以随时退出。',ready:'研究正在准备中。问答服务和持久存储验证完成后，将开放参与入口。',preview:'研究者预览',previewKey:'研究者访问密钥',group:'组别',resume:'继续你的任务',demo:'演示',questionnaire:'问卷',completed:'完成',you:'你',ai:'AI 队友',turn:'回合',remaining:'剩余回合',score:'协作得分',actions:'操作',wait:'等待',up:'上',down:'下',left:'左',right:'右',controls:'每次操作推进一步并平滑播放。不操作时，游戏不会自动前进。',rules:'任务规则',replay:'回看一个回合',current:'回到当前回合',events:'刚才发生了什么',noEvents:'下一次共同动作的结果会显示在这里。',ask:'向 AI 队友提问',questionPlaceholder:'可以问当前回合、之前的决定，或另一种行动会怎样……',send:'发送问题',asking:'正在核对这个回合……',why:'你为什么等待？',help:'我怎样配合你？',whatif:'如果我等待会怎样？',closed:'当前不提供解释',baseline:'Task 1 中，通过共同完成任务来了解队友的行为。',transfer:'Task 3 中，请运用已经学到的内容与队友配合。新问题和之前的回答均不可查看。',control:'使用相同的操作和回放功能与队友协作。',task2Intro:'Task 2 中，可以询问已经经历过的任意回合。提问不消耗游戏回合。',demoPlay:'开始播放',demoPause:'暂停',demoSkip:'跳过播放，直接进入任务',demoComplete:'演示已完成',startTask:'开始 Task 1',demoNote:'独立教学演示，演示得分不计入任务成绩。',step:'步骤',finishTitle:'任务完成',finishSub:'成绩已保存。准备好后继续。',nextTask:'继续 Task',toSurvey:'进入问卷',readOnly:'你正在回看历史回合。回到当前回合后才能行动。',surveyTitle:'这次配合感觉如何？',surveySub:'请结合三个任务回答。评分题没有对错。',disagree:'非常不同意',agree:'非常同意',na:'未使用／不适用',checkTitle:'几个新情境',checkSub:'根据学到的内容选择最合适的答案。研究过程中不显示正确答案。',feedback:'还有什么想告诉我们？（可选）',submit:'提交问卷',thanks:'感谢参与。',saved:'三个任务与问卷已保存。',homeButton:'返回任务选择',task:'Task',holding:'手持',empty:'无',battery:'电量',handoff:'交接台',orders:'订单',raw:'原料',chopped:'已备料',cooked:'熟食',plated:'已装盘',tomato:'番茄',onion:'洋葱',wave:'波次',teamBall:'合作球',smallBall:'普通球',publicInfo:'共同可见信息',prepare:'准备',served:'已上菜',expired:'已过期',delivered:'已送达',deadline:'截止',noScore:'不计分',notMeasured:'真人协作表现的提升幅度尚未测量。',loading:'正在准备任务……',close:'关闭',retry:'请重试。'};
Object.assign(EN,{egg:'Egg',meat:'Meat',pepper:'Pepper',egg_tomato:'Tomato and eggs',pepper_meat:'Pepper and meat',interact:'Interact',prepared:'Prepared',protein_cooked:'Cooked first ingredient',output:'Serving container',dish:'Finished dish',temporary_plate:'Temporary plate',output_container:'Serving container',raw_score:'Score',facing:'Facing',buffers:'Counters',ready_food:'Ready',cooking:'Cooking',burnt:'Burnt',emptyPot:'Empty pan',cooked_protein:'Cooked first ingredient on a temporary plate',cooked_vegetable:'Cooked vegetable',finished:'Finished dish in a serving container',formal_plate:'Formal serving plate',score_delivery:'Deliveries',score_robot_collision:'Collisions',score_shutdown:'Shutdown',score_time:'Moves',score_human_detour:'Detours',score_shared_charger_occupancy:'Charger occupancy'});
Object.assign(ZH,{egg:'鸡蛋',meat:'肉',pepper:'辣椒',egg_tomato:'番茄炒鸡蛋',pepper_meat:'辣椒炒肉',interact:'交互',prepared:'已备料',protein_cooked:'已炒熟主料',output:'出锅容器',dish:'成品菜',temporary_plate:'熟料临时盘',output_container:'出锅容器',raw_score:'得分',facing:'朝向',buffers:'暂存台',ready_food:'已熟',cooking:'烹饪中',burnt:'已烧糊',emptyPot:'空锅',cooked_protein:'熟主料（临时盘）',cooked_vegetable:'已炒熟辅料',finished:'成品菜（出锅容器）',formal_plate:'正式餐盘',score_delivery:'配送',score_robot_collision:'碰撞',score_shutdown:'断电',score_time:'步数',score_human_detour:'绕路',score_shared_charger_occupancy:'占用充电站'});
Object.assign(EN,{entryUnavailable:'The study service is temporarily unavailable, so new tasks cannot start yet. We will check again automatically; your entries will stay here.',entryNetwork:'We could not reach the service. We will check again automatically; your entries will stay here.',entryChecking:'Checking the service…',entryRetry:'Check again',entryOpen:'The service is ready. You can begin.'});
Object.assign(ZH,{entryUnavailable:'研究服务暂时不可用，因此目前无法开始新任务。系统会自动重试，并保留你填写的内容。',entryNetwork:'暂时无法连接服务。系统会自动重试，并保留你填写的内容。',entryChecking:'正在重新检查服务……',entryRetry:'重新检查',entryOpen:'服务已就绪，可以开始。'});
const tr=k=>(lang==='zh'?ZH:EN)[k]||EN[k]||k;
Object.assign(EN,{tutorial:'Kitchen practice',tutorialStart:'Start practice',tutorialResume:'Resume practice',tutorialPause:'Pause practice',tutorialRetry:'Retry this section',tutorialSkip:'Skip tutorial and start Task 1',tutorialComplete:'Practice complete',tutorialReady:'Practice is complete. Select Start Task 1 when you are ready.',tutorialPaused:'Practice is paused. Start when you are ready.',tutorialNote:'Practice only. These actions and scores do not count toward your tasks.',operationHelp:'Controls and public rules',practiceTurn:'Practice turn',tutorialGoal:'Your current goal'});
Object.assign(ZH,{tutorial:'厨房练习',tutorialStart:'开始练习',tutorialResume:'继续练习',tutorialPause:'暂停练习',tutorialRetry:'重练当前段',tutorialSkip:'跳过教程，进入 Task 1',tutorialComplete:'练习完成',tutorialReady:'练习已完成。准备好后，点击“开始 Task 1”。',tutorialPaused:'练习已暂停，准备好后开始。',tutorialNote:'独立练习。操作与分数均不计入正式任务。',operationHelp:'操作帮助与公共规则',practiceTurn:'练习回合',tutorialGoal:'当前目标'});
function isKitchenTutorial(){return domain==='kitchen'&&view?.stage==='demo'&&!!view.tutorial;}
function stageLabel(stage){return stage==='demo'&&domain==='kitchen'?tr('tutorial'):tr(stage);}
function localized(value){return typeof value==='string'?value:(value?.[lang]||value?.en||'');}
function kitchenRules(s){return s?.rule_metadata||view?.rule_metadata||{};}
function actionState(){return isKitchenTutorial()?(view.tutorial.state||view.state):view?.state;}
function availableActions(){return view?.actions||view?.tutorial?.actions||[];}
const meta={warehouse:{name:'Warehouse',zh:'协作仓库',en:'Coordinate deliveries, share an aisle, and keep both robots moving.',desc:'协调配送、共用通道，让两台机器人都能顺利完成任务。'},pong:{name:'Cooperative Pong',zh:'合作接球',en:'Find your side, prepare for the next ball, and make the catch together.',desc:'找准分工位置，提前准备，共同接住来球。'},kitchen:{name:'Cooperative Kitchen',zh:'合作厨房',en:'Prepare, cook, and serve. A good handoff keeps the kitchen running.',desc:'备料、烹饪、上菜，用顺畅交接完成出餐。'}};
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,6500);}
const errors={prolific_submission_closed:["This Prolific submission has ended. Please contact the researcher on Prolific if you need help.","此 Prolific 提交已结束。如需帮助，请在 Prolific 联系研究者。"],participant_mode_conflict:['This ID belongs to another study mode. Please use a separate study ID.','此编号属于另一种研究模式，请使用独立研究编号。'],question_busy:['A question is already being answered. Please wait for it to finish.','已有问题正在回答，请等待完成。'],session_required:['Please enter your study ID to begin.','请输入研究编号开始。'],consent_required:['Please read and accept the study information.','请阅读并同意研究说明。'],invalid_participant_id:['Use an anonymous ID with 3–64 letters, numbers, hyphens or underscores.','请使用3–64位英文字母、数字、连字符或下划线作为匿名编号。'],participant_exists_use_recovery:['This ID already exists. Use the original browser or contact the researcher.','此编号已经存在，请使用原浏览器或联系研究者。'],researcher_access_required:['A valid researcher key is required for preview.','预览需要有效的研究者密钥。'],study_not_ready:['The study is not open yet. Please return later.','研究尚未开放，请稍后再来。'],another_domain_active:['Finish or resume your active domain before starting another one.','请先完成或继续正在进行的领域任务。'],stale_state:['The state changed in another window. The latest turn has been restored.','另一窗口已改变任务状态，已恢复最新回合。'],explanations_unavailable:['Questions and previous answers are available only while Task 2 is active.','仅 Task 2 进行中可提问和查看回答。'],incomplete_questionnaire:['Please answer every required question.','请回答所有必填问题。'],finish_demo:['Please complete every demonstration step first.','请先完成全部演示步骤。'],finish_task:['Please finish this task before continuing.','请先完成本任务。'],server_error:['The service could not complete that action. Please try again.','服务未能完成操作，请重试。'],release_changed:['The tasks have been updated. Refresh to start the new version; your earlier records are saved.','玩法已更新。请刷新页面进入新版；旧版记录已保留。']};
async function api(path,payload,extra={}){const options={credentials:'same-origin',cache:'no-store',headers:{...extra}};if(payload){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(payload);}const response=await fetch(path,options);const result=await response.json();if(!response.ok){const e=new Error(result.error||'server_error');e.code=result.error;throw e;}return result;}
function report(e){toast(errors[e.code]?.[lang==='zh'?1:0]||tr('retry'));}
function entryPageOpen(){return !!meta[domain]&&!view?.instance_id&&!!$('startForm');}
function stopEntryCheck(){if(entryTimer!==null){clearTimeout(entryTimer);entryTimer=null;}}
function entryStatusText(){return tr(entryChecking?'entryChecking':entryFailed?'entryNetwork':release?.study_ready?'entryOpen':'entryUnavailable');}
function updateEntryStatus(){
 const box=$('entryStatus'),message=$('entryStatusMessage'),retry=$('entryRetryButton'),start=$('startButton');
 if(box)box.hidden=preview||(!entryChecking&&!entryFailed&&!!release?.study_ready);
 if(message)message.textContent=entryStatusText();
 if(retry){retry.textContent=tr('entryRetry');retry.disabled=entryChecking;}
 if(start)start.disabled=busy||(!preview&&!release?.study_ready);
}
function scheduleEntryCheck(){
 stopEntryCheck();if(!entryPageOpen()||preview||document.hidden||entryChecking||release?.study_ready)return;
 const delay=Math.max(0,ENTRY_RECHECK_MS-(performance.now()-entryCheckedAt));
 entryTimer=setTimeout(()=>{entryTimer=null;checkEntryReadiness();},delay);
}
async function loadEntryRelease(){
 entryCheckedAt=performance.now();const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),12000);
 try{const response=await fetch('/api/release',{credentials:'same-origin',cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('release_unavailable');return await response.json();}
 finally{clearTimeout(timeout);}
}
async function checkEntryReadiness(){
 if(!entryPageOpen()||preview||document.hidden||entryChecking)return;
 if(performance.now()-entryCheckedAt<ENTRY_MANUAL_COOLDOWN_MS){scheduleEntryCheck();return;}
 stopEntryCheck();entryChecking=true;updateEntryStatus();
 try{release=await loadEntryRelease();entryFailed=false;}catch{entryFailed=true;}
 finally{entryChecking=false;updateEntryStatus();scheduleEntryCheck();}
}
function captureEntryDraft(){
 if(!$('startForm'))return null;
 return {participant:$('participantInput').value,group:$('groupInput').value,consent:$('consentInput').checked,key:$('adminInput')?.value};
}
function restoreEntryDraft(draft){
 if(!draft||!$('startForm'))return;
 $('participantInput').value=draft.participant;$('groupInput').value=draft.group;$('consentInput').checked=draft.consent;
 if($('adminInput'))$('adminInput').value=draft.key||'';
}

function languageUI(){if(prolificEntry||view?.prolific){const brand=document.querySelector('.brand');if(brand)brand.href='/prolific/';}if($('headerContext'))$('headerContext').textContent=view?.instance_id?title()+' · '+(view.stage.startsWith('task')?'Task '+view.stage.slice(-1):stageLabel(view.stage)):'HUMAN + AI';document.documentElement.lang=lang==='zh'?'zh-CN':'en';$('englishButton').classList.toggle('active',lang==='en');$('chineseButton').classList.toggle('active',lang==='zh');$('studyBadge').textContent=tr('pilot');$('footerText').textContent=tr('footer');}
async function setLanguage(next){stopHeld();if(busy||animationPending)return;draftQuestion=$('questionInput')?.value||draftQuestion;lang=next;languageUI();if(view?.instance_id)await command('language',{language:lang});else render();}
$('englishButton').onclick=()=>setLanguage('en');$('chineseButton').onclick=()=>setLanguage('zh');
function clearChat(){stopHeld();chatEpoch++;if(view){view.questions=[];view.can_ask=false;}draftQuestion='';asking=false;const box=$('chatLog');if(box)box.replaceChildren();const input=$('questionInput');if(input)input.value='';if($('chatPanel'))$('chatPanel').hidden=true;}
async function refresh(){
 if((!domain&&!prolificEntry)||document.hidden)return;if(busy||animationPending){pendingRefresh=true;return;}
 const epoch=chatEpoch,sequence=++refreshSequence;
 try{const next=await api(prolificEntry?'/api/prolific/session'+location.search:'/api/study/view?domain='+encodeURIComponent(domain));if(epoch!==chatEpoch||sequence!==refreshSequence||document.hidden)return;
 participantId=next.participant_id||participantId;if(prolificEntry&&next.domain)domain=next.domain;
 if(next.instance_id){if(view?.instance_id===next.instance_id&&next.revision<view.revision)return;const entering=!view?.instance_id;if(view?.can_ask&&!next.can_ask)clearChat();view=next;if(entering&&view.language!==lang){await command('language',{language:lang});}else{lang=view.language;languageUI();await render();}}
 else{clearChat();replay=null;view=next;await render();}}
 catch(e){if(e.code!=='session_required')report(e);}
}
async function command(kind,extra={}){
 if(busy||!view?.instance_id)return;const movement=kind==='action'||(kind==='tutorial'&&extra.command==='action');if(!movement)stopHeld();busy=true;updateControlLocks();const epoch=chatEpoch;
 try{const old=view;const next=await api('/api/study/'+kind,{instance_id:view.instance_id,revision:view.revision,command_id:crypto.randomUUID(),timings:takeTimings(),...extra});
 if(view.instance_id===next.instance_id&&next.revision<view.revision)return;
 if(old.can_ask&&!next.can_ask)clearChat();if(epoch!==chatEpoch||document.hidden){next.questions=[];next.can_ask=false;pendingRefresh=true;}
 if(next.stage!==old.stage){replay=null;draftQuestion='';tutorialPauseRequested=false;}if(next.state?.terminal||next.tutorial?.index!==old.tutorial?.index||next.tutorial?.completed)stopHeld();
 view=next;channel?.postMessage({id:view.instance_id,revision:view.revision});await render({animate:movement&&next.tutorial?.index===old.tutorial?.index});
 }catch(e){stopHeld();report(e);if(e.code==='stale_state')pendingRefresh=true;}
 finally{busy=false;updateControlLocks();if(pendingRefresh){pendingRefresh=false;await refresh();}if(tutorialPauseRequested&&isKitchenTutorial()&&view.tutorial.playing){tutorialPauseRequested=false;await tutorialCommand('pause');}else scheduleHeld();}
}
function tutorialCommand(kind,action){
 if(!isKitchenTutorial())return;
 if(kind!=='action')stopHeld();
 if(kind==='start'||kind==='retry')tutorialPauseRequested=false;
 return command('tutorial',{command:kind,...(action?{action}:{})});
}
function pauseTutorial(){stopHeld();tutorialPauseRequested=true;updateControlLocks();if(!busy&&!animationPending&&isKitchenTutorial()&&view.tutorial.playing){tutorialPauseRequested=false;return tutorialCommand('pause');}}
function cardArt(d){if(d==='warehouse')return '<svg viewBox="0 0 220 140"><g fill="#d8e0ed"><rect x="28" y="16" width="45" height="25" rx="5"/><rect x="148" y="100" width="45" height="25" rx="5"/><rect x="99" y="16" width="23" height="42" rx="4"/><rect x="99" y="88" width="23" height="37" rx="4"/></g><path d="M51 80h56V68h66" fill="none" stroke="#a8b8d7" stroke-width="3" stroke-dasharray="5 6"/><rect x="31" y="62" width="35" height="35" rx="11" fill="#4f6ff0"/><rect x="156" y="50" width="35" height="35" rx="11" fill="#f56b3d"/><g fill="white"><circle cx="41" cy="75" r="3"/><circle cx="55" cy="75" r="3"/><circle cx="166" cy="63" r="3"/><circle cx="181" cy="63" r="3"/></g></svg>';if(d==='pong')return '<svg viewBox="0 0 220 140"><g stroke="#ded9c3" stroke-dasharray="3 6"><path d="M42 15v105M90 15v105M135 15v105M180 15v105"/></g><rect x="72" y="30" width="88" height="24" rx="12" fill="#e7b460"/><circle cx="83" cy="42" r="6" fill="#fff9e7"/><circle cx="149" cy="42" r="6" fill="#fff9e7"/><circle cx="40" cy="70" r="9" fill="#7b8daa"/><rect x="57" y="105" width="49" height="14" rx="7" fill="#4f6ff0"/><rect x="127" y="105" width="49" height="14" rx="7" fill="#f56b3d"/></svg>';return '<svg viewBox="0 0 220 140"><rect x="85" y="17" width="43" height="106" rx="8" fill="#ead5c3"/><rect x="86" y="57" width="41" height="29" rx="5" fill="#bd987e"/><rect x="25" y="20" width="34" height="24" rx="5" fill="#dedbc3"/><circle cx="157" cy="32" r="17" fill="#bbb8a5"/><circle cx="157" cy="32" r="11" fill="#d8845d"/><rect x="26" y="76" width="33" height="33" rx="11" fill="#4f6ff0"/><rect x="155" y="82" width="33" height="33" rx="11" fill="#f56b3d"/><circle cx="107" cy="70" r="9" fill="#e4966a"/><path d="M103 60l5-5 4 5" fill="#6e84a2"/></svg>';}
function title(){return meta[domain]?(lang==='zh'?meta[domain].zh:meta[domain].name):'PolicyLens';}
function home(){return `<div class="home-top"><div class="intro"><div class="eyebrow">${tr('eyebrow')}</div><h1>${tr('home')}</h1><p>${tr('homeSub')}</p></div><div class="home-meta"><b>01—03</b><br>${lang==='zh'?'三个任务 · 逐步协作':'Three tasks · One turn at a time'}<br>${lang==='zh'?'演示 → 任务 → 问卷':'Demo → Tasks → Questionnaire'}</div></div><div class="domain-grid">${Object.entries(meta).map(([d,m],i)=>`<a class="domain-card" href="/${d}/${preview?'?preview=1':''}"><span class="card-number">0${i+1} / ${d.toUpperCase()}</span><div class="card-art">${cardArt(d)}</div><h2>${lang==='zh'?m.zh:m.name}</h2><p>${lang==='zh'?m.desc:m.en}</p><div class="card-link">${lang==='zh'?'进入任务':'Enter task'}<span class="arrow">↗</span></div></a>`).join('')}</div>${!release?.study_ready?`<div class="notice">${tr('ready')}</div>`:''}<p class="quiet-note">${tr('notMeasured')}</p>`;}
function welcome(){return `<a class="back-link" href="/">${tr('back')}</a><div class="welcome-layout"><div><div class="eyebrow">${tr('eyebrow')}</div><h1>${title()}</h1><p class="muted">${lang==='zh'?meta[domain].desc:meta[domain].en}</p><div class="card-art">${cardArt(domain)}</div><p class="quiet-note">${domain==='kitchen'?(lang==='zh'?'先完成厨房练习，再进行三个任务，最后填写问卷。':'Try the kitchen controls, complete three tasks, then answer a short questionnaire.'):(lang==='zh'?'先观看演示，再完成三个任务，最后填写问卷。':'Watch the demonstration, complete three tasks, then answer a short questionnaire.')}</p></div><section class="panel"><h2>${lang==='zh'?'准备开始':'Before you begin'}</h2>${view?.previous_version_saved?`<p class="notice">${lang==='zh'?'玩法已更新。旧版记录已保留，请从新版演示开始。':'The tasks have been updated. Your previous records are saved; begin the new demonstration below.'}</p>`:''}<p class="muted">${tr('data')}</p><form id="startForm"><label class="field"><span>${tr('participant')}</span><input id="participantInput" autocomplete="off" minlength="3" maxlength="64" required value="${esc(participantId)}" placeholder="study-001"><small>${tr('participantHint')}</small></label>${preview?`<div class="notice">${tr('preview')} · ${lang==='zh'?'数据不计入真人研究':'Excluded from human-study analysis'}</div><label class="field"><span>${tr('previewKey')}</span><input id="adminInput" type="password" autocomplete="off" required></label>`:''}<label class="field"><span>${tr('group')}</span><select id="groupInput"><option value="A">A</option><option value="B">B</option></select><small>${lang==='zh'?'A 组仅 Task 2 可提问；B 组不提供问答。':'Group A can ask questions in Task 2 only; Group B has no Q&A.'}</small></label><label class="check-field"><input id="consentInput" type="checkbox" required><span>${tr('consent')}</span></label><div id="entryStatus" class="notice" role="status" aria-live="polite" ${preview||release?.study_ready?'hidden':''}><p id="entryStatusMessage">${entryStatusText()}</p><button id="entryRetryButton" type="button" class="button secondary">${tr('entryRetry')}</button></div><button id="startButton" class="button full" aria-describedby="entryStatusMessage" ${!release?.study_ready&&!preview?'disabled':''}>${domain==='kitchen'?tr('tutorialStart'):tr('start')} →</button></form></section></div>`;}
function progress(){const stages=['demo','task1','task2','task3','questionnaire','completed'];const i=stages.indexOf(view.stage);return `<ol class="progress">${stages.slice(0,5).map((s,n)=>`<li class="${n===i?'current':n<i?'done':''}">${n===0?stageLabel('demo'):n===4?tr('questionnaire'):'Task '+n}</li>`).join('')}</ol>`;}
function heading(){return `${domainNavigation()}${progress()}<div class="study-title"><div><div class="eyebrow">${title()}</div><h1>${view.stage.startsWith('task')?'Task '+view.stage.slice(-1):stageLabel(view.stage)}</h1><p>${view.stage==='task1'?tr('baseline'):view.stage==='task2'?(view.can_ask?tr('task2Intro'):tr('control')):view.stage==='task3'?tr('transfer'):''}</p></div><div class="phase-tag"><i class="dot"></i>${view.mode==='pilot'?tr('pilot'):tr('preview')}${view.group?' · '+view.group:''}</div></div>`;}
function itemText(item){if(!item)return tr('empty');if(typeof item==='string')return esc(item);const name=item.label_en||item.name_en;if(name)return esc(lang==='zh'?(item.label_zh||item.name_zh||name):name);return `${esc(tr(item.ingredient||item.recipe||'dish'))} · ${esc(tr(item.stage||'raw'))}`;}
function scoreSuffix(score){return score.score_max===null||score.score_scale==='raw'?'':`<small> / ${score.score_max||100}</small>`;}
function controlsText(){if(domain==='pong')return lang==='zh'?'A / D：左移 / 右移 · 空格：等待。按住方向键连续移动。':'A / D: left / right · Space: wait. Hold a direction to keep moving.';if(domain==='kitchen')return lang==='zh'?'WASD：移动与转向 · E：操作正前方工位 · 空格：等待。':'WASD: move and face a direction · E: use the station in front · Space: wait.';return lang==='zh'?'WASD：移动 · 空格：等待。':'WASD: move · Space: wait.';}
function board(s,isDemo=false){
 const score=s.score||{task_score:0};
 const practice=isDemo&&domain==='kitchen';
 return `<section class="panel board-panel"><div class="board-toolbar"><span>${isDemo?tr(practice?'tutorialNote':'demoNote'):tr('publicInfo')}</span><div class="legend"><span><i class="swatch"></i>${tr('you')}</span><span><i class="swatch ai"></i>${tr('ai')}</span></div></div>
 <div class="scorebar ${isDemo?'demo-scorebar':''} ${practice?'tutorial-scorebar':''}">${isDemo?'':`<div class="metric"><span>${tr('score')}</span><b>${Number(score.task_score||0).toFixed(0)}</b>${scoreSuffix(score)}${scoreFeedback(s)}</div>`}<div class="metric"><span>${tr(practice?'practiceTurn':'turn')}</span><b>${s.turn}</b>${practice?'':`<small> / ${s.max_turns}</small>`}</div>${practice?'':`<div class="metric"><span>${tr('remaining')}</span><b>${Math.max(0,s.max_turns-s.turn)}</b></div>`}</div>
 <div id="boardDrawing" class="board ${s.domain==='pong'?'pong-board':''}"><canvas id="studyCanvas" role="img" aria-label="${esc(title())}" tabindex="0"></canvas></div></section>`;
}
function scoreFeedback(s){const penalties=(s.events||[]).filter(e=>e.type==='waste'&&Number(e.score_delta)<0);return penalties.map(e=>`<strong class="score-penalty" role="status">${Number(e.score_delta)}</strong>`).join('');}
function kitchenMenu(s){
 if(!s?.orders)return '';
 const rules=kitchenRules(s),score=rules.score||{},count=rules.orders_per_task??s.orders.length,signed=value=>Number(value)>0?'+'+Number(value):String(Number(value));
 const scoring=[['served','按时上菜','Serve on time'],['step','每回合','Each turn'],['single_component_discard','丢配料','Trash ingredient'],['combined_dish_discard','丢成品','Trash dish']].filter(([key])=>Number.isFinite(score[key])).map(([key,zh,en])=>`${lang==='zh'?zh:en} ${signed(score[key])}`).join(' · ');
 return `<section class="panel kitchen-menu"><h3>${lang==='zh'?`本轮菜单 · ${count} 道菜`:`Menu · ${count} dishes`}</h3><ol>${s.orders.map((order,i)=>`<li class="menu-${esc(order.status)}"><span>${i+1}. ${esc(tr(order.recipe))}<small>${tr('deadline')} ${order.deadline}</small></span><b>${order.status==='completed'?'✓':order.status==='expired'?(lang==='zh'?'已截止':'Closed'):(lang==='zh'?'待上菜':'To serve')}</b></li>`).join('')}</ol>${scoring?`<p class="hint">${esc(scoring)}</p>`:''}</section>`;
}

function domainNavigation(){if(view?.prolific)return ''; return `<nav class="domain-navigation" aria-label="${lang==='zh'?'切换领域':'Switch domain'}">${Object.entries(meta).map(([key,m])=>`<a href="/${key}/${preview?'?preview=1':''}" ${key===domain?'aria-current="page"':''}>${esc(lang==='zh'?m.zh:m.name)}</a>`).join('')}</nav>`;}
function conciseRules(s){
 if(domain==='kitchen')return `<ul class="rules-list compact-rules">${(view?.public_help||s?.public_help||[]).map(rule=>`<li>${esc(localized(rule))}</li>`).join('')}</ul>`;
 const rules={warehouse:{en:['Pick up a package at A and deliver it to B. Keep an eye on your battery.','WASD: move. Space: wait.'],zh:['从 A 点拿货送到 B 点，同时关注自己的电量。','WASD 移动，空格等待。']},pong:{en:['Choose which balls to catch together: small balls score 1; large balls need both players and score 3. Some arrive together, so you cannot catch everything.','A / D: move left / right. Space: wait.'],zh:['一起取舍接球：小球 1 分；大球需两人配合，得 3 分。有些球同时落下，无法全部接住。','A / D 左右移动，空格等待。']}};
 let html=`<ul class="rules-list compact-rules">${rules[domain][lang].map(rule=>`<li>${esc(rule)}</li>`).join('')}</ul>`;

 return html;
}
const actionLabels={take_tomato:['Take tomato','取番茄'],take_onion:['Take onion','取洋葱'],chop:['Prepare ingredient','切菜备料'],plate:['Plate dish','装盘'],serve:['Serve dish','上菜'],interact:['Interact (E)','交互（E）'],discard:['Discard held item','丢弃手持物品'],interact_handoff:['Use handoff counter','交接台取放'],interact_buffer:['Use buffer counter','暂存台取放'],interact_human_buffer:['Use buffer counter','暂存台取放'],take_handoff:['Take from handoff','从交接台取物'],put_handoff:['Place on handoff','放到交接台'],take_buffer:['Take from buffer','从暂存台取物'],put_buffer:['Place on buffer','放到暂存台']};
function actionLabel(a){return view?.action_labels?.[a]||actionLabels[a]?.[lang==='zh'?1:0]||({up:tr('up'),down:tr('down'),left:tr('left'),right:tr('right'),wait:tr('wait')}[a])||a.replaceAll('_',' ');}
let hypotheticalLane=5;
function questionExamples(){
 const zh=lang==='zh';
 if(domain!=='pong')return [{id:'why',label:zh?'为什么做这个动作？':'Why this action?',text:zh?'你这一回合为什么这么做？':'Why did you take this action this turn?'},{id:'help',label:tr('help'),text:tr('help')},{id:'whatif',label:tr('whatif'),text:tr('whatif')}];
 return [
  {id:'why',label:zh?'为什么做这个动作？':'Why this action?',text:zh?'你这一回合为什么这么做？':'Why did you take this action this turn?'},
  {id:'alternative',label:zh?'为什么不左移？':'Why not move left?',text:zh?'你这一回合为什么没有向左移动？':'Why did you not move left this turn?'},
  {id:'whatif',label:zh?'如果我右移呢？':'What if I move right?',text:zh?'如果我向右移动一步，你会怎么移动？':'If I move one lane right, how will you move?'},
  {id:'help',label:zh?'我应该接哪个球？':'Which ball should I catch?',text:zh?'我离哪个球最近，还剩几回合？我怎样配合你？':'Which ball am I closest to, how many turns until it arrives, and how can I coordinate with you?'}
 ];
}
function exampleQuestion(id){
 if(id==='position')return lang==='zh'?`如果我现在在第 ${hypotheticalLane} 道，你会怎么移动？`:`If I were at lane ${hypotheticalLane} now, how would you move?`;
 return questionExamples().find(q=>q.id===id)?.text||'';
}
function questionPanel(){if(!view.can_ask)return `<section class="panel locked-panel"><h3>${tr('closed')}</h3>${view.stage==='task3'?tr('transfer'):view.stage==='task1'?tr('baseline'):tr('control')}</section>`;return `<section id="chatPanel" class="panel chat-panel"><h3>${tr('ask')}</h3><p class="hint">${tr('task2Intro')}</p><div id="chatLog" class="chat-log">${(view.questions||[]).slice(-1).map(q=>`<div class="chat-message"><div class="chat-question">${esc(q.question)}</div><div class="chat-answer">${esc(q.result?.answer||'')}</div><div class="chat-meta">${tr('turn')} ${q.target_turn}</div></div>`).join('')}</div><textarea id="questionInput" maxlength="2000" placeholder="${tr('questionPlaceholder')}" ${asking?'disabled':''}>${esc(draftQuestion)}</textarea><div class="examples">${questionExamples().map(q=>`<button class="example" data-example="${q.id}">${esc(q.label)}</button>`).join('')}</div>${domain==='pong'?`<label class="hypothetical-lane"><span>${lang==='zh'?'假设我的位置':'Imagine my position'}</span><select id="hypotheticalLane" aria-label="${lang==='zh'?'假设球道':'Hypothetical lane'}">${Array.from({length:9},(_,i)=>i+1).map(n=>`<option value="${n}" ${n===hypotheticalLane?'selected':''}>${lang==='zh'?'第 '+n+' 道':'Lane '+n}</option>`).join('')}</select><button class="example" data-example="position">${lang==='zh'?'你会怎么移动？':'How would you move?'}</button></label>`:''}<button id="askButton" class="button teal full" ${asking?'disabled':''}>${asking?tr('asking'):tr('send')} ↗</button></section>`;}
function replayPanel(s){return `<section class="panel replay-panel"><div class="replay-controls"><span class="replay-label">${tr('replay')}</span><select id="replayTask">${view.task_runs.map(r=>`<option value="${r.id}" ${r.id===(replay?.run_id||view.run_id)?'selected':''}>Task ${r.task}</option>`).join('')}</select><input id="replaySlider" aria-label="${tr('turn')}" type="range" min="0" max="${replay?.max_turn??view.state.turn}" value="${s.turn}"><span class="replay-label">${s.turn}</span><button id="currentButton" class="button secondary small">${tr('current')}</button></div><ul class="event-list">${(s.events||[]).length?s.events.map(e=>`<li>${esc(e[lang]||e.en||e.type)}</li>`).join(''):`<li>${tr('noEvents')}</li>`}</ul></section>`;}
function taskPage(){const s=replay?.state||view.state;if(view.run_status==='completed'&&!replay)return `${heading()}<section class="panel result-panel"><div class="eyebrow">${tr('finishTitle')}</div><div class="result-score">${Number(s.score.task_score).toFixed(0)}${scoreSuffix(s.score)}</div><p class="muted">${tr('finishSub')}</p>${view.stage==='task2'?`<p class="hint">${view.task2_explanation_notice?tr('closed')+'. '+tr('transfer'):tr('control')}</p>`:''}<button id="nextButton" class="button">${view.stage==='task3'?tr('toSurvey'):tr('nextTask')+' '+(Number(view.stage.slice(-1))+1)} →</button></section>${replayPanel(s)}`;return `${heading()}<div class="workspace"><div>${board(s)}${replayPanel(s)}</div><aside class="side-stack"><section class="panel"><h3>${tr('actions')}</h3>${replay?`<div class="notice">${tr('readOnly')}</div>`:''}<div class="action-grid">${view.actions.map(a=>`<button class="action-button ${selectedAction===a?'selected':''}" data-action="${a}" ${replay?'disabled':''}>${esc(actionLabel(a))}</button>`).join('')}</div><p class="control-keys">${controlsText()}</p>${domain==='kitchen'?`<p class="interaction-hint">${(typeof s.interaction==='object'&&s.interaction?.available)?'E: ':''}${esc(typeof s.interaction==='string'?s.interaction:(lang==='zh'?s.interaction?.label_zh:s.interaction?.label_en)||'')}</p>`:''}<p class="hint">${tr('controls')}</p></section>${domain==='kitchen'?kitchenMenu(s):''}${questionPanel()}<section class="panel"><details><summary>${domain==='kitchen'?tr('operationHelp'):tr('rules')}</summary>${conciseRules(s)}</details></section></aside></div>`;}
function demoCaption(text,state){return domain==='warehouse'?String(text).replace(/\btask_\d+\b/g,id=>{const index=(state.orders||[]).findIndex(order=>order.id===id);return index<0?id:'A'+(index+1);}):text;}
function demoStorageKey(){return view?.instance_id?'policylens.demo.'+view.release_id+'.'+view.instance_id:null;}
function demoFrame(){
 const d=view?.demo;if(!d)return 0;
 if(d.index>=d.captions.length)return d.frames.length-1;
 let saved=0;try{saved=Number(sessionStorage.getItem(demoStorageKey())||0);}catch{}
 const checkpoint=d.captions[Math.min(d.index,d.captions.length-1)]?.index||0;
 return Math.max(checkpoint,Math.min(d.frames.length-1,Number.isFinite(saved)?saved:0));
}
function saveDemoFrame(frame){try{sessionStorage.setItem(demoStorageKey(),String(frame));}catch{}}
function currentDemoCaption(frame){return [...view.demo.captions].reverse().find(c=>c.index<=frame)||view.demo.captions[0];}
function tutorialProgress(t){
 const checks=t.index===0?[['moved','Move','移动'],['faced_counter','Face a counter','面向工位'],['waited','Wait','等待']]:t.index===3?[['handoff_item_placed','Place on handoff','放到交接台'],['handoff_item_taken','Take from handoff','从交接台取回'],['human_buffer_item_placed','Place on your counter','放到暂存台'],['human_buffer_item_taken','Take from your counter','从暂存台取回']]:[];
 return checks.length?`<ul class="tutorial-checks">${checks.map(([key,en,zh])=>`<li class="${t.progress?.[key]?'done':''}"><span aria-hidden="true">${t.progress?.[key]?'✓':'○'}</span> ${esc(lang==='zh'?zh:en)}</li>`).join('')}</ul>`:'';
}
function tutorialPreparation(s){const p=s.human?.preparation;if(!p)return '';const label=lang==='zh'?p.label_zh:p.label_en;return `<p class="tutorial-preparation" role="status">${esc(label)} · ${p.completed} / ${p.required} · ${p.ready?tr('prepared'):(lang==='zh'?`还需 ${p.remaining} 次`:`${p.remaining} more interactions`)}</p>`;}
function tutorialPage(){
 const t=view.tutorial,s=actionState(),total=t.total_segments||6,index=Math.min(total-1,Math.max(0,t.index)),message=localized(t.feedback||t.message),interaction=typeof s.interaction==='string'?s.interaction:localized({en:s.interaction?.label_en,zh:s.interaction?.label_zh});
 const completed=!!t.completed,playing=t.playing&&!tutorialPauseRequested;
 const hintIds=[['controls'],[],['prep'],['capacity'],['containers'],['freshness']][index]||[];
 const hints=(view.public_help||[]).filter(rule=>hintIds.includes(rule.id));
 return `${heading()}<div class="workspace kitchen-tutorial"><div>${board(s,true)}</div><aside class="side-stack"><section class="panel tutorial-goal" aria-labelledby="tutorialGoalTitle"><span class="number-label">${tr('step')} ${index+1} / ${total}</span><div class="tutorial-segments" aria-hidden="true">${Array.from({length:total},(_,i)=>`<span class="${i<index||completed?'done':i===index?'current':''}"></span>`).join('')}</div><h2 id="tutorialGoalTitle">${completed?tr('tutorialComplete'):tr('tutorialGoal')}</h2><p id="tutorialGoal" class="tutorial-goal-text" aria-live="polite">${esc(completed?tr('tutorialReady'):localized(t.goal))}</p>${completed?'':tutorialProgress(t)}${index===2?tutorialPreparation(s):''}${!completed&&hints.length?`<div class="tutorial-context">${hints.map(rule=>`<p>${esc(localized(rule))}</p>`).join('')}</div>`:''}${message?`<p id="tutorialFeedback" class="tutorial-feedback" role="status">${esc(message)}</p>`:''}${!playing&&!completed?`<p class="tutorial-paused" role="status">${tr('tutorialPaused')}</p>`:''}<div class="button-row"><button id="tutorialStartButton" class="button" ${playing?'disabled':''}>${completed?tr('startTask'):(s.turn>0||index>0?tr('tutorialResume'):tr('tutorialStart'))}</button><button id="tutorialPauseButton" class="button secondary" ${!playing||completed?'disabled':''}>${tr('tutorialPause')}</button><button id="tutorialRetryButton" class="button secondary">${tr('tutorialRetry')}</button></div>${completed?'':`<button id="tutorialSkipButton" class="tutorial-skip">${tr('tutorialSkip')} →</button>`}</section><section class="panel"><h3>${tr('actions')}</h3><div class="action-grid">${availableActions().map(a=>`<button class="action-button" data-action="${esc(a)}" ${!playing||completed?'disabled':''}>${esc(actionLabel(a))}</button>`).join('')}</div><p class="control-keys">${controlsText()}</p><p class="interaction-hint">${s.interaction?.available?'E: ':''}${esc(interaction)}</p><p class="hint">${tr('controls')}</p></section><section class="panel"><details><summary>${tr('operationHelp')}</summary>${conciseRules(s)}</details></section></aside></div>`;
}
function demoPage(){if(domain==='kitchen')return view.tutorial?tutorialPage():`${heading()}<div class="loading">${tr('loading')}</div>`;const d=view.demo,frame=demoFrame(),done=d.index>=d.captions.length,caption=currentDemoCaption(frame),state=d.frames[frame];return `${heading()}<div class="workspace"><div id="demoBoard">${board(state,true)}</div><aside class="side-stack demo-sidebar"><section class="panel"><h3>${tr('rules')}</h3>${conciseRules(state)}</section><section class="panel demo-caption"><span id="demoProgress" class="number-label">${tr('turn')} ${frame} / ${d.frames.length-1}</span><p id="demoCaption">${esc(demoCaption(caption[lang]||caption.en,state))}</p><div class="button-row"><button id="demoPlayButton" class="button secondary" ${done?'disabled':''}>${done?tr('demoComplete'):tr('demoPlay')}</button><button id="demoPauseButton" class="button secondary" disabled>${tr('demoPause')}</button><button id="demoSkipButton" class="button">${done?tr('startTask'):tr('demoSkip')} →</button></div></section>${domain==='kitchen'?kitchenMenu(state):''}</aside></div>`;}

function surveyPage(){const survey=view.questionnaire;return `${heading()}<section class="panel questionnaire"><h2>${tr('surveyTitle')}</h2><p class="muted">${tr('surveySub')}</p>${view.questionnaire_optional?'<p>You may skip any question you do not wish to answer.</p>':''}<form id="surveyForm">${survey.items.map(q=>`<div class="question-row"><p>${esc(q.text)}</p><div class="rating">${[1,2,3,4,5,6,7].map(n=>`<label><input type="radio" name="rating_${q.id}" value="${n}" ${view.questionnaire_optional?'':'required'}>${n}</label>`).join('')}${q.allow_na?`<label><input type="radio" name="rating_${q.id}" value="na">${tr('na')}</label>`:''}</div><div class="rating-ends"><span>${tr('disagree')}</span><span>${tr('agree')}</span></div></div>`).join('')}<label class="field"><span>${tr('feedback')}</span><textarea name="feedback" rows="3" maxlength="4000"></textarea></label><button class="button">${tr('submit')} →</button></form></section>`;}
function completedPage(){return `${heading()}<div class="task-intro"><div class="eyebrow">${tr('completed')}</div><h1>${tr('thanks')}</h1><p>${tr('saved')}</p></div><div class="results-grid">${view.task_runs.map(r=>`<div class="panel"><div class="eyebrow">Task ${r.task}</div><b>${Number(r.score.task_score).toFixed(0)}</b>${scoreSuffix(r.score)}</div>`).join('')}</div><div class="task-intro">${view.prolific?(view.completion_url?`<p>Your responses have been saved. Return to Prolific to submit your completion for review.</p><a class="button" href="${esc(view.completion_url)}">Return to Prolific →</a><p>If needed, enter this completion code on Prolific: <strong>${esc(view.completion_code)}</strong></p>`:'<p>Your data is saved. Please contact the researcher on Prolific for completion assistance.</p>'):`<a class="button secondary" href="/">${tr('homeButton')} →</a>`}</div>`;}
function nodeKey(node){return node?.nodeType===1?(node.id||node.getAttribute('data-key')):null;}
function reconcileChildren(parent,desired){
 const wanted=[...desired.childNodes];
 for(let i=0;i<wanted.length;i++){
  const fresh=wanted[i],key=nodeKey(fresh);let current=parent.childNodes[i];
  if(key&&nodeKey(current)!==key){const found=[...parent.childNodes].find(n=>nodeKey(n)===key);if(found){parent.insertBefore(found,current||null);current=found;}}
  if(!current){parent.append(fresh.cloneNode(true));continue;}
  if(current.nodeType!==fresh.nodeType||current.nodeName!==fresh.nodeName||nodeKey(current)!==key){parent.replaceChild(fresh.cloneNode(true),current);continue;}
  if(current.nodeType===3){if(current.nodeValue!==fresh.nodeValue)current.nodeValue=fresh.nodeValue;continue;}
  if(current.nodeType!==1)continue;
  if(current.tagName==='CANVAS')continue;
  for(const attr of [...current.attributes])if(!fresh.hasAttribute(attr.name)&&!(current.tagName==='DETAILS'&&attr.name==='open'))current.removeAttribute(attr.name);
  for(const attr of fresh.attributes)if(current.getAttribute(attr.name)!==attr.value)current.setAttribute(attr.name,attr.value);
  if(current.tagName==='TEXTAREA'){if(current!==document.activeElement)current.value=fresh.textContent;continue;}
  reconcileChildren(current,fresh);
 }
 while(parent.childNodes.length>wanted.length)parent.lastChild.remove();
}
async function render({animate=false,displayState=null}={}){
 const entryDraft=captureEntryDraft();saveSurveyDraft();
 if(view?.stage==='questionnaire'&&surveyKey()){try{surveyDraft=JSON.parse(sessionStorage.getItem(surveyKey())||'{}');}catch{surveyDraft={};}}else if(view?.stage==='completed'){sessionStorage.removeItem(surveyKey());surveyDraft={};}
 if(demoPlaying)pauseDemo();
 languageUI();let html;
 if(prolificEntry&&!view?.instance_id)html=window.ProlificEntry.render();else if(!meta[domain])html=home();else if(!view?.instance_id)html=welcome();else if(view.stage==='demo')html=demoPage();else if(view.stage==='questionnaire')html=surveyPage();else if(view.stage==='completed')html=completedPage();else html=taskPage();
 const template=document.createElement('template');template.innerHTML=html;reconcileChildren($('app'),template.content);restoreEntryDraft(entryDraft);bind();
 if($('surveyForm'))for(const[name,value]of Object.entries(surveyDraft))for(const input of $('surveyForm').elements){if(input.name!==name)continue;if(input.type==='radio')input.checked=input.value===value;else input.value=value;}
 const canvas=$('studyCanvas');if(!canvas){boardRenderer?.destroy();boardRenderer=null;return;}
 if(!boardRenderer||boardRenderer.canvas!==canvas){boardRenderer?.destroy();boardRenderer=new window.StudyBoard(canvas);}
 const state=displayState||replay?.state||(isKitchenTutorial()?actionState():view?.stage==='demo'?view.demo?.frames[demoFrame()]:view.state);
 if(state){animationPending=animate;updateControlLocks();await boardRenderer.setState(isKitchenTutorial()?{...state,tutorial_highlights:view.tutorial.highlights||[]}:state,{animate,language:lang});animationPending=false;updateControlLocks();}
}
function bind(){if(prolificEntry&&!view?.instance_id)window.ProlificEntry.bind(async next=>{view=next;domain=next.domain;participantId=next.participant_id;await render();},render);if($('startForm'))$('startForm').onsubmit=async e=>{e.preventDefault();if(busy||(!preview&&!release?.study_ready))return;busy=true;updateEntryStatus();adminKey=$('adminInput')?.value||'';try{view=await api('/api/study/session',{domain,participant_id:$('participantInput').value,language:lang,consent:$('consentInput').checked,mode:preview?'preview':'pilot',group:$('groupInput').value},adminKey?{Authorization:'Bearer '+adminKey}:{});participantId=view.participant_id;adminKey='';render();}catch(err){if(err.code==='study_not_ready'){release={...release,study_ready:false};entryFailed=false;entryCheckedAt=-Infinity;}report(err);}finally{busy=false;updateEntryStatus();scheduleEntryCheck();}};
 if($('entryRetryButton'))$('entryRetryButton').onclick=checkEntryReadiness;updateEntryStatus();scheduleEntryCheck();
 document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{stopHeld();submitAction(b.dataset.action);});
 document.querySelectorAll('.domain-navigation a').forEach(link=>link.onclick=stopHeld);
 if($('nextButton'))$('nextButton').onclick=()=>command('next');
 if($('demoPlayButton'))$('demoPlayButton').onclick=playDemo;
 if($('demoPauseButton'))$('demoPauseButton').onclick=pauseDemo;
 if($('demoSkipButton'))$('demoSkipButton').onclick=()=>{pauseDemo();command('demo_skip');};
 if($('tutorialStartButton'))$('tutorialStartButton').onclick=()=>view.tutorial.completed?command('next'):tutorialCommand('start');
 if($('tutorialPauseButton'))$('tutorialPauseButton').onclick=pauseTutorial;
 if($('tutorialRetryButton'))$('tutorialRetryButton').onclick=()=>tutorialCommand('retry');
 if($('tutorialSkipButton'))$('tutorialSkipButton').onclick=()=>{stopHeld();command('demo_skip');};
 if($('questionInput')){$('questionInput').oninput=e=>draftQuestion=e.target.value;$('questionInput').onfocus=stopHeld;}
 document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{if(busy||animationPending||replayLoading||asking||!view?.can_ask)return;const input=$('questionInput');if(!input)return;draftQuestion=exampleQuestion(b.dataset.example);input.value=draftQuestion;input.focus();});
 if($('hypotheticalLane'))$('hypotheticalLane').onchange=e=>{stopHeld();hypotheticalLane=Number(e.target.value);};
 if($('askButton'))$('askButton').onclick=ask;
 if($('currentButton'))$('currentButton').onclick=returnToCurrentFrame;
 if($('replaySlider'))$('replaySlider').onchange=e=>loadFrame($('replayTask').value,Number(e.target.value));
 if($('replayTask'))$('replayTask').onchange=e=>loadFrame(e.target.value,0);
 if($('surveyForm')){$('surveyForm').oninput=saveSurveyDraft;$('surveyForm').onchange=saveSurveyDraft;}
 if($('surveyForm'))$('surveyForm').onsubmit=e=>{e.preventDefault();const form=new FormData(e.target),answers={};for(const [k,v]of form){if(k.startsWith('rating_'))answers[k.slice(7)]=v==='na'?'na':Number(v);}command('questionnaire',{answers,feedback:form.get('feedback')});};
}
function updateDemoControls(){
 if($('demoPlayButton'))$('demoPlayButton').disabled=busy||demoPlaying||view?.demo?.index>=view?.demo?.captions.length;
 if($('demoPauseButton'))$('demoPauseButton').disabled=!demoPlaying;
 if($('demoSkipButton'))$('demoSkipButton').disabled=busy;
}
function pauseDemo(){demoPlaying=false;demoGeneration++;updateDemoControls();}
async function playDemo(){
 if(busy||demoPlaying||view?.stage!=='demo'||!view.demo)return;
 stopHeld();demoPlaying=true;updateDemoControls();const generation=++demoGeneration,d=view.demo;
 let frame=demoFrame();
 while(frame<d.frames.length-1&&generation===demoGeneration&&view.stage==='demo'&&!document.hidden){
  frame++;saveDemoFrame(frame);const state=d.frames[frame],container=$('demoBoard');if(!container)break;
  const template=document.createElement('template');template.innerHTML=board(state,true);reconcileChildren(container,template.content);
  const caption=currentDemoCaption(frame);if($('demoCaption'))$('demoCaption').textContent=demoCaption(caption[lang]||caption.en,state);
  if($('demoProgress'))$('demoProgress').textContent=tr('turn')+' '+frame+' / '+(d.frames.length-1);
  if(!boardRenderer||boardRenderer.canvas!==$('studyCanvas')){boardRenderer?.destroy();boardRenderer=new window.StudyBoard($('studyCanvas'));}
  await boardRenderer.setState(state,{animate:true,language:lang});
  if(generation===demoGeneration&&frame<d.frames.length-1)await new Promise(resolve=>setTimeout(resolve,250));
 }
 if(generation===demoGeneration){demoPlaying=false;updateDemoControls();if(view.stage==='demo'&&frame===d.frames.length-1)await command('demo_finish');}
}

function cancelReplayLoad(){replayRequestSequence++;replayLoading=false;updateControlLocks();}
function returnToCurrentFrame(){stopHeld();cancelReplayLoad();replay=null;render();}
async function loadFrame(run,turn){
 stopHeld();if(busy||animationPending||asking||!view?.instance_id)return;
 const sequence=++replayRequestSequence,instance=view.instance_id,revision=view.revision;
 replayLoading=true;updateControlLocks();
 try{
  const result=await api(`/api/study/frame?instance_id=${instance}&run_id=${run}&turn=${turn}`);
  if(sequence!==replayRequestSequence||view?.instance_id!==instance||view.revision!==revision||document.hidden)return;
  const metaRun=view.task_runs.find(r=>r.id===run);
  const maxTurn=run===view.run_id?view.state.turn:(metaRun?.turn??result.state.max_turns);
  replay={state:result.state,run_id:run,max_turn:maxTurn};if(!result.can_ask)clearChat();await render();
 }catch(e){if(sequence===replayRequestSequence)report(e);}
 finally{if(sequence===replayRequestSequence){replayLoading=false;updateControlLocks();}}
}
async function ask(){stopHeld();if(asking||busy||animationPending||replayLoading||!view.can_ask)return;const question=$('questionInput').value.trim();if(!question)return;const epoch=chatEpoch;asking=true;draftQuestion=question;render();const target=replay?.run_id||view.run_id,turn=(replay?.state||view.state).turn;const started=performance.now();try{const answer=await api('/api/study/ask',{instance_id:view.instance_id,authorized_run:view.run_id,target_run:target,turn,question,question_id:crypto.randomUUID(),language:lang});if(epoch!==chatEpoch||document.hidden)return;await refresh();if(view.can_ask&&epoch===chatEpoch){await api('/api/study/answer-displayed',{instance_id:view.instance_id,question_id:answer.id});draftQuestion='';render();$('chatLog')?.scrollTo(0,$('chatLog').scrollHeight);} }catch(e){report(e);}finally{if(epoch===chatEpoch){asking=false;render();}if(view?.can_ask&&!busy)command('timing',{kind:'explanation_wait',seconds:(performance.now()-started)/1000});}}
function stopHeld(){heldDirection=null;if(heldTimer){clearTimeout(heldTimer);heldTimer=null;}}
function editing(){return /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)||document.activeElement?.isContentEditable;}
function inputStageActive(){return isKitchenTutorial()?view.tutorial.playing&&!view.tutorial.completed&&!tutorialPauseRequested:view?.stage?.startsWith('task')&&!view.state?.terminal;}
function canMove(){return !busy&&!animationPending&&!replayLoading&&!asking&&!replay&&!document.hidden&&!editing()&&inputStageActive()&&!!actionState();}
function updateControlLocks(){updateDemoControls();for(const id of ['englishButton','chineseButton'])if($(id))$(id).disabled=busy||animationPending;for(const button of document.querySelectorAll('[data-action]'))button.disabled=busy||animationPending||replayLoading||asking||!!replay||!inputStageActive()||!availableActions().includes(button.dataset.action);for(const id of ['tutorialStartButton','tutorialRetryButton','tutorialSkipButton'])if($(id))$(id).disabled=busy||animationPending||(id==='tutorialStartButton'&&view?.tutorial?.playing&&!view?.tutorial?.completed);if($('tutorialPauseButton'))$('tutorialPauseButton').disabled=!view?.tutorial?.playing||view.tutorial.completed||tutorialPauseRequested;if($('hypotheticalLane'))$('hypotheticalLane').disabled=busy||animationPending||replayLoading||asking||!view?.can_ask;if($('askButton'))$('askButton').disabled=busy||animationPending||replayLoading||asking||!view?.can_ask;for(const button of document.querySelectorAll('[data-example]'))button.disabled=busy||animationPending||replayLoading||asking||!view?.can_ask;}
function submitAction(action){if(!canMove())return;if(!availableActions().includes(action)&&!(isKitchenTutorial()&&action==='interact')){if(domain==='kitchen'&&action==='interact')toast(lang==='zh'?actionState().interaction?.label_zh:actionState().interaction?.label_en);return;}selectedAction=action;return isKitchenTutorial()?tutorialCommand('action',action):command('action',{run_id:view.run_id,turn:view.state.turn,action});}
function scheduleHeld(){if(heldTimer||!heldDirection||!canMove())return;if(!availableActions().includes(heldDirection)){stopHeld();return;}heldTimer=setTimeout(()=>{heldTimer=null;if(heldDirection&&canMove())submitAction(heldDirection);},0);}
function keyAction(key){key=key.toLowerCase();if(key===' ')return 'wait';if(domain==='pong')return {a:'left',d:'right'}[key];return {a:'left',d:'right',w:'up',s:'down',arrowleft:'left',arrowright:'right',arrowup:'up',arrowdown:'down',...(domain==='kitchen'?{e:'interact'}:{})}[key];}
document.addEventListener('keydown',e=>{
 if(e.ctrlKey||e.metaKey||e.altKey||editing()||!inputStageActive()||replay||asking||document.hidden)return;
 const action=keyAction(e.key);if(!action)return;e.preventDefault();if(e.repeat)return;
 if(['up','down','left','right'].includes(action)){heldDirection=action;if(canMove())submitAction(action);}else{stopHeld();if(canMove())submitAction(action);}
});
document.addEventListener('keyup',e=>{if(keyAction(e.key)===heldDirection)stopHeld();});
window.addEventListener('blur',()=>{stopHeld();pauseDemo();});
document.addEventListener('focusin',e=>{if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))stopHeld();});
channel&&(channel.onmessage=e=>{if(view?.instance_id===e.data.id&&e.data.revision!==view.revision){clearChat();refresh();}});
document.addEventListener('visibilitychange',()=>{sampleTime();if(document.hidden){stopEntryCheck();stopHeld();cancelReplayLoad();pauseDemo();boardRenderer?.stop();clearChat();}else{refresh();scheduleEntryCheck();}});
window.addEventListener('pageshow',e=>{if(e.persisted){clearChat();refresh();}});
window.addEventListener('focus',()=>{if(view?.can_ask){clearChat();refresh();}});
window.addEventListener('offline',()=>{cancelReplayLoad();clearChat();});
(async()=>{if(prolificEntry){try{await window.ProlificEntry.load();}catch{} }languageUI();try{release=await loadEntryRelease();entryFailed=false;}catch{entryFailed=true;}try{if(domain||prolificEntry)await refresh();render();}catch(e){$('app').textContent=tr('retry');report(e);}})();
