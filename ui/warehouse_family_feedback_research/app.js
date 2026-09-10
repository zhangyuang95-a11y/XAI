/* Server-authoritative native warehouse UI. No policy, scoring or explanation generation here. */
(() => {
  "use strict";
  const FRONTEND_VERSION="warehouse-family-feedback-research.v2";
  const PENDING_KEY="warehouse-family-feedback-fresh.v2.pending", LANGUAGE_KEY="warehouse-family-feedback-fresh.v2.lang";
  function local(value,language){return typeof value==="string"?value:value?.[language] || value?.[language==="zh"?"zh-CN":"en"] || value?.en || value?.zh || "";}
  function accessibilityLabels(language){return language==="en"?{
    publicHistoryPanel:"Previous confirmed transition",warehouseCanvas:"Warehouse map: blue player 1 and orange AI teammate 2",previousFrame:"Previous frame",historySlider:"Playback frame",workflow:"Study workflow",languageButton:"Switch to Chinese",
  }:{publicHistoryPanel:"上一回合公开记录",warehouseCanvas:"仓库地图：蓝色玩家 1，橙色 AI 队友 2",previousFrame:"上一帧",historySlider:"回放帧",workflow:"研究流程",languageButton:"切换为英文"};}
  function releaseMessageKey(view){const r=view?.release || {};if(!view)return "connecting";if(view.study_version_mismatch)return "studyVersionChanged";if(view.verification_only && r.test_fixture)return "testFixtureMessage";if(r.capability_status==="failed")return view.verification_only?"verificationFailedMessage":"capabilityFailedMessage";if(view.verification_only)return "verificationMessage";if(r.study_ready===true)return "localStudyMessage";if(r.capability_status==="unknown" && r.model_ready)return "capabilityUnknownMessage";if(r.formal_ready)return "readyMessage";if(r.model_ready)return "candidateMessage";return /fail|rejected/.test(String(r.status))?"failedGate":"modelGate";}
  function releaseBadgeKey(view){const r=view?.release || {},failed=/fail|rejected/.test(String(r.status));return !view?"connecting":view.verification_only?"verification":r.capability_status==="failed"?"capabilityFailed":r.study_ready===true?"localStudy":r.formal_ready?"ready":r.model_ready?"candidate":failed?"failedModel":r.status && r.status!=="unknown"?"training":"unknown";}
  function normalizeView(value){
    const run=value?.run || {}, inner=run.view || {};
    return {...inner,...run,...value,run_id:value?.run_id ?? run.id ?? run.run_id ?? null,
      state:value?.state ?? run.state ?? inner.state ?? null,map:value?.map ?? run.map ?? inner.map ?? null,
      metrics:value?.metrics ?? run.metrics ?? inner.metrics ?? {},
      flow:value?.flow ?? run.flow ?? {mode:value?.mode || run.mode || "freeplay",stage:value?.stage || run.stage || "freeplay"},
      release:value?.release || {status:"unknown",model_ready:false,explanation_ready:false,formal_ready:false}};
  }
  function phase(view){return view?.flow?.stage || "freeplay";}
  function displayStage(view){const p=phase(view);return study(view)?(p==="task1_complete"?"task1":p):view?.run_id?p:"notStarted";}
  function study(view){return view?.flow?.mode==="study";}
  function ended(view){return Boolean(view?.state?.terminated || view?.state?.truncated || view?.done || view?.ended || view?.run?.done);}
  function verificationFlow(view){return view?.verification_only===true && view?.verification_flow_allowed===true && view?.release?.test_fixture===true;}
  function playable(view){return !view?.study_version_mismatch && (view?.release?.model_ready===true || verificationFlow(view));}
  function canEnter(view,mode){if(view?.study_version_mismatch)return false;if(mode==="freeplay")return playable(view);return mode==="study" && ((view?.release?.study_ready===true || view?.release?.formal_ready===true) && view?.study_allowed===true || verificationFlow(view));}
  function allowed(view,kind){
    if(view?.study_version_mismatch)return false;
    const list=view?.allowed_kinds || view?.allowed_commands || view?.flow?.allowed_kinds;
    if(Array.isArray(list))return list.includes(kind);
    if(view?.permissions && typeof view.permissions[kind]==="boolean")return view.permissions[kind];
    if(kind==="action")return view?.release?.model_ready===true && !!view.run_id && !view.version_mismatch && !ended(view) && ["freeplay","practice","task1","task2"].includes(phase(view));
    if(kind==="end")return !study(view) && !!view?.run_id && !ended(view);
    if(kind==="next")return study(view) && ended(view) && ["practice","task1","task1_complete","task2"].includes(phase(view));
    return true;
  }
  function explanationsAllowed(view){return !view?.study_version_mismatch && allowed(view,"question") && view?.explain_allowed===true && (view?.release?.explanation_ready===true || verificationFlow(view)) && ["freeplay","task1","task1_complete"].includes(phase(view));}
  function visibleAnswers(view){
    if(!explanationsAllowed(view))return [];
    return (Array.isArray(view.answers)?view.answers:[]).filter(a=>!a.run_id || a.run_id===view.run_id).map(a=>({
      id:a.id || a.answer_id || a.question_id,status:a.status || (a.text?"ready":"pending"),
      question:String(a.question || ""),text:typeof a.text==="string"?a.text:typeof a.answer==="string"?a.answer:"",
      frame:a.frame ?? a.selected_frame ?? null,sources:Array.isArray(a.sources)?a.sources.filter(s=>typeof s==="string"):[],
    }));
  }
  function keyboardAction(event){
    if(event.ctrlKey || event.metaKey || event.altKey)return null;
    const tag=String(event.target?.tagName || "").toUpperCase();
    if(["INPUT","TEXTAREA","SELECT"].includes(tag) || event.target?.isContentEditable)return null;
    if(["BUTTON","SUMMARY"].includes(tag) && [" ","Enter"].includes(event.key))return null;
    return {ArrowUp:"UP",w:"UP",W:"UP",ArrowDown:"DOWN",s:"DOWN",S:"DOWN",ArrowLeft:"LEFT",a:"LEFT",A:"LEFT",ArrowRight:"RIGHT",d:"RIGHT",D:"RIGHT"," ":"WAIT"}[event.key] || null;
  }
  function canAct(view,replay,busy,pending,historyLoading=false){return !!view?.run_id && playable(view) && !view.version_mismatch && !ended(view) && ["freeplay","practice","task1","task2"].includes(phase(view)) && allowed(view,"action") && replay===null && !busy && !pending && !historyLoading;}
  // A lost terminal response may already have changed the permitted stage.
  // Confirm its unchanged operation ID; never send it across a bound-version change.
  function canSubmit(view,kind,retry=false){return !!view && !view.study_version_mismatch && (retry || allowed(view,kind));}
  function readLocked(view,busy,pending){return busy || !!pending && !view?.study_version_mismatch;}
  function frameMetrics(frame){const m=frame?.metrics || {},s=frame?.state || {};return {
    deliveries:m.deliveries ?? s.total_deliveries ?? 0,score:m.score ?? null,legacy_score:m.legacy_score ?? null,
    steps:m.steps ?? s.frame ?? 0,collisions:m.collisions ?? s.robot_collision_events ?? s.collision_count ?? 0,
    shutdowns:m.shutdowns ?? s.shutdown_count ?? 0,
  };}
  const MOTION_DURATION_MS=380;
  function confirmedMotion(before,after,kind,replay){
    if(kind!=="action" || replay!==null || !before || !after
        || before.session_id!==after.session_id || before.run_id!==after.run_id
        || !Number.isInteger(before.version) || after.version!==before.version+1
        || !Number.isInteger(before.state?.frame) || after.state?.frame!==before.state.frame+1)return null;
    const previous=new Map((before.state?.agents || []).map(agent=>[agent.id,agent]));
    const agents={};let moved=false;
    for(const agent of after.state?.agents || []){
      const old=previous.get(agent.id),from=old?.position,to=agent.position;
      if(!Array.isArray(from) || !Array.isArray(to) || from.length!==2 || to.length!==2
          || !from.every(Number.isFinite) || !to.every(Number.isFinite))continue;
      agents[agent.id]={from:[...from],to:[...to]};
      if(from[0]!==to[0] || from[1]!==to[1])moved=true;
    }
    return moved?{agents,duration:MOTION_DURATION_MS}:null;
  }
  function easeMotion(progress){const t=Math.max(0,Math.min(1,Number(progress) || 0));return t*t*(3-2*t);}
  function interpolateMotion(motion,progress){const t=easeMotion(progress),positions={};for(const [id,path] of Object.entries(motion?.agents || {}))positions[id]=[path.from[0]+(path.to[0]-path.from[0])*t,path.from[1]+(path.to[1]-path.from[1])*t];return positions;}
  function readStorage(storage, key) { try { return storage.getItem(key); } catch { return null; } }
  function makeTransport({fetchImpl,storage,onPending=()=>{}}) {
    let pending = null, sending = false;
    try { const old = JSON.parse(readStorage(storage,PENDING_KEY)); if (old?.body?.operation_id && Number.isInteger(old.body.expected_version)) pending = old; } catch { /* no previous request */ }
    function persist() {
      try { if(pending) storage.setItem(PENDING_KEY,JSON.stringify(pending)); else storage.removeItem(PENDING_KEY); }
      catch { /* in-memory retry remains available if browser storage is disabled */ }
      onPending(pending);
    }
    async function request(path, options={}) {
      const signal = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = signal ? setTimeout(() => signal.abort(),15000) : null;
      try {
        const response = await fetchImpl(path,{cache:"no-store",credentials:"same-origin",...options,headers:{"Content-Type":"application/json",...(options.headers || {})},...(signal ? {signal:signal.signal} : {})});
        let value;
        try { value = await response.json(); } catch { const e=new Error("invalid_response");e.status=response.status;throw e; }
        if(!response.ok) { const e=new Error(typeof value.error === "string" ? value.error : value.error?.message || "request_failed");e.status=response.status;e.payload=value;throw e; }
        return value;
      } finally { if(timer) clearTimeout(timer); }
    }
    async function send() {
      if(!pending || sending) throw new Error("request_pending");
      sending = true;
      try {
        const result=await request("/api/command",{method:"POST",body:JSON.stringify(pending.body)});
        pending=null;persist();return result;
      } catch(error) {
        // These are explicit rejections, not uncertain server/network outcomes.
        if([400,401,403,404,409,410,422].includes(error.status)){pending=null;persist();}
        throw error;
      } finally { sending=false; }
    }
    return {request,pending:()=>pending,sending:()=>sending,
      async submit(body,sessionId){if(pending || sending)throw new Error("request_pending");pending={session_id:sessionId,body:{...body}};persist();return send();},
      retry:send,
      bindSession(sessionId){if(pending?.session_id && sessionId && pending.session_id!==sessionId){pending=null;persist();return false;}return true;},
    };
  }
  function framePublicHistory(frame){return frame?.state?.public_feedback ?? frame?.public_feedback ?? null;}
  function publicHistoryLines(history,language){
    const zh=language!=="en";
    if(history?.valid!==true)return [zh?"尚无上一回合的已确认记录。":"No preceding confirmed transition.","—","—"];
    const actions=zh?{UP:"上",DOWN:"下",LEFT:"左",RIGHT:"右",WAIT:"等待"}:{UP:"Up",DOWN:"Down",LEFT:"Left",RIGHT:"Right",WAIT:"Wait"};
    const count=n=>Number.isInteger(n) && n>=0?String(n):"—";
    const canceled=id=>{const flag=history.move_canceled?.[id],n=count(history.consecutive_move_canceled?.[id]);return (flag===true?(zh?"是":"Yes"):flag===false?(zh?"否":"No"):"—")+(zh?`（连续 ${n}）`:` (${n} consecutive)`);};
    const kinds=zh?{none:"无",same_target:"争用同格",swap:"交换位置",occupied_stationary:"目标格被停留者占用"}:{none:"None",same_target:"Same destination",swap:"Position swap",occupied_stationary:"Stationary occupant"};
    const a=id=>actions[history.submitted_actions?.[id]] || "—";
    return zh?[`提交动作：你 ${a("robot_1")} · AI 2 ${a("robot_2")}`,`移动取消：你 ${canceled("robot_1")} · AI 2 ${canceled("robot_2")}`,`机器人冲突：${kinds[history.collision_kind] || "—"} · 连续 ${count(history.consecutive_collision)} 步`]:[`Submitted: You ${a("robot_1")} · AI 2 ${a("robot_2")}`,`Move canceled: You ${canceled("robot_1")} · AI 2 ${canceled("robot_2")}`,`Robot conflict: ${kinds[history.collision_kind] || "—"} · ${count(history.consecutive_collision)} consecutive turns`];
  }
  if(typeof module!=="undefined" && module.exports)module.exports={FRONTEND_VERSION,MOTION_DURATION_MS,verificationFlow,playable,framePublicHistory,publicHistoryLines,PENDING_KEY,local,accessibilityLabels,releaseMessageKey,releaseBadgeKey,normalizeView,phase,displayStage,study,ended,canEnter,allowed,explanationsAllowed,visibleAnswers,keyboardAction,canAct,canSubmit,readLocked,frameMetrics,confirmedMotion,easeMotion,interpolateMotion,makeTransport};
  if(typeof document==="undefined")return;
  const $=id=>document.getElementById(id);
  const WORDS={
    zh:{title:"仓库神经队友",connecting:"正在连接",warehouse:"物流仓库",delivery:"协作配送",confirmed:"已确认步数",consent:"参与说明与同意",practice:"操作练习",questionnaire:"问卷",completed:"完成",deliveries:"配送",score:"得分",steps:"步数",collisions:"碰撞",shutdowns:"停机",legacyScore:"原计分（参考）",you:"你 · 1",shelf:"货架",jobs:"取货／交付",charger:"充电",live:"回到当前",currentStage:"当前阶段",freeplay:"自由试玩",study:"研究流程",separateRuns:"新回合单独保存，旧记录保留。",participantId:"用户 ID",consentText:"研究流程会记录操作、问答和问卷。请阅读研究者提供的参与说明后确认。",agree:"我已阅读参与说明并同意参加。",startFreeplay:"开始试玩",startStudy:"同意并开始练习",controls:"操作控制",controlHint:"点击地图后：方向键／WASD 移动，空格等待，每次一步。按钮聚焦时，空格操作按钮。",up:"上",down:"下",left:"左",right:"右",wait:"等待",end:"结束本局",next:"继续",retry:"重试待确认操作",explanation:"行为解释",why:"为什么刚才这样行动",alternative:"为什么没有选另一动作",counterfactual:"如果我等待三步会怎样",questionFocus:"问题对象",executed:"已发生的动作",nextState:"所选状态的下一步",questionInput:"输入问题",ask:"提问",draftNotice:"可保存草稿，刷新后继续。",previewQuestion:"查看题目地图",predictionCandidate:"请根据题目所选帧的地图和公开记录作答。",predictionFailed:"行为预测题库检查未通过；当前问卷仅含自评。",predictionUnavailable:"行为预测题库尚未就绪；当前问卷仅含自评。",predictionMismatch:"题库版本与本次登记不一致，暂不能提交。",questionMap:"题目地图 · 帧",saveDraft:"保存草稿",submitQuestionnaire:"提交问卷",completedText:"本次流程已完成，记录已保存。",feedback:"试玩反馈",clarity:"操作清晰度",cooperation:"合作感",note:"备注",saveFeedback:"保存反馈",savedRuns:"已保存回合",details:"版本与状态",modelVersion:"模型版本",runId:"回合 ID",technicalNote:"候选试玩不等于正式研究验收；原计分若未提供则显示“—”。",verification:"流程验证 · 非正式实验",verificationMessage:"隔离流程验证环境；若为测试权重，不代表训练或能力验收。",testFixtureMessage:"界面与流程测试 · 随机初始化测试权重，不代表模型能力。",verificationFailedMessage:"流程验证 · 能力未通过，仅用于候选诊断试玩。",capabilityFailed:"能力未通过 · 候选",capabilityFailedMessage:"能力未通过 · 仅用于候选诊断试玩。",capabilityUnknownMessage:"能力尚未验收 · 仅用于候选试玩。",candidate:"候选试玩 · 非正式实验",localStudy:"本地预实验 · 已核验",localStudyMessage:"本地预实验 · 已核验。当前记录不作为正式研究样本。",studyVersionChanged:"本次研究绑定的版本已变更，请联系研究者；已有记录保留。",ready:"研究版本已就绪",training:"模型训练／检查中",failedModel:"模型检查未通过",unknown:"模型状态尚未确认",modelGate:"当前模型尚不能试玩；请等待服务端明确开放。",failedGate:"模型检查未通过，继续刷新不会使它自动开放。",studyGate:"本地预实验核验尚未通过，研究入口暂不开放。",explanationGate:"解释检查尚未完成，本次试玩不提供解释。",candidateMessage:"可进行候选试玩；当前尚非正式研究版本。",readyMessage:"请按研究流程完成练习、任务与问卷。",questionPending:"问题已保存，正在获取核验答案。",answerUnavailable:"本题尚无可核验答案。",answerFrame:"回答绑定帧",source:"来源",ai:"AI · 2",frame:"所选帧",replay:"回放",liveFrame:"当前帧",paused:"回放期间操作暂停",ended:"本局已结束",battery:"电量",empty:"空载",carrying:"携带",last:"上步实际动作",synced:"已同步",busy:"正在确认…",pending:"操作未确认；重试不会重复推进。",requestFailed:"请求未完成，请检查本地服务。",conflict:"另一页面更新了进度，请检查后再操作。",oldSession:"原会话已失效，旧请求没有重放。",invalidSeed:"请选择当前开放范围内的关卡编号。",scene:"关卡编号",consentRequired:"请填写用户 ID 并确认参与说明。",participantTaken:"此用户 ID 已登记，请换一个编号。",participantInvalid:"用户 ID 须以英文字母开头，包含 3–32 位字母、数字、下划线或连字符。",questionRequired:"请输入问题。",questionnaireMissing:"问卷尚未加载，暂不能提交。",answersRequired:"请完成所有必答题。",feedbackRequired:"请选择两项评分。",noHistory:"历史记录暂不可用",runtimeChanged:"运行版本已变更，请开始新局；旧记录保留。",notStarted:"尚未开始",selectOption:"请选择",round:"小回合",rules:"操作规则"},
    en:{title:"Warehouse neural teammate",connecting:"Connecting",warehouse:"Warehouse",delivery:"Collaborative delivery",confirmed:"Confirmed steps",consent:"Information & consent",practice:"Practice",questionnaire:"Questionnaire",completed:"Completed",deliveries:"Deliveries",score:"Score",steps:"Steps",collisions:"Collisions",shutdowns:"Shutdowns",legacyScore:"Legacy score (ref.)",you:"You · 1",shelf:"Shelf",jobs:"Pickup / delivery",charger:"Charger",live:"Return live",currentStage:"Current stage",freeplay:"Freeplay",study:"Study",separateRuns:"New runs preserve earlier attempts.",participantId:"Participant ID",consentText:"The study records actions, questions and questionnaire responses. Read the participation information supplied by the researcher before agreeing.",agree:"I have read the participation information and consent to take part.",startFreeplay:"Start freeplay",startStudy:"Consent and begin practice",controls:"Controls",controlHint:"Click the map: arrows / WASD move; Space waits. One input, one step. Space activates focused buttons.",up:"Up",down:"Down",left:"Left",right:"Right",wait:"Wait",end:"End run",next:"Continue",retry:"Retry pending request",explanation:"Behavior explanation",why:"Why did you do that?",alternative:"Why not another action?",counterfactual:"What if I wait three steps?",questionFocus:"Question reference",executed:"An executed action",nextState:"Next step from selected state",questionInput:"Enter a question",ask:"Ask",draftNotice:"Save a draft and resume after refresh.",previewQuestion:"View question map",predictionCandidate:"Use the selected question frame and its public history to answer.",predictionFailed:"Prediction-bank checks failed. This questionnaire contains self-reports only.",predictionUnavailable:"Prediction bank is unavailable. This questionnaire contains self-reports only.",predictionMismatch:"The bank version differs from enrollment. Submission is unavailable.",questionMap:"Question map · Frame",saveDraft:"Save draft",submitQuestionnaire:"Submit questionnaire",completedText:"This study is complete. Your records have been saved.",feedback:"Freeplay feedback",clarity:"Control clarity",cooperation:"Sense of cooperation",note:"Note",saveFeedback:"Save feedback",savedRuns:"Saved runs",details:"Version and status",modelVersion:"Model version",runId:"Run ID",technicalNote:"Candidate freeplay is not formal research approval. An unavailable legacy score is shown as “—”.",verification:"Flow verification · Not a study",verificationMessage:"Isolated flow verification. Test weights are not training or capability evidence.",testFixtureMessage:"UI and flow tests · Random-initialized test weights, not model capability evidence.",verificationFailedMessage:"Flow verification · Capability failed. Diagnostic candidate play only.",capabilityFailed:"Capability failed · Candidate",capabilityFailedMessage:"Capability checks failed · Candidate diagnostic play only.",capabilityUnknownMessage:"Capability is not yet verified · Candidate play only.",candidate:"Candidate freeplay · Not formal",localStudy:"Local pilot · Verified",localStudyMessage:"Local pilot · Verified. These records are not formal research samples.",studyVersionChanged:"The version bound to this study has changed. Contact the researcher; existing records are preserved.",ready:"Study version ready",training:"Model training / checks",failedModel:"Model checks failed",unknown:"Model status unconfirmed",modelGate:"The model is not playable yet. Wait for explicit server readiness.",failedGate:"Model checks failed. Refreshing will not automatically unlock play.",studyGate:"Local pilot checks are incomplete. Study enrollment remains closed.",explanationGate:"Explanation checks are incomplete. Explanations are unavailable for this run.",candidateMessage:"Candidate freeplay is available. This is not a formal study release.",readyMessage:"Follow the practice, task and questionnaire sequence.",questionPending:"Question saved. A verified answer is being prepared.",answerUnavailable:"No verified answer is available for this question.",answerFrame:"Answer frame",source:"Source",ai:"AI · 2",frame:"Selected frame",replay:"Replay",liveFrame:"Live frame",paused:"Controls paused during replay",ended:"Run ended",battery:"Battery",empty:"Empty",carrying:"Carrying",last:"Last executed actions",synced:"Synced",busy:"Confirming…",pending:"Request unconfirmed. Retry is safe.",requestFailed:"Request incomplete. Check the local service.",conflict:"Another page updated progress. Review before acting.",oldSession:"Earlier session expired; its request was not replayed.",invalidSeed:"Choose a scene number within the available range.",scene:"Scene number",consentRequired:"Enter an ID and confirm the participation information.",participantTaken:"This ID is already registered. Choose another ID.",participantInvalid:"Use 3–32 letters, digits, underscores or hyphens, beginning with a letter.",questionRequired:"Enter a question.",questionnaireMissing:"The questionnaire has not loaded. Submission is unavailable.",answersRequired:"Complete every required question.",feedbackRequired:"Select both ratings.",noHistory:"History is unavailable",runtimeChanged:"Runtime changed. Start a new run; earlier records are preserved.",notStarted:"Not started",selectOption:"Select an option",round:"Round",rules:"Controls"},
  };
  Object.assign(WORDS.zh,{publicHistory:"上一回合公开记录",publicRules:"玩法规则",publicRulesText:"前往 A 自动取货，运到同色 B 自动交付。每局最多 120 步，方向键移动，空格等待。成功移动一格消耗 2% 电量；在充电格实际停留一步最多恢复 10%，上限 100%。撞墙、争用同格或交换位置会取消移动，仍消耗一步。任一机器人在充电格外耗尽电量时结束本局。",publicScoreText:"得分：每次配送 +100，机器人碰撞 −200，断电 −50，每步 −1。回放不推进时间。",candidateMessage:"真实 RL 候选队友；目前仅开放自由试玩。",capabilityFailedMessage:"真实 RL 候选队友，能力尚未达标；目前仅开放自由试玩。",explanationGate:"当前回合不开放行为解释。"});
  Object.assign(WORDS.en,{publicHistory:"Previous confirmed transition",publicRules:"Task rules",publicRulesText:"Visit A to pick up automatically, then deliver to the matching-color B. A run lasts at most 120 turns. Arrows move; Space waits. Each successful move costs 2% battery. Staying on the charger restores up to 10% per turn, capped at 100%. Walls, conflicting destinations and position swaps cancel moves but consume a turn. A run ends if either robot runs out of battery away from the charger.",publicScoreText:"Score: +100 per delivery, −200 per robot collision, −50 per shutdown, −1 per turn. Playback does not advance time.",candidateMessage:"Trained RL candidate. Freeplay is available; study enrollment is closed.",capabilityFailedMessage:"Trained RL candidate; capability checks have not passed. Freeplay only.",explanationGate:"Behavior explanations are unavailable in this run."});
  Object.assign(WORDS.zh,{historyLoading:"正在加载历史；操作已暂停。",historyLoadFailed:"历史加载失败；回放未切换，可以重试。",historyChanged:"当前会话或进度已变更；旧历史请求已取消。"});
  Object.assign(WORDS.en,{historyLoading:"Loading history; controls are paused.",historyLoadFailed:"History could not be loaded. Playback did not change; retry.",historyChanged:"The session or progress changed; the old history request was canceled."});
  const ui={view:null,entry:"study",language:readStorage(localStorage,LANGUAGE_KEY)==="en"?"en":"zh",busy:false,error:null,replay:null,history:null,historyToken:0,historyLoading:null,refreshing:false,userNavigationEpoch:0,viewportEpoch:0,initialized:false,seen:new Set(),seenQueue:new Set(),seenTimer:null,questionnaireDirty:false,questionnairePreview:null,feedbackRun:null,answerSignature:null,motion:null,motionToken:0};
  const tr=k=>WORDS[ui.language][k] || k;
  const transport=makeTransport({fetchImpl:(...a)=>fetch(...a),storage:localStorage,onPending:()=>{if(ui.initialized)render();}});
  function shown(){if(study(ui.view) && phase(ui.view)==="questionnaire"){const item=questionnaireItems(ui.view).find(i=>i.id===ui.questionnairePreview) || questionnaireItems(ui.view).find(i=>i.preview);if(item?.preview)return item.preview;}return ui.replay===null?ui.view:{...ui.view,...ui.history?.frames?.[ui.replay],map:ui.history?.map || ui.view?.map};}
  function count(){return Math.max(1,Number(ui.view?.history_count || (ui.view?.state?.frame ?? 0)+1));}
  function actionLabel(a){return {UP:tr("up"),DOWN:tr("down"),LEFT:tr("left"),RIGHT:tr("right"),WAIT:tr("wait")}[a] || "—";}
  function format(n){return n==null?"—":Number.isInteger(Number(n))?String(Number(n)):Number(n).toFixed(1);}
  function stable(update){const left=window.scrollX,top=window.scrollY,focus=document.activeElement,nav=ui.userNavigationEpoch,revision=++ui.viewportEpoch;const restore=()=>{if(nav!==ui.userNavigationEpoch || revision!==ui.viewportEpoch)return;if(focus?.isConnected && !focus.disabled && document.activeElement!==focus && [document.body,document.documentElement].includes(document.activeElement))focus.focus({preventScroll:true});if(window.scrollX!==left || window.scrollY!==top)window.scrollTo({left,top,behavior:"instant"});};const result=update();restore();requestAnimationFrame(restore);return result;}
  function disable(el,value){el.setAttribute("aria-disabled",String(value));el.disabled=value && !(document.activeElement===el && el.tagName==="BUTTON" && (ui.busy || transport.pending()));}
  function setLanguage(language){ui.language=language;try{localStorage.setItem(LANGUAGE_KEY,language);}catch{}document.documentElement.lang=language==="zh"?"zh-CN":"en";for(const [id,label] of Object.entries(accessibilityLabels(language)))$(id).setAttribute("aria-label",label);document.title=`PolicyLens · ${tr("title")}`;document.querySelectorAll("[data-i18n]").forEach(n=>n.textContent=tr(n.dataset.i18n));$("languageButton").textContent=language==="zh"?"EN":"中";ui.answerSignature=null;render();}
  function render(){return stable(paint);}
  function paint(){
    if(!ui.initialized)return;
    const v=ui.view,s=v?.state,f=shown(),p=phase(v),isStudy=study(v),pending=transport.pending(),locked=ui.busy || ui.historyLoading!==null || !!pending,r=v?.release || {},m=frameMetrics(f);
    const failed=/fail|rejected/.test(String(r.status));const badge=releaseBadgeKey(v);
    $("releaseBadge").textContent=v?tr(badge):tr("connecting");$("releaseBadge").classList.toggle("failed",failed && !r.model_ready);
    $("releaseMessage").textContent=tr(releaseMessageKey(v));
    $("fullReleaseMessage").textContent=local(r.message || v?.notice,ui.language) || $("releaseMessage").textContent;
    const displayedStage=study(v)?displayStage(v):!v?.run_id && ui.entry==="study"?"consent":displayStage(v);$("stageTitle").textContent=displayedStage==="task1"?"Task 1":displayedStage==="task2"?"Task 2":tr(displayedStage);
    $("roundLabel").textContent=[v?.flow?.participant_id?`${tr("participantId")}: ${v.flow.participant_id}`:"",v?.flow?.round_count && ["practice","task1","task1_complete","task2"].includes(p)?`${tr("round")} ${v.flow.round_index ?? 1} / ${v.flow.round_count}`:""].filter(Boolean).join("\n");
    const stages=["consent","practice","task1","task2","questionnaire","completed"],index=stages.indexOf(p==="task1_complete"?"task1":p);
    $("workflow").classList.toggle("hidden",!isStudy);document.querySelectorAll("[data-stage]").forEach(n=>{n.classList.toggle("active",n.dataset.stage===(p==="task1_complete"?"task1":p));n.classList.toggle("done",isStudy && stages.indexOf(n.dataset.stage)<index);});
    $("stepValue").textContent=s?`${s.frame} / ${v.horizon || 120}`:"—";
    for(const [id,key] of [["deliveriesValue","deliveries"],["scoreValue","score"],["stepsValue","steps"],["collisionsValue","collisions"],["shutdownsValue","shutdowns"],["legacyScoreValue","legacy_score"]])$(id).textContent=v?.run_id?format(m[key]):"—";
    $("emptyMessage").classList.toggle("hidden",!!v?.map && !!v.run_id || p==="questionnaire" && !!f?.state);$("emptyMessage").textContent=v?.study_version_mismatch?tr("studyVersionChanged"):["questionnaire","completed","consent"].includes(p)?tr(p):v?(!playable(v)?tr(failed?"failedGate":"modelGate"):tr(ui.entry==="study"?"consent":"startFreeplay")):tr("connecting");
    const consentStage=isStudy && p==="consent";const entryAllowed=!isStudy || consentStage;
    ui.entry="study";
    document.querySelector(".agreement").hidden=!consentStage;
    $("participantInput").value=v?.flow?.participant_id || $("participantInput").value;
    $("consentText").hidden=!consentStage;
    $("entryPanel").classList.toggle("hidden",!entryAllowed);$("freeplaySetup").classList.toggle("hidden",ui.entry!=="freeplay");$("studySetup").classList.toggle("hidden",ui.entry!=="study");
    $("freeplayTab").classList.toggle("active",ui.entry==="freeplay");$("studyTab").classList.toggle("active",ui.entry==="study");$("startButton").textContent=consentStage?tr("startStudy"):(ui.language==="zh"?"登记用户 ID":"Register participant ID");
    $("entryGate").textContent=v?.study_version_mismatch?tr("studyVersionChanged"):ui.entry==="study" && !canEnter(v,"study")?tr("studyGate"):!canEnter(v,"freeplay")?tr(failed?"failedGate":"modelGate"):!r.explanation_ready?tr("explanationGate"):"";
    disable($("startButton"),locked || !v || !canEnter(v,ui.entry) || !entryAllowed || !allowed(v,consentStage?"next":"start"));
    for(const id of ["seedInput","consentInput"])disable($(id),locked);
    for(const id of ["participantInput","freeplayTab","studyTab"])disable($(id),locked || consentStage);
    $("seedInput").max=String(Math.max(0,(v?.play_scene_count || 1)-1));
    if(v?.flow?.consent_text)$("consentText").textContent=local(v.flow.consent_text,ui.language);
    $("operationPanel").classList.toggle("hidden",!v?.run_id || ["consent","questionnaire","completed"].includes(p));
    document.querySelectorAll("[data-action]").forEach(b=>disable(b,!canAct(v,ui.replay,ui.busy,pending,ui.historyLoading!==null)));
    disable($("endButton"),locked || !allowed(v,"end") || ui.replay!==null);$("endButton").classList.toggle("hidden",!allowed(v,"end"));
    disable($("nextButton"),locked || !allowed(v,"next") || ui.replay!==null);$("nextButton").classList.toggle("hidden",!allowed(v,"next"));
    $("robotStatus").classList.toggle("hidden",!v?.run_id);
    for(const [id,role] of [["humanStatus","robot_1"],["aiStatus","robot_2"]]){const a=f?.state?.agents?.find(a=>a.id===role);$(id).textContent=a?`${tr("battery")} ${Math.round(a.battery)}% · ${a.carrying_label?tr("carrying")+" "+a.carrying_label:tr("empty")}`:"—";}
    const agents=Number(f?.state?.frame)>0?f.state.agents || []:[];$("lastActions").textContent=`${tr("last")}: 1 ${actionLabel(agents.find(a=>a.id==="robot_1")?.last_executed_action)} · 2 ${actionLabel(agents.find(a=>a.id==="robot_2")?.last_executed_action)}`;
    publicHistoryLines(framePublicHistory(f),ui.language).forEach((line,i)=>$( ["submittedHistory","canceledHistory","collisionHistory"][i]).textContent=line);
    $("historySlider").max=String(count()-1);$("historySlider").value=String(ui.replay===null?count()-1:ui.replay);disable($("historySlider"),readLocked(v,ui.busy || ui.historyLoading!==null,pending) || !v?.run_id || count()<2);disable($("previousFrame"),readLocked(v,ui.busy || ui.historyLoading!==null,pending) || !v?.run_id || Number($("historySlider").value)<=0);disable($("liveButton"),readLocked(v,ui.busy || ui.historyLoading!==null,pending) || ui.replay===null);$("historyLabel").textContent=v?.run_id?`${f?.state?.frame ?? 0} / ${s?.frame ?? 0}`:"—";
    $("frameNote").textContent=ui.historyLoading!==null?tr("historyLoading"):p==="questionnaire" && f?.state?`${tr("questionMap")} ${f.state.frame}`:ui.replay!==null?`${tr("replay")} ${f?.state?.frame ?? ui.replay} · ${tr("confirmed")} ${s?.frame ?? 0} · ${tr("paused")}`:v?.run_id?`${ended(v)?tr("ended"):tr("liveFrame")} ${s?.frame ?? 0}`:"";
    $("statusText").textContent=v?.study_version_mismatch?tr("studyVersionChanged"):ui.historyLoading!==null?tr("historyLoading"):ui.busy?tr("busy"):pending?tr("pending"):ui.error?tr(ui.error):v?.version_mismatch?tr("runtimeChanged"):v?tr("synced"):tr("connecting");$("statusText").parentElement.classList.toggle("error",!!pending || !!ui.error || !!v?.study_version_mismatch);$("retryButton").classList.toggle("hidden",!pending || !!v?.study_version_mismatch);disable($("retryButton"),ui.busy || ui.historyLoading!==null || !canSubmit(v,pending?.body?.kind,true));
    renderExplanations(v,locked);renderQuestionnaire(v,locked || !allowed(v,"questionnaire"));
    $("completedPanel").classList.toggle("hidden",p!=="completed");$("feedbackPanel").classList.toggle("hidden",isStudy || !v?.run_id || !ended(v));
    if(ui.feedbackRun!==v?.run_id){ui.feedbackRun=v?.run_id;const fb=v?.feedback || v?.run?.feedback || {};$("clarityInput").value=fb.clarity || "";$("cooperationInput").value=fb.cooperation || "";$("feedbackNote").value=fb.note || "";}
    disable($("saveFeedback"),locked || !allowed(v,"feedback"));renderRuns(v,locked || !allowed(v,"select_run"));$("modelVersion").textContent=r.model_version || r.version || "—";$("runId").textContent=v?.run_id || "—";
    drawWarehouse(f,ui.motion?.positions);drawQuestionMarkers(f);document.body.dataset.stage=p;document.body.dataset.version=String(v?.version ?? 0);document.body.dataset.frame=String(s?.frame ?? 0);document.body.dataset.replay=String(ui.replay!==null);document.body.dataset.historyLoading=String(ui.historyLoading!==null);document.body.dataset.frontendVersion=FRONTEND_VERSION;document.body.dataset.explanationAllowed=String(explanationsAllowed(v));
  }
  function renderExplanations(view,locked){
    const permitted=explanationsAllowed(view);$("explanationPanel").classList.toggle("hidden",!permitted);
    if(!permitted){$("answerList").replaceChildren();$("questionInput").value="";ui.answerSignature=null;ui.seenQueue.clear();return;}
    $("questionFrame").textContent=`${tr("ai")} · ${tr("frame")} ${shown()?.state?.frame ?? 0}`;
    for(const b of document.querySelectorAll("[data-question]"))disable(b,locked);disable($("askButton"),locked);disable($("questionInput"),locked);disable($("questionFocus"),locked);
    const answers=visibleAnswers(view),signature=JSON.stringify([answers,ui.language]);
    if(signature!==ui.answerSignature){ui.answerSignature=signature;$("answerList").replaceChildren(...answers.map(a=>{const box=document.createElement("article");box.className="answer";const q=document.createElement("strong");q.textContent=a.question;const body=document.createElement("p");body.textContent=["pending","queued","running"].includes(a.status)?tr("questionPending"):a.text || tr("answerUnavailable");const meta=document.createElement("small");meta.textContent=`${tr("ai")} · ${tr("answerFrame")} ${a.frame ?? "—"}${a.sources.length?" · "+tr("source")+": "+a.sources.join("; "):""}`;box.append(q,body,meta);return box;}));}
    for(const a of answers)if(a.id && a.text && !["pending","queued","running"].includes(a.status) && !ui.seen.has(a.id))ui.seenQueue.add(a.id);
    if(ui.seenQueue.size && !ui.seenTimer)ui.seenTimer=setTimeout(()=>{ui.seenTimer=null;void flushSeen();},0);
  }
  async function flushSeen(){if(ui.busy || ui.historyLoading!==null || transport.pending() || !explanationsAllowed(ui.view) || !allowed(ui.view,"answer_seen"))return;const id=ui.seenQueue.values().next().value;if(!id)return;ui.seenQueue.delete(id);ui.seen.add(id);await execute({kind:"answer_seen",answer_id:id});}
  function questionnaireItems(view){return Array.isArray(view?.questionnaire?.items)?view.questionnaire.items:[];}
  function renderQuestionnaire(view,locked){const visible=study(view) && phase(view)==="questionnaire";$("questionnairePanel").classList.toggle("hidden",!visible);if(!visible)return;const items=questionnaireItems(view),signature=JSON.stringify(items);const box=$("questionnaireFields");if(box.dataset.signature!==signature){box.dataset.signature=signature;box.replaceChildren(...items.map(item=>{const label=document.createElement("label");label.className="questionnaire-item";const prompt=document.createElement("span");prompt.dataset.promptId=item.id;let input;if(item.type==="text"){input=document.createElement("textarea");input.rows=3;input.maxLength=item.max_length || 4000;}else{input=document.createElement("select");const blank=document.createElement("option");blank.value="";input.append(blank);const choices=item.options || (item.type==="scale"?[1,2,3,4,5]:[]);for(const option of choices){const o=document.createElement("option");o.value=String(typeof option==="object"?option.value ?? option.id:option);o.dataset.choice=JSON.stringify(typeof option==="object"?option.label ?? option.text ?? o.value:option);input.append(o);}}input.dataset.qId=item.id;input.dataset.qType=item.type;input.addEventListener("input",()=>{ui.questionnaireDirty=true;});label.append(prompt,input);if(item.preview){const preview=document.createElement("button");preview.type="button";preview.className="secondary";preview.dataset.previewId=item.id;preview.addEventListener("click",event=>{event.preventDefault();ui.questionnairePreview=item.id;render();});label.append(preview);}return label;}));ui.questionnaireDirty=false;}
    for(const item of items){const label=[...box.querySelectorAll("[data-prompt-id]")].find(n=>n.dataset.promptId===item.id);if(label)label.textContent=local(item.prompt || item.label,ui.language);const input=[...box.querySelectorAll("[data-q-id]")].find(n=>n.dataset.qId===item.id);if(!input)continue;if(!ui.questionnaireDirty)input.value=view.questionnaire?.draft?.[item.id] ?? "";for(const o of input.querySelectorAll?.("option") || [])o.textContent=o.value===""?tr("selectOption"):local(JSON.parse(o.dataset.choice),ui.language) || o.value;disable(input,locked);}
    for(const button of box.querySelectorAll("[data-preview-id]"))button.textContent=tr("previewQuestion");
    $("questionnaireBankNotice").textContent=view.study_version_mismatch?tr("studyVersionChanged"):view.questionnaire?.blocked?tr("predictionMismatch"):view.questionnaire?.bank?.available?tr("predictionCandidate"):view.questionnaire?.bank?.status==="candidate_failed"?tr("predictionFailed"):tr("predictionUnavailable");
    disable($("saveQuestionnaire"),locked || !items.length || !!view.questionnaire?.blocked);disable($("submitQuestionnaire"),locked || !items.length || !!view.questionnaire?.blocked);if(!items.length)box.textContent=tr("questionnaireMissing");
  }
  function renderRuns(view,locked){const visible=!study(view) && !!view?.run_id;$("savedRunsField").classList.toggle("hidden",!visible);if(!visible)return;const rows=view.runs || [],signature=JSON.stringify([rows,view.run_id,ui.language]);if($("runSelect").dataset.signature!==signature){$("runSelect").dataset.signature=signature;$("runSelect").replaceChildren(...rows.filter(r=>r.mode!=="study").map((r,i)=>{const o=document.createElement("option");o.value=r.id || r.run_id;o.textContent=`${rows.length-i} · seed ${r.seed ?? "—"} · ${r.steps ?? r.metrics?.steps ?? 0}`;return o;}));$("runSelect").value=view.run_id;}disable($("runSelect"),locked);}
  function adopt(raw){
    const next=normalizeView(raw);if(!Number.isInteger(next.version) || !next.session_id)throw new Error("invalid_response");
    const old=ui.view;if(old?.session_id===next.session_id && next.version<old.version)return;
    if(!transport.bindSession(next.session_id))ui.error="oldSession";
    if(old?.session_id!==next.session_id){ui.seen.clear();ui.seenQueue.clear();}
    const changed=old?.session_id!==next.session_id || old?.run_id!==next.run_id || old?.history_count!==next.history_count || phase(old)!==phase(next) || old?.study_version_mismatch!==next.study_version_mismatch;
    if(changed || ui.historyLoading!==null && old?.version!==next.version){
      if(ui.historyLoading!==null)ui.error="historyChanged";
      ui.history=null;ui.replay=null;ui.historyToken++;ui.historyLoading=null;
    }
    if(old?.run_id!==next.run_id){ui.answerSignature=null;ui.questionnaireDirty=false;$("seedInput").value=String(next.seed ?? next.run?.seed ?? 0);if(next.flow?.participant_id)$("participantInput").value=next.flow.participant_id;}
    if(study(next))ui.entry="study";ui.view=next;if(!explanationsAllowed(next))ui.view={...next,answers:[]};render();
  }
  function animateConfirmedAction(before,after,kind){
    const motion=confirmedMotion(before,after,kind,ui.replay);if(!motion)return Promise.resolve();
    const token=++ui.motionToken;ui.motion={token,progress:0,positions:interpolateMotion(motion,0)};drawWarehouse(shown(),ui.motion.positions);drawQuestionMarkers(shown());
    return new Promise(resolve=>{let started=null;const tick=now=>{if(token!==ui.motionToken){resolve();return;}if(started===null)started=now;const progress=Math.min(1,(now-started)/motion.duration);ui.motion={token,progress,positions:interpolateMotion(motion,progress)};drawWarehouse(shown(),ui.motion.positions);drawQuestionMarkers(shown());if(progress<1){requestAnimationFrame(tick);return;}ui.motion=null;resolve();};requestAnimationFrame(tick);});
  }
  async function refresh(){if(ui.busy || ui.refreshing)return;ui.refreshing=true;try{adopt(await transport.request("/api/view"));}catch{if(!transport.pending())ui.error="requestFailed";render();}finally{ui.refreshing=false;}}
  async function execute(payload,retry=false){if(ui.busy || ui.historyLoading!==null || (!retry && transport.pending()) || !canSubmit(ui.view,payload.kind,retry))return;ui.busy=true;ui.error=null;render();try{const before=ui.view,submitted=retry?transport.pending()?.body:payload,result=retry?await transport.retry():await transport.submit({operation_id:crypto.randomUUID(),expected_version:ui.view.version,...payload},ui.view.session_id),next=normalizeView(result);if(payload.kind==="questionnaire")ui.questionnaireDirty=false;await animateConfirmedAction(before,next,submitted?.kind);adopt(next);}catch(e){ui.motionToken++;ui.motion=null;ui.error=["study_version_mismatch","study_version_changed"].includes(e.message)?"studyVersionChanged":e.message==="participant_id_taken"?"participantTaken":e.message==="invalid_participant_id"?"participantInvalid":e.status===409?"conflict":"requestFailed";if(!transport.pending())try{adopt(await transport.request("/api/view"));}catch{ui.error="requestFailed";}}finally{ui.motionToken++;ui.motion=null;ui.busy=false;render();}}
  async function selectFrame(index){
    if(!ui.view?.run_id || readLocked(ui.view,ui.busy,transport.pending()))return;
    if(!Number.isInteger(index) || index<0)return;
    if(index>=count()-1){ui.historyToken++;ui.historyLoading=null;ui.replay=null;render();return;}
    const ticket=++ui.historyToken,run=ui.view.run_id,session=ui.view.session_id,version=ui.view.version;
    // Acquire before any await: both command dispatch and rendered controls see this lock.
    ui.historyLoading=ticket;ui.error=null;render();
    try{
      let history=ui.history;
      if(!history || history.run_id!==run || history.frames.length<count())history=await transport.request("/api/history");
      if(ticket!==ui.historyToken || ui.view.run_id!==run || ui.view.session_id!==session || ui.view.version!==version)return;
      if(history.run_id && history.run_id!==run)throw new Error("history_mismatch");
      if(!Array.isArray(history.frames) || !history.frames[index])throw new Error("history_missing");
      ui.history={...history,run_id:run};ui.replay=index;ui.error=null;
    }catch{
      if(ticket===ui.historyToken)ui.error="historyLoadFailed";
    }finally{
      // A late response cannot release a newer history request's lock.
      if(ui.historyLoading===ticket){ui.historyLoading=null;render();}
    }
  }
  const COLORS=["#009E73","#CC79A7","#B79F00","#7A5AF8","#5D6B7A"];
  function taskColor(id,i){const n=String(id || "").match(/(\d+)$/);return COLORS[(n?Math.max(0,Number(n[1])-1):i)%COLORS.length];}
  function drawWarehouse(view,visualPositions=null){const canvas=$("warehouseCanvas"),width=canvas.clientWidth,height=canvas.clientHeight,ratio=window.devicePixelRatio || 1;if(!width || !height)return;const pixelWidth=Math.round(width*ratio),pixelHeight=Math.round(height*ratio);if(canvas.width!==pixelWidth)canvas.width=pixelWidth;if(canvas.height!==pixelHeight)canvas.height=pixelHeight;const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);canvas.dataset.animationRunning=String(ui.motion!==null);canvas.dataset.animationProgress=String(ui.motion?.progress ?? 1);if(!view?.map)return;const map=view.map,rows=map.rows,cols=map.cols,size=Math.min((width-34)/cols,(height-34)/rows),ox=(width-cols*size)/2,oy=(height-rows*size)/2;const cell=p=>[ox+p[1]*size,oy+p[0]*size];ctx.fillStyle="#f8fafc";ctx.fillRect(ox,oy,cols*size,rows*size);ctx.strokeStyle="#dce4ef";ctx.lineWidth=1;for(let r=0;r<=rows;r++){ctx.beginPath();ctx.moveTo(ox,oy+r*size);ctx.lineTo(ox+cols*size,oy+r*size);ctx.stroke();}for(let c=0;c<=cols;c++){ctx.beginPath();ctx.moveTo(ox+c*size,oy);ctx.lineTo(ox+c*size,oy+rows*size);ctx.stroke();}for(const p of map.shelves || []){const[x,y]=cell(p);ctx.fillStyle="#9aa8ba";ctx.fillRect(x+2,y+2,size-4,size-4);}ctx.textAlign="center";ctx.textBaseline="middle";if(map.charger_position){const[x,y]=cell(map.charger_position);ctx.fillStyle="#6558e8";ctx.fillRect(x+4,y+4,size-8,size-8);ctx.fillStyle="#fff";ctx.font=`bold ${size*.43}px sans-serif`;ctx.fillText("⚡",x+size/2,y+size/2);}const tasks=view.state?.tasks || [];for(const [i,t] of tasks.entries()){ctx.font=`bold ${size*.25}px sans-serif`;for(const [key,label] of [["pickup_position",`A${i+1}`],["delivery_position",`B${i+1}`]]){if(key==="pickup_position" && t.status!=="available")continue;if(!t[key])continue;const[x,y]=cell(t[key]);ctx.fillStyle=taskColor(t.task_id,i);ctx.beginPath();ctx.arc(x+size/2,y+size/2,size*.3,0,Math.PI*2);ctx.fill();ctx.fillStyle="#fff";ctx.fillText(label,x+size/2,y+size/2);}}
    const agents=view.state?.agents || [],display=agent=>visualPositions?.[agent.id] || agent.position;for(const a of agents){const position=display(a);canvas.dataset[a.id==="robot_1"?"robot1Position":"robot2Position"]=JSON.stringify(position);const overlap=agents.some(b=>b.id!==a.id && display(b)?.join()===position?.join());let[x,y]=cell(position);if(overlap)x+=(a.id==="robot_1"?-1:1)*size*.17;const human=a.id==="robot_1",w=size*(overlap?.54:.68);ctx.fillStyle=a.active?(human?"#4f6ff0":"#f56b3d"):"#d9485f";ctx.beginPath();ctx.roundRect(x+size/2-w/2,y+size*.19,w,size*.64,size*.12);ctx.fill();ctx.fillStyle="#fff";ctx.font=`900 ${size*.3}px sans-serif`;ctx.fillText(human?"1":"2",x+size/2,y+size*.49);const battery=`${Math.round(a.battery)}%`;ctx.font=`bold ${Math.max(10,size*.17)}px sans-serif`;const pw=Math.max(35,ctx.measureText(battery).width+8),ph=Math.max(16,size*.23);ctx.fillStyle="#fff";ctx.strokeStyle=a.battery<=20?"#d9485f":"#cbd6e4";ctx.beginPath();ctx.roundRect(x+size/2-pw/2,Math.max(oy+1,y-2),pw,ph,5);ctx.fill();ctx.stroke();ctx.fillStyle="#26324a";ctx.fillText(battery,x+size/2,Math.max(oy+1,y-2)+ph/2);if(a.carrying_label){ctx.fillStyle="#26324a";ctx.font=`bold ${Math.max(10,size*.17)}px sans-serif`;ctx.fillText(a.carrying_label,x+size/2,y+size*.72);}}
  }
  function drawQuestionMarkers(view){const canvas=$("warehouseCanvas"),map=view?.map;if(!map || !view.question_markers)return;const width=canvas.clientWidth,height=canvas.clientHeight,size=Math.min((width-34)/map.cols,(height-34)/map.rows),ox=(width-map.cols*size)/2,oy=(height-map.rows*size)/2,ctx=canvas.getContext("2d");ctx.textAlign="center";ctx.textBaseline="middle";ctx.font=`bold ${Math.max(11,size*.19)}px sans-serif`;for(const marker of view.question_markers){const[r,c]=marker.position,x=ox+(c+.81)*size,y=oy+(r+.8)*size;ctx.fillStyle="#fff";ctx.strokeStyle="#26324a";ctx.beginPath();ctx.arc(x,y,size*.15,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle="#26324a";ctx.fillText(marker.label,x,y);}}
  $("warehouseCanvas").addEventListener("click",()=>$("warehouseCanvas").focus({preventScroll:true}));
  $("languageButton").addEventListener("click",()=>setLanguage(ui.language==="zh"?"en":"zh"));
  for(const mode of ["freeplay","study"])$(mode+"Tab").addEventListener("click",()=>{if(ui.busy || ui.historyLoading!==null || transport.pending())return;ui.entry=mode;render();});
  $("startButton").addEventListener("click",()=>{
    if(!canEnter(ui.view,"study"))return;
    if(study(ui.view) && phase(ui.view)==="consent"){
      if(!$("consentInput").checked){ui.error="consentRequired";render();return;}
      void execute({kind:"next",consent:true});return;
    }
    const id=$("participantInput").value.trim();
    if(!/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(id)){ui.error="participantInvalid";render();return;}
    void execute({kind:"start",mode:"study",participant_id:id});
  });
  document.querySelectorAll("[data-action]").forEach(b=>b.addEventListener("click",()=>{if(canAct(ui.view,ui.replay,ui.busy,transport.pending(),ui.historyLoading!==null))void execute({kind:"action",action:b.dataset.action});}));
  $("endButton").addEventListener("click",()=>{if(allowed(ui.view,"end") && ui.replay===null)void execute({kind:"end"});});$("nextButton").addEventListener("click",()=>{if(allowed(ui.view,"next") && ui.replay===null)void execute({kind:"next",...(phase(ui.view)==="consent"?{consent:true}:{})});});
  $("retryButton").addEventListener("click",()=>void execute({},true));$("historySlider").addEventListener("input",e=>void selectFrame(Number(e.target.value)));$("previousFrame").addEventListener("click",()=>void selectFrame(Math.max(0,Number($("historySlider").value)-1)));$("liveButton").addEventListener("click",()=>{if(readLocked(ui.view,ui.busy,transport.pending()))return;ui.historyToken++;ui.historyLoading=null;ui.replay=null;render();});
  $("runSelect").addEventListener("change",e=>{if(e.target.value && e.target.value!==ui.view?.run_id && !study(ui.view))void execute({kind:"select_run",run_id:e.target.value});});
  function ask(question,focus){if(!explanationsAllowed(ui.view) || ui.busy || ui.historyLoading!==null || transport.pending())return;const q=question.trim();if(!q){ui.error="questionRequired";render();return;}$("questionInput").value=q;void execute({kind:"question",question:q,focus,language:ui.language,run_id:ui.view.run_id,frame:shown()?.state?.frame ?? 0});}
  document.querySelectorAll("[data-question]").forEach(b=>b.addEventListener("click",()=>{$("questionFocus").value=b.dataset.focus;ask(tr(b.dataset.question),b.dataset.focus);}));$("askButton").addEventListener("click",()=>ask($("questionInput").value,$("questionFocus").value));
  function saveQuestionnaire(submit){if(!allowed(ui.view,"questionnaire"))return;const items=questionnaireItems(ui.view);if(!items.length){ui.error="questionnaireMissing";render();return;}const answers={};for(const input of $("questionnaireFields").querySelectorAll("[data-q-id]"))answers[input.dataset.qId]=input.value===""?null:input.dataset.qType==="scale"?Number(input.value):input.value;if(submit && items.some(item=>item.required!==false && (answers[item.id]==null || answers[item.id]===""))){ui.error="answersRequired";render();return;}void execute({kind:"questionnaire",answers,submit});}
  $("saveQuestionnaire").addEventListener("click",()=>saveQuestionnaire(false));$("submitQuestionnaire").addEventListener("click",()=>saveQuestionnaire(true));
  $("saveFeedback").addEventListener("click",()=>{const clarity=Number($("clarityInput").value),cooperation=Number($("cooperationInput").value);if(!clarity || !cooperation){ui.error="feedbackRequired";render();return;}void execute({kind:"feedback",feedback:{clarity,cooperation,note:$("feedbackNote").value.trim()}});});
  window.addEventListener("keydown",event=>{const action=keyboardAction(event);if(!action || !["freeplay","practice","task1","task2"].includes(phase(ui.view)))return;event.preventDefault();if(event.repeat || !canAct(ui.view,ui.replay,ui.busy,transport.pending(),ui.historyLoading!==null))return;void execute({kind:"action",action});});
  for(const type of ["wheel","touchstart","touchmove","pointerdown","keydown"])window.addEventListener(type,()=>{ui.userNavigationEpoch++;},{capture:true,passive:true});window.addEventListener("pointermove",e=>{if(e.buttons)ui.userNavigationEpoch++;},{capture:true,passive:true});window.addEventListener("resize",()=>{ui.userNavigationEpoch++;drawWarehouse(shown(),ui.motion?.positions);drawQuestionMarkers(shown());});window.addEventListener("online",()=>void refresh());
  ui.initialized=true;setLanguage(ui.language);void refresh();setInterval(()=>{if(document.visibilityState!=="hidden")void refresh();},5000);
})();
