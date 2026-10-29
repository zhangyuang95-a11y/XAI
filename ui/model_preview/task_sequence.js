'use strict';
// Local navigation for manual play only; server frames and bindings remain authoritative.
class PreviewTaskSequence {
  constructor({domain,variant,seed,binding=null,runs=[]}) {
    this.check(['warehouse','pong','kitchen'].includes(domain),'unknown sequence domain');
    this.check(['raw','regularized'].includes(variant),'unknown sequence variant');
    this.check(Number.isSafeInteger(seed)&&seed>=0,'invalid sequence seed');
    this.domain=domain;this.variant=variant;this.seed=seed;
    this._binding=binding===null?null:this.copy(binding);this._runs=[];
    if(this._binding)this.checkBinding(this._binding);
    this.check(Array.isArray(runs)&&runs.length<=3,'invalid sequence runs');
    for(const run of runs){
      this.check(this._binding!==null,'restored runs require their exact binding');
      this.checkRun(run);
      this.check(run.task===this._runs.length+1,'sequence tasks must be consecutive');
      this.check(!this._runs.length||this._runs.at(-1).terminal,'previous task is incomplete');
      this.check(!this._runs.length||run.binding_identity_sha256===this._runs[0].binding_identity_sha256,'sequence binding identity changed');
      this.check(!this._runs.some(r=>r.token===run.token),'tasks must have distinct session tokens');
      this._runs.push(this.copy(run));
    }
  }
  check(ok,message){if(!ok)throw new Error(message);}
  copy(value){return JSON.parse(JSON.stringify(value));}
  canonical(value){
    return JSON.stringify(value,(_key,item)=>item&&typeof item==='object'&&!Array.isArray(item)
      ?Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])):item);
  }
  checkBinding(binding){
    this.check(binding&&typeof binding==='object'&&!Array.isArray(binding)&&binding.variant===this.variant,'sequence model variant differs');
    this.check(binding.domain===undefined||binding.domain===this.domain,'sequence model domain differs');
  }
  checkRun(run){
    this.check(run&&Number.isInteger(run.task)&&run.task>=1&&run.task<=3,'invalid sequence task');
    this.check(typeof run.token==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(run.token),'invalid sequence session token');
    this.check(typeof run.terminal==='boolean'&&Number.isSafeInteger(run.turn)&&run.turn>=0,'invalid sequence frame');
    this.check(typeof run.score==='number'&&Number.isFinite(run.score),'invalid sequence score');
    this.check(typeof run.binding_identity_sha256==='string'&&run.binding_identity_sha256.length>0,'missing sequence binding identity');
  }
  get binding(){return this._binding===null?null:this.copy(this._binding);}
  get runs(){return this.copy(this._runs);}
  get nextTask(){const last=this._runs.at(-1);return !last?1:last.terminal&&last.task<3?last.task+1:null;}
  get finished(){return this._runs.length===3&&this._runs.every(run=>run.terminal);}
  accept(view,token){
    const state=view?.state,binding=view?.binding;
    this.check(state?.domain===this.domain,'sequence frame domain differs');this.checkBinding(binding);
    this.check(this._binding===null||this.canonical(binding)===this.canonical(this._binding),'sequence frozen model binding changed');
    const run={task:state.task,token,terminal:state.terminal,score:state.score?.task_score,
      turn:state.turn,binding_identity_sha256:state.binding_identity_sha256};
    this.checkRun(run);
    const last=this._runs.at(-1);
    if(last&&last.task===run.task){
      this.check(last.token===token,'current sequence session changed');
      this.check(run.turn>=last.turn&&(!last.terminal||run.terminal),'sequence frame regressed');
      this.check(last.binding_identity_sha256===run.binding_identity_sha256,'sequence binding identity changed');
    }else{
      this.check(run.task===this.nextTask,'complete the current task before advancing');
      this.check(!this._runs.some(r=>r.token===token),'tasks must have distinct session tokens');
      this.check(!last||last.binding_identity_sha256===run.binding_identity_sha256,'sequence binding identity changed');
    }
    if(this._binding===null)this._binding=this.copy(binding);
    if(last?.task===run.task)this._runs[this._runs.length-1]=run;else this._runs.push(run);
    return this;
  }
  toJSON(){return {domain:this.domain,variant:this.variant,seed:this.seed,binding:this.binding,runs:this.runs};}
}
if(typeof module!=='undefined')module.exports=PreviewTaskSequence;
else window.PreviewTaskSequence=PreviewTaskSequence;
