'use strict';
// Play confirmed frames in order. Networking may overlap animation, never policy execution.
window.PreviewBoard=class PreviewBoard extends StudyBoard {
  constructor(canvas){super(canvas);this.frames=[];this.capacity=[];this.activeFrame=null;this.duration=380;}
  get pendingFrames(){return this.frames.length+(this.activeFrame?1:0);}
  ready(){return this.pendingFrames<2?Promise.resolve():new Promise(resolve=>this.capacity.push(resolve));}
  notifyCapacity(){if(this.pendingFrames<2)for(const resolve of this.capacity.splice(0))resolve();}
  stop(){
    super.stop();
    if(this.activeFrame){this.activeFrame.resolve();this.activeFrame=null;}
    for(const frame of this.frames||[])frame.resolve();
    this.frames=[];this.notifyCapacity();
  }
  setState(state,{animate=false,language='en',onPresent=()=>{}}={}){
    if(!animate)this.stop();
    return new Promise(resolve=>{this.frames.push({state,animate,language,onPresent,resolve});this.drain();});
  }
  drain(){
    if(this.activeFrame||!this.frames.length)return;
    const item=this.activeFrame=this.frames.shift(),{state,animate,language,onPresent}=item,old=this.state;
    this.state=state;this.lang=language;onPresent();
    const finish=()=>{if(this.activeFrame!==item)return;this.animation=null;this.canvas.dataset.animating='false';
      this.activeFrame=null;item.resolve();this.notifyCapacity();this.drain();};
    if(!animate||!old||old.domain!==state.domain||old.task!==state.task||state.turn!==old.turn+1||matchMedia('(prefers-reduced-motion: reduce)').matches){
      this.draw(state,state,1);finish();return;
    }
    let started=null;
    const animation={resolve:()=>{},frame:null};this.animation=animation;this.canvas.dataset.animating='true';
    const frame=now=>{if(this.activeFrame!==item)return;if(started===null)started=now;
      const p=Math.min(1,(now-started)/this.duration);this.draw(old,state,p);
      if(p<1)animation.frame=requestAnimationFrame(frame);else finish();};
    animation.frame=requestAnimationFrame(frame);
  }
};
