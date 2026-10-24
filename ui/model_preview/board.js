'use strict';
// Preview-only animation: interpolate confirmed frames; never predict NN moves.
window.PreviewBoard=class PreviewBoard extends StudyBoard {
  setState(state,{animate=false,language='en'}={}){
    const old=this.state;this.stop();this.state=state;this.lang=language;
    if(!animate||!old||old.domain!==state.domain||old.task!==state.task||state.turn!==old.turn+1||matchMedia('(prefers-reduced-motion: reduce)').matches){
      this.draw(state,state,1);return Promise.resolve();
    }
    return new Promise(resolve=>{
      const start=performance.now(),animation={resolve,frame:null};this.animation=animation;this.canvas.dataset.animating='true';
      const frame=now=>{if(this.animation!==animation)return;const p=Math.min(1,(now-start)/120);this.draw(old,state,p);
        if(p<1)animation.frame=requestAnimationFrame(frame);else{this.animation=null;this.canvas.dataset.animating='false';resolve();}};
      animation.frame=requestAnimationFrame(frame);
    });
  }
};
