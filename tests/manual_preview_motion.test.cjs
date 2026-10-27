const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm'),fs=require('node:fs');
function harness(reduced=false){
  let now=0,id=0;const callbacks=new Map(),draws=[],shown=[];
  class StudyBoard{
    constructor(canvas){this.canvas=canvas;this.state=null;this.animation=null;}
    stop(){if(this.animation){callbacks.delete(this.animation.frame);this.animation.resolve();this.animation=null;}this.canvas.dataset.animating='false';}
    draw(a,b,p){draws.push([a.turn,b.turn,p]);}
  }
  const context=vm.createContext({window:{},StudyBoard,matchMedia:()=>({matches:reduced}),
    requestAnimationFrame:fn=>{callbacks.set(++id,fn);return id;}});
  vm.runInContext(fs.readFileSync('ui/model_preview/board.js','utf8'),context);
  const board=new context.window.PreviewBoard({dataset:{}});
  const show=(turn,animate=true,domain='warehouse')=>board.setState({turn,task:1,domain},{animate,onPresent:()=>shown.push(turn)});
  const frame=delta=>{now+=delta;const pending=[...callbacks.values()];callbacks.clear();for(const fn of pending)fn(now);};
  return {board,show,frame,draws,shown,callbacks};
}
test('fast confirmed responses do not interrupt or skip frames, and cap visual backlog',async()=>{
  const h=harness();await h.show(0,false);
  const a=h.show(1);h.frame(0);h.frame(190);const b=h.show(2);
  let ready=false;const capacity=h.board.ready().then(()=>ready=true);await Promise.resolve();
  assert.equal(ready,false);assert.deepEqual(h.shown,[0,1]);
  h.frame(190);await capacity;assert(ready);assert.deepEqual(h.shown,[0,1,2]);
  assert(h.draws.some(([a,b,p])=>a===0&&b===1&&p===1));
  h.frame(0);h.frame(380);await Promise.all([a,b]);
  assert.deepEqual(h.draws.at(-1),[1,2,1]);assert.equal(h.board.pendingFrames,0);
});
test('new session cancels old presentation callbacks and releases backpressure',async()=>{
  const h=harness();await h.show(0,false);const a=h.show(1),b=h.show(2),ready=h.board.ready();
  h.frame(0);await h.show(0,false,'pong');await Promise.all([a,b,ready]);
  h.frame(1000);assert.deepEqual(h.shown,[0,1,0]);assert.equal(h.board.state.domain,'pong');assert.equal(h.board.pendingFrames,0);
});
test('reduced motion and restored nonconsecutive frames show only confirmed endpoints',async()=>{
  const h=harness(true);await h.show(0,false);await h.show(1);await h.show(8);
  assert.deepEqual(h.draws,[[0,0,1],[1,1,1],[8,8,1]]);assert.equal(h.callbacks.size,0);
});
