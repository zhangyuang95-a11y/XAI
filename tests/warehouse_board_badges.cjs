/* Execute the real canvas renderer with recorded drawing commands; not a browser screenshot test. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const labels=[],shapes=[];
let drawing=[];
const ctx={
  setTransform(){},clearRect(){},fillRect(){},stroke(){},strokeRect(){},moveTo(){},lineTo(){},closePath(){},ellipse(){},
  beginPath(){drawing=[];},
  arc(x,y,r){drawing.push({kind:'circle',x,y,r});},
  roundRect(x,y,w,h){drawing.push({kind:'rect',x,y,w,h});},
  fill(){shapes.push(...drawing.map(shape=>({...shape,color:this.fillStyle})));},
  fillText(text,x,y){labels.push({text:String(text),x,y,color:this.fillStyle});},
};
const canvas={width:0,height:0,style:{},dataset:{},getContext(){return ctx;}};
const context=vm.createContext({window:{devicePixelRatio:1},ResizeObserver:class{observe(){}disconnect(){}},
  matchMedia(){return {matches:false};},performance,requestAnimationFrame(){throw Error('Unexpected animation scheduling');},cancelAnimationFrame(){}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../study_v3/web/board.js'),'utf8'),context);
const board=new context.window.StudyBoard(canvas);
const fixture=(ids=['task_1','task_2'])=>({domain:'warehouse',task:2,turn:3,width:7,height:6,walls:[],chargers:[],
  orders:ids.map((id,i)=>({id,pickup:[i+1,1],dropoff:[i+1,2],status:'carried'})),
  human:{x:3,y:0,battery:100,carrying:ids[0]},ai:{x:4,y:5,battery:20,carrying:ids[1]}});
function render(before,after=before,progress=1){
  labels.length=0;shapes.length=0;board.draw(before,after,progress);
  assert(!labels.some(label=>label.text.includes('task_')),'Internal job IDs must never appear on canvas badges');
  return labels.filter(label=>/^A\d+$/.test(label.text));
}
function circleAt(label){return shapes.find(shape=>shape.kind==='circle'&&shape.x===label.x&&shape.y===label.y);}
const original=fixture();const originalJson=JSON.stringify(original);
let badges=render(original);
assert.deepEqual(badges.map(label=>label.text),['A1','A2'],'Both robots carry the displayed pickup labels');
assert.equal(JSON.stringify(original),originalJson,'Rendering must not rewrite IDs or authoritative state');

// Replenished IDs are unrelated to the two current display slots.
const replenished=fixture(['task_5','task_12']);replenished.turn=22;
badges=render(replenished);
assert.deepEqual(badges.map(label=>label.text),['A1','A2']);
for(const badge of badges){
  const dropoff=labels.find(label=>label.text==='B'+badge.text.slice(1));
  assert.equal(circleAt(badge).color,circleAt(dropoff).color,'Cargo and its matching destination must share color');
}

// The battery plate must not cover the cargo badge, including the clamped top row.
for(const badge of badges){
  const cargo=circleAt(badge);
  for(const plate of shapes.filter(shape=>shape.kind==='rect'&&shape.color==='#fff')){
    const overlaps=plate.x<cargo.x+cargo.r&&plate.x+plate.w>cargo.x-cargo.r&&plate.y<cargo.y+cargo.r&&plate.y+plate.h>cargo.y-cargo.r;
    assert.equal(overlaps,false,'Battery rectangle overlaps the cargo circle');
  }
}
for(const percent of labels.filter(label=>label.text.endsWith('%'))){
  assert(shapes.some(shape=>shape.kind==='rect'&&shape.color==='#fff'&&percent.x>shape.x&&percent.x<shape.x+shape.w&&percent.y>shape.y&&percent.y<shape.y+shape.h),'Battery text must remain inside its plate');
}

// A replay frame uses its own order slot mapping, never the latest live mapping.
const historical=fixture(['task_12','task_5']);historical.turn=12;
historical.human.carrying='task_5';historical.ai.carrying='task_12';
badges=render(historical);
assert.deepEqual(badges.map(label=>label.text),['A2','A1']);

// Delivery/replenishment can renumber the remaining job. Cargo changes together
// with the displayed map at the completed transition, not halfway through it.
const next=structuredClone(original);next.turn++;
next.orders=[original.orders[1],{id:'task_5',pickup:[1,3],dropoff:[2,3],status:'available'}];
next.human.carrying=null;
badges=render(original,next,.75);
assert.deepEqual(badges.map(label=>label.text),['A1','A2']);
render(original,next,1);
const aiBadge=labels.find(label=>label.text==='A1');
assert(aiBadge&&circleAt(aiBadge),'Remaining AI cargo is now A1');
assert(!labels.some(label=>label.text==='task_2'));

// An inconsistent external frame should show an unknown badge, not invent A99.
const unknown=fixture();unknown.human.carrying='task_99';
render(unknown);
assert(labels.some(label=>label.text==='?'));
assert(!labels.some(label=>label.text==='A99'));
console.log('Warehouse board badges: both actors, replenishment, replay, transition consistency and battery separation passed.');

// A settled Pong transition must display the newly confirmed countdown, not
// reuse the previous-frame labels after the balls have reached their new height.
const pongBefore={domain:'pong',task:2,turn:4,lanes:9,human:{x:2},ai:{x:6},balls:[
  {id:'s1',kind:'ordinary',contacts:[2],y:6,vy:2,remaining:3},
  {id:'t1',kind:'cooperative',contacts:[1,5],y:9,vy:1,remaining:3},
  {id:'t2',kind:'cooperative',contacts:[3,7],y:9,vy:1,remaining:3}]};
const pongAfter=structuredClone(pongBefore);pongAfter.turn++;
for(const ball of pongAfter.balls){ball.y+=ball.vy;ball.remaining--;}
render(pongBefore,pongAfter,.5);
assert(labels.some(row=>row.text==='s1 · 3t'));
render(pongBefore,pongAfter,1);
for(const id of ['s1','t1','t2'])assert(labels.some(row=>row.text===id+' · 2t'));
assert(!labels.some(row=>row.text.endsWith('3t')));
board.lang='zh';render(pongBefore,pongAfter,1);
assert(labels.some(row=>row.text==='t1 · 2步'));
console.log('Pong settled animation countdowns match the confirmed frame in both languages.');
