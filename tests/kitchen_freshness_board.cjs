'use strict';
// Record actual canvas labels for every public prepared-food location.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const labels=[],strokes=[];
const ctx={setTransform(){},clearRect(){},fillRect(){},beginPath(){},roundRect(){},fill(){},arc(){},ellipse(){},moveTo(){},lineTo(){},stroke(){},strokeRect(...args){strokes.push({args,color:this.strokeStyle});},fillText(text,x,y){labels.push({text:String(text),x,y,color:this.fillStyle});}};
const canvas={width:0,height:0,style:{},dataset:{},getContext:()=>ctx};
const context=vm.createContext({window:{devicePixelRatio:1},ResizeObserver:class{observe(){}disconnect(){}},matchMedia:()=>({matches:false}),performance,cancelAnimationFrame(){}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../study_v3/web/board.js'),'utf8'),context);
const board=new context.window.StudyBoard(canvas),food=(id,stage='prepared')=>({id,ingredient:'tomato',stage,freshness_basis:'prepared',prepared_turn:10,expires_turn:30});
const s={domain:'kitchen',task:1,turn:25,width:9,height:7,walls:[],orders:[],pots:[],tutorial_highlights:['prep'],
 stations:[{id:'prep',x:1,y:3},{id:'handoff',x:4,y:3},{id:'human_buffer',x:1,y:4},{id:'ai_raw',x:7,y:3}],
 human:{x:2,y:3,facing:'left',holding:food('human'),preparation:{ready:true,remaining:0}},ai:{x:6,y:3,facing:'right',holding:food('ai')},handoff:food('handoff'),buffers:{human:food('human_buffer'),ai_raw:[food('slot1'),food('slot2')],protein:{}},food_freshness:['human','ai','handoff','human_buffer','slot1','slot2'].map(item_id=>({item_id,status:'fresh',remaining:5,expires_turn:30}))};
const original=JSON.stringify(s);board.draw(s,s,1);
const orange=labels.filter(x=>x.color==='#a34400');assert.equal(orange.length,6,'Both hands, handoff, human counter and both AI slots show the same expiry.');
assert(orange.every(x=>x.text.includes('5')));assert(strokes.some(x=>x.color==='#b36b00'),'Tutorial station is highlighted.');assert.equal(JSON.stringify(s),original,'Drawing cannot reset timestamps.');
// More than five turns is green; only prepared portions receive this clock.
assert.equal(board.freshnessLabel(food('fresh'),{turn:20,food_freshness:[]}).status,'fresh');
assert.equal(board.freshnessLabel({...food('cooked'),stage:'cooked_protein',freshness_basis:'in_pan'},s),null);
const moved={...food('human')};assert.equal(board.freshnessLabel(moved,s).text,board.freshnessLabel(s.human.holding,s).text);
s.human.holding.stage='spoiled';s.buffers.ai_raw[1].stage='spoiled';board.lang='zh';labels.length=0;board.draw(s,s,1);
assert.equal(labels.filter(x=>x.text==='已变质'&&x.color==='#b4233b').length,2);
// Replay uses the selected saved frame, and animations use the same frame's
// item and expiry together rather than mixing old cargo with the new clock.
assert(board.freshnessLabel(food('old'),{turn:11,food_freshness:[]}).text.includes('19'));
s.human.holding.stage='raw';s.human.preparation={ingredient:'egg',ready:false,completed:1,required:4,remaining:3};labels.length=0;board.draw(s,s,1);assert(labels.some(x=>x.text==='打散还需 3 次'));
s.human.preparation.ingredient='meat';labels.length=0;board.draw(s,s,1);assert(labels.some(x=>x.text==='切配还需 3 次'));
console.log('Kitchen freshness canvas: all six locations, warning boundary, spoilage, transfer/replay identity and highlights passed.');
