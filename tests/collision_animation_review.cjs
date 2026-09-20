'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context=vm.createContext({window:{}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../study_v3/web/board.js'),'utf8'),context);
const board=Object.create(context.window.StudyBoard.prototype);
const before={domain:'warehouse',human:{x:2,y:4},ai:{x:4,y:4}};
const after=structuredClone(before);
after.collision_animation={kind:'same_target',human:{from:[2,4],attempted:[3,4],to:[2,4]},ai:{from:[4,4],attempted:[3,4],to:[4,4]}};
const unchanged=JSON.stringify([before,after]);
for(const frames of [24,60,144]) {
 const track=Array.from({length:frames+1},(_,i)=>board.actorPosition(before,after,'human',i/frames)[0]);
 assert.equal(track[0],2);assert.equal(track.at(-1),2);
 assert(Math.max(...track)>2.6&&Math.max(...track)<3);
 assert(track[frames/2]>track[frames/4]);
}
assert.equal(JSON.stringify([before,after]),unchanged);
const ordinary=structuredClone(before);
for(const p of [0,.25,.5,.75,1])assert.equal(board.actorPosition(before,ordinary,'human',p)[0],2);
after.collision_animation={kind:'occupied_stationary',human:{from:[2,4],attempted:[3,4],to:[2,4]},ai:{from:[3,4],attempted:[3,4],to:[3,4]}};
before.ai.x=after.ai.x=3;
assert.equal(board.actorPosition(before,after,'ai',.5)[0],3);
assert.equal(board.actorPosition(before,after,'human',1)[0],2);
console.log('Server-confirmed collision rebound, stationary teammate and frame-rate independence checks passed.');
