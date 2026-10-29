const {test}=require('node:test');
const assert=require('node:assert/strict');
const Sequence=require('../ui/model_preview/task_sequence.js');
const config={domain:'warehouse',variant:'regularized',seed:40786};
const binding={domain:'warehouse',variant:'regularized',actor_sha256:'frozen-original-pair',programs:[]};
const frame=(task,terminal=false,turn=0)=>({binding:structuredClone(binding),state:{domain:'warehouse',task,turn,terminal,score:{task_score:turn-3},binding_identity_sha256:'bound-pair-identity'}});

test('three tasks require each previous terminal and finish only after task 3',()=>{
  const sequence=new Sequence(config);assert.equal(sequence.nextTask,1);assert(!sequence.finished);
  for(let task=1;task<=3;task++){
    sequence.accept(frame(task),`session-${task}`);assert.equal(sequence.nextTask,null);assert(!sequence.finished);
    sequence.accept(frame(task,true,100),`session-${task}`);
    assert.equal(sequence.nextTask,task<3?task+1:null);assert.equal(sequence.finished,task===3);
  }
  assert.deepEqual(sequence.runs.map(r=>r.task),[1,2,3]);
  assert.deepEqual(sequence.runs.map(r=>r.score),[97,97,97]);
});
test('early advancement, skipped tasks and regressed or replaced sessions fail',()=>{
  const sequence=new Sequence(config);
  assert.throws(()=>sequence.accept(frame(2),'session-2'),/current task/);
  sequence.accept(frame(1,false,2),'session-1');
  assert.throws(()=>sequence.accept(frame(2),'session-2'),/current task/);
  assert.throws(()=>sequence.accept(frame(1,false,1),'session-1'),/regressed/);
  assert.throws(()=>sequence.accept(frame(1,false,2),'replacement'),/session changed/);
  sequence.accept(frame(1,true,3),'session-1');
  assert.throws(()=>sequence.accept(frame(1,false,4),'session-1'),/regressed/);
  assert.throws(()=>sequence.accept(frame(3),'session-3'),/current task/);
  assert.throws(()=>sequence.accept(frame(2),'session-1'),/distinct session/);
});
test('every task and resumed frame preserve exact frozen binding',()=>{
  const sequence=new Sequence(config).accept(frame(1,true,10),'session-1');
  for(const mutate of [v=>v.binding.actor_sha256='another-pair',v=>v.binding.variant='raw',v=>v.state.binding_identity_sha256='another-identity',v=>v.state.domain='pong']){
    const changed=frame(2);mutate(changed);assert.throws(()=>sequence.accept(changed,'session-2'));
  }
  const saved=sequence.toJSON(),restored=new Sequence(saved);
  assert.deepEqual(restored.toJSON(),saved);assert.equal(restored.nextTask,2);
  restored.accept(frame(2,false,2),'session-2');
  assert.equal(sequence.runs.length,1);assert.equal(restored.runs.length,2);
  const external=restored.binding;external.actor_sha256='mutated';assert.deepEqual(restored.binding,binding);
});
test('restore rejects incomplete prefixes, missing bindings and malformed progress',()=>{
  const sequence=new Sequence(config).accept(frame(1,true,10),'session-1').accept(frame(2),'session-2');
  const saved=sequence.toJSON();
  for(const mutate of [s=>s.runs[0].terminal=false,s=>s.runs[1].task=3,s=>s.binding=null,s=>s.runs[0].token='',s=>s.runs[1].binding_identity_sha256='different',s=>s.runs[1].score=Infinity]){
    const bad=structuredClone(saved);mutate(bad);assert.throws(()=>new Sequence(bad));
  }
  assert.throws(()=>new Sequence({...config,seed:NaN}));assert.throws(()=>new Sequence({...config,variant:'unknown'}));
});
