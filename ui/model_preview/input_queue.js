'use strict';
// One acknowledged command at a time. Discrete actions are FIFO, never coalesced.
class PreviewInputQueue {
  constructor({send, changed=()=>{}, failed=()=>{}, full=()=>{}}) {
    Object.assign(this,{send,changed,failed,full});
    this.items=[];this.current=null;this.failedItem=null;this.waiters=[];
  }
  enqueue(action,held=false) {
    if(held&&this.items.some(x=>x.held))return true;
    if(this.items.length>=8){if(!held)this.full();return false;}
    this.items.push({action,held,payload:null});this.changed();this.pump();return true;
  }
  release(){this.items=this.items.filter(x=>!x.held);this.changed();}
  clear(){const n=this.items.length+(this.failedItem?1:0);this.items=[];this.failedItem=null;this.changed();return n;}
  retry(){if(this.failedItem){this.items.unshift(this.failedItem);this.failedItem=null;this.pump();}}
  idle(){return this.current?new Promise(resolve=>this.waiters.push(resolve)):Promise.resolve();}
  async pump(){
    if(this.current||this.failedItem||!this.items.length)return;
    const item=this.current=this.items.shift();this.changed();
    try{await this.send(item);}catch(error){this.failedItem=item;this.failed(error);}
    finally{
      this.current=null;this.changed();for(const resolve of this.waiters.splice(0))resolve();
      if(!this.failedItem)this.pump();
    }
  }
}
if(typeof module!=='undefined')module.exports=PreviewInputQueue;
else window.PreviewInputQueue=PreviewInputQueue;
