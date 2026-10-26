'use strict';
(()=>{
 const clone=x=>JSON.parse(JSON.stringify(x));
 class EditHistory{
  constructor(limit=80){this.limit=limit;this.past=[];this.future=[];this.current=null}
  reset(value){this.current=clone(value);this.past=[];this.future=[]}
  adopt(value){this.current=clone(value)}
  record(value,group=null,now=Date.now(),before=this.current){
   if(this.current===null){this.reset(value);return false}if(JSON.stringify(this.current)===JSON.stringify(value))return false;
   const last=this.past[this.past.length-1];
   if(group&&last?.group===group&&now-last.time<750&&!this.future.length){last.after=clone(value);last.time=now}
   else this.past.push({before:clone(before),after:clone(value),group,time:now});
   if(this.past.length>this.limit)this.past.shift();this.future=[];this.current=clone(value);return true;
  }
  undo(){const item=this.past.pop();if(!item)return null;this.future.push(item);this.current=clone(item.before);return clone(this.current)}
  redo(){const item=this.future.pop();if(!item)return null;this.past.push(item);this.current=clone(item.after);return clone(this.current)}
 }
 window.DublyEditHistory=EditHistory;
})();
