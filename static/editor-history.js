'use strict';
(()=>{
 const clone=x=>JSON.parse(JSON.stringify(x));
 const canonical=x=>JSON.stringify(x,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
 class EditHistory{
  constructor(limit=80,key=null){this.limit=limit;this.key=key;this.past=[];this.future=[];this.current=null}
  persist(){
   if(!this.key)return;
   for(;;)try{window.localStorage.setItem(this.key,JSON.stringify({version:1,current:this.current,past:this.past,future:this.future}));return}catch(error){
    if(error?.name!=='QuotaExceededError')return;
    if(this.past.length)this.past.shift();else if(this.future.length)this.future.shift();else return;
   }
  }
  restore(value){
   if(this.key)try{
    const saved=JSON.parse(window.localStorage.getItem(this.key));
    if(saved?.version===1&&Array.isArray(saved.past)&&Array.isArray(saved.future)&&[...saved.past,...saved.future].every(item=>item&&item.before&&item.after)&&canonical(saved.current)===canonical(value)){
     this.current=clone(value);this.past=saved.past.slice(-this.limit);this.future=saved.future.slice(-this.limit);this.persist();return true;
    }
   }catch{}
   this.reset(value);return false;
  }
  reset(value){this.current=clone(value);this.past=[];this.future=[];this.persist()}
  adopt(value){this.current=clone(value);this.persist()}
  record(value,group=null,now=Date.now(),before=this.current){
   if(this.current===null){this.reset(value);return false}if(JSON.stringify(this.current)===JSON.stringify(value))return false;
   const last=this.past[this.past.length-1];
   if(group&&last?.group===group&&now-last.time<750&&!this.future.length){last.after=clone(value);last.time=now}
   else this.past.push({before:clone(before),after:clone(value),group,time:now});
   if(this.past.length>this.limit)this.past.shift();this.future=[];this.current=clone(value);this.persist();return true;
  }
  undo(){const item=this.past.pop();if(!item)return null;this.future.push(item);this.current=clone(item.before);this.persist();return clone(this.current)}
  redo(){const item=this.future.pop();if(!item)return null;this.past.push(item);this.current=clone(item.after);this.persist();return clone(this.current)}
 }
 window.DublyEditHistory=EditHistory;
})();
