const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');const ctx=vm.createContext({window:{}});vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/editor-history.js'),'utf8'),ctx);const History=ctx.window.DublyEditHistory;
test('undo and redo restore independent snapshots including deleted and added rows',()=>{
 const h=new History(),base={segments:[{id:0,deleted:false,text:'before'}]};h.reset(base);const added={segments:[{id:0,deleted:false,text:'before'},{id:1,deleted:false,text:'new'}]};h.record(added);h.record({segments:[{id:0,deleted:true,text:'before'},{id:1,deleted:false,text:'new'}]});
 const undone=h.undo();assert.equal(undone.segments[0].deleted,false);undone.segments[0].text='mutated outside';assert.equal(h.current.segments[0].text,'before');assert.equal(h.undo().segments.length,1);assert.equal(h.redo().segments.length,2);assert.equal(h.redo().segments[0].deleted,true);
});
test('typing coalesces, a new action clears redo, and history is bounded',()=>{
 const h=new History(2);h.reset({text:''});h.record({text:'a'},'text',10);h.record({text:'ab'},'text',20);assert.equal(h.past.length,1);assert.equal(h.undo().text,'');h.record({text:'new'},null,30);assert.equal(h.redo(),null);h.record({text:'two'},null,40);h.record({text:'three'},null,50);assert.equal(h.past.length,2);assert.equal(h.undo().text,'two');assert.equal(h.undo().text,'new');assert.equal(h.undo(),null);
});

test('regeneration undo restores the audio and its original translation together',()=>{
 const h=new History(),old={segments:[{id:0,translated_text:'old translation'}],audio_versions:{0:'old-audio'}};
 h.reset(old);h.record({segments:[{id:0,translated_text:'new translation'}],audio_versions:{0:'old-audio'}});
 h.record({segments:[{id:0,translated_text:'new translation'}],audio_versions:{0:'new-audio'}},null,Date.now(),old);
 assert.equal(h.undo().segments[0].translated_text,'old translation');assert.equal(h.current.audio_versions[0],'old-audio');
 assert.equal(h.redo().segments[0].translated_text,'new translation');assert.equal(h.current.audio_versions[0],'new-audio');
});
function persistentHistory(storage){const sandbox=vm.createContext({window:{localStorage:storage}});vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/editor-history.js'),'utf8'),sandbox);return sandbox.window.DublyEditHistory}
function memoryStorage(){const data=new Map();return {getItem:key=>data.get(key)||null,setItem:(key,value)=>data.set(key,value)}}
test('reload preserves both undo and redo per video, including split audio versions',()=>{
 const storage=memoryStorage(),H=persistentHistory(storage),base={segments:[{id:0,text:'whole'}],audio_versions:{0:'full'}},split={segments:[{id:0,text:'left'},{id:1,text:'right'}],audio_versions:{0:'left',1:'right'}};
 const first=new H(80,'video-a');first.reset(base);first.record(split);
 const reload=new H(80,'video-a');assert.equal(reload.restore(split),true);assert.equal(reload.undo().audio_versions[0],'full');reload.adopt(base);
 const again=new H(80,'video-a');assert.equal(again.restore(base),true);assert.equal(again.redo().audio_versions[1],'right');
 const other=new H(80,'video-b');assert.equal(other.restore(base),false);assert.equal(other.past.length,0);
});
test('history ignores key order but discards incompatible server state and corrupt storage',()=>{
 const storage=memoryStorage(),H=persistentHistory(storage),h=new H(80,'video');h.reset({a:1,b:2});h.record({a:2,b:3});
 const next=new H(80,'video');assert.equal(next.restore({b:3,a:2}),true);assert.equal(next.past.length,1);
 assert.equal(next.restore({a:99,b:3}),false);assert.equal(next.past.length,0);
 storage.setItem('video','invalid');assert.equal(new H(80,'video').restore({a:1}),false);
});
test('unavailable browser storage does not break editing; quota keeps recent history',()=>{
 const H=persistentHistory({getItem(){throw Error('blocked')},setItem(){throw Error('blocked')}}),h=new H(80,'video');h.restore({n:0});h.record({n:1});assert.equal(h.undo().n,0);
 let limited=false;const storage=memoryStorage(),set=storage.setItem;storage.setItem=(key,value)=>{if(limited&&JSON.parse(value).past.length>1)throw Object.assign(Error('full'),{name:'QuotaExceededError'});set(key,value)};
 const Q=persistentHistory(storage),q=new Q(80,'video');q.reset({n:0});q.record({n:1});limited=true;q.record({n:2});assert.equal(q.past.length,1);assert.equal(q.undo().n,1);
});
