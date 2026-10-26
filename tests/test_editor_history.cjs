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
