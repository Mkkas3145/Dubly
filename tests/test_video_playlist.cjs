const {test}=require('node:test'),assert=require('node:assert/strict');
const {groupVideos,adjacent,nextPlayable}=require('../static/video-playlist.js');
test('split siblings stay ordered inside a single group regardless of history sorting',()=>{
 const videos=[{id:'b',filename:'Long (2).mp4',created:3,split:{group_id:'a',title:'Long.mp4',part:2,parts:2}},{id:'x',filename:'Long (3).mp4',created:2},{id:'a',filename:'Long (1).mp4',created:3,split:{group_id:'a',title:'Long.mp4',part:1,parts:2}}];
 for(const descending of [true,false]){
  const groups=groupVideos(videos,descending),group=groups.find(g=>g.split);
  assert.equal(groups.length,2);assert.deepEqual(group.items.map(v=>v.id),['a','b']);assert.equal(group.title,'Long.mp4');
 }
 assert.equal(videos[0].id,'b');
});
test('autoplay stops at an unready next part instead of skipping ahead',()=>{
 const items=[{id:'a',playable:true},{id:'b',playable:false},{id:'c',playable:true}];
 assert.equal(nextPlayable(items,'a'),null);assert.equal(adjacent(items,'a',1).id,'b');
 assert.equal(nextPlayable(items,'b').id,'c');assert.equal(nextPlayable(items,'c'),null);assert.equal(nextPlayable(items,'unknown'),null);
});
