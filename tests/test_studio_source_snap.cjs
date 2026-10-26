const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const scope={window:{}};vm.createContext(scope);vm.runInContext(fs.readFileSync('static/studio-preview.js','utf8'),scope);
const {speechRegions,groupRegions,boundaries,nearest,move}=scope.window.DublySourceSnap;
test('quiet edges come from audio peaks rather than misplaced ASR segments',()=>{
 const peaks=Array(1000).fill(.0001);for(let i=536;i<552;i++)peaks[i]=.2;
 const regions=speechRegions({start:40,end:50,peaks});assert.equal(regions.length,1);
 assert.ok(Math.abs(regions[0].start-45.345)<1e-7);assert.ok(Math.abs(regions[0].end-45.545)<1e-7);
});
test('tiny isolated noise is excluded and short intra-word closures are bridged',()=>{
 const peaks=Array(200).fill(0);for(let i=20;i<30;i++)peaks[i]=.2;for(let i=35;i<45;i++)peaks[i]=.2;peaks[100]=1;
 const regions=speechRegions({start:0,end:2,peaks});assert.equal(regions.length,1);assert.ok(regions[0].end>.44);
});
test('viewport crop edges are never artificial snap targets',()=>{
 const regions=speechRegions({start:10,end:11,peaks:Array(100).fill(.2)});
 assert.equal(regions.length,1);assert.equal(boundaries(regions).length,0);
});
test('snapping uses screen pixels, can be bypassed, and releases beyond hysteresis',()=>{
 assert.equal(nearest(10.04,[10],200).time,10);
 assert.equal(nearest(10.06,[10],200).time,10.06);
 assert.equal(nearest(10.04,[10],200,{disabled:true}).time,10.04);
 assert.equal(nearest(10.06,[10],200,{locked:10}).time,10);
 assert.equal(nearest(10.08,[10],200,{locked:10}).time,10.08);
 assert.equal(nearest(10.06,[10],100).time,10);
});
test('moving aligns either edge while preserving clip duration',()=>{
 assert.equal(move(9.96,2,[10],200).position,10);
 assert.equal(move(7.96,2,[10],200).position,8);
 assert.equal(move(7.96,2,[10],200,{disabled:true}).position,7.96);
});

test('phrase groups bridge short pauses and expose only outer snap boundaries',()=>{
 const fine=[{start:1,end:1.3},{start:1.5,end:2},{start:2.2,end:3},{start:4,end:5}];
 const grouped=groupRegions(fine);
 assert.equal(grouped.length,2);assert.equal(grouped[0].parts.length,3);
 assert.deepEqual(Array.from(boundaries(grouped)),[1,3,4,5]);
 assert.equal(fine[0].end,1.3);
});
test('short replies from another voice retain their own boundaries',()=>{
 const fine=[{start:44.6,end:44.8},{start:45.34,end:45.56},{start:46,end:47.7}];
 const turns=[{start:44.48,end:45.23,speaker_id:'0'},{start:45.34,end:45.56,speaker_id:'2'},{start:46.03,end:47.74,speaker_id:'1'}];
 assert.equal(groupRegions(fine,turns).length,3);
});
test('an unknown span cannot bridge two known different voices',()=>{
 const fine=[{start:1,end:1.2},{start:1.4,end:1.6},{start:1.8,end:2.2}];
 const turns=[{start:1,end:1.2,speaker_id:'0'},{start:1.8,end:2.2,speaker_id:'1'}];
 assert.equal(groupRegions(fine,turns).length,2);
});

test('both syllables of a short reply belong to its voice after punctuation-tail repair',()=>{
 const fine=[{start:44.418,end:44.778},{start:45.081,end:45.273},{start:45.349,end:45.604}];
 const turns=[{start:44.49,end:44.71,speaker_id:'0'},{start:45.08,end:45.57,speaker_id:'2'}];
 const groups=groupRegions(fine,turns);
 assert.equal(groups.length,2);assert.equal(groups[0].parts.length,1);
 assert.equal(groups[1].parts.length,2);assert.equal(groups[1].start,45.081);
 assert.equal(groups[1].voice,'2');
});
