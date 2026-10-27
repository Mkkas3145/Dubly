const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context=vm.createContext({window:{}});vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/studio-preview.js'),'utf8'),context);const math=context.window.DublyPreviewMath;
const info={duration:3,sampleRate:1000,peaks:Array.from({length:300},(_,i)=>i<100||i>=200?.8:0),pauses:[{id:0,start:1,end:2,cut:1.5}]};
test('preview uses the server fixed gain once, including speed-rendered clips and cache changes',async()=>{
 const raw=new Float32Array(1000).fill(.1),buffer={numberOfChannels:1,sampleRate:1000,duration:1,getChannelData:()=>raw};let gainDb=4;
 const preview={edited:new Map(),getDoc:()=>({tts_gain_db:gainDb,revision:1}),getWave:async()=>({duration:1,sampleRate:1000,peaks:[.1],pauses:[]}),buffer:async()=>buffer,save:async()=>{},context:()=>({createBuffer:(channels,length,sampleRate)=>({duration:length/sampleRate,copyToChannel(){}})})};
 const clip=context.window.DublyLivePreview.prototype.clip;
 for(const speed of [1,2]){const value=await clip.call(preview,{id:speed,audio_key:'same',audio_edit:{speed}});assert.ok(Math.abs(value.gain-10**(4/20))<1e-12)}
 gainDb=0;const changed=await clip.call(preview,{id:1,audio_key:'same',audio_edit:{speed:1}});assert.equal(changed.gain,1);assert.equal(raw[0],Math.fround(.1));
});
test('pause expansion shifts later speech and waveform together; removing it reverses the change',()=>{
 const source=new Float32Array(3000);source.fill(.25,0,1000);source.fill(.75,2000);
 const edit={pauses:[{id:0,seconds:2}]},result=math.editPCM([source],1000,info,edit)[0],view=math.waveform(info,edit);
 assert.equal(result.length,4000);assert.deepEqual(Array.from(result.slice(0,1000)),Array.from(source.slice(0,1000)));assert.deepEqual(Array.from(result.slice(3000)),Array.from(source.slice(2000)));assert.equal(view.duration,4);assert.equal(view.map(2.5),3.5);assert.equal(view.unmap(3.5),2.5);
 assert.equal(math.waveform(info,{}).duration,3);assert.equal(view.peaks.filter(p=>p===0).length,200);
});
test('pause contraction preserves both surrounding speech regions and trim coordinates',()=>{
 const source=new Float32Array(3000);source.fill(.3,0,1000);source.fill(.7,2000);const edit={trim_start:.25,trim_end:2.75,pauses:[{id:0,seconds:.5}]},result=math.editPCM([source],1000,info,edit)[0];
 assert.equal(result.length,2000);assert.deepEqual(Array.from(result.slice(0,750)),Array.from(source.slice(250,1000)));assert.deepEqual(Array.from(result.slice(-750)),Array.from(source.slice(2000,2750)));assert.equal(math.waveform(info,edit).duration,2);
});
test('mix weights preserve original fallback lines and use max for overlapping fades',()=>{
 const windows=[{start:1,end:3},{start:2,end:4}];assert.equal(math.weight(2.5,windows),1);assert.equal(math.weight(2.5,windows,[{start:2,end:3}]),0);assert.equal(math.weight(0,windows),0);assert.ok(Math.abs(math.weight(.95,windows)-.5)<1e-8);
});
test('edge fitting changes quiet samples before accelerating; it never trims speech',()=>{
 const small=math.fitDuration(info,{},2.4),short=math.fitDuration(info,{},1.5),long=math.fitDuration(info,{},4);
 assert.equal(small.speed,1);assert.ok(Math.abs(small.pauses[0].seconds-.4)<1e-8);assert.ok(Math.abs(math.waveform(info,small).duration-2.4)<1e-8);
 assert.ok(short.speed>1);assert.equal(short.pauses[0].seconds,.01);assert.ok(Math.abs(math.waveform(info,short).duration-1.5)<1e-8);assert.equal(short.trim_start,undefined);assert.equal(short.trim_end,undefined);
 assert.equal(long.speed,1);assert.equal(long.pauses[0].seconds,2);assert.equal(math.waveform(info,long).duration,4);
 const source=new Float32Array(3000);source.fill(.3,0,1000);source.fill(.7,2000);const out=math.editPCM([source],1000,info,short)[0];assert.deepEqual(Array.from(out.slice(0,1000)),Array.from(source.slice(0,1000)));assert.deepEqual(Array.from(out.slice(-1000)),Array.from(source.slice(-1000)));
});
test('extending a clip with no internal pause pads the chosen edge and preserves every sample',()=>{
 const plain={duration:2,sampleRate:1000,peaks:[.5,.5],pauses:[]},source=new Float32Array(2000).fill(.25),left=math.fitDuration(plain,{},3,'start'),right=math.fitDuration(plain,{},3,'end');
 assert.equal(left.pad_start,1);assert.equal(right.pad_end,1);assert.equal(left.speed,1);assert.equal(right.speed,1);
 const padded=math.editPCM([source],1000,plain,left)[0];assert.equal(padded.length,3000);assert.ok(padded.slice(0,1000).every(x=>x===0));assert.deepEqual(Array.from(padded.slice(1000)),Array.from(source));assert.equal(math.waveform(plain,left).duration,3);
});

test('dialogue edits and cuts add no gain envelope',()=>{
 const rate=1000,plain={duration:2,sampleRate:rate,pauses:[],edge_fades:{start:true,end:true}},source=new Float32Array(2000).fill(.5);
 for(const edit of [{},{trim_start:.2,trim_end:1.4},{speed:2},{position:5}]){
  const channels=[source];assert.equal(math.fadePCM(channels,rate,plain,edit),channels);
  assert.deepEqual({...math.editedEdgeFades(plain,edit)},{start:false,end:false});
 }
 assert.equal(source[0],.5);assert.equal(source.at(-1),.5);
});

test('source background transition has 100ms OUTSIDE the complete replacement interval',()=>{
 const windows=[{start:1,end:3}];
 assert.ok(Math.abs(math.weight(.9,windows))<1e-10);
 assert.ok(Math.abs(math.weight(.95,windows)-.5)<1e-10);
 for(const t of [1,1.05,2.95,3])assert.equal(math.weight(t,windows),1);
 assert.ok(Math.abs(math.weight(3.05,windows)-.5)<1e-10);
 assert.ok(Math.abs(math.weight(3.1,windows))<1e-10);
});


test('explicit user fades preserve full-gain middle and match the sine envelope',()=>{
 const source=new Float32Array(1000).fill(.5),plain={duration:1,sampleRate:1000,pauses:[]},edit={fade_in:.2,fade_out:.3};
 const out=math.fadePCM([source],1000,plain,edit)[0];assert.equal(out.length,source.length);
 assert.equal(out[0],0);assert.equal(out.at(-1),0);assert.equal(out[200],.5);assert.equal(out[699],.5);
 assert.ok(Math.abs(out[100]-.5*Math.sin(100/199*Math.PI/2))<1e-7);assert.equal(source[0],.5);
 const f=math.fadeLengths({fade_in:.8,fade_out:.8},1);assert.equal(f.in,.5);assert.equal(f.out,.5);
});

test('fade handle geometry tracks pauses, trim, padding, and output speed',()=>{
 const plain={duration:4,sampleRate:1000,peaks:new Array(400).fill(.5),pauses:[{id:0,start:1,end:2,cut:1.5}]};
 const edit={trim_start:.5,trim_end:3.5,speed:2,pad_start:.2,pad_end:.3,pauses:[{id:0,seconds:2}],fade_in:.25,fade_out:.5};
 const g=math.fadeGeometry(plain,edit,824);assert.equal(g.duration,2);
 assert.ok(Math.abs((g.inEnd-g.start)/(824-24)*g.view.duration-.5)<1e-10);
 assert.ok(Math.abs((g.end-g.outStart)/(824-24)*g.view.duration-1)<1e-10);
 assert.equal(g.fadeIn,.25);assert.equal(g.fadeOut,.5);
 const peaks=math.waveform(plain,{fade_in:1},true).peaks;assert.ok(peaks[0]<.01);assert.ok(peaks.at(-1)>.49);
});
