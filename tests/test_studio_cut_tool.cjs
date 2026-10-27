const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const source=fs.readFileSync('static/studio.js','utf8');
function fn(name){const begin=source.indexOf('function '+name+'('),body=source.indexOf('{',begin);let depth=0;for(let i=body;i<source.length;i++){if(source[i]==='{')depth++;if(source[i]==='}'&&!--depth)return source.slice(begin,i+1)}}

test('single click centers script without opening a popup; double click opens it',()=>{
 let centered=null,opened=null;
 const state={scrubDrag:null,timelineDrag:{id:3,moved:false},timelineSnap:null,$:()=>({hidden:false}),drawTimeline(){},centerDialogue:id=>centered=id,openClipEditor:id=>opened=id,
 timeline:{},cutHeld:false,splittingClip:null,timelineHit:()=>({row:{id:3}})};
 vm.createContext(state);vm.runInContext(fn('finishTimeline'),state);state.finishTimeline({type:'pointerup'});
 assert.equal(centered,3);assert.equal(opened,null);
 const begin=source.indexOf('timeline.ondblclick=');vm.runInContext(source.slice(begin,source.indexOf('timeline.onpointerup=',begin)),state);
 state.timeline.ondblclick({preventDefault(){}});assert.equal(opened,3);
});

test('centering scrolls only the dialogue panel with smooth motion',()=>{
 let options=null;
 const card={dataset:{id:'2'},clientHeight:200,getBoundingClientRect:()=>({top:600})};
 const list={children:[card],scrollTop:150,clientHeight:500,getBoundingClientRect:()=>({top:100}),scrollTo:value=>options=value};
 const state={$:()=>list,filter:'',matchMedia:()=>({matches:false})};vm.createContext(state);vm.runInContext(fn('centerDialogue'),state);state.centerDialogue(2);
 assert.equal(options.top,500);assert.equal(options.behavior,'smooth');
});

test('holding C cuts at the waveform time instead of dragging or opening editor',()=>{
 let cut=null;const row={id:4};
 const state={doc:{},sourceWave:{},cutHeld:true,busy:false,timeline:{},$:()=>({}),timelineHit:()=>({row,pos:5,end:9}),timeAt:()=>7.25,splitTimelineClip:(r,t)=>cut=[r.id,t],
 selectRow(){throw Error('Cutting must not select or open the editor')}};
 vm.createContext(state);const begin=source.indexOf('timeline.onpointerdown=');vm.runInContext(source.slice(begin,source.indexOf('timeline.onpointermove=',begin)),state);
 state.timeline.onpointerdown({button:0,preventDefault(){}});assert.deepEqual(cut,[4,7.25]);
});


test('cut target outline and guide agree with pointer after scroll and zoom',()=>{
 for(const [offset,width] of [[0,2120],[677,2120],[1300,4120]]){
  const scroll={scrollLeft:offset,clientWidth:1000};let drawn=null;const boxes=[],labels=[];
  const ctx={save(){},restore(){},rect(){},clip(){},fillRect(){},fillText:t=>labels.push(t),measureText:t=>({width:t.length*6}),roundRect:(...r)=>boxes.push(r),setLineDash(){},beginPath(){},moveTo(x){drawn=x},lineTo(){},stroke(){},fill(){}};
  const row={id:3,speaker_id:'voice',audio_ready:true};
  const state={doc:{duration:100},timeline:{getBoundingClientRect:()=>({left:200})},$:()=>scroll,timelineDocument:()=>({duration:100}),layoutTimeline:()=>({width}),
   allSpeakers:()=>['voice'],TRACK_HEIGHT:70,stamp:t=>t.toFixed(2),T:s=>s,cutHeld:true,cutHover:null,timelinePointer:{clientX:610},busy:false,splittingClip:null,splitPosition:null,
   timelineHit:()=>({row,pos:0,end:100}),rowRegenerating:()=>false};
  vm.createContext(state);vm.runInContext(fn('timeAt')+'\n'+fn('cutPreview')+'\n'+fn('drawCutPreview'),state);
  state.cutHover=state.timeAt(state.timelinePointer);state.drawCutPreview(ctx,{duration:100},width,300,1000,offset);
  assert.ok(Math.abs(drawn-410)<1e-7,`scroll=${offset}: guide=${drawn}`);
  assert.equal(boxes[0][0],110-offset);assert.equal(boxes[0][1],108);
  assert.ok(labels[0].startsWith('#4 · '));assert.ok(labels[0].endsWith('자르기'));
  assert.ok(boxes.at(-1)[0]>=110&&boxes.at(-1)[0]+boxes.at(-1)[2]<=996,'label remains in visible timeline');
 }
});

test('cut preview only targets a splittable interior and remains while splitting after C is released',()=>{
 const row={id:2,speaker_id:'0',start:20,audio_ready:true};
 const state={cutHeld:true,cutHover:23.45,timelinePointer:{},busy:false,splittingClip:null,splitPosition:null,
  timelineHit:()=>({row,pos:20,end:25}),rowRegenerating:()=>false,doc:{segments:[null,null,row]},waveCache:new Map(),durationOf:()=>5};
 vm.createContext(state);vm.runInContext(fn('cutPreview'),state);
 assert.equal(state.cutPreview().row.id,2);state.cutHover=20;assert.equal(state.cutPreview(),null);
 state.cutHover=23.45;row.keep_original=true;assert.equal(state.cutPreview(),null);row.keep_original=false;
 state.cutHeld=false;assert.equal(state.cutPreview(),null);
 state.splittingClip=2;state.splitPosition=23.45;state.timelinePointer=null;
 assert.equal(state.cutPreview().processing,true);assert.equal(state.cutPreview().time,23.45);
});

test('pressing C immediately shows the cut guide at the last pointer position',()=>{
 const point={clientX:430,clientY:180};let paints=0;
 const state={cutHeld:false,cutHover:null,timelinePointer:point,timeline:{classList:{toggle(){}}},
  timelineHit:p=>p===point?{row:{id:1}}:null,timeAt:()=>23.45,drawTimeline(){paints++}};
 vm.createContext(state);vm.runInContext(fn('refreshCutHover')+'\n'+fn('setCutMode'),state);
 state.setCutMode(true);assert.equal(state.cutHover,23.45);assert.equal(paints,1);
 state.setCutMode(false);assert.equal(state.cutHover,null);
 state.timelinePointer=null;state.setCutMode(true);assert.equal(state.cutHover,null);
});


test('playhead follow scrolls at both edges and stays clamped at video ends',()=>{
 const state={};vm.createContext(state);vm.runInContext(fn('playheadScrollLeft'),state);
 const follow=state.playheadScrollLeft;
 assert.equal(follow(500,3000,1000,0),0);
 assert.equal(follow(950,3000,1000,0),30);
 assert.equal(follow(1110,3000,1000,1000),920);
 assert.equal(follow(110,3000,1000,1000),0);
 assert.equal(follow(2990,3000,1000,0),2000);
 assert.equal(follow(300,1000,1000,0),0);
});


test('space toggles playback from buttons and panels, but preserves text input and IME',()=>{
 let toggles=0,prevented=0;
 const state={loading:false,toggleVideoPlayback:()=>toggles++};vm.createContext(state);
 vm.runInContext(fn('editingText')+'\n'+fn('studioPlaybackKey'),state);
 const event=target=>({code:'Space',target,preventDefault(){prevented++},stopPropagation(){}});
 const button={closest:()=>null};
 state.studioPlaybackKey(event(button));assert.equal(toggles,1);assert.equal(prevented,1);
 state.studioPlaybackKey({...event(button),repeat:true});assert.equal(toggles,1);
 state.studioPlaybackKey(event({closest:()=>({})}));assert.equal(toggles,1);
 state.studioPlaybackKey(event({isContentEditable:true}));assert.equal(toggles,1);
 state.studioPlaybackKey({...event(button),isComposing:true});assert.equal(toggles,1);
 state.studioPlaybackKey({...event(button),ctrlKey:true});assert.equal(toggles,1);
});


test('actual playhead updates follow forward and backward seeks, not just the scroll formula',()=>{
 const scroll={scrollLeft:0,clientWidth:1000},head={style:{},hidden:true},clock={};
 const state={doc:{duration:100},sourceWave:null,video:{currentTime:90,paused:true},timelineDrag:null,
  timelineDocument:()=>({duration:100}),layoutTimeline:()=>({width:3000}),$:id=>id==='timeline-scroll'?scroll:id==='timeline-playhead'?head:clock,
  performance:{now:()=>100},lastClockPaint:0,stamp:String};
 vm.createContext(state);vm.runInContext(fn('playheadScrollLeft')+'\n'+fn('updatePlayhead'),state);
 state.updatePlayhead(true);assert.equal(scroll.scrollLeft,1782);assert.equal(head.hidden,false);
 state.video.currentTime=1;state.updatePlayhead(true);assert.equal(scroll.scrollLeft,0);assert.equal(head.hidden,false);
 scroll.scrollLeft=400;state.updatePlayhead(false);assert.equal(scroll.scrollLeft,400);
});

test('scrubbing continues scrolling even with a stationary pointer and a paused video',()=>{
 let frames=0,follow=null;
 const state={video:{paused:true,currentTime:0},scrubDrag:{x:980,y:80},lastPlayheadFrame:0,raf:0,
  seekTimeline:()=>state.video.currentTime=75,updatePlayhead:(...args)=>follow=args,requestAnimationFrame:()=>++frames};
 vm.createContext(state);vm.runInContext(fn('playheadFrame'),state);state.playheadFrame(100);
 assert.equal(state.video.currentTime,75);assert.equal(frames,1);assert.equal(follow[0],true);assert.ok(follow[1]>0&&follow[1]<20);
 state.scrubDrag=null;state.playheadFrame(116);assert.equal(state.raf,0);
});

test('manual playhead snapping uses generated edited clip edges, hysteresis, and Alt bypass',()=>{
 const rows=[{id:0,start:1,end:2,audio_ready:true,audio_edit:{position:10,trim_start:.2,trim_end:1.7,speed:1.5}},
  {id:1,start:5,audio_ready:true,deleted:true},{id:2,start:20,audio_ready:true,audio_error:'failed'},
  {id:3,start:21,audio_ready:true,keep_original:true},{id:4,start:22,audio_ready:false},
  {id:5,start:30,audio_ready:true,audio_seconds:1.5}];
 const state={window:{},doc:{duration:40,segments:rows},waveCache:new Map([[0,{duration:2,pauses:[]}]]),scrubDrag:{snapBoundary:null},video:{currentTime:0},timelineSnap:null,
  layoutTimeline:()=>({width:4120}),timeAt:e=>e.time,timeline:{getBoundingClientRect:()=>({top:0})},TRACK_HEIGHT:70,allSpeakers:()=>[]};
 vm.createContext(state);vm.runInContext(fs.readFileSync('static/studio-preview.js','utf8'),state);
 vm.runInContext(fn('durationOf')+'\n'+fn('clipSnapCandidates')+'\n'+fn('playheadSnapTargets')+'\n'+fn('seekTimeline'),state);
 assert.deepEqual(Array.from(state.playheadSnapTargets()),[10,11,30,31.5]);
 state.seekTimeline({time:11.04});assert.equal(state.video.currentTime,11);assert.equal(state.timelineSnap,11);
 state.seekTimeline({time:11.12});assert.equal(state.video.currentTime,11);
 state.seekTimeline({time:11.16});assert.equal(state.video.currentTime,11.16);assert.equal(state.timelineSnap,null);
 state.seekTimeline({time:30.04});assert.equal(state.video.currentTime,30);
 state.seekTimeline({time:30.04,altKey:true});assert.equal(state.video.currentTime,30.04);assert.equal(state.timelineSnap,null);
});

test('dragging the white playhead starts scrubbing even over a clip',()=>{
 let dragged=0,stopped=0;const head={};
 const state={timeline:{},$:()=>head,doc:{},sourceWave:null,cutHeld:false,beginScrub(){dragged++}};
 vm.createContext(state);const start=source.indexOf("$('timeline-playhead').onpointerdown=");vm.runInContext(source.slice(start,source.indexOf('timeline.onpointerenter=',start)),state);
 head.onpointerdown({button:0,preventDefault(){},stopPropagation(){stopped++}});
 assert.equal(dragged,1);assert.equal(stopped,1);
 state.cutHeld=true;head.onpointerdown({button:0});assert.equal(dragged,1);
});
