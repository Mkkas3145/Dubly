const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const source=fs.readFileSync('static/studio.js','utf8');
function fn(name){const begin=source.indexOf('function '+name+'('),start=source.lastIndexOf('async ',begin)===begin-6?begin-6:begin;let depth=0,body=source.indexOf('{',begin);for(let i=body;i<source.length;i++){if(source[i]==='{')depth++;if(source[i]==='}'&&!--depth)return source.slice(start,i+1)}throw Error(name)}
test('automatic saving preserves the objects used by text and speaker event handlers',async()=>{
 const row={id:0,translated_text:'old',speaker_id:'0'};
 const state={api:"/api/test",doc:{revision:1,segments:[row]},dirty:true,saving:null,saveTimer:null,busy:false,
 $:()=>({}),clearTimeout(){},payload:()=>({segments:[{...row}]}),request:async()=>({revision:2,updated:1,segments:[{...row}],render_stale:false}),editHistory:{adopt(){}},editableSnapshot(){},syncAudioInputs(){},updateAudioBadges(){},drawTimeline(){},livePreview:{refresh(){}}};
 vm.createContext(state);vm.runInContext(fn('saveDraft'),state);
 await state.saveDraft();assert.equal(state.doc.segments[0],row);
 row.translated_text='edited after saving';state.dirty=true;await state.saveDraft();assert.equal(state.doc.segments[0].translated_text,'edited after saving');
 row.speaker_id='1';state.dirty=true;await state.saveDraft();assert.equal(state.doc.segments[0].speaker_id,'1');
});
test('translation and speaker changes immediately show stale audio and reverting clears it',()=>{
 const badge={textContent:'',classList:{toggle(){}}},row={id:0,original_text:'original',translated_text:'old',speaker_id:'0',audio_key:'sha',audio_ready:true,audio_stale:false};
 const state={busy:false,processing:false,rowRegenerating:()=>false,doc:{segments:[row]},audioInputs:new Map(),updateBatchButton(){},$:()=>({children:[{dataset:{id:'0'},querySelector:()=>badge}]})};vm.createContext(state);
 vm.runInContext([fn('audioPending'),fn('audioState'),fn('ttsSignature'),fn('syncAudioInputs'),fn('updateAudioBadges')].join('\n'),state);
 state.syncAudioInputs();row.translated_text='new';state.updateAudioBadges(true);assert.equal(badge.textContent,'재생성 필요');
 row.translated_text='old';state.updateAudioBadges(true);assert.equal(badge.textContent,'생성됨');
 row.speaker_id='1';state.updateAudioBadges(true);assert.equal(badge.textContent,'재생성 필요');
});

test('vertical clip drag changes speaker track while preserving its timeline position',()=>{
 const row={id:0,start:1,speaker_id:'1',audio_edit:{}},info={duration:2,sha256:'sha'},tracks=['0','1','2'];
 const state={cutHeld:false,cutHover:null,timelineSnapTargets:()=>[],doc:{duration:10,segments:[row]},timelineDrag:{id:0,x:200,y:228,time:1,position:1,edge:'start',info,base:{},speaker:'1',moved:false},scrubDrag:null,busy:false,waveCache:new Map(),TRACK_HEIGHT:80,timeline:{style:{},getBoundingClientRect:()=>({top:0})},allSpeakers:()=>tracks,timeAt:()=>1,durationOf:()=>2,updateAudioBadges(){},drawTimeline(){},selected:null,wave:null,livePreview:{refresh(){}}};
 Object.assign(state,{layoutTimeline:()=>({width:1120}),sourceEdges:[],window:{DublySourceSnap:{move:p=>({position:p,boundary:null})}},timelineSnap:null});
 vm.createContext(state);const begin=source.indexOf('timeline.onpointermove=e=>{'),end=source.indexOf('function finishTimeline',begin);vm.runInContext(source.slice(begin,end),state);
 state.timeline.onpointermove({clientX:200,clientY:148});assert.equal(row.speaker_id,'0');assert.equal(row.audio_edit.position,1);assert.equal(state.timelineDrag.moved,true);
 state.timeline.onpointermove({clientX:200,clientY:308});assert.equal(row.speaker_id,'2');assert.equal(row.audio_edit.position,1);
});

test('an ungenerated clip moves in time and across speaker tracks without a waveform',()=>{
 const row={id:0,start:1,end:3,speaker_id:'1',audio_ready:false,audio_edit:{}},tracks=['0','1'];let marks=0;
 const state={cutHeld:false,cutHover:null,timelineSnapTargets:()=>[],doc:{duration:10,segments:[row]},timelineDrag:{id:0,x:200,y:228,time:1,position:1,end:3,edge:null,base:{},speaker:'1',moved:false},
  scrubDrag:null,busy:false,waveCache:new Map(),TRACK_HEIGHT:80,timeline:{style:{},getBoundingClientRect:()=>({top:0})},allSpeakers:()=>tracks,
  timeAt:e=>e.clientX/100-1,durationOf:r=>r.end-r.start,updateAudioBadges(){},drawTimeline(){},selected:null,wave:null,livePreview:{refresh(){}},
  $:()=>({}),markDirty(){marks++},renderScript(){},openClipEditor(){throw Error('Dragging must not open the editor')}};
 Object.assign(state,{layoutTimeline:()=>({width:1120}),sourceEdges:[],window:{DublySourceSnap:{move:p=>({position:p,boundary:null})}},timelineSnap:null});
 vm.createContext(state);const begin=source.indexOf('timeline.onpointermove=e=>{'),end=source.indexOf('timeline.onpointerup=finishTimeline',begin);
 vm.runInContext(source.slice(begin,end),state);
 state.timeline.onpointermove({clientX:400,clientY:148});assert.equal(row.audio_edit.position,3);assert.equal(row.speaker_id,'0');
 assert.deepEqual(Object.keys(row.audio_edit),['position']);assert.equal(state.timelineDrag.moved,true);
 state.finishTimeline({type:'pointerup'});assert.equal(marks,1);
});

test('only pending TTS clips stay gray as completed clips become available during the same job',()=>{
 const state={busy:true,processing:true,job:null,rowRegenerating:()=>false};vm.createContext(state);vm.runInContext([fn('audioPending'),fn('audioState')].join('\n'),state);
 const pending={audio_ready:false};assert.equal(state.audioPending(pending),true);assert.equal(state.audioState(pending),'음성 생성 중…');
 const ready={audio_ready:true};assert.equal(state.audioPending(ready),false);assert.equal(state.audioState(ready),'생성됨');
 assert.equal(state.audioPending({...ready,audio_processing:true}),true);
 assert.equal(state.audioPending({...pending,keep_original:true}),false);
 assert.equal(state.audioPending({...pending,audio_error:'failed'}),false);
 state.busy=false;state.processing=false;assert.equal(state.audioPending(pending),false);assert.equal(state.audioState(pending),'미생성');
});

test('ungenerated clip edges store a target range without inventing waveform trim values',()=>{
 const row={start:1,end:3,audio_edit:{}},state={doc:{duration:10}};vm.createContext(state);
 vm.runInContext([fn('applyClipLength'),fn('durationOf')].join('\n'),state);
 state.applyClipLength(row,null,{},1,3,'end',6);
 assert.equal(row.audio_edit.position,1);assert.equal(state.durationOf(row,null),5);
 assert.equal(row.audio_edit.trim_end,undefined);
 state.applyClipLength(row,null,{},1,3,'start',.5);
 assert.equal(row.audio_edit.position,.5);assert.equal(state.durationOf(row,null),2.5);
 state.applyClipLength(row,null,{},1,3,'end',50);assert.equal(state.durationOf(row,null),9);
});

test('failed TTS markers preserve the source dialogue span instead of collapsing to one sample',()=>{
 const state={};vm.createContext(state);vm.runInContext(fn('durationOf'),state);
 const row={start:21.94,end:22.87,audio_seconds:1/44100,audio_error:'generated_dialogue_not_aligned',audio_edit:{}};
 assert.ok(Math.abs(state.durationOf(row,null)-.93)<1e-9);
 assert.ok(Math.abs(state.durationOf(row,{duration:1/44100,pauses:[]})-.93)<1e-9);
 row.audio_edit={position:20,target_duration:2};assert.equal(state.durationOf(row,null),2);
 delete row.audio_error;row.audio_edit={};row.audio_seconds=1.4;assert.equal(state.durationOf(row,null),1.4);
});

test('expected play interruption is ignored while genuine playback errors remain visible',()=>{
 const notices=[],state={say:text=>notices.push(text)};vm.createContext(state);vm.runInContext(fn('reportPlaybackError'),state);
 state.reportPlaybackError({name:'AbortError',message:'play interrupted by pause'});assert.equal(notices.length,0);
 state.reportPlaybackError({name:'NotSupportedError'});assert.equal(notices[0],'오디오 또는 영상을 재생할 수 없습니다.');
 state.reportPlaybackError({name:'NetworkError',message:'connection failed'});assert.equal(notices[1],'connection failed');
});
test('closing the clip editor while saving prevents delayed audio playback',async()=>{
 let saved,plays=0;const dialog={open:true},state={clipPlaybackEpoch:0,selected:0,api:'/api/test',doc:{revision:1,segments:[{start:0}]},saveDraft:()=>new Promise(resolve=>saved=resolve),audio:{pause(){},play(){plays++;return Promise.resolve()},dataset:{}},video:{pause(){}},$:()=>dialog,reportPlaybackError(){throw Error('unexpected error')}};
 vm.createContext(state);vm.runInContext(fn('playClip'),state);const playback=state.playClip(0,'edited');dialog.open=false;state.clipPlaybackEpoch++;saved();await playback;assert.equal(plays,0);
});
test('switching clip source quickly gives only the latest request control of playback',async()=>{
 const pending=[],notices=[],status={},dialog={open:true},state={clipPlaybackEpoch:0,selected:0,api:'/api/test',doc:{revision:1,segments:[{start:1}]},saveDraft:()=>Promise.resolve(),audio:{pause(){},play(){return new Promise((resolve,reject)=>pending.push({resolve,reject}))},dataset:{}},video:{pause(){}},$:id=>id==='clip-dialog'?dialog:status,reportPlaybackError:e=>notices.push(e)};
 vm.createContext(state);vm.runInContext(fn('playClip'),state);
 const first=state.playClip(0,'reference');await new Promise(setImmediate);const second=state.playClip(0,'edited');await new Promise(setImmediate);
 pending[0].reject({name:'AbortError'});pending[1].resolve();await Promise.all([first,second]);assert.equal(notices.length,0);assert.equal(status.textContent,'편집 음성 재생 중');assert.equal(state.audio.dataset.kind,'edited');
});

test('source waveform loads before the draft and is replaced when separation becomes ready',async()=>{
 const loads=[],state={doc:null,sourceWave:null,sourceWaveKey:null,sourceWaveLoading:false,sourceDetail:{},sourceDetailRegions:[1],detailRequest:0,sourceRegions:[],api:'/studio/test',
 request:async()=>loads.shift(),window:{DublySourceSnap:{speechRegions:()=>[{start:1,end:2}]}},updateSourceSpeech(){},drawTimeline(){},queueSourceDetail(){},$:()=>({classList:{add(){}}})};
 vm.createContext(state);vm.runInContext([fn('timelineDocument'),fn('refreshSourceWave')].join('\n'),state);
 loads.push({duration:90,kind:'original',key:'original',peaks:[.3]});await state.refreshSourceWave({key:'original'});
 assert.equal(state.timelineDocument().duration,90);assert.equal(state.doc,null);assert.equal(state.sourceRegions.length,0);
 loads.push({duration:90,kind:'clean',key:'clean',peaks:[.1]});await state.refreshSourceWave({key:'clean'});
 assert.equal(state.sourceWave.kind,'clean');assert.equal(state.sourceRegions.length,1);assert.equal(state.sourceDetail,null);
 await state.refreshSourceWave({key:'clean'});assert.equal(loads.length,0);
});

test('timeline click and drag seek before a draft exists, without editing any rows',()=>{
 const state={cutHeld:false,cutHover:null,doc:null,sourceWave:{duration:90},busy:true,sourceWaveKey:'original',timelineDrag:null,scrubDrag:null,
 timeline:{getBoundingClientRect:()=>({left:0}),setPointerCapture(){},style:{}},video:{currentTime:0},
 $:()=>({scrollLeft:0}),layoutTimeline:()=>({width:1020}),startPlayhead(){},drawTimeline(){}};
 vm.createContext(state);vm.runInContext([fn('timelineDocument'),fn('timeAt'),fn('timelineHit'),fn('seekTimeline'),fn('beginScrub')].join('\n'),state);
 const down=source.indexOf('timeline.onpointerdown=e=>{'),move=source.indexOf('timeline.onpointermove=e=>{',down),end=source.indexOf('function finishTimeline',move);
 vm.runInContext(source.slice(down,end),state);
 state.timeline.onpointerdown({button:0,clientX:210,pointerId:1});assert.equal(state.video.currentTime,10);
 state.timeline.onpointermove({clientX:410});assert.equal(state.video.currentTime,30);assert.equal(state.doc,null);assert.equal(state.timelineDrag,null);
 state.timeline.onpointermove({clientX:1500});assert.equal(state.video.currentTime,90);
});

test('single-row regeneration keeps editing open while bulk processing still locks edits',()=>{
 const controls=new Map(),state={processing:false,busy:false,lastGenerationStatus:'',doc:{},dirty:true,job:{workflow_action:'tts',editor_index:0,regenerating_indices:[0]},wave:null,$:id=>{if(!controls.has(id))controls.set(id,{classList:{toggle(){}},disabled:false});return controls.get(id)},updateBatchButton(){},renderScript(){},updateSkeletonAnimation(){}};
 vm.createContext(state);vm.runInContext(fn('setBusy'),state);state.setBusy(true);
 assert.equal(state.processing,true);assert.equal(state.busy,false);assert.equal(controls.get('save').disabled,false);assert.equal(controls.get('search').disabled,false);
 state.job={workflow_action:'tts',editor_index:null};state.setBusy(true);assert.equal(state.busy,true);assert.equal(controls.get('save').disabled,true);
});
test('incoming generated metadata preserves only local changed fields during editing',()=>{
 const baseline={revision:1,segments:[{id:0,translated_text:'old',audio_edit:{position:1},audio_key:'old-audio'}]},state={lastRemoteDoc:baseline,doc:JSON.parse(JSON.stringify(baseline))};state.doc.segments[0].translated_text='typed while generating';
 vm.createContext(state);vm.runInContext(fn('mergeRemoteDocument'),state);
 const merged=state.mergeRemoteDocument({revision:2,segments:[{id:0,translated_text:'old',audio_edit:{position:1},audio_key:'new-audio'}]});
 assert.equal(merged.revision,2);assert.equal(merged.segments[0].audio_key,'new-audio');assert.equal(merged.segments[0].translated_text,'typed while generating');
});
test('script skeleton is retained before a draft but is never prepended to an existing pending dialogue',()=>{
 assert.match(fn('renderDraftSkeleton'),/draft-skeleton/);
 assert.doesNotMatch(fn('renderScript'),/skeleton-audio|card.append\(skeleton\)/);
});
