'use strict';
(()=>{
const T=(s,v)=>window.DublyI18n?.t(s,v)||s;
const $=id=>document.getElementById(id),videoId=location.pathname.split('/')[2],api=`/api/studio/${videoId}`;
let doc=null,job=null,selected=null,dirty=false,saving=null,saveTimer=null,polling=false,started=false,loading=true,busy=false,filter='',wave=null,selectedPause=null,sourceWave=null,zoom=2,clipDrag=null,timelineDrag=null,raf=0;
const editHistory=new window.DublyEditHistory();const audioInputs=new Map(),speakerTrackOrder=[];let historyApplying=false;
const waveCache=new Map(),wavePending=new Set(),waveFailures=new Set();let waveWorkers=0;
let studioHydrated=false,processing=false,lastRemoteDoc=null,lastGenerationStatus='';const generationSubmissions=new Set();
function rowRegenerating(id){return generationSubmissions.has(id)||(job?.regenerating_indices||[]).includes(id)}
function audioPending(row){return !row.deleted&&!row.keep_original&&!row.audio_error&&(rowRegenerating(row.id)||(processing&&(row.audio_processing||(!job?.editor_index&&job?.editor_index!==0&&!row.audio_ready))))}
function audioState(row){return audioPending(row)?'음성 생성 중…':row.audio_error?'원음 유지':row.audio_stale?'재생성 필요':row.audio_ready?'생성됨':'미생성'}
// Keep shimmer on compositor layers; repaint the waveform only when timeline data changes.
const pendingClipNodes=new Map();let pendingClipLayer=null;
function updateSkeletonAnimation(){drawTimeline()}
function syncPendingClips(clips,viewport,height,offset){
 if(!pendingClipLayer){pendingClipLayer=document.createElement('div');pendingClipLayer.className='timeline-pending-layer';pendingClipLayer.setAttribute('aria-hidden','true');$('timeline-content').append(pendingClipLayer)}
 pendingClipLayer.style.width=viewport+'px';pendingClipLayer.style.height=height+'px';pendingClipLayer.style.transform=`translateX(${offset}px)`;
 const visible=new Set();
 for(const {id,x,y,w,h} of clips){visible.add(id);let node=pendingClipNodes.get(id);if(!node){node=document.createElement('div');node.className='timeline-pending-shimmer';pendingClipLayer.append(node);pendingClipNodes.set(id,node)}node.style.left=(x-offset)+'px';node.style.top=y+'px';node.style.width=w+'px';node.style.height=h+'px'}
 for(const [id,node] of pendingClipNodes)if(!visible.has(id)){node.remove();pendingClipNodes.delete(id)}
}
const video=$('video'),audio=$('clip-audio'),timeline=$('timeline'),clip=$('clip-wave'),previewAudio=$('preview-audio');let previewMode='dubbed',previewVersion=null;
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const stamp=t=>`${String(Math.floor(Math.max(0,t)/60)).padStart(2,'0')}:${Math.max(0,t%60).toFixed(2).padStart(5,'0')}`;
let messageTimer=null,messageCleanup=null;
function say(text){const toast=$('message');clearTimeout(messageTimer);clearTimeout(messageCleanup);if(!text){toast.classList.remove('is-visible');messageCleanup=setTimeout(()=>{toast.textContent=''},220);return}toast.textContent=String(text);toast.classList.add('is-visible');messageTimer=setTimeout(()=>say(''),3500)}
let clipPlaybackEpoch=0;
function reportPlaybackError(error){if(error?.name==='AbortError')return;say(error?.name==='NotAllowedError'?'재생 버튼을 눌러 다시 시도해 주세요.':error?.name==='NotSupportedError'?'오디오 또는 영상을 재생할 수 없습니다.':error?.message||'미디어를 재생하지 못했습니다.')}
function stopClipPlayback(){clipPlaybackEpoch++;audio.pause()}
const videoSource=new window.DublyVideoSource(video,api+'/source',text=>say(T(text)),`/samples/speed/browser-previews/${videoId}/video.mp4`);
function toggleVideoPlayback(){if(video.paused){stopClipPlayback();videoSource.play().catch(reportPlaybackError)}else video.pause()}
async function request(url,options={}){const r=await fetch(url,options),data=await r.json();if(!r.ok){const error=Error(typeof data.detail==='string'?data.detail:'요청을 처리하지 못했습니다.');error.status=r.status;throw error;}return data}
const post=(url,body)=>request(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
$('open-player').onclick=async()=>{try{await saveDraft();location.href=`/videos/${videoId}/watch?t=${video.currentTime||0}`}catch(e){say(e.message)}};
const exportKey='dubly-export-'+videoId;let pendingExport=null,displayedJobError='';try{pendingExport=JSON.parse(sessionStorage.getItem(exportKey)||'null')}catch{}
function rememberExport(value){pendingExport=value;try{if(value)sessionStorage.setItem(exportKey,JSON.stringify(value));else sessionStorage.removeItem(exportKey)}catch{}}
function finishExport(result){if(!pendingExport)return;const revision=result.document?.render_revision;if(result.video.state==='done'&&result.has_output&&revision>=pendingExport.revision){rememberExport(null);const link=document.createElement('a');link.href=`/api/videos/${videoId}/download`;link.download='';link.hidden=true;document.body.append(link);link.click();link.remove();say('MP4가 준비되었습니다. 다운로드가 시작됩니다.')}else if(['error','cancelled'].includes(result.video.state)&&result.video.updated>=pendingExport.requested){rememberExport(null);say(result.video.error||'내보내기에 실패했습니다.')}}

window.DublyHistory?.remember(videoId);
function editableSnapshot(){const p=payload();delete p.revision;p.audio_versions=Object.fromEntries(doc.segments.map(r=>[r.id,r.audio_ready?r.audio_key:null]));return p}
function updateHistoryButtons(){$('undo').disabled=busy||historyApplying||!editHistory.past.length;$('redo').disabled=busy||historyApplying||!editHistory.future.length}
function ttsSignature(row){return JSON.stringify([row.original_text,row.translated_text,row.speaker_id,!!row.keep_original])}
function syncAudioInputs(){if(!doc)return;for(const row of doc.segments){const old=audioInputs.get(row.id);if(!old||old.audioKey!==row.audio_key)audioInputs.set(row.id,{audioKey:row.audio_key,signature:ttsSignature(row.audio_input||row),stale:!row.audio_input&&!!row.audio_stale})}}
let batchSubmitting=false;
function pendingSpeechRows(){return doc?.segments.filter(row=>!row.deleted&&!row.keep_original&&(!row.audio_ready||row.audio_stale||row.audio_error))||[]}
function updateBatchButton(){const count=pendingSpeechRows().length,button=$('generate-stale');button.textContent=T('필요한 음성 모두 생성 ({count})',{count});button.disabled=processing||batchSubmitting||!count}
function updateAudioBadges(local=false){if(!doc)return;for(const row of doc.segments){const input=audioInputs.get(row.id);if(local&&row.audio_ready&&input)row.audio_stale=input.stale||input.signature!==ttsSignature(row);const card=[...$('dialogues').children].find(c=>Number(c.dataset.id)===row.id),badge=card?.querySelector('.audio-state');if(badge){badge.textContent=audioState(row);badge.classList.toggle('stale',!!row.audio_stale)}}updateBatchButton()}
function markDirty(track=true){updateSourceSpeech();updateAudioBadges(true);if(track&&doc){const a=document.activeElement;editHistory.record(editableSnapshot(),a?.tagName==='TEXTAREA'?'text:'+a.getAttribute('aria-label'):null)}updateHistoryButtons();dirty=true;$('save-state').textContent='저장하지 않은 변경';$('save').disabled=false;clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveDraft().catch(e=>say(e.message)),900);drawTimeline();livePreview.refresh()}
function payload(){return {revision:doc.revision,segments:doc.segments.map(r=>({id:r.id,start:r.start,end:r.end,deleted:!!r.deleted,original_text:r.original_text,translated_text:r.translated_text,speaker_id:r.speaker_id,keep_original:!!r.keep_original,audio_edit:r.audio_edit||{}})),speaker_names:doc.speaker_names||{},speaker_ids:[...allSpeakers()]}}
function mergeRemoteDocument(remote){
 if(!doc){lastRemoteDoc=JSON.parse(JSON.stringify(remote));return remote}
 const fields=['start','end','original_text','translated_text','speaker_id','keep_original','audio_edit','deleted'];
 const merged={...remote,segments:remote.segments.map((row,i)=>{const local=doc.segments[i],base=lastRemoteDoc?.segments[i];if(!local)return row;const next={...row};for(const key of fields)if(!base||JSON.stringify(local[key])!==JSON.stringify(base[key]))next[key]=local[key];return next})};
 for(const row of doc.segments.slice(remote.segments.length))merged.segments.push(row);
 for(const key of ['speaker_names','speaker_ids'])if(JSON.stringify(doc[key])!==JSON.stringify(lastRemoteDoc?.[key]))merged[key]=doc[key];
 lastRemoteDoc=JSON.parse(JSON.stringify(remote));return merged;
}
async function saveDraft(){
 if(saving){await saving;if(dirty)return saveDraft();return}
 if(!dirty||!doc)return;
 if(busy)throw Error('처리가 끝난 후 저장해 주세요.');
 clearTimeout(saveTimer);const data=payload();dirty=false;$('save-state').textContent='저장 중…';
 saving=(async()=>{
  let result;
  try{result=await request(api+'/draft',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})}
  catch(error){if(error.status!==409)throw error;const fresh=await request(api);if(!fresh.document)throw error;doc=mergeRemoteDocument(fresh.document);result=await request(api+'/draft',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())})}
  doc.revision=result.revision;doc.updated=result.updated;doc.render_stale=result.render_stale;
  if(!dirty){doc.segments=result.segments.map((row,i)=>Object.assign(doc.segments[i]||{},row));editHistory.adopt(editableSnapshot());syncAudioInputs();updateAudioBadges();$('save-state').textContent='저장됨';$('save').disabled=true}
  lastRemoteDoc=JSON.parse(JSON.stringify(result));drawTimeline();livePreview.refresh();
 })().catch(e=>{dirty=true;$('save-state').textContent='저장 실패';throw e}).finally(()=>saving=null);
 await saving;
}
$('save').onclick=()=>saveDraft().catch(e=>say(e.message));
window.addEventListener('beforeunload',e=>{if(dirty||saving){e.preventDefault();e.returnValue=''}});
const palette=['126,169,244','232,155,190','144,198,159','223,181,116','176,158,231','104,195,201','226,144,132','182,199,126','141,176,201','202,163,215','208,190,163','146,195,182'];
function voiceColor(id){let hash=0;for(const c of String(id))hash=(Math.imul(hash,31)+c.charCodeAt(0))>>>0;const number=/^\d+$/.test(String(id))?Number(id):hash;if(number<palette.length)return palette[number];const h=(number*137.508)%360,s=.52,l=.68,a=s*Math.min(l,1-l),f=n=>{const k=(n+h/30)%12;return Math.round(255*(l-a*Math.max(-1,Math.min(k-3,9-k,1))))};return [f(0),f(8),f(4)].join(',')}
function speakerName(id){const alias=doc.speaker_names?.[id];return alias?`${alias} · ID ${id}`:T('화자 {id}',{id})}
function allSpeakers(){const ids=[...new Set([...(doc.speaker_ids||[]),...doc.segments.filter(r=>!r.deleted).map(r=>r.speaker_id),...Object.keys(doc.speaker_names||{})])];for(const id of ids)if(!speakerTrackOrder.includes(id))speakerTrackOrder.push(id);for(let i=speakerTrackOrder.length-1;i>=0;i--)if(!ids.includes(speakerTrackOrder[i]))speakerTrackOrder.splice(i,1);doc.speaker_ids=[...speakerTrackOrder];return speakerTrackOrder}
function selectRow(id,seek=false){if(selected!==id)stopClipPlayback();selected=id;for(const row of $('dialogues').children)row.classList.toggle('active',Number(row.dataset.id)===id);const row=doc.segments[id];$('clip-name').textContent=`${id+1}번 · ${speakerName(row.speaker_id)}`;$('audio-editor').hidden=false;if(seek)video.currentTime=row.start;selectedPause=null;$('pause-editor').hidden=true;loadClip(id);drawTimeline()}
function renderDraftSkeleton(){const list=$('dialogues');if(list.querySelector('.draft-skeleton'))return;list.replaceChildren();$('dialogue-count').textContent=T('대본을 준비하고 있습니다');for(let i=0;i<4;i++){const card=el('article',undefined,'dialogue draft-skeleton');card.setAttribute('aria-hidden','true');for(const cls of ['skeleton-heading','skeleton-text','skeleton-text short','skeleton-audio'])card.append(el('div',undefined,'skeleton '+cls));list.append(card)}list.setAttribute('aria-busy','true')}
function renderScript(){if(!doc){renderDraftSkeleton();return}updateSourceSpeech();
 $('dialogues').setAttribute('aria-busy',String(busy));
 updateBatchButton();const list=$('dialogues');list.replaceChildren();$('dialogue-count').textContent=`${doc.segments.filter(r=>!r.deleted).length}개 대사`;
 for(const row of [...doc.segments].sort((a,b)=>a.start-b.start||a.id-b.id)){
  if(row.deleted)continue;
  if(filter&&!`${row.original_text} ${row.translated_text||''}`.toLowerCase().includes(filter))continue;
  const card=el('article',undefined,'dialogue'+(row.id===selected?' active':'')+(audioPending(row)?' audio-pending':''));card.dataset.id=row.id;card.style.setProperty('--voice',voiceColor(row.speaker_id));card.setAttribute('aria-busy',String(audioPending(row)));
  const top=el('div',undefined,'dialogue-top'),number=el('span',String(row.id+1).padStart(3,'0'),'dialogue-number'),speaker=el('button',undefined,'speaker-button');const aliasName=doc.speaker_names?.[row.speaker_id];speaker.append(el('span',aliasName||speakerName(row.speaker_id),'speaker-alias'));if(aliasName)speaker.append(el('span','ID '+row.speaker_id,'speaker-id'));speaker.title=speakerName(row.speaker_id);speaker.style.borderColor=`rgba(${voiceColor(row.speaker_id)},.5)`;speaker.type='button';speaker.disabled=busy;speaker.setAttribute('aria-haspopup','listbox');
  const time=el('button',`${stamp(row.start)} — ${stamp(row.end)}`,'time');time.style.cssText='border:0;background:none;padding:0';time.onclick=()=>selectRow(row.id,true);
  speaker.onclick=()=>{const old=top.querySelector('.speaker-menu');document.querySelectorAll('.speaker-menu').forEach(n=>n.remove());if(old)return;const menu=el('div',undefined,'speaker-menu');menu.setAttribute('role','listbox');
   function choose(id){row.speaker_id=id;menu.remove();markDirty();renderScript();if(selected===row.id)$('clip-name').textContent=`${row.id+1}번 · ${speakerName(id)}`;drawTimeline()}
   for(const id of allSpeakers()){const b=el('button',doc.speaker_names?.[id]?speakerName(id):speakerName(id)+' · ID '+id);b.dataset.noI18n='';b.type='button';b.setAttribute('role','option');b.setAttribute('aria-selected',String(id===row.speaker_id));b.onclick=e=>{e.stopPropagation();choose(id)};menu.append(b)}
   const aliasLabel=el('div','현재 화자 별명','alias-label'),alias=el('input'),aliasSave=el('button','적용'),aliasBox=el('div',undefined,'add-speaker');alias.value=doc.speaker_names?.[row.speaker_id]||'';alias.placeholder='별명';alias.maxLength=64;alias.setAttribute('aria-label','화자 '+row.speaker_id+' 별명');aliasSave.onclick=()=>{doc.speaker_names=doc.speaker_names||{};if(alias.value.trim())doc.speaker_names[row.speaker_id]=alias.value.trim();else delete doc.speaker_names[row.speaker_id];markDirty();renderScript();if(selected===row.id)$('clip-name').textContent=`${row.id+1}번 · ${speakerName(row.speaker_id)}`;drawTimeline()};aliasBox.append(alias,aliasSave);menu.append(aliasLabel,aliasBox);
   top.append(menu);
  };
  top.append(number,speaker,time);
  const pair=el('div',undefined,'text-pair');for(const [key,title] of [['original_text','원문 전사'],['translated_text','번역 대사']]){const label=el('label',title),text=el('textarea');text.value=row[key]||'';text.disabled=busy;text.maxLength=8000;text.setAttribute('aria-label',`${row.id+1}번 ${title}`);text.onfocus=()=>{if(selected!==row.id)selectRow(row.id)};text.oninput=()=>{row[key]=text.value;markDirty()};label.append(text);pair.append(label)}
  const bottom=el('div',undefined,'dialogue-bottom'),generate=el('button',row.audio_ready?'다시 생성':'음성 생성'),status=el('span',audioState(row),'audio-state'+(row.audio_stale?' stale':''));
  generate.disabled=busy||rowRegenerating(row.id)||!!row.keep_original;generate.dataset.regenerateId=row.id;if(rowRegenerating(row.id))generate.textContent='재생성 중…';generate.onclick=()=>generateRow(row.id);bottom.append(generate,status);
  const keep=el('label'),check=el('input');check.type='checkbox';check.checked=!!row.keep_original;check.disabled=busy;check.onchange=()=>{row.keep_original=check.checked;markDirty();generate.disabled=busy||rowRegenerating(row.id)||check.checked};keep.append(check,document.createTextNode('원음 유지'));bottom.append(keep);card.append(top,pair,bottom);card.oncontextmenu=e=>{e.preventDefault();showContext(e,row)};card.onclick=e=>{if(!e.target.closest('button,input,textarea,.speaker-menu,details,summary'))selectRow(row.id,true)};list.append(card);
 }
}

const livePreview=new window.DublyLivePreview({video,api,getDoc:()=>doc?{...doc,segments:doc.segments.filter(r=>!r.deleted).sort((a,b)=>a.start-b.start||a.id-b.id)}:null,getWave,getWaveSync:id=>waveCache.get(id),save:saveDraft,status:text=>$('preview-status').textContent=text});livePreview.getWaveSync=id=>waveCache.get(id);
function updatePreview(){if(!doc)return;$('mode-dub').disabled=false;video.muted=previewMode==='dubbed';livePreview.enabled=previewMode==='dubbed';$('mode-source').setAttribute('aria-pressed',String(previewMode==='source'));$('mode-dub').setAttribute('aria-pressed',String(previewMode==='dubbed'))}
$('mode-source').onclick=()=>{previewMode='source';livePreview.stop();updatePreview();$('preview-status').textContent='원음'};
$('mode-dub').onclick=()=>{previewMode='dubbed';updatePreview();livePreview.refresh();$('preview-status').textContent='자동 미리듣기'};
video.onplay=()=>{updatePlayerIcon();startPlayhead();if(previewMode==='dubbed')livePreview.start()};video.onseeking=()=>{livePreview.stop()};video.onseeked=()=>livePreview.refresh();video.onpause=()=>{livePreview.stop();updatePlayerIcon();stopPlayhead()};video.onended=()=>{livePreview.stop();updatePlayerIcon();stopPlayhead()};video.onratechange=()=>livePreview.refresh();video.onloadedmetadata=()=>livePreview.warm().catch(()=>{});
video.addEventListener('loadeddata',()=>$('studio').classList.add('has-source-video'));

function openClipEditor(id){selectRow(id);$('clip-dialog').showModal();drawClip();syncAudioControls()}
$('close-clip').onclick=()=>$('clip-dialog').close();$('clip-dialog').onclick=e=>{if(e.target!==$('clip-dialog'))return;const r=$('clip-dialog').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('clip-dialog').close()};$('clip-dialog').onclose=stopClipPlayback;
timeline.onkeydown=e=>{if(e.key==='Enter'&&selected!==null){e.preventDefault();openClipEditor(selected)}else if(e.shiftKey&&['ArrowLeft','ArrowRight'].includes(e.key)&&selected!==null&&!busy){const row=doc.segments[selected],info=waveCache.get(selected);if(!info)return;e.preventDefault();const pos=row.audio_edit?.position??row.start,end=pos+durationOf(row,info);applyClipLength(row,info,row.audio_edit||{},pos,end,'end',Math.min(doc.duration,end+(e.key==='ArrowRight'?.1:-.1)));markDirty();if(wave)syncAudioControls()}};
function updatePlayerIcon(){$('play-symbol').setAttribute('d',video.paused?'m7 4 9 6-9 6Z':'M5 4h4v12H5ZM12 4h4v12h-4Z')}
$('player-play').onclick=toggleVideoPlayback;video.onclick=()=>$('player-play').click();
let previousVolume=1;$('player-volume').oninput=e=>{video.volume=Number(e.target.value);if(video.volume>0)previousVolume=video.volume};$('player-mute').onclick=()=>{video.volume=video.volume>0?0:previousVolume;$('player-volume').value=video.volume};video.onvolumechange=()=>{if(livePreview.output)livePreview.output.gain.value=video.volume;$('player-volume').value=video.volume;$('player-mute').setAttribute('aria-pressed',String(video.volume===0))};
function restoreEdits(snapshot){const focus=document.activeElement,focusLabel=focus?.tagName==='TEXTAREA'?focus.getAttribute('aria-label'):null,caret=focus?.selectionStart;const current=doc.segments;doc.speaker_names=JSON.parse(JSON.stringify(snapshot.speaker_names||{}));doc.speaker_ids=[...(snapshot.speaker_ids||[])];speakerTrackOrder.length=0;doc.segments=current.map((row,i)=>{const saved=snapshot.segments.find(r=>r.id===i);return saved?{...row,...JSON.parse(JSON.stringify(saved))}:{...row,deleted:true}});for(const row of snapshot.segments)if(!doc.segments[row.id])doc.segments[row.id]={...JSON.parse(JSON.stringify(row)),audio_ready:false,audio_stale:false};if(selected!==null&&doc.segments[selected]?.deleted)selected=doc.segments.find(r=>!r.deleted)?.id??null;renderScript();if($('clip-dialog').open){if(selected===null)$('clip-dialog').close();else selectRow(selected)}markDirty(false);drawTimeline();if(focusLabel){const next=[...$('dialogues').querySelectorAll('textarea')].find(t=>t.getAttribute('aria-label')===focusLabel);if(next){next.focus();next.setSelectionRange(Math.min(caret||0,next.value.length),Math.min(caret||0,next.value.length))}}}
async function applyHistory(direction){if(busy||historyApplying)return;historyApplying=true;updateHistoryButtons();let state=null;try{await saveDraft();state=direction==='undo'?editHistory.undo():editHistory.redo();if(!state)return;const audioChanges=Object.entries(state.audio_versions||{}).filter(([id,key])=>{const row=doc.segments[Number(id)];return row&&(row.audio_ready?row.audio_key:null)!==key});if(audioChanges.length){const draft={...state,revision:doc.revision};delete draft.audio_versions;doc=await request(api+'/draft',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(draft)});for(const [id,key] of audioChanges){doc=await post(`${api}/segments/${id}/restore`,{revision:doc.revision,audio_key:key});waveCache.delete(Number(id));waveFailures.delete(Number(id))}}syncAudioInputs();restoreEdits(state);await saveDraft()}catch(e){if(state)(direction==='undo'?editHistory.redo():editHistory.undo());say(e.message)}finally{historyApplying=false;updateHistoryButtons()}}
$('undo').onclick=()=>applyHistory('undo');$('redo').onclick=()=>applyHistory('redo');


document.addEventListener('keydown',e=>{if(!(e.ctrlKey||e.metaKey)||e.altKey)return;if(e.key.toLowerCase()==='s'){e.preventDefault();saveDraft().catch(e=>say(e.message));return}if(e.target.closest('input,[contenteditable]'))return;if(e.key.toLowerCase()==='z'){e.preventDefault();(e.shiftKey?$('redo'):$('undo')).click()}else if(e.key.toLowerCase()==='y'){e.preventDefault();$('redo').click()}});
function deleteDialogue(row){if(busy||!row||row.deleted)return;row.deleted=true;if($('clip-dialog').open&&selected===row.id)$('clip-dialog').close();if(selected===row.id)selected=doc.segments.find(r=>!r.deleted)?.id??null;markDirty();renderScript();drawTimeline()}
function showContext(e,row=null,time=null,speaker=null){if(busy)return;const menu=$('timeline-menu');menu.replaceChildren();const button=el('button',row?'삭제':'대사 추가');button.type='button';button.setAttribute('role','menuitem');button.onclick=()=>{menu.hidden=true;if(row)deleteDialogue(row);else openAdd(time??video.currentTime,speaker)};menu.append(button);menu.style.left=Math.max(8,Math.min(window.innerWidth-175,e.clientX))+'px';menu.style.top=Math.max(8,Math.min(window.innerHeight-65,e.clientY))+'px';menu.hidden=false;button.focus()}
document.addEventListener('pointerdown',e=>{if(!e.target.closest('#timeline-menu'))$('timeline-menu').hidden=true});document.addEventListener('keydown',e=>{if(e.key==='Escape')$('timeline-menu').hidden=true});
function openAdd(time,speaker){if(busy)return;const start=Math.max(0,Math.min(doc.duration-.05,time));$('add-start').value=start.toFixed(2);$('add-end').value=Math.min(doc.duration,start+2).toFixed(2);$('add-speaker').value=speaker??allSpeakers()[0]??'0';$('add-source').value='';$('add-target').value='';$('add-error').textContent='';$('add-dialog').showModal();$('add-source').focus()}
function showSpeakerContext(e){if(busy)return;const menu=$('timeline-menu');menu.replaceChildren();const button=el('button','화자 ID 추가');button.type='button';button.setAttribute('role','menuitem');button.onclick=()=>{menu.hidden=true;const used=allSpeakers();let next=0;while(used.includes(String(next)))next++;$('new-speaker-id').value=String(next);$('speaker-error').textContent='';$('speaker-dialog').showModal();$('new-speaker-id').focus()};menu.append(button);menu.style.left=Math.max(8,Math.min(window.innerWidth-175,e.clientX))+'px';menu.style.top=Math.max(8,Math.min(window.innerHeight-65,e.clientY))+'px';menu.hidden=false;button.focus()}
$('speaker-tracks').oncontextmenu=e=>{e.preventDefault();showSpeakerContext(e)};
$('speaker-tracks').onkeydown=e=>{if(e.key==='ContextMenu'||(e.shiftKey&&e.key==='F10')){e.preventDefault();const r=$('speaker-tracks').getBoundingClientRect();showSpeakerContext({clientX:r.left+20,clientY:Math.max(110,r.top+15)})}};
$('cancel-speaker').onclick=()=>$('speaker-dialog').close();
$('speaker-form').onsubmit=e=>{e.preventDefault();if(busy)return;const id=$('new-speaker-id').value.trim();if(!id)return;if(allSpeakers().includes(id)){$('speaker-error').textContent='이미 존재하는 화자 ID입니다.';return}doc.speaker_ids.push(id);$('speaker-dialog').close();markDirty();renderScript();drawTimeline()};
$('cancel-add').onclick=()=>$('add-dialog').close();$('add-form').onsubmit=async e=>{e.preventDefault();const start=Number($('add-start').value),end=Number($('add-end').value),speaker=$('add-speaker').value.trim(),original=$('add-source').value.trim(),translation=$('add-target').value.trim();if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>doc.duration){$('add-error').textContent='대사 시간이 영상 범위를 벗어납니다.';return}if(!speaker||!original||!translation)return;const id=doc.segments.length;doc.segments.push({id,start,end,speaker_id:speaker,original_text:original,translated_text:translation,deleted:false,keep_original:false,audio_edit:{},audio_ready:false,audio_stale:false});$('add-dialog').close();markDirty();renderScript();selectRow(id,true);try{await saveDraft()}catch(error){say(error.message)}};
$('clip-play').onclick=()=>{if(!audio.src||audio.dataset.row!==String(selected)||audio.dataset.revision!==String(doc.revision)||dirty){if(selected!==null)playClip(selected,audio.dataset.row===String(selected)?audio.dataset.kind||'edited':'edited');return}if(audio.paused){video.pause();audio.play().catch(reportPlaybackError)}else stopClipPlayback()};
function syncClipTransport(){$('clip-play-symbol').setAttribute('d',audio.paused?'m7 4 9 6-9 6Z':'M5 4h4v12H5ZM12 4h4v12h-4Z');const duration=Number.isFinite(audio.duration)?audio.duration:0;$('clip-play-time').textContent=stamp(audio.currentTime||0)+' / '+stamp(duration);$('clip-seek').value=duration?audio.currentTime/duration:0;paintRange($('clip-seek'))}
audio.onplay=syncClipTransport;audio.onpause=syncClipTransport;audio.ontimeupdate=syncClipTransport;audio.onloadedmetadata=syncClipTransport;$('clip-seek').oninput=e=>{if(Number.isFinite(audio.duration))audio.currentTime=Number(e.target.value)*audio.duration};$('clip-volume').oninput=e=>audio.volume=Number(e.target.value);
function paintRange(input){const min=Number(input.min)||0,max=Number(input.max)||1;input.style.setProperty('--range-progress',Math.max(0,Math.min(100,(Number(input.value)-min)/(max-min)*100))+'%')}
document.addEventListener('input',e=>{if(e.target.matches('input[type=range]'))paintRange(e.target)});for(const input of document.querySelectorAll('input[type=range]'))paintRange(input);
$('search').oninput=e=>{filter=e.target.value.toLowerCase();renderScript()};
document.addEventListener('pointerdown',e=>{if(!e.target.closest('.speaker-button,.speaker-menu'))document.querySelectorAll('.speaker-menu').forEach(n=>n.remove())});
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.querySelectorAll('.speaker-menu').forEach(n=>n.remove())});
async function generateRow(id){if(busy||rowRegenerating(id))return;generationSubmissions.add(id);renderScript();try{await saveDraft();const result=await post(`${api}/segments/${id}/generate`,{confirmed:true,revision:doc.revision});say(result.queued?'대사 재생성을 대기열에 추가했습니다.':result.duplicate?'이미 재생성 중인 대사입니다.':'선택한 대사의 음성을 생성합니다.');await poll()}catch(e){say(e.message)}finally{generationSubmissions.delete(id);renderScript()}}
async function generatePending(){
 if(processing||batchSubmitting||!doc)return;
 batchSubmitting=true;updateBatchButton();
 try{
  do{await saveDraft()}while(dirty);
  const count=pendingSpeechRows().length;
  if(!count){say('생성이 필요한 음성이 없습니다.');return}
  setBusy(true);
  await post(api+'/generate-stale',{confirmed:true,revision:doc.revision});
  say(T('{count}개 대사의 음성을 생성합니다.',{count}));
  await poll();
 }catch(error){setBusy(false);say(error.message)}finally{batchSubmitting=false;updateBatchButton()}
}
$('generate-stale').onclick=generatePending;
$('export').onclick=async()=>{try{await saveDraft();await post(api+'/generate',{confirmed:true,revision:doc.revision});rememberExport({revision:doc.revision,requested:Date.now()/1000-2});say('변경된 음성을 반영하여 MP4를 생성합니다.');await poll()}catch(e){say(e.message)}};
async function playClip(id,kind){const epoch=++clipPlaybackEpoch;try{await saveDraft();if(epoch!==clipPlaybackEpoch||selected!==id||!$('clip-dialog').open)return;audio.pause();video.pause();video.currentTime=kind==='reference'?doc.segments[id].start:(doc.segments[id].audio_edit?.position??doc.segments[id].start);audio.dataset.kind=kind;audio.dataset.row=String(id);audio.dataset.revision=String(doc.revision);audio.src=`${api}/segments/${id}/audio?kind=${kind==='edited'?'playback':kind}&v=${doc.revision}`;await audio.play();if(epoch!==clipPlaybackEpoch)return;$('clip-status').textContent=kind==='reference'?'원본 참조 재생 중':'편집 음성 재생 중'}catch(e){if(epoch===clipPlaybackEpoch)reportPlaybackError(e)}}
$('play-reference').onclick=()=>selected!==null&&playClip(selected,'reference');$('play-edited').onclick=()=>selected!==null&&playClip(selected,'edited');
function settings(){const row=doc.segments[selected];if(!row.audio_edit||!Object.keys(row.audio_edit).length)row.audio_edit={trim_start:0,trim_end:wave.duration,speed:1,position:row.start,pauses:[],audio_sha256:wave.sha256};return row.audio_edit}
function durationOf(row,info){if(row.audio_edit?.target_duration)return row.audio_edit.target_duration;if(row.audio_error)return row.end-row.start;if(!info)return row.audio_seconds||row.end-row.start;const edit=row.audio_edit||{},lo=edit.trim_start||0,hi=edit.trim_end??info.duration;let length=hi-lo+(edit.pad_start||0)+(edit.pad_end||0);for(const change of edit.pauses||[]){const p=info.pauses.find(p=>p.id===change.id);if(p&&p.start>=lo&&p.end<=hi)length+=change.seconds-(p.end-p.start)}return Math.max(.001,length/(edit.speed||1))}
function syncAudioControls(){
 for(const id of ['trim-start','trim-end','clip-position','clip-speed','speed-slider','pause-seconds','pause-reset','reset-audio'])$(id).disabled=busy||!wave;
 if(!wave||selected===null)return;const row=doc.segments[selected],edit=row.audio_edit||{};
 $('trim-start').value=(edit.trim_start||0).toFixed(2);$('trim-end').value=(edit.trim_end??wave.duration).toFixed(2);$('clip-position').value=(edit.position??row.start).toFixed(2);$('clip-speed').value=edit.speed||1;$('speed-slider').value=Math.min(4,edit.speed||1);$('speed-value').textContent=(edit.speed||1).toFixed(2)+'×';
 for(const id of ['trim-start','trim-end','clip-position','clip-speed','speed-slider','pause-seconds','pause-reset','reset-audio'])$(id).disabled=busy;paintRange($('speed-slider'));
 $('clip-length').textContent=`기본 ${wave.duration.toFixed(2)}초 → 편집 ${durationOf(row,wave).toFixed(2)}초`;
 if(selectedPause!==null){const p=wave.pauses.find(x=>x.id===selectedPause),saved=edit.pauses?.find(x=>x.id===selectedPause);$('pause-seconds').value=(saved?.seconds??(p.end-p.start)).toFixed(2)}
 drawClip();drawTimeline();
}
async function loadClip(id){
 wave=null;$('clip-status').textContent='파형을 불러오는 중…';
 const row=doc.segments[id];if(!row.audio_ready||row.audio_error){syncAudioControls();$('clip-status').textContent=row.audio_error?'생성이 실패해 원음을 유지합니다. 해당 대사를 다시 생성할 수 있습니다.':'음성이 아직 생성되지 않았습니다.';drawClip();return}
 try{const info=await getWave(id);if(selected!==id)return;wave=info;$('clip-status').textContent=row.audio_stale?'대본이 바뀌었습니다. 기존 생성 음성의 파형입니다.':'쉼 영역에 마우스를 올리면 길이 조절 위치가 표시됩니다.';syncAudioControls()}catch(e){$('clip-status').textContent=e.message}
}
async function getWave(id){const row=doc.segments[id],key=row.audio_key,cached=waveCache.get(id);if(cached&&(!key||cached.sha256===key))return cached;const data=await request(`${api}/segments/${id}/waveform`);if(doc.segments[id]?.audio_key!==key)return getWave(id);waveCache.set(id,data);return data}
for(const [id,key] of [['trim-start','trim_start'],['trim-end','trim_end'],['clip-position','position'],['clip-speed','speed'],['speed-slider','speed']])$(id).onchange=()=>{
 if(!wave||selected===null||busy)return;let value=Number($(id).value);if(!Number.isFinite(value))return;const edit=settings();
 if(key==='trim_start')value=Math.max(0,Math.min(value,(edit.trim_end??wave.duration)-.01));if(key==='trim_end')value=Math.min(wave.duration,Math.max(value,(edit.trim_start||0)+.01));if(key==='position')value=Math.max(0,Math.min(value,doc.duration-.01));if(key==='speed')value=Math.max(1,Math.min(8,value));edit[key]=value;markDirty();syncAudioControls();
};
function setPause(seconds){const edit=settings(),p=wave.pauses.find(p=>p.id===selectedPause);if(!p)return;seconds=Math.max(.01,Math.min(doc.duration,seconds));edit.pauses=edit.pauses.filter(x=>x.id!==p.id);edit.pauses.push({id:p.id,seconds});markDirty();syncAudioControls()}
$('pause-seconds').onchange=()=>{const n=Number($('pause-seconds').value);if(Number.isFinite(n))setPause(n)};
$('pause-reset').onclick=()=>{if(!wave||selectedPause===null||busy)return;settings().pauses=settings().pauses.filter(x=>x.id!==selectedPause);markDirty();syncAudioControls()};
$('reset-audio').onclick=()=>{if(selected===null||busy)return;doc.segments[selected].audio_edit={};selectedPause=null;$('pause-editor').hidden=true;markDirty();syncAudioControls()};
function canvasSize(canvas,height){const dpr=Math.min(devicePixelRatio||1,3),width=canvas.parentElement.clientWidth;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.getContext('2d').setTransform(dpr,0,0,dpr,0,0);return width}
function paintWave(ctx,peaks,x,y,width,height,alpha=1,color='220,220,220'){
 if(!peaks?.length||width<=0)return;let max=.001;for(const p of peaks)max=Math.max(max,p);
 const transform=ctx.getTransform(),scale=transform.a||1,visibleLo=-transform.e/scale,visibleHi=visibleLo+ctx.canvas.width/scale;
 const columns=Math.max(1,Math.ceil(width*scale)),first=Math.max(0,Math.floor((visibleLo-x)/width*columns)),last=Math.min(columns,Math.ceil((visibleHi-x)/width*columns));ctx.fillStyle=`rgba(${color},${alpha})`;
 for(let j=first;j<last;j++){const lo=Math.floor(j/columns*peaks.length),hi=Math.min(peaks.length,Math.max(lo+1,Math.ceil((j+1)/columns*peaks.length)));let value=0;for(let k=lo;k<hi;k++)value=Math.max(value,peaks[k]);if(value<=.00001)continue;const h=Math.max(1/scale,value/max*height);ctx.fillRect(x+j/columns*width,y-h/2,width/columns+.15/scale,h)}
}
function clipView(){return window.DublyPreviewMath.waveform(wave,doc.segments[selected].audio_edit||{},false)}
function drawClip(){const width=canvasSize(clip,150),ctx=clip.getContext('2d');ctx.clearRect(0,0,width,150);if(!wave)return;const view=clipView();paintWave(ctx,view.peaks,12,76,width-24,88,1,voiceColor(doc.segments[selected].speaker_id));const edit=doc.segments[selected].audio_edit||{},x=t=>12+view.map(t)/view.duration*(width-24),lo=edit.trim_start||0,hi=edit.trim_end??wave.duration;ctx.fillStyle='#000b';ctx.fillRect(0,0,x(lo),150);ctx.fillRect(x(hi),0,width-x(hi),150);for(const p of wave.pauses){ctx.fillStyle=p.id===selectedPause?'#ffffff30':'#ffffff0b';ctx.fillRect(x(p.start),10,Math.max(6,x(p.end)-x(p.start)),130);const change=edit.pauses?.find(c=>c.id===p.id);if(change){ctx.fillStyle='#ddd';ctx.font='10px Pretendard';ctx.fillText(change.seconds.toFixed(2)+'s',x(p.start),20)}}for(const t of [lo,hi]){ctx.fillStyle='#eee';ctx.fillRect(x(t)-1,8,2,134);ctx.fillRect(x(t)-4,8,8,17);ctx.fillRect(x(t)-4,125,8,17)}ctx.fillStyle='#999';ctx.font='10px Pretendard';ctx.fillText('0.00',12,146);ctx.fillText(view.duration.toFixed(2)+'s',width-58,146)}
function clipTime(e){const r=clip.getBoundingClientRect(),view=clipView();return Math.max(0,Math.min(wave.duration,view.unmap((e.clientX-r.left-12)/(r.width-24)*view.duration)))}

function pauseAt(t){return wave.pauses.find(p=>p.start-.015<=t&&t<=p.end+.015)}
clip.onpointerdown=e=>{if(!wave||busy)return;const t=clipTime(e),edit=settings(),r=clip.getBoundingClientRect(),slack=9/(r.width-24)*wave.duration;let type=null;if(Math.abs(t-edit.trim_start)<slack)type='start';else if(Math.abs(t-edit.trim_end)<slack)type='end';else{const p=pauseAt(t);if(p){selectedPause=p.id;$('pause-editor').hidden=false;type='pause';syncAudioControls()}else return}clipDrag={type,x:e.clientX,viewDuration:clipView().duration,edit:JSON.parse(JSON.stringify(edit)),pause:selectedPause};clip.setPointerCapture(e.pointerId)};
clip.onpointermove=e=>{if(!wave)return;const t=clipTime(e),p=pauseAt(t),tip=$('wave-tip');if(clipDrag){const edit=settings();if(clipDrag.type==='start')edit.trim_start=Math.min(t,edit.trim_end-.01);else if(clipDrag.type==='end')edit.trim_end=Math.max(t,edit.trim_start+.01);else{const slot=wave.pauses.find(x=>x.id===clipDrag.pause),base=clipDrag.edit.pauses.find(x=>x.id===slot.id)?.seconds??(slot.end-slot.start),delta=(e.clientX-clipDrag.x)/Math.max(100,clip.clientWidth-24)*(clipDrag.viewDuration||wave.duration);const seconds=Math.max(.01,base+delta);edit.pauses=edit.pauses.filter(x=>x.id!==slot.id);edit.pauses.push({id:slot.id,seconds})}syncAudioControls();livePreview.refresh();return}tip.hidden=!p;clip.style.cursor=p?'ew-resize':'crosshair';if(p){tip.textContent=`쉼 ${(p.end-p.start).toFixed(2)}초 · 드래그로 조절`;tip.style.left=Math.min(clip.clientWidth-160,Math.max(5,e.clientX-clip.getBoundingClientRect().left))+'px'}};
function finishClip(){if(clipDrag){clipDrag=null;markDirty()}}clip.onpointerup=finishClip;clip.onpointercancel=finishClip;clip.onpointerleave=()=>{$('wave-tip').hidden=true};
const TRACK_HEIGHT=64;
let sourceDetail=null,detailRequest=0,detailTimer,timelineHover=null;
let sourceRegions=[],sourceDetailRegions=[],sourceSpeech=[],sourceFine=[],sourceEdges=[],timelineSnap=null;
function updateSourceSpeech(){sourceFine=(sourceDetail?sourceRegions.filter(r=>r.end<sourceDetail.start||r.start>sourceDetail.end).concat(sourceDetailRegions):sourceRegions).sort((a,b)=>a.start-b.start);sourceSpeech=window.DublySourceSnap.groupRegions(sourceFine,doc?.segments||[]);sourceEdges=window.DublySourceSnap.boundaries(sourceSpeech)}
function drawSourceSpeech(ctx,px,lo,hi){for(const r of sourceSpeech){if(r.end<lo||r.start>hi)continue;const x=px(r.start),end=px(r.end);ctx.fillStyle='#ffffff06';ctx.fillRect(x,33,Math.max(1,end-x),64);ctx.fillStyle='#71959a88';ctx.fillRect(x,96,Math.max(1,end-x),3)}}

function snapTimelineTime(time,e,edge){const scale=(layoutTimeline().width-120)/doc.duration,locked=timelineDrag?.snapEdge===edge?timelineDrag?.snapBoundary:null,result=window.DublySourceSnap.nearest(time,sourceEdges,scale,{disabled:e.altKey,locked:locked??null});timelineSnap=result.boundary;if(timelineDrag){timelineDrag.snapBoundary=result.boundary;timelineDrag.snapEdge=edge}return result.time}
let sourceWaveKey=null,sourceWaveLoading=false;
function timelineDocument(){return doc||{duration:sourceWave?.duration||1,segments:[],speaker_ids:[],speaker_names:{}}}
async function refreshSourceWave(info){
 if(!info||sourceWaveLoading||sourceWaveKey===info.key)return;
 sourceWaveLoading=true;
 try{const data=await request(api+'/waveform');sourceWave=data;sourceWaveKey=data.key||info.key;sourceDetail=null;sourceDetailRegions=[];detailRequest++;sourceRegions=data.kind==='original'?[]:window.DublySourceSnap.speechRegions(data);updateSourceSpeech();$('studio').classList.add('has-source-wave');drawTimeline();if(doc)queueSourceDetail()}
 catch{}finally{sourceWaveLoading=false}
}
function layoutTimeline(){const viewport=$('timeline-scroll').clientWidth;return {width:Math.max(viewport,timelineDocument().duration*5*zoom),height:Math.max($('timeline-scroll').clientHeight-10,108+(doc?allSpeakers().length:0)*TRACK_HEIGHT+12)}}
function paintPendingClip(ctx,x,y,w,h,id){ctx.save();ctx.beginPath();ctx.roundRect(x,y,w,h,9);ctx.fillStyle='#2b2c2e';ctx.fill();ctx.strokeStyle='#48494c';ctx.lineWidth=1;ctx.stroke();ctx.clip();ctx.fillStyle='#989a9f';ctx.font='10px Pretendard';if(w>18)ctx.fillText(String(id+1),x+7,y+15);if(w>110){ctx.textAlign='center';ctx.fillText(T('음성 생성 중…'),x+w/2,y+h/2+4)}ctx.restore()}
function drawTimeline(){
 if((!doc&&!sourceWave)||loading)return;const view=timelineDocument();const {width,height}=layoutTimeline(),scroll=$('timeline-scroll'),viewport=scroll.clientWidth,dpr=Math.min(devicePixelRatio||1,3),offset=scroll.scrollLeft;
 $('timeline-content').style.width=width+'px';$('timeline-content').style.height=height+'px';$('speaker-tracks').style.height=(height-108)+'px';$('speaker-tracks').style.transform=`translateX(${offset}px)`;
 timeline.width=Math.ceil(viewport*dpr);timeline.height=Math.ceil(height*dpr);timeline.style.width=viewport+'px';timeline.style.height=height+'px';
 const ctx=timeline.getContext('2d'),px=t=>t/view.duration*(width-120)+110;ctx.setTransform(dpr,0,0,dpr,-offset*dpr,0);ctx.fillStyle='#101010';ctx.fillRect(offset,0,viewport,height);ctx.font='10px Pretendard';
 const step=Math.max(.2,Math.ceil(view.duration/Math.max(1,(width-120)/100)/.2)*.2),lo=Math.max(0,(offset-110)/(width-120)*view.duration),hi=(offset+viewport)/(width-120)*view.duration;
 for(let t=Math.floor(lo/step)*step;t<=hi+step;t+=step){ctx.fillStyle='#999';ctx.fillText(stamp(t),Math.round(px(t))+4,18);ctx.fillStyle='#ffffff0d';ctx.fillRect(Math.round(px(t)),25,1/dpr,height-25)}
 if(sourceWave)paintWave(ctx,sourceWave.peaks,110,65,width-120,48,.8);
 if(sourceDetail){const x=px(sourceDetail.start),w=px(sourceDetail.end)-x;ctx.fillStyle='#101010';ctx.fillRect(x,33,w,63);paintWave(ctx,sourceDetail.peaks,x,65,w,48,.95)}
 drawSourceSpeech(ctx,px,lo,hi);
 const pendingClips=[];const speakers=doc?allSpeakers():[];for(let i=0;i<speakers.length;i++){const y=108+i*TRACK_HEIGHT;ctx.fillStyle=i%2?'#ffffff03':'#ffffff06';ctx.fillRect(110,y,width-120,TRACK_HEIGHT-3)}
 for(const row of view.segments){if(row.deleted)continue;const info=waveCache.get(row.id),pos=row.audio_edit?.position??row.start,length=durationOf(row,info);if(pos+length<lo||pos>hi)continue;const y=108+speakers.indexOf(row.speaker_id)*TRACK_HEIGHT,x=px(pos),w=Math.max(3,px(pos+length)-x);if(audioPending(row)){paintPendingClip(ctx,x,y,w,TRACK_HEIGHT-5,row.id);pendingClips.push({id:row.id,x,y,w,h:TRACK_HEIGHT-5});continue;}const rgb=voiceColor(row.speaker_id).split(',').map(Number),pastel=rgb.map(v=>Math.round(v*.55+255*.45)).join(','),focused=row.id===selected||row.id===timelineHover,ink=rgb.map(v=>Math.round(v*(focused?.23:.46))).join(',');ctx.fillStyle=`rgba(${pastel},.96)`;ctx.beginPath();ctx.roundRect(x,y,w,TRACK_HEIGHT-5,9);ctx.fill();ctx.strokeStyle=`rgba(${voiceColor(row.speaker_id)},${focused?1:.8})`;ctx.lineWidth=1;ctx.stroke();
 const showHandles=row.id===timelineHover,cap=showHandles?Math.min(14,w/3):0;if(showHandles){ctx.save();ctx.beginPath();ctx.roundRect(x,y,w,TRACK_HEIGHT-5,9);ctx.clip();ctx.fillStyle=`rgba(${rgb.map(v=>Math.round(v*.4+255*.6)).join(',')},.98)`;ctx.fillRect(x,y,cap,TRACK_HEIGHT-5);ctx.fillRect(x+w-cap,y,cap,TRACK_HEIGHT-5);ctx.strokeStyle=`rgba(${voiceColor(row.speaker_id)},.7)`;ctx.lineWidth=.75;ctx.beginPath();ctx.moveTo(x+cap,y);ctx.lineTo(x+cap,y+TRACK_HEIGHT-5);ctx.moveTo(x+w-cap,y);ctx.lineTo(x+w-cap,y+TRACK_HEIGHT-5);ctx.stroke();ctx.restore()}if(info){ctx.save();ctx.beginPath();ctx.rect(x+cap+2,y+16,Math.max(0,w-2*cap-4),TRACK_HEIGHT-23);ctx.clip();paintWave(ctx,window.DublyPreviewMath.waveform(info,row.audio_edit).peaks,x,y+TRACK_HEIGHT/2+6,w,TRACK_HEIGHT-25,1,ink);ctx.restore()}if(w>36){ctx.fillStyle=`rgb(${ink})`;ctx.fillText(String(row.id+1),x+cap+5,y+12)}if(showHandles&&!row.audio_error){ctx.fillStyle=`rgb(${voiceColor(row.speaker_id)})`;const grip=Math.min(4,cap/3);for(const gx of [x+cap/2-grip/2,x+w-cap/2-grip/2]){ctx.beginPath();ctx.roundRect(gx,y+(TRACK_HEIGHT-5-26)/2,grip,26,2);ctx.fill()}if((row.id===selected||row.id===timelineHover)&&w>34){const middle=x+w/2;ctx.fillStyle='#d9dfe5';ctx.strokeStyle='#96a3b0';ctx.lineWidth=.8;ctx.beginPath();ctx.roundRect(middle-13,y-6,26,17,7);ctx.fill();ctx.stroke();ctx.strokeStyle='#35404a';ctx.lineWidth=1.3;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(middle-5,y-1);ctx.lineTo(middle-8,y+2);ctx.lineTo(middle-5,y+5);ctx.moveTo(middle+5,y-1);ctx.lineTo(middle+8,y+2);ctx.lineTo(middle+5,y+5);ctx.stroke()}}}
 if(timelineSnap!==null){const sx=px(timelineSnap);ctx.strokeStyle='#b9ece4';ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(sx,30);ctx.lineTo(sx,height);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#b9ece4';ctx.fillRect(sx-3,97,7,5)}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#171717';ctx.fillRect(0,25,106,height-25);ctx.fillStyle='#aaa';ctx.fillText(T(sourceWave?.kind==='original'?'원본 오디오':'원본 대사'),10,60);ctx.fillStyle='#84a9ac';ctx.font='9px Pretendard';ctx.fillText(T('발화 구간'),10,78);ctx.font='10px Pretendard';for(let i=0;i<speakers.length;i++){const y=108+i*TRACK_HEIGHT;ctx.fillStyle=`rgb(${voiceColor(speakers[i])})`;ctx.fillRect(0,y,3,TRACK_HEIGHT-5);ctx.save();ctx.beginPath();ctx.rect(8,y,92,TRACK_HEIGHT);ctx.clip();const alias=view.speaker_names?.[speakers[i]];ctx.fillText(alias||speakerName(speakers[i]),10,y+24);if(alias){ctx.fillStyle='#aaa';ctx.fillText('ID '+speakers[i],10,y+41)}ctx.restore()}
 syncPendingClips(pendingClips,viewport,height,offset);
 updatePlayhead();
}
function fetchSourceDetail(){
 if(!doc||loading)return;const scroll=$('timeline-scroll'),{width}=layoutTimeline(),span=scroll.clientWidth/(width-120)*doc.duration;if(span>180){sourceDetail=null;sourceDetailRegions=[];updateSourceSpeech();drawTimeline();return}
 const start=Math.max(0,(scroll.scrollLeft-110)/(width-120)*doc.duration-span*.15),end=Math.min(doc.duration,start+span*1.3),requestId=++detailRequest;
 request(`${api}/waveform?start=${start.toFixed(3)}&end=${end.toFixed(3)}&points=${Math.min(12000,Math.ceil(scroll.clientWidth*(devicePixelRatio||1)*1.3))}`).then(data=>{if(requestId!==detailRequest)return;sourceDetail=data;sourceDetailRegions=window.DublySourceSnap.speechRegions(data);updateSourceSpeech();drawTimeline()}).catch(()=>{});
}
function queueSourceDetail(){clearTimeout(detailTimer);detailTimer=setTimeout(fetchSourceDetail,160)}

function fetchVisibleWaves(){if(!doc)return;const scroll=$('timeline-scroll'),{width}=layoutTimeline(),lo=Math.max(0,(scroll.scrollLeft-110)/(width-120)*doc.duration),hi=(scroll.scrollLeft+scroll.clientWidth)/(width-120)*doc.duration;const needed=doc.segments.filter(r=>!r.deleted&&r.audio_ready&&!r.audio_error&&r.start<=hi&&r.end>=lo&&!waveCache.has(r.id)&&!wavePending.has(r.id)&&!waveFailures.has(r.id));for(const row of needed){if(waveWorkers>=4)break;waveWorkers++;wavePending.add(row.id);getWave(row.id).catch(()=>waveFailures.add(row.id)).finally(()=>{waveWorkers--;wavePending.delete(row.id);drawTimeline();fetchVisibleWaves()})}}
let scrollTimer; $('timeline-scroll').onscroll=()=>{drawTimeline();clearTimeout(scrollTimer);scrollTimer=setTimeout(fetchVisibleWaves,100);queueSourceDetail()};
function setZoom(value,anchor=.5){if(!doc&&!sourceWave)return;const duration=timelineDocument().duration;const scroll=$('timeline-scroll'),old=layoutTimeline().width,time=(scroll.scrollLeft+scroll.clientWidth*anchor-110)/(old-120)*duration;zoom=Math.max(.2,Math.min(40,value));$('zoom').value=zoom;drawTimeline();scroll.scrollLeft=time/duration*(layoutTimeline().width-120)+110-scroll.clientWidth*anchor;drawTimeline();fetchVisibleWaves();queueSourceDetail()}
$('zoom').oninput=e=>setZoom(Number(e.target.value));
$('timeline-scroll').addEventListener('wheel',e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();const r=timeline.getBoundingClientRect();setZoom(zoom*Math.exp(-e.deltaY*.004),Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)))},{passive:false});
function timeAt(e){const duration=timelineDocument().duration,r=timeline.getBoundingClientRect(),{width}=layoutTimeline();return Math.max(0,Math.min(duration,(e.clientX-r.left+$('timeline-scroll').scrollLeft-110)/(width-120)*duration))}
function timelineHit(e){if(!doc)return null;const t=timeAt(e),rect=timeline.getBoundingClientRect(),screenX=e.clientX-rect.left;if(screenX<106)return null;const track=Math.floor((e.clientY-rect.top-108)/TRACK_HEIGHT),speaker=allSpeakers()[track],scale=(layoutTimeline().width-120)/doc.duration,slack=8/scale;
 const row=doc.segments.find(r=>{const pos=r.audio_edit?.position??r.start;return r.speaker_id===speaker&&t>=pos-slack&&t<=pos+durationOf(r,waveCache.get(r.id))+slack&&!r.deleted});if(!row)return null;
 const pos=row.audio_edit?.position??row.start,end=pos+durationOf(row,waveCache.get(row.id)),left=Math.abs(t-pos)*scale,right=Math.abs(t-end)*scale,edge=!row.audio_error&&Math.min(left,right)<=14?(left<right?'start':'end'):null;return {row,t,pos,end,edge};
}
function applyClipLength(row,info,base,position,end,edge,time){
 if(!info){const start=edge==='start'?Math.max(0,Math.min(end-.01,time)):position,finish=edge==='end'?Math.max(position+.01,Math.min(doc.duration,time)):end;row.audio_edit={position:start,target_duration:Math.max(.01,finish-start)};return row.audio_edit.target_duration;}
 const target=edge==='start'?Math.max(.01,end-time):Math.max(.01,time-position),edit=window.DublyPreviewMath.fitDuration(info,base,target,edge),view=window.DublyPreviewMath.waveform(info,edit);
 edit.position=edge==='start'?Math.max(0,end-view.duration):position;row.audio_edit=edit;return view.duration;
}
let scrubDrag=null;
timeline.oncontextmenu=e=>{e.preventDefault();if(!doc)return;const rect=timeline.getBoundingClientRect();if(e.clientX-rect.left<106&&e.clientY-rect.top>=108){showSpeakerContext(e);return}const hit=timelineHit(e),track=Math.floor((e.clientY-rect.top-108)/TRACK_HEIGHT);showContext(e,hit?.row||null,timeAt(e),allSpeakers()[track])};
timeline.onpointerdown=e=>{if((!doc&&!sourceWave)||e.button!==0)return;const hit=timelineHit(e);if(hit){const {row,t,pos,end,edge}=hit;selectRow(row.id);if(!busy){const info=waveCache.get(row.id);timelineDrag={id:row.id,time:t,x:e.clientX,y:e.clientY,speaker:row.speaker_id,position:pos,end,edge,base:JSON.parse(JSON.stringify(row.audio_edit||{})),info,moved:false};timeline.setPointerCapture(e.pointerId)}}else{video.currentTime=timeAt(e);scrubDrag={x:e.clientX};timeline.setPointerCapture(e.pointerId);drawTimeline()}};
timeline.onpointermove=e=>{
 if(scrubDrag){video.currentTime=timeAt(e);drawTimeline();return}
 if(!timelineDrag||busy){const hit=timelineHit(e),next=hit?.row.id??null;timeline.style.cursor=hit?(hit.edge?'ew-resize':'grab'):'crosshair';if(next!==timelineHover){timelineHover=next;drawTimeline()}return}
 const drag=timelineDrag,row=doc.segments[drag.id],info=drag.info||waveCache.get(row.id);if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<3&&!drag.moved)return;drag.info=info;
 if(drag.edge&&Math.abs(e.clientY-drag.y)>6&&Math.abs(e.clientY-drag.y)>Math.abs(e.clientX-drag.x)*1.2){drag.edge=null;row.audio_edit=JSON.parse(JSON.stringify(drag.base))}
 if(drag.edge){applyClipLength(row,info,drag.base,drag.position,drag.end,drag.edge,snapTimelineTime(timeAt(e),e,drag.edge));timeline.style.cursor='ew-resize';const length=durationOf(row,info),tip=$('timeline-tip');tip.textContent=T('길이 {seconds}초 · {speed}×',{seconds:length.toFixed(2),speed:(row.audio_edit.speed||1).toFixed(2)});tip.style.left=Math.max(10,Math.min(window.innerWidth-170,e.clientX-75))+'px';tip.style.top=Math.max(8,e.clientY-40)+'px';tip.hidden=false}
 else{if(!row.audio_edit)row.audio_edit={};if(info&&!Object.keys(row.audio_edit).length)row.audio_edit={trim_start:0,trim_end:info.duration,speed:1,position:row.start,pauses:[],audio_sha256:info.sha256};const length=durationOf(row,info),rawPosition=Math.max(0,Math.min(doc.duration-length,drag.position+timeAt(e)-drag.time)),scale=(layoutTimeline().width-120)/doc.duration,candidate=window.DublySourceSnap.move(rawPosition,length,sourceEdges,scale,{disabled:e.altKey}),position=Math.max(0,Math.min(doc.duration-length,candidate.position));row.audio_edit.position=position;timelineSnap=Math.abs(position-candidate.position)<1e-7?candidate.boundary:null;const track=Math.floor((e.clientY-timeline.getBoundingClientRect().top-108)/TRACK_HEIGHT),speaker=allSpeakers()[track];if(speaker!==undefined)row.speaker_id=speaker;timeline.style.cursor='grabbing';updateAudioBadges(true)}
 drag.moved=true;drawTimeline();if(selected===row.id&&wave)syncAudioControls();if((row.audio_edit?.speed||1)===1)livePreview.refresh();
};
function finishTimeline(e){scrubDrag=null;const drag=timelineDrag;timelineDrag=null;timelineSnap=null;$('timeline-tip').hidden=true;if(drag?.moved){markDirty();if(doc.segments[drag.id].speaker_id!==drag.speaker)renderScript()}drawTimeline();if(drag&&!drag.moved&&e?.type!=='pointercancel')openClipEditor(drag.id)}timeline.onpointerup=finishTimeline;timeline.onpointercancel=finishTimeline;timeline.onpointerleave=()=>{if(!timelineDrag){timelineHover=null;drawTimeline()}};

let lastClockPaint=0;
function updatePlayhead(){if(!doc&&!sourceWave)return;const duration=timelineDocument().duration;const {width}=layoutTimeline(),position=110+(video.currentTime||0)/duration*(width-120),scroll=$('timeline-scroll'),head=$('timeline-playhead'),screenX=position-scroll.scrollLeft;head.hidden=screenX<106||screenX>scroll.clientWidth;head.style.transform=`translate3d(${position}px,0,0)`;const now=performance.now();if(now-lastClockPaint>80||video.paused){$('time-display').textContent=stamp(video.currentTime)+' / '+stamp(duration);lastClockPaint=now}}
function playheadFrame(){updatePlayhead();raf=video.paused?0:requestAnimationFrame(playheadFrame)}
function startPlayhead(){if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(playheadFrame)}
function stopPlayhead(){if(raf)cancelAnimationFrame(raf);raf=0;updatePlayhead()}
video.ontimeupdate=updatePlayhead;
$('toggle-play').onclick=toggleVideoPlayback;
window.addEventListener('resize',()=>{clampWorkspace();drawClip();drawTimeline();queueSourceDetail()});
function setBusy(value){const previousProcessing=processing,generationStatus=JSON.stringify(job?.regenerating_indices||[]),rowsChanged=lastGenerationStatus!==generationStatus;lastGenerationStatus=generationStatus;processing=value;const blocking=value&&!(doc&&Array.isArray(job?.regenerating_indices)&&job?.workflow_action==='tts'&&Number.isInteger(job?.editor_index)),changed=busy!==blocking||previousProcessing!==processing||rowsChanged;busy=blocking;$('studio').classList.toggle('is-working',processing);updateBatchButton();$('export').disabled=processing||!doc;$('save').disabled=busy||!doc||!dirty;$('open-player').disabled=!doc;$('search').disabled=!doc;$('work-progress').hidden=!processing&&!!doc&&!job?.error;if(changed&&doc)renderScript();updateSkeletonAnimation();if(wave)syncAudioControls()}
function hydrateStudio(){if(!doc||studioHydrated)return;studioHydrated=true;$('studio').classList.remove('draft-loading');$('save-state').textContent=busy?'생성 중…':'저장됨';updatePreview();renderScript();clampWorkspace();drawTimeline();fetchVisibleWaves();queueSourceDetail();if(doc.segments.length)selectRow(0)}
function enterStudio(){if(!loading)return;loading=false;$('loading-screen').hidden=true;$('studio').hidden=false;$('studio').classList.toggle('draft-loading',!doc);history.replaceState(null,'',`/videos/${videoId}/edit`);videoSource.start();renderScript();clampWorkspace();hydrateStudio()}
function timelineBounds(){const h=window.innerHeight;return {min:140,max:Math.max(140,h-230)}}
function clampWorkspace(){const bounds=timelineBounds(),current=parseFloat($('studio').style.getPropertyValue('--timeline-height'))||window.innerHeight*.38;const height=Math.max(bounds.min,Math.min(bounds.max,current));$('studio').style.setProperty('--timeline-height',height+'px');$('timeline-resize').setAttribute('aria-valuenow',Math.round(height));$('timeline-resize').setAttribute('aria-valuemax',Math.round(bounds.max))}
function persistLayout(){try{localStorage.setItem('dubly-studio-layout-v1',JSON.stringify({timeline:parseFloat($('studio').style.getPropertyValue('--timeline-height')),script:$('studio').style.getPropertyValue('--script-width')}))}catch{}}
try{const saved=JSON.parse(localStorage.getItem('dubly-studio-layout-v1')||'null');if(saved){if(Number.isFinite(saved.timeline))$('studio').style.setProperty('--timeline-height',saved.timeline+'px');if(typeof saved.script==='string'&&/^\d+(\.\d+)?%$/.test(saved.script))$('studio').style.setProperty('--script-width',Math.max(24,Math.min(65,parseFloat(saved.script)))+'%')}}catch{}
for(const [id,axis] of [['timeline-resize','y'],['panel-resize','x']]){const handle=$(id);let drag=null;
 handle.onpointerdown=e=>{e.preventDefault();drag={x:e.clientX,y:e.clientY,height:document.querySelector('.timeline-section').getBoundingClientRect().height,width:document.querySelector('.script-pane').getBoundingClientRect().width};handle.setPointerCapture(e.pointerId);handle.classList.add('dragging');document.body.classList.add('workspace-dragging')};
 handle.onpointermove=e=>{if(!drag)return;if(axis==='y'){$('studio').style.setProperty('--timeline-height',drag.height+drag.y-e.clientY+'px');clampWorkspace()}else{const total=document.querySelector('.studio-layout').clientWidth,percent=(drag.width+e.clientX-drag.x)/total*100;$('studio').style.setProperty('--script-width',Math.max(24,Math.min(65,percent))+'%')}drawClip();drawTimeline()};
 const finish=()=>{if(!drag)return;drag=null;handle.classList.remove('dragging');document.body.classList.remove('workspace-dragging');persistLayout();fetchVisibleWaves();queueSourceDetail()};handle.onpointerup=finish;handle.onpointercancel=finish;
 handle.onkeydown=e=>{if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home'].includes(e.key))return;e.preventDefault();if(axis==='y'){const n=parseFloat($('studio').style.getPropertyValue('--timeline-height'))||window.innerHeight*.38;$('studio').style.setProperty('--timeline-height',(e.key==='Home'?window.innerHeight*.38:n+(e.key==='ArrowUp'?30:-30))+'px');clampWorkspace()}else{const n=parseFloat($('studio').style.getPropertyValue('--script-width'))||38;$('studio').style.setProperty('--script-width',Math.max(24,Math.min(65,e.key==='Home'?38:n+(e.key==='ArrowRight'?3:-3)))+'%')}persistLayout();drawClip();drawTimeline();queueSourceDetail()};
 handle.ondblclick=()=>{$('studio').style.setProperty(axis==='y'?'--timeline-height':'--script-width',axis==='y'?window.innerHeight*.38+'px':'38%');clampWorkspace();persistLayout();drawClip();drawTimeline();queueSourceDetail()};
}
document.addEventListener('keydown',e=>{if(e.target.closest('input,textarea,button,[contenteditable]')||loading)return;if(e.code==='Space'){e.preventDefault();$('toggle-play').click()}else if(e.key==='+'||e.key==='='){e.preventDefault();setZoom(zoom*1.3)}else if(e.key==='-'){e.preventDefault();setZoom(zoom/1.3)}});
const bootTime=Date.now();
async function poll(){
 if(polling)return;polling=true;
 try{
  const result=await request(api);job=result.video;$('project-name').textContent=job.filename;
  const working=['queued','running'].includes(job.state);setBusy(working);updateHistoryButtons();
  if(!doc&&result.document){doc=result.document;lastRemoteDoc=JSON.parse(JSON.stringify(doc));syncAudioInputs();editHistory.reset(editableSnapshot())}
  else if(result.document&&!dirty&&!saving&&!historyApplying&&doc.updated!==result.document.updated){const audioChanged=doc.segments.some(r=>r.audio_key!==result.document.segments[r.id]?.audio_key),before=editableSnapshot();if(audioChanged)for(const old of doc.segments){if(old.audio_key===result.document.segments[old.id]?.audio_key)continue;const saved=old.audio_input||(audioInputs.get(old.id)&&JSON.parse(audioInputs.get(old.id).signature));if(saved){const input=Array.isArray(saved)?{original_text:saved[0],translated_text:saved[1],speaker_id:saved[2],keep_original:saved[3]}:saved;Object.assign(before.segments.find(r=>r.id===old.id),input)}}doc=result.document;lastRemoteDoc=JSON.parse(JSON.stringify(doc));syncAudioInputs();if(audioChanged)editHistory.record(editableSnapshot(),null,Date.now(),before);else editHistory.adopt(editableSnapshot());updateHistoryButtons();for(const row of doc.segments){if(waveCache.get(row.id)?.sha256!==row.audio_key){waveCache.delete(row.id);waveFailures.delete(row.id)}}if(!loading){renderScript();if(selected!==null)loadClip(selected);drawTimeline();fetchVisibleWaves();if(audioChanged)livePreview.refresh()}}
  if(result.document&&doc&&!saving&&dirty&&doc.updated!==result.document.updated){doc=mergeRemoteDocument(result.document);syncAudioInputs();updateAudioBadges(true);drawTimeline()}
  enterStudio();hydrateStudio();refreshSourceWave(result.source_waveform);setBusy(working);updateSkeletonAnimation();
  $('save-state').textContent=!doc?'대본 준비 중…':working?'생성 중…':dirty?'저장하지 않은 변경':'저장됨';
  $('work-bar').style.width=Math.min(100,Math.max(0,job.progress||0))+'%';
  const completed=doc?.segments.filter(r=>!r.deleted&&!r.keep_original&&r.audio_ready&&!r.audio_processing&&!r.audio_error).length||0,total=doc?.segments.filter(r=>!r.deleted&&!r.keep_original).length||0;
  $('work-status').replaceChildren(el('span',job.error||window.DublyProgress?.task(job)||'스튜디오 준비 중'),document.createTextNode(' · '+T('전체 {value}%',{value:Math.round(job.progress||0)})+' · '),el('span',window.DublyProgress?.stageText(job)||''),...(working&&doc?[el('span',T('음성 {done}/{total}',{done:completed,total}))]:[]));
  $('loading-retry').hidden=!['error','cancelled'].includes(job.state);
  if(job.error&&(!doc||job.updated>=doc.updated)){if(displayedJobError!==job.error){displayedJobError=job.error;say(job.error)}}else if(displayedJobError&&$('message').textContent===displayedJobError){say('');displayedJobError=''}
  if(job.state==='uploaded'&&!started){started=true;await post(api+'/analyze')}
  if(!working){fetchVisibleWaves();updatePreview()}
  finishExport(result);$('download').hidden=!result.has_output;$('download').href=`/api/videos/${videoId}/download`;$('download').textContent=doc?.render_stale?'이전 MP4 다운로드 ↗':'MP4 다운로드 ↗';
 }catch(e){enterStudio();$('work-progress').hidden=false;$('work-status').textContent=e.message;$('loading-retry').hidden=false;say(e.message)}finally{polling=false}
}
$('loading-retry').onclick=async()=>{try{await post(`/api/videos/${videoId}/retry`);await post('/api/start');await poll()}catch(e){say(e.message)}};
window.addEventListener('dubly:localechange',()=>{if(doc&&!loading){renderScript();if(selected!==null)$('clip-name').textContent=T('{id}번 · {name}',{id:selected+1,name:speakerName(doc.segments[selected].speaker_id)});drawClip();drawTimeline()}});
setBusy(true);enterStudio();poll();setInterval(poll,1500);
})();
