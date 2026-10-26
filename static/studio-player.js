'use strict';
(()=>{
 const $=id=>document.getElementById(id),video=$('player'),parts=window.DublyPlaylists;
 let id=location.pathname.split('/')[2],api='/api/studio/'+id;
 let timer=null,playlistTimer=null,requested=null,ready=false,epoch=0,playlist=null;
 let startTime=Math.max(0,Number(new URLSearchParams(location.search).get('t'))||0),resumeOnReady=false;
 const source=new window.DublyVideoSource(video,api+'/watch-video',text=>fail(text));
 $('back').href='/videos/'+id+'/edit';
 try{$('autoplay-next').checked=localStorage.getItem('dubly-playlist-autoplay')!=='false'}catch{}
 $('autoplay-next').onchange=()=>{try{localStorage.setItem('dubly-playlist-autoplay',String($('autoplay-next').checked))}catch{}};
 async function request(url,options){const r=await fetch(url,options),data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'영상을 준비하지 못했습니다.');return data}
 function fail(message){clearTimeout(timer);$('loading').hidden=false;$('status').textContent=message;$('spinner').hidden=true;$('retry').hidden=false;$('refresh').disabled=false}
 function later(token){if(token!==epoch)return;clearTimeout(timer);timer=setTimeout(check,1500)}
 async function check(){const token=epoch,currentApi=api;try{
  const result=await request(currentApi);if(token!==epoch)return;
  const doc=result.document,job=result.video;
  $('title').textContent=job.filename;document.title=job.filename+' · Dubly';
  if(!doc){fail('아직 편집할 영상이 준비되지 않았습니다.');return}
  if(doc.player_revision===doc.revision&&doc.player_volume_version===doc.volume_policy_version){
   ready=true;$('loading').hidden=true;$('refresh').disabled=false;
   const stale=doc.segments.filter(r=>!r.deleted&&!r.keep_original&&r.audio_stale).length;
   $('note').textContent=stale?`현재 생성된 음성으로 재생 · 재생성 필요한 대사 ${stale}개`:'저장된 편집 내용으로 재생합니다.';
   source.url=currentApi+'/watch-video?v='+doc.revision+'&volume='+doc.volume_policy_version;
   if(!source.started)source.start();else source.load(false);return;
  }
  if(job.state==='queued'){$('status').textContent='재생 준비 대기 중이에요. 앞선 작업이 끝나면 시작합니다.';later(token);return}
  if(job.state==='running'){$('status').textContent=(job.workflow_action==='watch'?'재생용 영상을 준비하고 있어요.':job.phase||'현재 작업을 마치는 중이에요.')+(job.phase_progress?' '+Math.min(99,Math.round(job.phase_progress))+'%':'');later(token);return}
  if(requested===doc.revision){fail(job.error||'영상을 준비하지 못했습니다. 다시 시도해 주세요.');return}
  if(doc.segments.some(r=>!r.deleted&&!r.keep_original&&!r.audio_ready)){fail('생성되지 않은 대사가 있습니다. 편집 화면에서 음성을 생성한 뒤 다시 열어 주세요.');return}
  requested=doc.revision;$('status').textContent='편집한 음성과 영상을 합치고 있어요.';
  await request(currentApi+'/watch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:true,revision:doc.revision})});later(token);
 }catch(e){if(token===epoch)fail(e.message)}}
 function refresh(){epoch++;clearTimeout(timer);video.pause();ready=false;requested=null;$('loading').hidden=false;$('spinner').hidden=false;$('retry').hidden=true;$('refresh').disabled=true;check()}
 function selectPart(item,autoplay=true,push=true){
  if(!item||!item.playable||item.id===id)return;
  clearTimeout(timer);clearTimeout(source.timer);source.timer=null;video.pause();ready=false;
  source.started=false;source.pending=null;source.attempts=0;source.resolving=null;
  video.removeAttribute('src');video.load();id=item.id;api='/api/studio/'+id;source.url=api+'/watch-video';startTime=0;resumeOnReady=autoplay;
  $('back').href=item.edit_url;$('playlist-note').textContent='';$('title').textContent=item.filename;
  if(push)history.pushState(null,'',item.watch_url);drawPlaylist();refresh();loadPlaylist();
 }
 const duration=n=>Number.isFinite(n)?`${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,'0')}`:'';
 function drawPlaylist(){
  $('playlist').hidden=!playlist;if(!playlist)return;
  $('playlist-title').textContent=playlist.title;
  const current=playlist.items.find(item=>item.id===id);
  $('playlist-count').textContent=`${current?.part||1} / ${playlist.parts} · ${playlist.items.length}개 영상`;
  $('playlist-items').replaceChildren();
  for(const item of playlist.items){
   const li=document.createElement('li'),button=document.createElement('button'),meta=document.createElement('span'),edit=document.createElement('a');
   li.setAttribute('aria-current',String(item.id===id));button.type='button';button.className='playlist-part';button.textContent=`${item.part}번 · ${item.filename}`;
   button.disabled=!item.playable||item.id===id;button.onclick=()=>selectPart(item);
   meta.textContent=`${duration(item.duration)} · ${item.id===id?'현재 영상':item.playable?'재생 가능':item.phase||'처리 대기'}`;button.append(meta);
   edit.href=item.edit_url;edit.textContent='편집';edit.setAttribute('aria-label',`${item.part}번 영상 편집`);li.append(button,edit);$('playlist-items').append(li);
  }
  const previous=parts.adjacent(playlist.items,id,-1),next=parts.adjacent(playlist.items,id,1);
  $('previous-part').disabled=!previous?.playable;$('next-part').disabled=!next?.playable;
  $('previous-part').onclick=()=>selectPart(previous);$('next-part').onclick=()=>selectPart(next);
 }
 async function loadPlaylist(){
  clearTimeout(playlistTimer);const current=id;
  try{const result=await request(api+'/playlist');if(current!==id)return;playlist=result.playlist;drawPlaylist()}
  catch{if(current!==id)return}
  if(playlist)playlistTimer=setTimeout(loadPlaylist,10000);
 }
 $('retry').onclick=refresh;$('refresh').onclick=()=>{startTime=video.currentTime||startTime;resumeOnReady=!video.paused;refresh()};
 video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(startTime,Math.max(0,video.duration-.01));startTime=video.currentTime});
 video.addEventListener('canplay',()=>{if(!ready||!resumeOnReady)return;resumeOnReady=false;source.play().catch(error=>{if(error.name!=='AbortError')$('note').textContent='재생 버튼을 눌러 이어 보세요.'})});
 video.addEventListener('error',()=>{if(ready)fail('영상을 불러오지 못했습니다. 다시 시도해 주세요.')});
 video.addEventListener('ended',()=>{
  if(!playlist||!$('autoplay-next').checked)return;
  const next=parts.nextPlayable(playlist.items,id);
  if(next)selectPart(next);
  else $('playlist-note').textContent=parts.adjacent(playlist.items,id,1)?'다음 영상은 아직 처리 중이거나 준비되지 않았습니다.':'재생목록의 마지막 영상입니다.';
 });
 window.addEventListener('popstate',()=>{const next=location.pathname.split('/')[2];if(next===id)return;const item=playlist?.items.find(item=>item.id===next);if(item?.playable)selectPart(item,false,false);else location.reload()});
 document.addEventListener('keydown',e=>{if(!ready||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('button,a,input,textarea'))return;if(e.code==='Space'){e.preventDefault();if(video.paused)source.play().catch(error=>{if(error.name!=='AbortError')fail(error.message)});else video.pause()}else if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();video.currentTime=Math.max(0,Math.min(video.duration,video.currentTime+(e.key==='ArrowRight'?5:-5)))}else if(e.key.toLowerCase()==='f'){if(document.fullscreenElement)document.exitFullscreen();else video.requestFullscreen?.()}});
 refresh();loadPlaylist();
})();
