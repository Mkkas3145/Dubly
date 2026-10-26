(()=>{
 const T=s=>window.DublyI18n?.t(s)||s;
 const KEY='dubly-projects-v1',names={ko:'한국어',en:'영어',ja:'일본어',auto:'자동 감지'};
 const read=()=>{try{const data=JSON.parse(localStorage.getItem(KEY)||'[]');return Array.isArray(data)?data.filter(x=>typeof x==='string'&&/^[a-f0-9-]{36}$/.test(x)).slice(0,200):[]}catch{return []}};
 function remember(id){try{localStorage.setItem(KEY,JSON.stringify([id,...read().filter(x=>x!==id)].slice(0,200)));window.dispatchEvent(new CustomEvent('dubly:projectschange',{detail:{remembered:id}}));return true}catch{return false}}
 function hide(id){try{localStorage.setItem(KEY,JSON.stringify(read().filter(value=>value!==id)));let queue=[];try{queue=JSON.parse(localStorage.getItem('dubly-upload-queue-v1')||'[]')}catch{}if(Array.isArray(queue))localStorage.setItem('dubly-upload-queue-v1',JSON.stringify(queue.filter(item=>item?.id!==id)));window.dispatchEvent(new CustomEvent('dubly:projecthidden',{detail:{id}}));window.dispatchEvent(new CustomEvent('dubly:projectschange'));return true}catch{return false}}
 function merge(ids){const previous=read(),next=[...new Set([...ids,...previous])].filter(id=>typeof id==='string'&&/^[a-f0-9-]{36}$/.test(id)).slice(0,200);if(JSON.stringify(next)===JSON.stringify(previous))return true;try{localStorage.setItem(KEY,JSON.stringify(next));window.dispatchEvent(new CustomEvent('dubly:projectschange'));return true}catch{throw Error('브라우저 기록을 저장하지 못했습니다.')}}
 window.DublyHistory={remember,hide,merge,ids:read};
 const section=document.getElementById('history-section');if(!section)return;
 section.hidden=!read().length;if(read().length)document.body.classList.add('has-history');let videos=[],descending=true,refreshing=false;const stopping=new Set(),expanded=new Set();
 const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
 const duration=seconds=>seconds?`${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`:'분석 중';
 function draw(){
  const box=document.getElementById('history-items');box.replaceChildren();const query=document.getElementById('history-search').value.toLowerCase();
  const visible=new Set(read()),groups=window.DublyPlaylists.groupVideos(videos.filter(v=>visible.has(v.id)),descending);
  let count=0;
  for(const group of groups){
   const filtered=group.items.filter(v=>group.title.toLowerCase().includes(query)||v.filename.toLowerCase().includes(query));if(!filtered.length)continue;
   let container=box;count+=filtered.length;
   if(group.split){
    const details=el('details',undefined,'history-playlist'),summary=el('summary'),image=el('img');image.src=`/api/studio/${group.items[0].id}/thumbnail`;image.alt='';image.loading='lazy';
    const name=el('div'),title=el('strong',group.title),meta=el('small',`${group.items.length}개 영상 · ${group.total}개로 나눈 원본`);title.dataset.noI18n='';name.append(title,meta);
    const play=el('a','재생목록 열기','history-playlist-play');play.href=`/videos/${group.items[0].id}/watch`;play.onclick=e=>e.stopPropagation();
    summary.append(image,name,play);details.append(summary);container=el('div',undefined,'history-playlist-items');details.append(container);details.open=expanded.has(group.id)||!!query;
    details.addEventListener('toggle',()=>{if(!details.isConnected||query)return;if(details.open)expanded.add(group.id);else expanded.delete(group.id)});box.append(details);
   }
   for(const v of filtered){
   const row=el('article',undefined,'history-row'),image=el('img');image.src=`/api/studio/${v.id}/thumbnail`;image.alt='';image.loading='lazy';image.onerror=()=>image.style.opacity='.1';
   const name=el('div'),link=el('a',v.filename);link.dataset.noI18n='';link.href=`/videos/${v.id}/edit`;name.append(link,el('small',new Date(v.created*1000).toLocaleDateString(window.DublyI18n?.dateLocale||'ko-KR')+' · '+T(window.DublyProgress?.task(v)||v.phase||'')));
   if(v.stopError)name.append(el('small',T(v.stopError)));
   const language=el('span',(v.source_language==='auto'&&v.detected_source_language?(window.DublyI18n?.t('자동 감지: {language}',{language:window.DublyLanguages?.languages[v.detected_source_language]?.labels[window.DublyI18n?.locale||'ko']||v.detected_source_language})||v.detected_source_language):(window.DublyLanguages?.languages[v.source_language]?.labels[window.DublyI18n?.locale||'ko']||T(names[v.source_language]||v.source_language||'자동')))+' → '+(window.DublyLanguages?.languages[v.target_language]?.labels[window.DublyI18n?.locale||'ko']||T(names[v.target_language]||v.target_language||'한국어')),'history-meta'),length=el('span',duration(v.duration),'history-meta'),remove=el('button',undefined,'history-hide');remove.type='button';const icon=document.createElementNS('http://www.w3.org/2000/svg','svg');icon.setAttribute('viewBox','0 0 24 24');icon.setAttribute('aria-hidden','true');for(const d of ['M3 3l18 18','M10.6 10.6a2 2 0 0 0 2.8 2.8','M9.9 5.2A10.8 10.8 0 0 1 12 5c5.8 0 9 7 9 7a17.4 17.4 0 0 1-3.3 4.3','M6.2 6.2C3.5 8.1 2 12 2 12s3.4 7 10 7c1.7 0 3.2-.5 4.5-1.2']){const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',d);icon.append(path)}remove.append(icon);const actions=el('div',undefined,'history-actions'),tooltip=el('span','이 브라우저의 기록에서 숨기기','history-tooltip');tooltip.id='history-hide-'+v.id;tooltip.setAttribute('role','tooltip');remove.setAttribute('aria-describedby',tooltip.id);if(['running','queued','uploaded'].includes(v.state)){const stop=el('button',stopping.has(v.id)?'중단 중…':v.state==='running'?'중단':'대기 취소','history-stop');stop.type='button';stop.disabled=stopping.has(v.id);stop.onclick=async()=>{stopping.add(v.id);delete v.stopError;draw();try{const response=await fetch(`/api/videos/${v.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:true})});const data=await response.json();if(!response.ok)throw Error(data.detail||'작업을 중단하지 못했습니다. 다시 시도해 주세요.');v.state=data.state;v.phase=data.state==='cancelled'?'중단됨':v.phase;window.dispatchEvent(new CustomEvent('dubly:queuechange'))}catch(error){v.stopError=error.message}finally{stopping.delete(v.id);draw()}};actions.append(stop)}actions.append(remove,tooltip);remove.setAttribute('aria-label',window.DublyI18n?.t('{name} 기록 숨기기',{name:v.filename})||v.filename+' 기록 숨기기');remove.onclick=()=>{if(hide(v.id)){videos=videos.filter(x=>x.id!==v.id);draw()}else document.getElementById('history-message').textContent='브라우저 기록을 저장하지 못했습니다.'};
   row.append(image,name,language,length,actions);container.append(row);
   }
  }
  document.getElementById('history-message').textContent=count?'다른 컴퓨터에서도 보려면 상단의 영상 공유에서 공유 코드를 사용하세요.':'표시할 작업이 없습니다.';
 }
 window.addEventListener('dubly:localechange',draw);
 document.getElementById('history-search').oninput=draw;document.getElementById('history-sort').onclick=e=>{descending=!descending;e.target.textContent=descending?'최신순 ↓':'오래된순 ↑';draw()};
 async function refresh(){if(refreshing)return;refreshing=true;try{const response=await fetch('/api/studio/history',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:read()})});if(!response.ok)throw Error();const data=await response.json();videos=data.videos.filter(v=>read().includes(v.id));if(videos.length){section.hidden=false;document.body.classList.add('has-history')}draw()}catch{document.getElementById('history-message').textContent='이전 작업을 불러오지 못했습니다. 새로고침해 주세요.'}finally{refreshing=false}}
 window.addEventListener('storage',e=>{if(e.key===KEY)refresh()});window.addEventListener('dubly:queuechange',refresh);window.addEventListener('dubly:projectschange',refresh);setInterval(()=>{if(videos.some(v=>['uploaded','queued','running'].includes(v.state)))refresh()},2000);refresh();
})();
