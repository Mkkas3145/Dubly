'use strict';
const $=id=>document.getElementById(id);
let uploading=false,uploaded=false,dragDepth=0;
const maxFileBytes=10*1024**3;
const catalog=window.DublyLanguages,sourceLanguages=new Set(['auto',...(catalog?.source||['en','ja','ko'])]),targetLanguages=new Set(catalog?.target||['en','ja','ko']);
const languages=new Set([...sourceLanguages,...targetLanguages]),languageNames={en:'영어',ja:'일본어',ko:'한국어'};
const languageCodes=kind=>kind==='source'?sourceLanguages:targetLanguages;
function languageLabel(code){if(code==='auto')return window.DublyI18n?.t('자동 감지')||'자동 감지';return catalog?.languages[code]?.labels[window.DublyI18n?.locale||'ko']||languageNames[code]||catalog?.languages[code]?.name||code}
function filterLanguageMenu(kind){const query=$(kind+'-search')?.value.trim().toLowerCase()||'';for(const code of languageCodes(kind)){const row=catalog?.languages[code],option=$(kind+'-option-'+code);option.hidden=!!query&&![code,languageLabel(code),row?.name,row?.nativeName,...Object.values(row?.labels||{})].some(v=>String(v||'').toLowerCase().includes(query))}}
function buildLanguageMenus(){if(!catalog)return;for(const kind of ['source','target']){const select=$(kind+'-language'),selected=select.value,menu=$(kind+'-menu');select.replaceChildren();menu.replaceChildren();menu.setAttribute('role','group');const search=document.createElement('input');search.id=kind+'-search';search.type='search';search.placeholder='언어 검색…';search.setAttribute('aria-label',kind==='source'?'원본 언어 검색':'출력 언어 검색');search.autocomplete='off';search.oninput=()=>filterLanguageMenu(kind);search.onkeydown=e=>{if(e.key==='Escape'){e.preventDefault();closeLanguageMenus();$(kind+'-trigger').focus()}else if(e.key==='ArrowDown'){e.preventDefault();const first=[...languageCodes(kind)].find(code=>!$(kind+'-option-'+code).hidden&&!$(kind+'-option-'+code).disabled);if(first)$(kind+'-option-'+first).focus()}};menu.append(search);const list=document.createElement('div');list.className='language-options';list.setAttribute('role','listbox');list.setAttribute('aria-label',kind==='source'?'원본 언어':'출력 언어');menu.append(list);
 for(const code of languageCodes(kind)){const native=document.createElement('option');native.value=code;native.textContent=languageLabel(code);select.append(native);const button=document.createElement('button');button.type='button';button.id=kind+'-option-'+code;button.setAttribute('role','option');button.tabIndex=-1;const name=document.createElement('span');name.className='option-name';name.dataset.noI18n='';name.textContent=languageLabel(code);const short=document.createElement('span');short.className='option-code';short.dataset.noI18n='';short.textContent=code.toUpperCase();const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),path=document.createElementNS('http://www.w3.org/2000/svg','path');svg.setAttribute('viewBox','0 0 16 16');svg.setAttribute('aria-hidden','true');path.setAttribute('d','m3 8 3 3 7-7');svg.append(path);button.append(name,short,svg);list.append(button)}select.value=languageCodes(kind).has(selected)?selected:[...languageCodes(kind)][0]}}
function browserTargetLanguage(){const preferences=typeof navigator==='undefined'?[]:[...(navigator.languages||[]),navigator.language],aliases={no:'nb',tl:'fil',jw:'jv'};for(const preference of preferences){if(typeof preference!=='string')continue;const base=preference.toLowerCase().replace('_','-').split('-')[0],code=aliases[base]||base;if(targetLanguages.has(code))return code}return targetLanguages.has('ko')?'ko':[...targetLanguages][0]}
buildLanguageMenus();
$('target-language').value=browserTargetLanguage();
const presetCodes=['creative','balanced','consistent'];let ttsPreset=window.DublyTTSPresets?.default||'creative';
function choosePreset(code){if(uploading||uploaded||!presetCodes.includes(code))return;ttsPreset=code;for(const key of presetCodes){const button=$('preset-'+key);button.setAttribute('aria-checked',String(key===code));button.tabIndex=key===code?0:-1}const preset=window.DublyTTSPresets?.presets[code];if(preset)$('preset-description').textContent=preset.description}
for(const code of presetCodes){const button=$('preset-'+code);button.onclick=()=>choosePreset(code);button.onkeydown=e=>{let next;if(['ArrowRight','ArrowDown'].includes(e.key))next=(presetCodes.indexOf(code)+1)%3;else if(['ArrowLeft','ArrowUp'].includes(e.key))next=(presetCodes.indexOf(code)+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;if(next!==undefined){e.preventDefault();choosePreset(presetCodes[next]);$('preset-'+presetCodes[next]).focus()}}}
choosePreset(ttsPreset);
function closeLanguageMenus(){for(const kind of ['source','target']){$(kind+'-menu').hidden=true;$(kind+'-trigger').setAttribute('aria-expanded','false')}}
function openLanguageMenu(kind,focus=false){
    if(uploading||uploaded)return;
    const wasOpen=!$(kind+'-menu').hidden;closeLanguageMenus();
    if(wasOpen&&!focus)return;
    $(kind+'-menu').hidden=false;$(kind+'-trigger').setAttribute('aria-expanded','true');if(catalog){$(kind+'-search').value='';filterLanguageMenu(kind)}
    if(focus)$(kind+'-option-'+$(kind+'-language').value).focus();
}
function syncLanguages(){
    const source=$('source-language').value,target=$('target-language');
    if(target.value===source)target.value=[...targetLanguages].find(code=>code!==source);
    for(const option of target.options)option.disabled=option.value===source;
    for(const kind of ['source','target']){
        const selected=$(kind+'-language').value;$(kind+'-language-label').textContent=languageLabel(selected);
        for(const code of languageCodes(kind)){const option=$(kind+'-option-'+code);option.setAttribute('aria-selected',String(code===selected));option.disabled=kind==='target'&&code===source;if(catalog){option.querySelector('.option-name').textContent=languageLabel(code);for(const native of $(kind+'-language').options)if(native.value===code)native.textContent=languageLabel(code)}}
    }
}
$('source-language').onchange=syncLanguages;
$('target-language').onchange=syncLanguages;
syncLanguages();
for(const kind of ['source','target']){
    const trigger=$(kind+'-trigger');trigger.onclick=()=>openLanguageMenu(kind);
    trigger.onkeydown=e=>{if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();openLanguageMenu(kind,true)}else if(e.key==='Escape'){closeLanguageMenus()}};
    for(const code of languageCodes(kind)){
        const option=$(kind+'-option-'+code);
        option.onclick=()=>{if(option.disabled||uploading||uploaded)return;$(kind+'-language').value=code;syncLanguages();closeLanguageMenus();trigger.focus()};
        option.onkeydown=e=>{
            if(e.key==='Escape'){e.preventDefault();closeLanguageMenus();trigger.focus();return}
            if(e.key==='Tab'){closeLanguageMenus();return}
            if(!['ArrowDown','ArrowUp','Home','End'].includes(e.key))return;
            e.preventDefault();const choices=Array.from(languageCodes(kind)).filter(c=>!$(kind+'-option-'+c).disabled&&!$(kind+'-option-'+c).hidden);let at=choices.indexOf(code);
            at=e.key==='Home'?0:e.key==='End'?choices.length-1:(at+(e.key==='ArrowUp'?-1:1)+choices.length)%choices.length;
            $(kind+'-option-'+choices[at]).focus();
        };
    }
}
window.addEventListener('pointerdown',e=>{if(!e.target.closest('.language-control'))closeLanguageMenus()});
function lockLanguages(locked){closeLanguageMenus();for(const code of presetCodes)$('preset-'+code).disabled=locked;for(const kind of ['source','target']){$(kind+'-language').disabled=locked;$(kind+'-trigger').disabled=locked}}
function notify(message,error=false){$('notice').textContent=message;$('notice').dataset.error=String(error)}
function showProgress(percent){const value=Math.max(0,Math.min(100,Math.round(percent)));$('upload-line').style.width=value+'%';$('upload-line').setAttribute('aria-valuenow',String(value));$('drop-subtitle').textContent=`업로드 중 · ${value}%`}
const queueItems=[],queueStorageKey='dubly-upload-queue-v1';let queueTimer=null,queuePolling=false;
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
function rememberQueue(){try{localStorage.setItem(queueStorageKey,JSON.stringify(queueItems.filter(r=>r.id).map(r=>({id:r.id,name:r.name,source:r.source,target:r.target,preset:r.preset})).slice(-200)))}catch{}}
function syncQueueVisibility(hiddenId=null){const visible=window.DublyHistory?new Set(window.DublyHistory.ids()):null;for(let i=queueItems.length-1;i>=0;i--){const id=queueItems[i].id;if(id&&(id===hiddenId||(!hiddenId&&visible&&!visible.has(id))))queueItems.splice(i,1)}rememberQueue();drawQueue()}
function drawQueue(){
 let removed=false;for(let i=queueItems.length-1;i>=0;i--){if(['ready','done'].includes(queueItems[i].state)){queueItems.splice(i,1);removed=true}}if(removed)rememberQueue();if(!queueItems.length&&queueTimer!==null){if(typeof clearInterval==='function')clearInterval(queueTimer);queueTimer=null}
 const section=$('queue-section'),box=$('queue-items');section.hidden=!queueItems.length;box.replaceChildren();
 let waiting=0,running=0;
 for(const item of queueItems){
  const state=item.state||'waiting';if(state==='running')running++;else if(!['error','cancelled'].includes(state))waiting++;
  const row=node('article',undefined,'queue-row'),head=node('div',undefined,'queue-row-heading'),name=node(item.id?'a':'span',item.name,'queue-name');name.dataset.noI18n='';name.title=item.name;if(item.id)name.href=`/videos/${item.id}/loading`;
  const status=node('span',state==='waiting'?'업로드 대기':state==='uploading'?`업로드 중 · ${Math.round(item.percent||0)}%`:state==='uploaded'?'처리 대기':state==='queued'?'처리 대기':state==='running'?'처리 중':state==='ready'?'편집 준비 완료':state==='done'?'완료':state==='error'?'실패':'취소됨','queue-status');head.append(name,status);row.append(head);
  const track=node('div',undefined,'queue-track'),bar=node('div');bar.style.width=Math.max(0,Math.min(100,['ready','done'].includes(state)?100:item.percent||0))+'%';track.append(bar);row.append(track);
  if(state==='running'){const detail=node('div',undefined,'queue-detail');detail.append(node('span',window.DublyProgress?.task(item)||'스튜디오 준비 중'),node('span',`전체 ${Math.round(item.percent||0)}%`));row.append(detail);const stage=node('div',window.DublyProgress?.stageText(item)||'현재 단계','queue-stage');row.append(stage)}
  if(['waiting','uploading','uploaded','queued','running'].includes(state)){const stop=node('button',item.cancelling?'중단 중…':state==='running'||state==='uploading'?'중단':'대기 취소','queue-stop');stop.type='button';stop.disabled=!!item.cancelling;stop.onclick=()=>cancelQueueItem(item);row.append(stop)}
  if(item.error){const error=node('p',item.error,'queue-error');row.append(error);if(item.file||item.id){const retry=node('button','다시 시도');retry.type='button';retry.disabled=uploading;retry.onclick=()=>processBatch([item]);row.append(retry)}}

  box.append(row);
 }
 $('queue-summary').textContent=`대기 ${waiting} · 처리 중 ${running}`;
}
async function cancelQueueItem(item){if(item.cancelling)return;item.cancelling=true;drawQueue();try{item.cancelled=true;if(item.xhr)item.xhr.abort();if(item.id){const result=await queueRequest(`/api/videos/${item.id}/cancel`,{confirmed:true});item.state=result.state}else item.state='cancelled';item.error=null;notify('작업을 중단했습니다.');window.dispatchEvent(new CustomEvent('dubly:queuechange'))}catch(error){item.cancelled=false;notify(error.message,true)}finally{item.cancelling=false;drawQueue()}}
async function queueRequest(url,body){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'요청을 처리하지 못했습니다.');return data}
$('queue-reset').onclick=async()=>{const button=$('queue-reset');if(button.disabled)return;button.disabled=true;try{await queueRequest('/api/queue/reset');await pollQueue();notify('대기열을 초기화했습니다. 대기 중인 작업을 다시 시작합니다.');window.dispatchEvent(new CustomEvent('dubly:queuechange'))}catch(error){notify(error.message,true)}finally{button.disabled=false}};
function startQueuePolling(){if(queueTimer===null)queueTimer=setInterval(pollQueue,2000);pollQueue()}
async function pollQueue(){if(queuePolling)return;const ids=queueItems.filter(r=>r.id).map(r=>r.id);if(!ids.length)return;queuePolling=true;try{const result=await queueRequest('/api/studio/history',{ids});for(const job of result.videos){const item=queueItems.find(r=>r.id===job.id);if(item&&!item.cancelling&&!(item.cancelled&&['queued','running','uploaded'].includes(job.state)))Object.assign(item,{state:job.state,phase:job.phase,percent:job.progress,error:job.error,detail:job.detail,phase_progress:job.phase_progress,resource_state:job.resource_state})}drawQueue()}catch{}finally{queuePolling=false}}
function uploadFile(item){return new Promise((resolve,reject)=>{
 const xhr=new XMLHttpRequest();item.xhr=xhr;xhr.open('POST','/api/videos');
 xhr.upload.onprogress=e=>{if(e.lengthComputable){item.percent=e.loaded/e.total*100;showProgress(item.percent);drawQueue()}};
 xhr.upload.onload=()=>{$('drop-subtitle').textContent='영상 확인·필요한 경우 분할 중…'};
 xhr.onload=()=>{let data;try{data=JSON.parse(xhr.responseText)}catch{return reject(Error('서버 응답을 확인하지 못했습니다. 다시 시도해 주세요.'))}if(xhr.status<200||xhr.status>=300)return reject(Error(typeof data.detail==='string'?data.detail:'업로드하지 못했습니다. 다시 시도해 주세요.'));if(!data.id)return reject(Error('업로드 결과를 확인하지 못했습니다.'));resolve(data)};
 xhr.onabort=()=>reject(Error('업로드를 중단했습니다.'));
 xhr.onerror=()=>reject(Error('연결이 끊겼습니다. 네트워크를 확인하고 다시 시도해 주세요.'));
 const form=new FormData();form.append('file',item.file);form.append('source_language',item.source);form.append('target_language',item.target);form.append('tts_preset',item.preset||'creative');xhr.send(form);
})}
async function chooseAudioTrack(item){
 const response=await fetch(`/api/videos/${item.id}/audio-tracks`),data=await response.json();if(!response.ok)throw Error(data.detail||'음성 트랙을 확인하지 못했습니다.');
 if(!data.selection_required)return true;
 const dialog=node('dialog',undefined,'audio-track-dialog'),form=node('form'),title=node('h2','사용할 음성을 선택하세요'),name=node('p',item.name,'audio-track-filename'),hint=node('p','선택한 음성으로 원본 재생과 전사·더빙을 진행합니다.','audio-track-hint'),select=node('select'),label=node('label','음성 트랙'),sample=node('audio'),listen=node('button','선택한 음성 들어보기'),actions=node('div',undefined,'audio-track-actions'),cancel=node('button','취소'),submit=node('button','이 트랙 사용');
 select.setAttribute('aria-label','음성 트랙');sample.controls=true;sample.preload='none';sample.hidden=true;listen.type='button';cancel.type='button';submit.type='submit';
 const names={jpn:'일본어',ja:'일본어',eng:'영어',en:'영어',kor:'한국어',ko:'한국어',zho:'중국어',chi:'중국어',zh:'중국어',fra:'프랑스어',fre:'프랑스어',deu:'독일어',ger:'독일어',spa:'스페인어',und:'언어 정보 없음'};
 for(const [i,t] of data.tracks.entries()){const option=node('option',[`트랙 ${i+1}`,names[t.language]||t.language,t.title,t.codec,t.channels?`${t.channels}채널`:''].filter(Boolean).join(' · '));option.value=t.index;select.append(option);if(t.default)select.value=String(t.index)}
 listen.onclick=()=>{sample.src=`/api/videos/${item.id}/audio-track-preview?index=${select.value}`;sample.hidden=false;sample.play().catch(()=>{})};select.onchange=()=>{sample.pause();sample.hidden=true;sample.removeAttribute('src')};
 label.append(select);actions.append(cancel,submit);form.append(title,name,hint,label,listen,sample,actions);dialog.append(form);document.body.append(dialog);
 return new Promise(resolve=>{let done=false;function finish(value){if(done)return;done=true;sample.pause();dialog.close();dialog.remove();resolve(value)}cancel.onclick=()=>finish(false);dialog.oncancel=e=>{e.preventDefault();finish(false)};form.onsubmit=async e=>{e.preventDefault();submit.disabled=true;try{await queueRequest(`/api/videos/${item.id}/audio-track`,{index:Number(select.value)});finish(true)}catch(error){hint.textContent=error.message;submit.disabled=false}};dialog.showModal()});
}
function adoptUploadedParts(item,result,items,queue){
 const parts=Array.isArray(result.videos)&&result.videos.length?result.videos:[result];
 const first=parts[0];const group=result.split?first.id:undefined;Object.assign(item,{id:first.id,name:first.filename||item.name,state:'uploaded',percent:0,splitGroup:group});
 const children=parts.slice(1).map(part=>({id:part.id,name:part.filename,source:item.source,target:item.target,preset:item.preset,splitGroup:group,state:'uploaded',percent:0,cancelled:!!item.cancelled}));
 if(children.length){items.splice(items.indexOf(item)+1,0,...children);if(queue!==items)queue.splice(queue.indexOf(item)+1,0,...children)}
 return [item,...children];
}
async function processBatch(items,navigate=false){
 if(uploading||uploaded)return;uploading=true;notify('');$('pick').disabled=true;$('files').disabled=true;lockLanguages(true);$('dropzone').classList.add('uploading');$('dropzone').setAttribute('aria-busy','true');$('upload-line').hidden=false;drawQueue();
 try{
  for(const item of items){
   if(item.cancelled)continue;
   try{
    item.error=null;
    if(!item.id){item.state='uploading';item.percent=0;$('drop-title').dataset.noI18n='';$('drop-title').textContent=item.name;showProgress(0);drawQueue();const result=await uploadFile(item);item.xhr=null;const added=adoptUploadedParts(item,result,items,queueItems);if(added.length>1)navigate=false;for(const part of added)window.DublyHistory?.remember(part.id);if(item.cancelled){for(const part of added){await queueRequest(`/api/videos/${part.id}/cancel`,{confirmed:true});part.state='cancelled';part.cancelled=true}rememberQueue();continue}rememberQueue();drawQueue()}
    if(!item.audioTrackChosen){const chosen=await chooseAudioTrack(item);if(!chosen){item.cancelled=true;await queueRequest(`/api/videos/${item.id}/cancel`,{confirmed:true});item.state='cancelled';if(item.splitGroup)for(const sibling of items){if(sibling!==item&&sibling.splitGroup===item.splitGroup&&sibling.state==='uploaded'){await queueRequest(`/api/videos/${sibling.id}/cancel`,{confirmed:true});sibling.cancelled=true;sibling.state='cancelled'}}drawQueue();continue}item.audioTrackChosen=true}
    if(navigate){uploaded=true;showProgress(100);$('dropzone').classList.add('complete');$('dropzone').dataset.videoId=item.id;$('drop-subtitle').textContent='스튜디오를 준비합니다…';notify('업로드 완료');if(window.location)window.location.assign(`/videos/${item.id}/loading`);return}
    await queueRequest(`/api/studio/${item.id}/${['error','cancelled'].includes(item.state)?'resume':'analyze'}`);if(item.cancelled){await queueRequest(`/api/videos/${item.id}/cancel`,{confirmed:true});item.state='cancelled'}else item.state='queued';drawQueue();startQueuePolling();
   }catch(error){item.xhr=null;if(item.cancelled){item.state='cancelled';item.error=null}else{item.state='error';item.error=error.message;notify(error.message,true)}drawQueue()}
  }
  if(items.some(r=>r.state==='queued')&&!items.some(r=>r.error))notify('대기열에 추가했습니다.');
 }finally{
  uploading=false;$('files').value='';lockLanguages(uploaded);$('dropzone').classList.remove('uploading');$('dropzone').setAttribute('aria-busy','false');
  if(!uploaded){$('pick').disabled=false;$('files').disabled=false;$('upload-line').hidden=true;delete $('drop-title').dataset.noI18n;$('drop-title').textContent='영상을 이곳에 놓아주세요';$('drop-subtitle').textContent='여러 MP4 영상을 선택하거나 놓아주세요'}drawQueue();
 }
}
function episodeOrder(name){
 const text=String(name).normalize('NFKC');
 let match=text.match(/\bS(\d+)\s*[._ -]*E(\d+(?:\.\d+)?)(?!\d)/i)||text.match(/\b(\d+)x(\d+(?:\.\d+)?)(?!\d)/i);
 if(match)return [Number(match[1]),Number(match[2])];
 match=text.match(/(?:\bepisode\s*|\bep[._ -]*|第\s*)(\d+(?:\.\d+)?)/i)||text.match(/(\d+(?:\.\d+)?)\s*(?:화|회|話)(?![가-힣])/);
 if(match)return [1,Number(match[1])];
 // Fansub names commonly end in " - 03 [1080p][...].mp4".
 match=text.match(/(?:^|\s-\s|_)(\d+(?:\.\d+)?)(?:v\d+)?\s*(?:\[[^\]]*\]\s*)*\.mp4$/i);
 return match?[1,Number(match[1])]:null;
}
function orderUploadFiles(files){
 const natural=new Intl.Collator(undefined,{numeric:true,sensitivity:'base'});
 return Array.from(files).map((file,index)=>({file,index,episode:episodeOrder(file.name)})).sort((a,b)=>{
  if(a.episode&&b.episode)return a.episode[0]-b.episode[0]||a.episode[1]-b.episode[1]||a.index-b.index;
  if(a.episode||b.episode)return a.episode?-1:1;
  return natural.compare(a.file.name,b.file.name)||a.index-b.index;
 }).map(item=>item.file);
}
async function uploadOne(files){
 if(uploading||uploaded)return;const list=orderUploadFiles(files||[]);if(!list.length)return;
 const source=$('source-language').value,target=$('target-language').value;
 if(!sourceLanguages.has(source)||!targetLanguages.has(target)||source===target){notify('서로 다른 원본 언어와 출력 언어를 선택해 주세요.',true);return}
 for(const file of list){if(!file.name.toLowerCase().endsWith('.mp4')){notify('MP4 영상을 선택해 주세요.',true);return}if(file.size>maxFileBytes){notify('파일 크기는 최대 10GB까지 가능합니다.',true);return}}
 const navigate=list.length===1&&!queueItems.some(r=>!['ready','done','error','cancelled'].includes(r.state));
 const items=list.map(file=>({file,name:file.name,source,target,preset:ttsPreset,state:'waiting',percent:0}));queueItems.push(...items);await processBatch(items,navigate);
}
try{const stored=JSON.parse(localStorage.getItem(queueStorageKey)||'[]');if(Array.isArray(stored))for(const item of stored.slice(-200))if(item&&typeof item.id==='string'&&/^[a-f0-9-]{36}$/.test(item.id))queueItems.push({...item,state:'waiting'})}catch{}
if(queueItems.length){syncQueueVisibility();if(queueItems.length)startQueuePolling()}
window.addEventListener('dubly:projecthidden',e=>syncQueueVisibility(e.detail.id));window.addEventListener('storage',e=>{if(e.key==='dubly-projects-v1')syncQueueVisibility()});
window.addEventListener('dubly:localechange',()=>{syncLanguages();for(const kind of ['source','target'])if(catalog)filterLanguageMenu(kind);if(queueItems.length)drawQueue()});window.addEventListener('dubly:queuechange',pollQueue);
$('pick').onclick=()=>$('files').click();
$('files').onchange=e=>uploadOne(e.target.files);
// Never let a dropped video navigate away from the upload page.
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>e.preventDefault());
$('dropzone').addEventListener('dragenter',e=>{e.preventDefault();if(!uploading&&!uploaded){dragDepth++;$('dropzone').classList.add('dragging')}});
$('dropzone').addEventListener('dragover',e=>{e.preventDefault();e.dataTransfer.dropEffect=uploading||uploaded?'none':'copy'});
$('dropzone').addEventListener('dragleave',e=>{e.preventDefault();dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('dropzone').classList.remove('dragging')});
$('dropzone').addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('dropzone').classList.remove('dragging');uploadOne(e.dataTransfer.files)});

