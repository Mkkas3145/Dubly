'use strict';
(()=>{
 const KEY='dubly-shared-projects-v1',validCode=code=>typeof code==='string'&&/^DUBLY-[A-Za-z0-9_-]{32}$/.test(code);
 class SharedProjectsClient{
  constructor({storage,history,request,changed=()=>{}}){Object.assign(this,{storage,history,request,changed});this.pendingSync=null}
  get state(){try{const value=JSON.parse(this.storage.getItem(KEY)||'null');return value&&validCode(value.code)?{code:value.code,pending:Array.isArray(value.pending)?value.pending:[],hidden:Array.isArray(value.hidden)?value.hidden:[]}:null}catch{return null}}
  save(state){try{if(state)this.storage.setItem(KEY,JSON.stringify(state));else this.storage.removeItem(KEY)}catch{throw Error('브라우저 기록을 저장하지 못했습니다.')}this.changed()}
  merge(ids,state){if(!Array.isArray(ids))throw Error('공유 목록을 불러오지 못했습니다.');this.history.merge(ids.filter(id=>!state.hidden.includes(id)))}
  async create(){const existing=this.state;if(existing){await this.sync();return existing.code}const result=await this.request('/api/project-share/create',{ids:this.history.ids()});if(!validCode(result.code))throw Error('공유 코드를 만들지 못했습니다.');const state={code:result.code,pending:[],hidden:[]};this.save(state);this.merge(result.ids,state);return state.code}
  async connect(code){code=code.trim();if(!validCode(code))throw Error('공유 코드를 확인해 주세요.');const result=await this.request('/api/project-share/sync',{code,ids:[]});const previous=this.state,state=previous?.code===code?previous:{code,pending:[],hidden:[]};this.save(state);this.merge(result.ids,state);return result.ids.length}
  remember(id){const state=this.state;if(!state)return;state.pending=[...new Set([...state.pending,id])].slice(-200);state.hidden=state.hidden.filter(value=>value!==id);this.save(state)}
  hide(id){const state=this.state;if(!state)return;state.pending=state.pending.filter(value=>value!==id);state.hidden=[...new Set([...state.hidden,id])].slice(-200);this.save(state)}
  disconnect(){this.save(null)}
  async sync(){
   const initial=this.state;if(!initial)return null;if(this.pendingSync)return this.pendingSync;
   this.pendingSync=(async()=>{const result=await this.request('/api/project-share/sync',{code:initial.code,ids:initial.pending});const current=this.state;if(!current||current.code!==initial.code)return null;current.pending=current.pending.filter(id=>!initial.pending.includes(id));this.save(current);this.merge(result.ids,current);return result.ids.length})().finally(()=>this.pendingSync=null);
   return this.pendingSync;
  }
 }
 window.DublySharedProjectsClient=SharedProjectsClient;
 if(!window.DublyHistory)return;
 const $=id=>document.getElementById(id),T=(text,values)=>window.DublyI18n?.t(text,values)||text;
 let status='',busy=false;
 const request=async(url,body)=>{const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),20000);try{const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:abort.signal}),result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'공유 목록을 불러오지 못했습니다.');return result}catch(error){if(error.name==='AbortError'||error instanceof TypeError)throw Error('연결을 확인한 뒤 다시 동기화해 주세요.');throw error}finally{clearTimeout(timer)}};
 function draw(){const state=client.state;if(!$('project-share-dialog'))return;$('share-code').value=state?.code||'';$('share-copy').disabled=busy||!state;$('share-create').hidden=!!state;$('share-disconnect').hidden=!state;$('share-refresh').hidden=!state;for(const id of ['share-create','share-connect','share-disconnect','share-refresh'])$(id).disabled=busy;$('share-status').textContent=status||T(state?'공유 코드가 연결되어 있습니다.':'공유 코드를 만들어 다른 컴퓨터에서 입력하세요.')}
 const client=new SharedProjectsClient({storage:localStorage,history:window.DublyHistory,request,changed:draw});window.DublyProjectShare=client;
 async function sync(){try{const count=await client.sync();if(count!==null)status=T('동기화됨 · 영상 {count}개',{count});draw()}catch(error){status=error.message;draw()}}
 window.addEventListener('dubly:projectschange',event=>{if(event.detail?.remembered){try{client.remember(event.detail.remembered);sync()}catch(error){status=error.message;draw()}}});
 window.addEventListener('dubly:projecthidden',event=>{try{client.hide(event.detail.id)}catch(error){status=error.message;draw()}});
 window.addEventListener('storage',event=>{if(event.key===KEY){draw();sync()}});window.addEventListener('focus',sync);document.addEventListener('visibilitychange',()=>{if(!document.hidden)sync()});setInterval(()=>{if(!document.hidden)sync()},10000);
 window.addEventListener('dubly:localechange',()=>{status='';draw()});
 async function action(operation){if(busy)return;busy=true;status=T('동기화 중…');draw();try{await operation()}catch(error){status=error.message}finally{busy=false;draw()}}
 if($('project-share-dialog')){
  $('share-open').onclick=()=>{draw();$('project-share-dialog').showModal();sync()};$('share-close').onclick=()=>$('project-share-dialog').close();
  $('share-create').onclick=()=>action(async()=>{await client.create();status=T('공유 코드가 준비됐습니다. 다른 컴퓨터에서 입력하세요.')});
  $('share-import').onsubmit=event=>{event.preventDefault();action(async()=>{const count=await client.connect($('share-input').value);$('share-input').value='';status=T('동기화됨 · 영상 {count}개',{count})})};
  $('share-refresh').onclick=()=>action(async()=>{const count=await client.sync();status=T('동기화됨 · 영상 {count}개',{count})});
  $('share-disconnect').onclick=()=>{client.disconnect();status=T('연결을 해제했습니다. 불러온 영상 기록은 유지됩니다.');draw()};
  $('share-copy').onclick=async()=>{const code=client.state?.code;if(!code)return;try{if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(code);else throw Error();status=T('공유 코드를 복사했습니다.')}catch{$('share-code').focus();$('share-code').select();status=T(document.execCommand('copy')?'공유 코드를 복사했습니다.':'코드를 선택했습니다. 복사해서 사용하세요.')}draw()};
  draw();
 }
 sync();
})();
