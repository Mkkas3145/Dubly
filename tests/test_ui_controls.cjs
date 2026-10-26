const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
class Element{
 constructor(){this.children=[];this.dataset={};this.style={};this.disabled=false;this.textContent='';this.listeners={};this.classes=new Set();this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x)}}
 setAttribute(k,v){this[k]=v}
 append(...xs){this.children.push(...xs)}
 replaceChildren(){this.children=[]}
 addEventListener(k,v){this.listeners[k]=v}
 click(){this.clicked=true}
 focus(){this.focused=true}
}
function fixture(){
 const elements=new Map(),requests=[],calls=[],storage=new Map(),events=new Map();
 class XHR{
  constructor(){this.upload={};requests.push(this)}
  open(method,url){this.method=method;this.url=url}
  send(body){this.body=body}
  abort(){if(this.onabort)this.onabort()}
 }
 const context=vm.createContext({
  document:{getElementById(id){if(!elements.has(id)){const el=new Element();if(id==='source-language'||id==='target-language'){el.value=id==='source-language'?'ja':'ko';el.options=['en','ja','ko'].map(value=>({value,disabled:false}))}elements.set(id,el)}return elements.get(id)},createElement(){return new Element()},createElementNS(){return new Element()}},
  window:{addEventListener(type,callback){if(!events.has(type))events.set(type,[]);events.get(type).push(callback)},dispatchEvent(event){for(const callback of events.get(event.type)||[])callback(event)}},localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value)},CustomEvent:class{constructor(type,options={}){this.type=type;this.detail=options.detail}},setInterval(){return 1},fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>url.endsWith('/cancel')?{state:'cancelled'}:{videos:[]}}},XMLHttpRequest:XHR,FormData:class{append(k,v){this[k]=v}}
 });
 for(const kind of ['source','target'])context.document.getElementById(kind+'-menu').hidden=true;
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'),context);
 return {context,elements,requests,calls,storage,run:code=>vm.runInContext(code,context)};
}
const one="[{name:'scene.mp4',size:1234}]";
test('sampling preset is captured for the batch and locked while uploading',async()=>{
 const f=fixture();assert.equal(f.elements.get('preset-creative')['aria-checked'],'true');
 f.elements.get('preset-consistent').onclick();
 const pending=f.run('uploadOne('+one+')');assert.equal(f.requests[0].body.tts_preset,'consistent');
 assert.equal(f.elements.get('preset-creative').disabled,true);
 f.elements.get('preset-creative').onclick();assert.equal(f.run('ttsPreset'),'consistent');
 const req=f.requests[0];req.status=201;req.responseText=JSON.stringify({id:'preset-video'});req.onload();await pending;
});
test('invalid files reject the batch before uploading anything',async()=>{
 const f=fixture();
 for(const input of ["[{name:'a.mp4',size:1},{name:'bad.txt',size:1}]","[{name:'x.txt',size:5}]","[{name:'big.mp4',size:21*1024**3}]"]){await f.run('uploadOne('+input+')');assert.equal(f.requests.length,0);assert.equal(f.elements.get('notice').dataset.error,'true')}
});
test('single upload reports progress, prevents duplicates, and does not start any processing',async()=>{
 const f=fixture();const pending=f.run('uploadOne('+one+')');const req=f.requests[0];
 assert.equal(req.method,'POST');assert.equal(req.url,'/api/videos');assert.equal(req.body.file.name,'scene.mp4');assert.equal(f.elements.get('pick').disabled,true);
 assert.equal(req.body.source_language,'ja');assert.equal(req.body.target_language,'ko');assert.equal(f.elements.get('source-language').disabled,true);
 await f.run('uploadOne('+one+')');assert.equal(f.requests.length,1);
 req.upload.onprogress({lengthComputable:true,loaded:50,total:100});assert.equal(f.elements.get('upload-line')['aria-valuenow'],'50');
 req.status=201;req.responseText=JSON.stringify({id:'video-id'});req.onload();await pending;
 assert.equal(f.elements.get('dropzone').dataset.videoId,'video-id');assert.equal(f.elements.get('dropzone').classes.has('complete'),true);assert.equal(f.elements.get('drop-subtitle').textContent,'스튜디오를 준비합니다…');
 await f.run('uploadOne('+one+')');assert.equal(f.requests.length,1);
});
test('source changes keep a valid pair; identical languages cannot be uploaded',async()=>{
 const f=fixture();f.elements.get('source-language').value='ko';f.run('syncLanguages()');
 assert.equal(f.elements.get('target-language').value,'en');assert.equal(f.elements.get('target-language').options.find(o=>o.value==='ko').disabled,true);
 f.elements.get('target-language').value='ko';await f.run('uploadOne('+one+')');assert.equal(f.requests.length,0);
});
test('custom dropdowns support selection, disabled identical output, keyboard focus and escape',()=>{
 const f=fixture();f.elements.get('source-menu').hidden=true;
 f.elements.get('source-trigger').onclick();assert.equal(f.elements.get('source-menu').hidden,false);
 f.elements.get('source-option-en').onclick();assert.equal(f.elements.get('source-language').value,'en');assert.equal(f.elements.get('source-language-label').textContent,'영어');assert.equal(f.elements.get('source-menu').hidden,true);assert.equal(f.elements.get('target-option-en').disabled,true);
 f.elements.get('target-option-en').onclick();assert.equal(f.elements.get('target-language').value,'ko');
 f.elements.get('target-trigger').onkeydown({key:'ArrowDown',preventDefault(){}});assert.equal(f.elements.get('target-option-ko').focused,true);
 f.elements.get('target-option-ko').onkeydown({key:'Escape',preventDefault(){}});assert.equal(f.elements.get('target-menu').hidden,true);assert.equal(f.elements.get('target-trigger').focused,true);
});
test('network and server errors restore the upload control for retry',async()=>{
 for(const kind of ['network','server','bad-json']){
  const f=fixture();const pending=f.run('uploadOne('+one+')');const req=f.requests[0];
  if(kind==='network')req.onerror();else{req.status=400;req.responseText=kind==='bad-json'?'not json':JSON.stringify({detail:'서버 오류'});req.onload()}
  await pending;assert.equal(f.elements.get('pick').disabled,false);assert.equal(f.elements.get('files').disabled,false);assert.equal(f.elements.get('notice').dataset.error,'true');assert.equal(f.run('uploading'),false);assert.equal(f.run('uploaded'),false);
 }
});

test('multiple videos upload in selection order and each enters the analysis queue',async()=>{
 const f=fixture(),pending=f.run("uploadOne([{name:'a.mp4',size:1},{name:'b.mp4',size:2}])");
 assert.equal(f.requests.length,1);assert.equal(f.requests[0].body.file.name,'a.mp4');
 f.requests[0].status=201;f.requests[0].responseText=JSON.stringify({id:'first'});f.requests[0].onload();
 await new Promise(setImmediate);assert.equal(f.requests.length,2);assert.equal(f.requests[1].body.file.name,'b.mp4');
 f.requests[1].status=201;f.requests[1].responseText=JSON.stringify({id:'second'});f.requests[1].onload();await pending;
 assert.deepEqual(f.calls.filter(c=>c.url.endsWith('/analyze')).map(c=>c.url),['/api/studio/first/analyze','/api/studio/second/analyze']);
 assert.equal(f.run('uploaded'),false);assert.equal(f.elements.get('pick').disabled,false);assert.equal(f.run('queueItems.length'),2);
});
test('a failed upload does not block the next video',async()=>{
 const f=fixture(),pending=f.run("uploadOne([{name:'a.mp4',size:1},{name:'b.mp4',size:2}])");f.requests[0].onerror();await new Promise(setImmediate);
 assert.equal(f.requests.length,2);f.requests[1].status=201;f.requests[1].responseText=JSON.stringify({id:'second'});f.requests[1].onload();await pending;
 assert.equal(f.run('queueItems[0].state'),'error');assert.equal(f.run('queueItems[1].id'),'second');assert.equal(f.elements.get('pick').disabled,false);
});

test('cancel a waiting upload without cancelling the other selected video',async()=>{
 const f=fixture(),pending=f.run("uploadOne([{name:'a.mp4',size:1},{name:'b.mp4',size:2}])");await f.run('cancelQueueItem(queueItems[1])');
 f.requests[0].status=201;f.requests[0].responseText=JSON.stringify({id:'first'});f.requests[0].onload();await pending;
 assert.equal(f.requests.length,1);assert.equal(f.run('queueItems[1].state'),'cancelled');assert.equal(f.run('queueItems[0].state'),'queued');
});
test('aborting the uploading file continues with the next file',async()=>{
 const f=fixture(),pending=f.run("uploadOne([{name:'a.mp4',size:1},{name:'b.mp4',size:2}])");await f.run('cancelQueueItem(queueItems[0])');await new Promise(setImmediate);
 assert.equal(f.requests.length,2);f.requests[1].status=201;f.requests[1].responseText=JSON.stringify({id:'second'});f.requests[1].onload();await pending;
 assert.equal(f.run('queueItems[0].state'),'cancelled');assert.equal(f.run('queueItems[1].state'),'queued');
});
test('stopping a running job calls only its individual cancellation endpoint',async()=>{
 const f=fixture();f.run("queueItems.push({id:'active',name:'a.mp4',state:'running'},{id:'other',name:'b.mp4',state:'queued'})");await f.run('cancelQueueItem(queueItems[0])');
 assert.equal(f.calls.filter(c=>c.url!=='/api/studio/history').length,1);assert.equal(f.calls[0].url,'/api/videos/active/cancel');assert.deepEqual(JSON.parse(f.calls[0].options.body),{confirmed:true});assert.equal(f.run('queueItems[0].state'),'cancelled');assert.equal(f.run('queueItems[1].state'),'queued');
});

test('hiding a record removes it from the visible and persisted queue without stopping work',()=>{
 const f=fixture(),lookup=f.context.document.getElementById;f.context.document.getElementById=id=>id==='history-section'?null:lookup(id);
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/history.js'),'utf8'),f.context);
 const first='11111111-1111-4111-8111-111111111111',second='22222222-2222-4222-8222-222222222222';
 f.run(`window.DublyHistory.remember('${first}');window.DublyHistory.remember('${second}');queueItems.push({id:'${first}',name:'a.mp4',state:'running'},{id:'${second}',name:'b.mp4',state:'cancelled'});rememberQueue();drawQueue()`);
 f.run(`window.DublyHistory.hide('${first}')`);
 assert.equal(f.run('queueItems.length'),1);assert.equal(f.run('queueItems[0].id'),second);
 assert.deepEqual(JSON.parse(f.storage.get('dubly-projects-v1')),[second]);assert.equal(JSON.parse(f.storage.get('dubly-upload-queue-v1')).length,1);assert.equal(f.calls.length,0);
 f.run(`window.DublyHistory.hide('${second}')`);assert.equal(f.elements.get('queue-section').hidden,true);assert.deepEqual(JSON.parse(f.storage.get('dubly-upload-queue-v1')),[]);
});
test('previously hidden records are removed when queue visibility is reconciled, and cancelled rows have no resume button',()=>{
 const f=fixture();f.run("window.DublyHistory={ids:()=>['visible']};queueItems.push({id:'hidden',name:'a.mp4',state:'done'},{id:'visible',name:'b.mp4',state:'cancelled'});syncQueueVisibility()");
 assert.equal(f.run('queueItems.length'),1);assert.equal(f.run('queueItems[0].id'),'visible');
 const labels=[];function visit(e){if(e.textContent)labels.push(e.textContent);for(const child of e.children||[])visit(child)}visit(f.elements.get('queue-items'));
 assert.equal(labels.includes('다시 시작'),false);
});

test('completed jobs leave the queue and browser queue storage while history remains',async()=>{
 const f=fixture();f.storage.set('dubly-projects-v1',JSON.stringify(['a','b','c']));
 f.run("queueItems.push({id:'a',name:'a.mp4',state:'running'},{id:'b',name:'b.mp4',state:'running'},{id:'c',name:'c.mp4',state:'running'});rememberQueue()");
 f.context.fetch=async()=>({ok:true,json:async()=>({videos:[{id:'a',state:'done',progress:100},{id:'b',state:'ready',progress:100},{id:'c',state:'running',progress:100}]})});
 await f.run('pollQueue()');assert.equal(f.run('queueItems.length'),1);assert.equal(f.run('queueItems[0].id'),'c');
 assert.deepEqual(JSON.parse(f.storage.get('dubly-upload-queue-v1')).map(x=>x.id),['c']);
 assert.deepEqual(JSON.parse(f.storage.get('dubly-projects-v1')),['a','b','c']);
 f.context.fetch=async()=>({ok:true,json:async()=>({videos:[{id:'c',state:'done',progress:100}]})});
 await f.run('pollQueue()');assert.equal(f.elements.get('queue-section').hidden,true);assert.deepEqual(JSON.parse(f.storage.get('dubly-upload-queue-v1')),[]);
});

test('auto-detect is selectable only as the source and is included in uploads',async()=>{
 const f=fixture();f.elements.get('source-option-auto').onclick();assert.equal(f.elements.get('source-language').value,'auto');assert.equal(f.elements.get('source-language-label').textContent,'자동 감지');
 const pending=f.run('uploadOne('+one+')');assert.equal(f.requests[0].body.source_language,'auto');assert.equal(f.requests[0].body.target_language,'ko');
 f.requests[0].status=201;f.requests[0].responseText=JSON.stringify({id:'auto-video'});f.requests[0].onload();await pending;
 const other=fixture();other.elements.get('target-language').value='auto';await other.run('uploadOne('+one+')');assert.equal(other.requests.length,0);
});
