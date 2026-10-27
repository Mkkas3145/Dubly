const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const source=fs.readFileSync('static/studio.js','utf8');
function extract(name){const start=source.indexOf('async function '+name+'('),body=source.indexOf('{',start);let depth=0;for(let i=body;i<source.length;i++){if(source[i]==='{')depth++;if(source[i]==='}'&&!--depth)return source.slice(start,i+1)}}
for(const direction of ['undo','redo'])test(`split ${direction} saves only editable fields and restores audio before trims`,async()=>{
 const editable=row=>Object.fromEntries(['id','start','end','original_text','translated_text','speaker_id','keep_original','audio_edit','deleted'].filter(k=>k in row).map(k=>[k,row[k]]));
 const row={id:0,start:0,end:2,original_text:'hello',translated_text:'안녕',speaker_id:'0',keep_original:false,audio_edit:{trim_end:2,audio_sha256:'original'}};
 const extra={...row,id:1,start:1,end:2,audio_edit:{position:1},audio_ready:true,audio_key:'right',audio_seconds:1,audio_input:{},audio_history_count:2};
 const snapshot=direction==='undo'?{segments:[row],audio_versions:{0:'original'}}:{segments:[{...row,end:1,audio_edit:{position:0}},{...editable(extra),deleted:false}],audio_versions:{0:'left',1:'right'}};
 let restored=false,error=null,submitted=null,final=null;
 const state={busy:false,historyApplying:false,doc:{revision:5,segments:[{...row,end:1,audio_edit:{position:0},audio_ready:true,audio_key:direction==='undo'?'left':'original'},extra]},api:'/api/studio/test',
 editHistory:{undo:()=>snapshot,redo:()=>snapshot},updateHistoryButtons(){},saveDraft:async()=>{},payload:()=>({segments:state.doc.segments.map(editable)}),
 request:async(url,options)=>{submitted=JSON.parse(options.body);for(const r of submitted.segments){assert.deepEqual(Object.keys(r).filter(k=>!Object.keys(editable(r)).includes(k)),[])}assert.deepEqual(submitted.segments[0].audio_edit,{});return {...state.doc,segments:submitted.segments}},
 post:async()=>{restored=true;return state.doc},waveCache:new Map(),waveFailures:new Set(),syncAudioInputs(){},restoreEdits:s=>{assert.equal(restored,true);final=s},say:message=>error=message};
 vm.createContext(state);vm.runInContext(extract('applyHistory'),state);await state.applyHistory(direction);
 assert.equal(error,null);assert.equal(final,snapshot);assert.equal(submitted.segments[1].deleted,direction==='undo'?true:false);assert.equal(state.historyApplying,false);
});
