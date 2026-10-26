'use strict';
(()=>{
 const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
 // The same quiet-sample insert/remove operation used by the export path.
 function edits(info,edit={},trim=true){
  const rate=info.sampleRate||44100,lo=trim?Math.round((edit.trim_start||0)*rate):0,hi=trim?Math.round((edit.trim_end??info.duration)*rate):Math.round(info.duration*rate),guard=Math.round(.005*rate);
  return (edit.pauses||[]).map(change=>{const p=info.pauses.find(p=>p.id===change.id);if(!p)return null;const a=Math.round(p.start*rate),b=Math.round(p.end*rate),amount=Math.round(change.seconds*rate)-(b-a);if(a<Math.max(lo,Math.round((edit.trim_start||0)*rate))||b>Math.min(hi,Math.round((edit.trim_end??info.duration)*rate))||!amount)return null;const at=amount<0?a+guard+Math.floor((b-a-2*guard+amount)/2):Math.round((p.cut??(p.start+p.end)/2)*rate);return {at,amount,remove:Math.max(0,-amount)}}).filter(Boolean).sort((a,b)=>a.at-b.at);
 }
 function mapTime(info,edit,time,inverse=false){
  const rate=info.sampleRate||44100;let shift=0;
  for(const op of edits(info,edit,false)){const at=op.at/rate,delta=op.amount/rate,cut=op.remove/rate;
   if(inverse){const mapped=at+shift;if(time<mapped)break;if(delta>0&&time<mapped+delta)return at;if(delta<0&&time===mapped)return at+cut;shift+=delta}
   else{if(time<at)break;if(cut&&time<at+cut)return at+shift;shift+=delta}
  }return time+(inverse?-shift:shift);
 }
 function fitDuration(info,base={},seconds,edge='end'){
  const edit={...base,pauses:(base.pauses||[]).map(p=>({...p})),speed:1};
  const lo=edit.trim_start||0,hi=edit.trim_end??info.duration,rate=info.sampleRate||44100;
  const slots=info.pauses.filter(p=>lo<=p.start&&p.end<=hi).map(p=>({id:p.id,seconds:edit.pauses.find(x=>x.id===p.id)?.seconds??(p.end-p.start)}));
  let length=hi-lo+(edit.pad_start||0)+(edit.pad_end||0)+slots.reduce((n,p)=>n+p.seconds-(info.pauses.find(s=>s.id===p.id).end-info.pauses.find(s=>s.id===p.id).start),0),delta=seconds-length;
  if(delta>=0){
   if(slots.length){const total=slots.reduce((n,p)=>n+p.seconds,0);for(const p of slots)p.seconds+=delta*p.seconds/total}
   else{const field=edge==='start'?'pad_start':'pad_end';edit[field]=(edit[field]||0)+delta}
  }else{
   let remove=-delta;for(const field of edge==='start'?['pad_start','pad_end']:['pad_end','pad_start']){const cut=Math.min(edit[field]||0,remove);edit[field]=(edit[field]||0)-cut;remove-=cut;length-=cut}
   const capacities=slots.map(p=>Math.max(0,p.seconds-.01)),available=capacities.reduce((n,c)=>n+c,0),shrink=Math.min(remove,available);
   if(available)slots.forEach((p,i)=>p.seconds-=shrink*capacities[i]/available);length-=shrink;remove-=shrink;
   if(remove>0)edit.speed=clamp(length/Math.max(.001,seconds),1,8);
  }
  for(const p of slots){edit.pauses=edit.pauses.filter(x=>x.id!==p.id);edit.pauses.push({id:p.id,seconds:Math.max(.01,Math.round(p.seconds*rate)/rate)})}
  edit.audio_sha256=info.sha256;return edit;
 }
 const waveViews=new WeakMap();
 function waveform(info,edit={},trim=true){
  const key=JSON.stringify([edit,trim]),cache=waveViews.get(info)||new Map();if(cache.has(key))return cache.get(key);edit={...edit,pauses:(edit.pauses||[]).map(p=>({...p}))};
  const padStart=edit.pad_start||0,padEnd=edit.pad_end||0,start=trim?(edit.trim_start||0):0,end=trim?(edit.trim_end??info.duration):info.duration,mappedStart=mapTime(info,edit,start),mappedEnd=mapTime(info,edit,end),duration=Math.max(.001,mappedEnd-mappedStart+padStart+padEnd),count=Math.max(100,Math.min(12000,Math.ceil(info.peaks.length*duration/info.duration))),out=new Array(count).fill(0);
  const operations=edits(info,edit,false),rate=info.sampleRate||44100;
  for(let i=0;i<count;i++){const a=mappedStart+i/count*duration-padStart,b=mappedStart+(i+1)/count*duration-padStart;if(b<=mappedStart||a>=mappedEnd)continue;
   let inserted=false,shift=0;for(const op of operations){const at=op.at/rate+shift;if(op.amount>0&&a>=at&&b<=at+op.amount/rate){inserted=true;break}shift+=op.amount/rate}if(inserted)continue;
   const rawA=mapTime(info,edit,a,true),rawB=mapTime(info,edit,b,true),lo=clamp(Math.floor(rawA/info.duration*info.peaks.length),0,info.peaks.length-1),hi=clamp(Math.max(lo+1,Math.ceil(rawB/info.duration*info.peaks.length)),1,info.peaks.length);for(let j=lo;j<hi;j++)out[i]=Math.max(out[i],info.peaks[j]);
  }const result={duration:duration/(trim?(edit.speed||1):1),peaks:out,map:t=>mapTime(info,edit,t)+padStart,unmap:t=>mapTime(info,edit,t-padStart,true)};cache.set(key,result);if(cache.size>4)cache.delete(cache.keys().next().value);waveViews.set(info,cache);return result;
 }
 function editPCM(channels,rate,info,edit={}){
  const lo=Math.round((edit.trim_start||0)*rate),hi=Math.round((edit.trim_end??info.duration)*rate),ops=edits({...info,sampleRate:rate},edit,true),length=hi-lo+ops.reduce((n,p)=>n+p.amount,0),guard=Math.round(.005*rate);
  const before=Math.round((edit.pad_start||0)*rate),after=Math.round((edit.pad_end||0)*rate);
  return channels.map(source=>{const out=new Float32Array(Math.max(1,length+before+after));let cursor=lo,offset=before;for(const op of ops){out.set(source.subarray(cursor,op.at),offset);offset+=op.at-cursor;if(op.amount<0)cursor=op.at-op.amount;else{const bridge=Math.min(guard,Math.floor(op.amount/2));if(bridge>1)for(let i=0;i<bridge;i++){out[offset+i]=source[op.at-1]*(1-i/(bridge-1));out[offset+op.amount-bridge+i]=source[op.at]*i/(bridge-1)}offset+=op.amount;cursor=op.at}}out.set(source.subarray(cursor,hi),offset);return out});
 }
 function weight(time,intervals,preserved=[]){if(preserved.some(r=>r.start<=time&&time<r.end))return 0;let value=0;for(const p of intervals)value=Math.max(value,clamp(Math.min((time-(p.start-.2))/.2,((p.end+.2)-time)/.2),0,1));return value}
 window.DublyPreviewMath={waveform,editPCM,mapTime,weight,fitDuration};

 class Preview{
  constructor({video,api,getDoc,getWave,save,status}){Object.assign(this,{video,api,getDoc,getWave,save,status});this.enabled=true;this.epoch=0;this.sources=[];this.buffers=new Map();this.pending=new Map();this.edited=new Map();this.next=0;this.timer=setInterval(()=>this.pump(),180)}
  context(){if(!this.ctx){this.ctx=new AudioContext();this.output=this.ctx.createGain();this.output.gain.value=this.video.volume;this.output.connect(this.ctx.destination)}return this.ctx}
  async buffer(key,url){if(this.buffers.has(key))return this.buffers.get(key);if(this.pending.has(key))return this.pending.get(key);const promise=fetch(url).then(async r=>{if(!r.ok)throw Error('미리듣기 오디오를 불러오지 못했습니다.');return this.context().decodeAudioData(await r.arrayBuffer())}).then(buffer=>{this.buffers.set(key,buffer);let bytes=0;for(const [k,b] of [...this.buffers].reverse()){bytes+=b.length*b.numberOfChannels*4;if(bytes>80*1024*1024)this.buffers.delete(k)}return buffer}).finally(()=>this.pending.delete(key));this.pending.set(key,promise);return promise}
  async clip(row){const edit=JSON.parse(JSON.stringify(row.audio_edit||{})),gainDb=this.getDoc().tts_gain_db??0,key=row.id+':'+row.audio_key+':'+gainDb+':'+JSON.stringify({...edit,position:undefined});if(this.edited.has(key))return this.edited.get(key);const info=await this.getWave(row.id);let buffer;
   if((edit.speed||1)>1){await this.save();buffer=await this.buffer(key,`${this.api}/segments/${row.id}/audio?kind=edited&v=${this.getDoc().revision}`)}
   else{const base=await this.buffer('base:'+row.id+':'+row.audio_key,`${this.api}/segments/${row.id}/audio?kind=base&v=${encodeURIComponent(row.audio_key||'')}`),channels=editPCM(Array.from({length:base.numberOfChannels},(_,c)=>base.getChannelData(c)),base.sampleRate,info,edit);buffer=this.context().createBuffer(channels.length,channels[0].length,base.sampleRate);channels.forEach((x,c)=>buffer.copyToChannel(x,c))}
   const value={buffer,gain:10**(gainDb/20)};this.edited.set(key,value);if(this.edited.size>24)this.edited.delete(this.edited.keys().next().value);return value;
  }
  stop(){this.epoch++;this.next=0;for(const source of this.sources){try{source.stop()}catch{}}this.sources=[];this.preparing=false}
  async warm(){if(!this.enabled||!this.getDoc())return;const t=this.video.currentTime,at=Math.floor(t/12)*12,end=Math.min(at+12,this.getDoc().duration);if(end<=at)return;const kind=this.getDoc().has_background?'background':'clean';await Promise.all([this.buffer('original:'+at,`${this.api}/preview-source?kind=original&start=${at}&end=${end}`),this.buffer(kind+':'+at,`${this.api}/preview-source?kind=${kind}&start=${at}&end=${end}`),...this.getDoc().segments.filter(r=>r.audio_ready&&!r.audio_error&&!r.keep_original&&r.start<t+15&&r.end>t-10).map(r=>this.clip(r))])}
  async start(){this.stop();if(!this.enabled||this.video.paused||!this.getDoc())return;const epoch=this.epoch;try{await this.context().resume();this.status('미리듣기 준비 중…');await this.warm();if(epoch!==this.epoch||this.video.paused)return;this.anchorMedia=this.video.currentTime;this.anchorClock=this.ctx.currentTime+.02;this.next=this.anchorMedia;await this.schedule(epoch);this.status('자동 미리듣기')}catch(e){if(epoch!==this.epoch)return;this.stop();this.video.pause();this.status(e.message)}}
  async schedule(epoch){if(this.preparing||epoch!==this.epoch)return;this.preparing=true;try{const doc=this.getDoc(),start=this.next,tile=Math.floor(start/12)*12,end=Math.min(tile+12,doc.duration);if(end<=start+.001)return;const kind=doc.has_background?'background':'clean',original=await this.buffer('original:'+tile,`${this.api}/preview-source?kind=original&start=${tile}&end=${end}`),bed=await this.buffer(kind+':'+tile,`${this.api}/preview-source?kind=${kind}&start=${tile}&end=${end}`);
   const rows=doc.segments.filter(r=>r.audio_ready&&!r.audio_error&&!r.keep_original),clips=new Map();await Promise.all(rows.filter(r=>{const at=r.audio_edit?.position??r.start;return at<end+.5&&at+(r.audio_seconds||r.end-r.start)+(r.audio_edit?.pauses||[]).reduce((n,p)=>n+p.seconds,0)>start-.5}).map(async r=>clips.set(r.id,await this.clip(r))));if(epoch!==this.epoch||this.video.paused)return;
   const rate=this.video.playbackRate,when=this.anchorClock+(start-this.anchorMedia)/rate;if(when<this.ctx.currentTime-.12){this.preparing=false;return this.start()}
   const lengths=doc.segments.map(r=>{const info=this.getWaveSync?.(r.id);return clips.get(r.id)?.buffer.duration||(info?waveform(info,r.audio_edit).duration:0)||((r.audio_edit?.trim_end??r.audio_seconds??r.end-r.start)-(r.audio_edit?.trim_start||0))/(r.audio_edit?.speed||1)}),intervals=[];
   for(let i=0;i<doc.segments.length;i++){const r=doc.segments[i];if(!r.audio_ready||r.audio_error||r.keep_original)continue;const pos=r.audio_edit?.position??r.start,hi=Math.max(r.end,pos+lengths[i]),previous=i?Math.max(doc.segments[i-1].end,(doc.segments[i-1].audio_edit?.position??doc.segments[i-1].start)+lengths[i-1]):0,left=i?Math.min(r.start,(previous+r.start)/2):0,right=i+1<doc.segments.length?Math.max(hi,(hi+doc.segments[i+1].start)/2):doc.duration;intervals.push({start:pos<r.start?Math.max(0,pos-.2):Math.max(left,r.start-.2),end:Math.min(right,hi+.2)})}
   const preserved=doc.segments.filter(r=>r.keep_original||r.audio_error||!r.audio_ready),curveCount=4096,curve=new Float32Array(curveCount);for(let i=0;i<curveCount;i++)curve[i]=weight(start+i/(curveCount-1)*(end-start),intervals,preserved);
   const create=(buffer,offset,length,gain=1)=>{const source=this.ctx.createBufferSource(),volume=this.ctx.createGain();source.buffer=buffer;source.playbackRate.value=rate;volume.gain.value=gain;source.connect(volume);volume.connect(this.output);this.sources.push(source);source.onended=()=>{this.sources=this.sources.filter(x=>x!==source)};source.start(Math.max(this.ctx.currentTime,when),offset,length);return {source,volume}};
   const main=create(original,start-tile,end-start),back=create(bed,start-tile,end-start,0);
   const duration=(end-start)/rate;if(doc.has_background){const inverse=Float32Array.from(curve,x=>1-x);main.volume.gain.setValueCurveAtTime(inverse,Math.max(when,this.ctx.currentTime),duration);back.volume.gain.setValueCurveAtTime(curve,Math.max(when,this.ctx.currentTime),duration)}else back.volume.gain.setValueCurveAtTime(Float32Array.from(curve,x=>-x),Math.max(when,this.ctx.currentTime),duration);
   for(const row of rows){const item=clips.get(row.id);if(!item)continue;const pos=row.audio_edit?.position??row.start,lo=Math.max(start,pos),hi=Math.min(end,pos+item.buffer.duration);if(hi<=lo)continue;const source=this.ctx.createBufferSource(),gain=this.ctx.createGain();source.buffer=item.buffer;source.playbackRate.value=rate;gain.gain.value=item.gain;const automation=new Float32Array(256);for(let n=0;n<automation.length;n++)automation[n]=preserved.some(r=>r.start<=lo+n/(automation.length-1)*(hi-lo)&&lo+n/(automation.length-1)*(hi-lo)<r.end)?0:item.gain;gain.gain.setValueCurveAtTime(automation,Math.max(this.ctx.currentTime,this.anchorClock+(lo-this.anchorMedia)/rate),(hi-lo)/rate);source.connect(gain);gain.connect(this.output);this.sources.push(source);source.onended=()=>{this.sources=this.sources.filter(x=>x!==source)};source.start(Math.max(this.ctx.currentTime,this.anchorClock+(lo-this.anchorMedia)/rate),lo-pos,hi-lo)}
   this.next=end;
  }finally{if(epoch===this.epoch)this.preparing=false}}
  pump(){if(!this.enabled||this.video.paused||!this.ctx||!this.next)return;if(Math.abs((this.ctx.currentTime-this.anchorClock)*this.video.playbackRate+this.anchorMedia-this.video.currentTime)>.15){this.start();return}if(this.next-this.video.currentTime<3&&this.next<this.getDoc().duration&&!this.preparing){const epoch=this.epoch;this.schedule(epoch).catch(e=>{if(epoch===this.epoch){this.stop();this.video.pause();this.status(e.message)}})}}
  refresh(){clearTimeout(this.refreshTimer);this.refreshTimer=setTimeout(()=>{if(this.video.paused)this.warm().catch(e=>this.status(e.message));else this.start()},80)}
 }
 window.DublyLivePreview=Preview;
})();

'use strict';
(()=>{
 const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
 function speechRegions(data){
  const peaks=data?.peaks||[],start=data?.start||0,end=data?.end??(start+(data?.duration||0)),step=(end-start)/peaks.length;
  if(!peaks.length||!(step>0))return [];
  const sorted=peaks.filter(Number.isFinite).sort((a,b)=>a-b);if(!sorted.length)return [];
  const level=Math.max(sorted[Math.floor((sorted.length-1)*.95)],sorted[sorted.length-1]*.2),floor=sorted[Math.floor((sorted.length-1)*.15)];
  const gate=Math.max(.00002,level*.025,(floor<level*.3?floor:0)*3);
  const spans=[];
  for(let i=0;i<peaks.length;i++){
   if(!(peaks[i]>gate))continue;
   const last=spans[spans.length-1];
   if(last&&(i-last[1])*step<=.10)last[1]=i+1;else spans.push([i,i+1]);
  }
  return spans.filter(([a,b])=>(b-a)*step>=.055).map(([a,b])=>({
   start:Math.max(start,start+a*step-.015),end:Math.min(end,start+b*step+.025),
   openStart:a===0,openEnd:b===peaks.length
  }));
 }
 function groupRegions(regions,turns=[],gapSeconds=.2){
  const rows=turns.filter(r=>!r.deleted),result=[];
  for(const region of regions){
   const voices=new Set(rows.filter(r=>Math.min(r.end,region.end)-Math.max(r.start,region.start)>.015).map(r=>String(r.speaker_id??r.speaker)));
   const voice=voices.size===1?[...voices][0]:null,mixed=voices.size>1;
   const last=result[result.length-1];
   const gap=last?region.start-last.end:Infinity;
   const different=last?.voice!=null&&voice!=null&&last.voice!==voice;
   if(last&&!last.mixed&&!mixed&&!different&&gap<=gapSeconds+1e-7){
    last.end=Math.max(last.end,region.end);last.openEnd=region.openEnd;last.voice=last.voice??voice;last.parts.push({...region});
   }else result.push({...region,voice,mixed,parts:[{...region}]});
  }
  return result;
 }
 function boundaries(regions){return [...new Set(regions.flatMap(r=>[...(!r.openStart?[r.start]:[]),...(!r.openEnd?[r.end]:[])]))].sort((a,b)=>a-b)}
 function nearest(time,edges,scale,{disabled=false,locked=null,pixels=9,releasePixels=14}={}){
  if(disabled||!(scale>0))return {time,boundary:null};
  if(locked!==null&&Math.abs(time-locked)*scale<=releasePixels)return {time:locked,boundary:locked};
  let distance=pixels/scale,target=null;
  for(const boundary of edges){const delta=Math.abs(time-boundary);if(delta<=distance){distance=delta;target=boundary}}
  return {time:target??time,boundary:target};
 }
 function move(position,duration,edges,scale,options={}){
  const start=nearest(position,edges,scale,options),end=nearest(position+duration,edges,scale,options);
  if(start.boundary!==null&&(end.boundary===null||Math.abs(start.time-position)<=Math.abs(end.time-position-duration)))return {position:start.time,boundary:start.boundary,edge:'start'};
  if(end.boundary!==null)return {position:end.time-duration,boundary:end.boundary,edge:'end'};
  return {position,boundary:null,edge:null};
 }
 window.DublySourceSnap={speechRegions,groupRegions,boundaries,nearest,move};
})();
