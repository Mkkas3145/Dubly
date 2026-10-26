/* Recover a stalled media request without resetting the editor or its history. */
(()=>{
 class VideoSource{
  constructor(video,url,notify=()=>{},previewUrl=null){
   Object.assign(this,{video,url,notify,previewUrl,attempts:0,timer:null,pending:null,started:false,resolving:null});
   video.addEventListener('loadstart',()=>this.watch());
   // A slow request that is still receiving data must not be restarted.
   for(const event of ['progress','loadeddata'])video.addEventListener(event,()=>this.progress());
   video.addEventListener('waiting',()=>this.watch());
   video.addEventListener('stalled',()=>this.watch());
   video.addEventListener('loadedmetadata',()=>{
    if(this.pending&&Number.isFinite(video.duration))video.currentTime=Math.max(0,Math.min(this.pending.time,video.duration-.01));
    this.progress();
   });
   video.addEventListener('canplay',()=>{
    clearTimeout(this.timer);this.timer=null;
    this.attempts=0;
    const resume=this.pending?.resume;this.pending=null;
    if(resume)video.play().catch(e=>{if(e.name!=='AbortError')this.notify('재생 버튼을 눌러 다시 시도해 주세요.')});
   });
   video.addEventListener('timeupdate',()=>{if(video.readyState>=2){clearTimeout(this.timer);this.timer=null}});
   video.addEventListener('seeking',()=>this.progress());
   video.addEventListener('seeked',()=>{if(video.readyState>=2){clearTimeout(this.timer);this.timer=null;this.attempts=0}});
   video.addEventListener('error',()=>this.retry());
  }
  start(){
   if(this.started)return;this.started=true;
   if(!this.previewUrl){this.load(false);return}
   this.resolving=fetch(this.previewUrl,{method:'HEAD'}).then(response=>{if(response.ok)this.url=this.previewUrl})
    .catch(()=>{}).finally(()=>{this.load(false);this.resolving=null});
  }
  load(retry){this.video.preload='metadata';this.video.src=this.url;this.video.load();this.watch()}
  progress(){clearTimeout(this.timer);this.timer=null;this.watch()}
  idle(){return this.video.paused&&!this.video.seeking&&!this.pending?.resume&&this.video.readyState>=1}
  watch(){if(!this.started||this.timer||this.idle())return;this.timer=setTimeout(()=>{this.timer=null;if(document.hidden){this.watch();return}if(this.idle())return;this.retry()},30000)}
  retry(){
   clearTimeout(this.timer);this.timer=null;
   if(this.attempts>=2){this.notify('영상을 불러오지 못했습니다. 재생 버튼을 눌러 다시 시도해 주세요.');return}
   this.pending=this.pending||{time:Number.isFinite(this.video.currentTime)?this.video.currentTime:0,resume:!this.video.paused};
   this.attempts++;this.load(true);
  }
  async play(){
   if(this.resolving)await this.resolving;
   if(this.video.error||this.attempts>=2&&this.video.readyState===0){this.attempts=0;this.pending={time:this.video.currentTime||0,resume:true};this.retry();return Promise.resolve()}
   return this.video.play();
  }
 }
 window.DublyVideoSource=VideoSource;
})();
