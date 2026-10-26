const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
function fixture(previewUrl=null,fetcher=undefined){
 const listeners={},timers=new Map(),notices=[];let seq=0;
 const video={currentTime:4,duration:100,paused:true,seeking:false,readyState:0,error:null,loads:0,plays:0,
  addEventListener:(name,fn)=>listeners[name]=fn,load(){this.loads++},play(){this.plays++;return Promise.resolve()}};
 const state={window:{},document:{hidden:false},fetch:fetcher,setTimeout:fn=>{timers.set(++seq,fn);return seq},clearTimeout:id=>timers.delete(id)};
 vm.createContext(state);vm.runInContext(fs.readFileSync('static/studio-media.js','utf8'),state);
 const controller=new state.window.DublyVideoSource(video,'/source',text=>notices.push(text),previewUrl);
 return {video,controller,listeners,timers,notices};
}
test('source starts once, preloads frames, and retries an interrupted request preserving position',()=>{
 const {video,controller,listeners}=fixture();controller.start();controller.start();assert.equal(video.loads,1);assert.equal(video.preload,'metadata');
 listeners.error();assert.equal(video.loads,2);assert.equal(video.src,'/source');video.currentTime=0;listeners.loadedmetadata();assert.equal(video.currentTime,4);
 listeners.canplay();assert.equal(video.plays,0);
});

test('prepared browser proxy is preferred and play waits for source selection',async()=>{
 let finish;const response=new Promise(resolve=>finish=resolve);
 const {video,controller}=fixture('/preview.mp4',()=>response);controller.start();const playing=controller.play();
 assert.equal(video.loads,0);finish({ok:true});await playing;
 assert.equal(video.src,'/preview.mp4');assert.equal(video.plays,1);
});

test('missing optional proxy falls back to the original source',async()=>{
 const {video,controller}=fixture('/preview.mp4',async()=>({ok:false}));controller.start();await controller.resolving;
 assert.equal(video.src,'/source');assert.equal(video.loads,1);
});
test('stalled loading has bounded recovery and playback resumes only when previously playing',async()=>{
 const {video,controller,listeners,timers,notices}=fixture();video.paused=false;controller.start();[...timers.values()][0]();
 listeners.loadedmetadata();listeners.canplay();assert.equal(video.plays,1);listeners.error();listeners.error();listeners.error();assert.equal(video.loads,4);assert.equal(notices.length,1);
});

test('download progress renews the timeout instead of restarting a healthy slow request',()=>{
 const {video,controller,listeners,timers}=fixture();controller.start();
 for(let i=0;i<5;i++){const before=[...timers.keys()][0];listeners.progress();assert.equal(timers.size,1);assert.equal(timers.has(before),false)}
 assert.equal(video.loads,1);
});

test('paused metadata preload is not mistaken for a stalled download',()=>{
 const {video,controller,listeners,timers}=fixture();controller.start();video.readyState=1;listeners.loadedmetadata();
 assert.equal(timers.size,0);assert.equal(video.loads,1);
});

test('successful recovery resets the retry budget for a later independent interruption',()=>{
 const {video,controller,listeners}=fixture();controller.start();listeners.error();listeners.error();listeners.canplay();
 assert.equal(controller.attempts,0);listeners.error();assert.equal(controller.attempts,1);assert.equal(video.loads,4);
});
test('a user can retry after the recovery limit without reloading their document',async()=>{
 const {video,controller,listeners}=fixture();controller.start();listeners.error();listeners.error();listeners.error();video.error={code:2};await controller.play();
 listeners.loadedmetadata();listeners.canplay();assert.equal(video.plays,1);assert.equal(video.loads,4);
});
