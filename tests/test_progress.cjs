const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context=vm.createContext({window:{}});vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../static/progress.js'),'utf8'),context);
const task=context.window.DublyProgress.task;
test('loading describes the activity without exposing models or worker startup',()=>{
 for(const [phase,detail,expected] of [
  ['1/4 음성 분리','Mel-Band RoFormer 모델 로드 중','음성 분리 중'],
  ['2/4 전체 음성 전사·번역','Qwen 전사·화자·별명 · 첫 결과 대기','대사와 화자 분석 중'],
  ['2/4 전체 음성 전사·번역','Gemini 번역 · 응답 대기','대사 번역 중'],
  ['최종 영상 생성','TTS 13/295','더빙 음성 생성 중'],
  ['최종 영상 생성','MP4 저장 중','영상 저장 중']])assert.equal(task({state:'running',phase,detail}),expected);
 assert.equal(task({state:'running',resource_state:'local_wait',detail:'GPU 모델 로드 중'}),'처리 순서를 기다리는 중');
});
test('stage percentage uses backend stage progress independently of overall progress',()=>{
 const p=context.window.DublyProgress;
 assert.equal(p.stageText({state:'running',phase:'2/4 전체 음성 전사·번역',phase_progress:68.4,progress:45.7}),'전사·번역 · 68%');
 assert.equal(p.stagePercent({state:'queued',phase_progress:100}),0);
 assert.equal(p.stagePercent({state:'running',phase:'준비',phase_progress:100}),0);
 assert.equal(p.stagePercent({state:'running',phase_progress:120}),100);
 assert.equal(p.stagePercent({state:'running',phase_progress:-3}),0);
 assert.equal(p.stagePercent({state:'running'}),null);
 assert.equal(p.stagePercent({state:'done',phase_progress:91}),100);
});

test('Qwen streamed region count is shown without exposing the model name',()=>{
 assert.equal(task({state:'running',phase:'2/4 전사',detail:'감정 전환·화자 보정 응답 수신 42/197개'}),'감정 전환·화자 보정 응답 수신 42/197개');
});


test('translation displays streamed completed lines and translation-only percentage',()=>{
 const p=context.window.DublyProgress;
 const job={state:'running',phase:'2/4 전체 음성 전사·번역',detail:'구간 번역 12/48개 수신 · 7초',phase_progress:61.25,progress:42};
 assert.equal(p.task(job),'대사 번역 중 · 12/48개 수신');
 assert.equal(p.stageText(job),'번역 · 25%');
 assert.equal(p.stageText({...job,detail:'구간 번역 0/48개 수신 · 0초'}),'번역 · 0%');
 assert.equal(p.stageText({...job,detail:'구간 번역 48/48개 수신 · 20초'}),'번역 · 100%');
});


test('audio analysis uses streamed source clock coverage, then a separate alignment stage',()=>{
 const p=context.window.DublyProgress;
 const job={state:'running',phase:'2/4 전체 음성 전사·번역',phase_progress:85,progress:49,
  detail:'Gemini 12개 대사 수신 · 41.3% · 08:15 / 20:00 · 23초 경과'};
 assert.equal(p.stageText(job),'전사·번역 · 41%');
 assert.equal(p.task(job),'대사 분석 중 · 12개 수신 · 08:15 / 20:00');
 assert.equal(p.stageText({...job,detail:'MMS-FA 원본 정렬 2/8개 구간'}),'대사 시간 정렬 · 25%');
 assert.equal(p.stageText({...job,detail:'MMS-FA 원본 정렬 완료 · 48개 대사'}),'대사 시간 정렬 · 100%');
 assert.equal(p.stagePercent({...job,detail:'Gemini 0개 대사 수신 · 0.0% · 00:00 / 20:00 · 0초 경과'}),0);
});


test('untimed Gemini stream shows each received count without inventing coverage',()=>{
 const p=context.window.DublyProgress;
 const job={state:'running',phase:'2/4 전체 음성 전사·번역',phase_progress:4.3,progress:30};
 for(const count of [0,1,17,192]){
  const live={...job,detail:`Gemini ${count}개 대사 수신 · 23초 경과`};
  assert.equal(p.task(live),`대사 전사·번역 중 · ${count}개 수신 · 23초 경과`);
  assert.equal(p.stageText(live),'전사·번역');
 }
 assert.equal(p.task({...job,state:'done',detail:'Gemini 192개 대사 수신 · 23초 경과'}),'완료');
});
