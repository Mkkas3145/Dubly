/* Product progress copy: model names and worker internals stay in server logs. */
(()=>{
 function translationCount(job={}){
  const match=(job.detail||'').match(/구간 번역 (\d+)\/(\d+)개 수신/);
  if(!match||job.state!=='running')return null;
  const total=Number(match[2]),done=Math.min(total,Number(match[1]));
  return total>0?{done,total,percent:Math.round(done/total*100)}:null;
 }
 function audioReceived(job={}){
  if(job.state!=='running')return null;
  const m=(job.detail||'').match(/^Gemini (\d+)개 대사 수신 · ([\d.]+)초 경과$/);
  return m?{count:Number(m[1]),seconds:Number(m[2])}:null;
 }
 function audioCoverage(job={}){
  if(job.state!=='running')return null;
  const m=(job.detail||'').match(/Gemini (\d+)개 대사 수신 · ([\d.]+)% · ([\d:]+) \/ ([\d:]+)/);
  return m?{count:Number(m[1]),percent:Math.max(0,Math.min(99,Number(m[2]))),position:m[3],duration:m[4]}:null;
 }
 function alignmentProgress(job={}){
  const detail=job.detail||'';
  if(job.state!=='running'||!/MMS-FA/.test(detail))return null;
  if(/정렬 완료/.test(detail))return 100;
  const m=detail.match(/원본 정렬 (\d+)\/(\d+)개 구간/);
  return m&&Number(m[2])>0?Math.round(Number(m[1])/Number(m[2])*100):0;
 }
 function task(job={}){
  const state=job.state||'',phase=job.phase||'',detail=job.detail||'';
  if(state==='error')return '처리에 실패했습니다.';
  if(state==='cancelled')return '중단됨';
  if(state==='done'||state==='ready')return '완료';
  if(['waiting','uploaded','queued'].includes(state))return '처리 대기 중';
  if(state==='uploading')return '업로드 중';
  if(job.resource_state==='local_wait')return '처리 순서를 기다리는 중';
  if(/MP4|영상 저장/.test(detail))return '영상 저장 중';
  if(/미리듣기/.test(detail))return '미리듣기 준비 중';
  if(/합치는|합성|믹싱/.test(detail)||/4\/4/.test(phase))return '배경음과 음성 합치는 중';
  if(/TTS|번역 음성|음성 생성/.test(detail))return '더빙 음성 생성 중';
  if(/참조|임베딩|화자 검증/.test(detail))return '화자 음성 준비 중';
  const received=audioReceived(job);
  if(received)return `대사 전사·번역 중 · ${received.count}개 수신 · ${received.seconds}초 경과`;
  const audio=audioCoverage(job);
  if(audio)return `대사 분석 중 · ${audio.count}개 수신 · ${audio.position} / ${audio.duration}`;
  if(alignmentProgress(job)!==null)return '대사 시간 정렬 중';
  if(/경계|파형 보정/.test(detail))return '대사 구간 정리 중';
  const translated=translationCount(job);
  if(translated)return `대사 번역 중 · ${translated.done}/${translated.total}개 수신`;
  if(/Gemini|번역/.test(detail))return '대사 번역 중';
  if(/감정·말투 단서/.test(detail))return detail;
  if(/감정 전환·화자 보정 응답/.test(detail))return detail;
  if(/전사|단서|화자|Qwen|MAI|MOSS|Voxtral/.test(detail))return '대사와 화자 분석 중';
  if(/분리|오디오 추출|음원/.test(phase+' '+detail))return '음성 분리 중';
  if(/전사|번역|2\/4/.test(phase))return '대사와 화자 분석 중';
  if(/음성|3\/4/.test(phase))return '더빙 음성 생성 중';
  if(/영상/.test(phase))return '영상 저장 중';
  return '스튜디오 준비 중';
 }
 function stage(job={}){
  if(audioReceived(job)||audioCoverage(job))return '전사·번역';
  if(alignmentProgress(job)!==null)return '대사 시간 정렬';
  if(translationCount(job))return '번역';
  const phase=job.phase||'';
  if(/1\/4|분리/.test(phase))return '음성 분리';
  if(/2\/4|전사|번역/.test(phase))return '전사·번역';
  if(/3\/4|구간|경계/.test(phase))return '대사 구간 정리';
  if(/4\/4|최종 영상/.test(phase))return '더빙·영상 생성';
  if(/음성 생성/.test(phase))return '음성 생성';
  return '현재 단계';
 }
 function stagePercent(job={}){
  if(audioReceived(job))return null;
  const audio=audioCoverage(job);if(audio)return Math.round(audio.percent);
  const alignment=alignmentProgress(job);if(alignment!==null)return alignment;
  const translated=translationCount(job);
  if(translated)return translated.percent;
  if(['ready','done'].includes(job.state))return 100;
  if(['waiting','uploaded','queued'].includes(job.state)||job.phase==='준비')return 0;
  return typeof job.phase_progress==='number'&&Number.isFinite(job.phase_progress)?Math.round(Math.max(0,Math.min(100,job.phase_progress))):null;
 }
 function stageText(job={}){if(audioReceived(job))return stage(job);return stage(job)+' · '+(stagePercent(job)??'—')+'%'}
 window.DublyProgress={task,stage,stagePercent,stageText};
})();
