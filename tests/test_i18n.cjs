const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');

function fixture(stored=null,language='ko-KR'){
 const all=[],storage=new Map(stored?[['dubly-ui-language',stored]]:[]);let callback;
 class Element{
  constructor(tag='div',protectedData=false){this.nodeType=1;this.tag=tag;this.attributes={};this.children=[];this.protectedData=protectedData;all.push(this)}
  closest(selector){return (this.protectedData&&selector.includes('[data-no-i18n]'))||selector.split(',').includes(this.tag)?this:null}
  hasAttribute(k){return k in this.attributes}getAttribute(k){return this.attributes[k]}setAttribute(k,v){this.attributes[k]=v}
  append(n){n.parentElement=this;this.children.push(n)}
 }
 const root=new Element('html'),head=new Element('head');root.append(head);
 const label=new Element(),text={nodeType:3,nodeValue:'저장됨',parentElement:label};label.children.push(text);root.append(label);
 const transcript=new Element('textarea');transcript.value='사용자가 입력한 추가 번역 지시';transcript.setAttribute('placeholder','예: 인물의 말투를 살리고, 이름은 원어 발음대로 표기해 주세요.');const userText={nodeType:3,nodeValue:'저장됨',parentElement:transcript};transcript.children.push(userText);root.append(transcript);
 const filename=new Element('a',true),fileText={nodeType:3,nodeValue:'저장됨',parentElement:filename};filename.children.push(fileText);root.append(filename);
 const input=new Element('input');input.value='원문 전사';input.setAttribute('placeholder','대사 검색…');root.append(input);
 const document={documentElement:root,head,createElement:tag=>new Element(tag),querySelector:()=>null,getElementById:()=>null,querySelectorAll:()=>[],createTreeWalker(node){const nodes=[];function collect(n){for(const child of n.children||[]){nodes.push(child);collect(child)}}collect(node);let at=0;return {currentNode:node,nextNode(){this.currentNode=nodes[at++];return !!this.currentNode}}}};
 const window={dispatchEvent(){}};
 const context=vm.createContext({window,document,localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},navigator:{language},MutationObserver:class{constructor(cb){callback=cb}observe(){}},NodeFilter:{SHOW_ELEMENT:1,SHOW_TEXT:4},CustomEvent:class{}});
 for(const file of ['locales.js','i18n.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
 return {i18n:window.DublyI18n,text,userText,fileText,input,textarea:transcript,storage,update(value){text.nodeValue=value;callback([{type:'characterData',target:text}])},messages:window.DublyMessages};
}
test('locale switches round trip without modifying transcripts, filenames or input values',()=>{
 const f=fixture();f.i18n.set('en');assert.equal(f.text.nodeValue,'Saved');assert.equal(f.input.getAttribute('placeholder'),'Search dialogue…');
 f.i18n.set('ja');assert.equal(f.text.nodeValue,'保存済み');f.i18n.set('ko');assert.equal(f.text.nodeValue,'저장됨');
 assert.equal(f.userText.nodeValue,'저장됨');assert.equal(f.fileText.nodeValue,'저장됨');assert.equal(f.input.value,'원문 전사');
});
test('new asynchronous progress text translates and remains reversible',()=>{
 const f=fixture('en');f.update('업로드 중 · 47%');assert.equal(f.text.nodeValue,'Uploading · 47%');f.i18n.set('ja');assert.equal(f.text.nodeValue,'アップロード中 · 47%');f.i18n.set('ko');assert.equal(f.text.nodeValue,'업로드 중 · 47%');
});
test('specific templates win over generic speaker patterns and preserve parameters literally',()=>{
 const f=fixture('en');assert.equal(f.i18n.t('화자 14 별명'),'Speaker 14 nickname');assert.equal(f.i18n.t('기본 1.25초 → 편집 0.95초'),'Original 1.25s → Edited 0.95s');assert.equal(f.i18n.t('화자 {id}',{id:'<actor>'}),'Speaker <actor>');
});
test('UI language persists independently, browser detection and unsupported locales work',()=>{
 const f=fixture(null,'ja-JP');assert.equal(f.i18n.locale,'ja');f.i18n.set('en');assert.equal(f.storage.get('dubly-ui-language'),'en');f.i18n.set('xx');assert.equal(f.i18n.locale,'en');assert.equal(fixture('unknown').i18n.locale,'ko');
});
test('every catalog translation preserves interpolation fields',()=>{
 for(const row of fixture().messages){assert.equal(row.length,3);const fields=s=>[...s.matchAll(/\{(\w+)\}/g)].map(m=>m[1]).sort();for(const copy of row.slice(1)){assert.ok(copy);assert.deepEqual(fields(copy),fields(row[0]),row[0])}}
});

test('studio, cut and player UI labels translate with parameters in English and Japanese',()=>{
 const f=fixture();f.i18n.set('en');
 assert.equal(f.i18n.t('발화 구간'),'Speech regions');assert.equal(f.i18n.t('자르기'),'Cut');
 assert.equal(f.i18n.t('플레이어'),'Player');assert.equal(f.i18n.t('최신 편집 불러오기'),'Load latest edits');
 assert.equal(f.i18n.t('현재 생성된 음성으로 재생 · 재생성 필요한 대사 3개'),'Playing current audio · 3 lines need regeneration');
 const hint='드래그: 이동·길이 조절 · 재생 헤드: 대사 경계 맞춤 · Alt: 자석 해제 · C: 자르기';assert.ok(!/[가-힣]/.test(f.i18n.t(hint)));
 f.i18n.set('ja');assert.equal(f.i18n.t('발화 구간'),'発話区間');assert.equal(f.i18n.t('자르기'),'分割');assert.equal(f.i18n.t('플레이어'),'プレーヤー');
 const html=fs.readFileSync(path.join(__dirname,'../static/studio-player.html'),'utf8');
 assert.ok(html.indexOf('/static/locales.js')<html.indexOf('/static/i18n.js'));
 assert.ok(html.indexOf('/static/i18n.js')<html.indexOf('/static/studio-player.js'));
});


test('translation textarea placeholder translates on first load and locale changes without touching entered text',()=>{
 const f=fixture('en');
 assert.equal(f.textarea.getAttribute('placeholder'),'Example: Preserve each character’s speaking style and use phonetic spellings for names.');
 f.i18n.set('ja');assert.equal(f.textarea.getAttribute('placeholder'),'例: 登場人物の話し方を生かし、名前は原語の発音に合わせて表記してください。');
 f.i18n.set('ko');assert.equal(f.textarea.getAttribute('placeholder'),'예: 인물의 말투를 살리고, 이름은 원어 발음대로 표기해 주세요.');
 assert.equal(f.textarea.value,'사용자가 입력한 추가 번역 지시');assert.equal(f.userText.nodeValue,'저장됨');
});
