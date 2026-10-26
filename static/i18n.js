'use strict';
(()=>{
 const messages=window.DublyMessages||[],supported=['ko','en','ja'],storageKey='dubly-ui-language';
 const column={ko:0,en:1,ja:2}, bindings=new WeakMap(),attrs=['placeholder','title','aria-label'];
 let locale='ko';try{locale=localStorage.getItem(storageKey)||navigator.language?.split('-')[0]||'ko'}catch{}
 if(!supported.includes(locale))locale='ko';
 const escape=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
 const compiled=messages.map(row=>{const keys=[];const source=row[0].split(/(\{\w+\})/).map(part=>/^\{\w+\}$/.test(part)?(keys.push(part.slice(1,-1)),'(.+?)'):escape(part)).join('');return {row,keys,regex:keys.length?new RegExp('^'+source+'$'):null}}).sort((a,b)=>b.row[0].replace(/\{\w+\}/g,'').length-a.row[0].replace(/\{\w+\}/g,'').length);
 const exact=new Map(messages.map(row=>[row[0],row]));
 function t(source,values={}){
  let row=exact.get(source),params=values;
  if(!row){for(const item of compiled){if(!item.regex)continue;const match=source.match(item.regex);if(match){row=item.row;params=Object.fromEntries(item.keys.map((key,i)=>[key,match[i+1]]));break}}}
  const result=row?row[column[locale]]:source;
  return result.replace(/\{(\w+)\}/g,(whole,key)=>Object.hasOwn(params,key)?String(params[key]):whole);
 }
 function protectedNode(node){const parent=node.nodeType===3?node.parentElement:node;return !parent||!!parent.closest('script,style,textarea,[data-no-i18n],.speaker-button');}
 function translateNode(node){
  if(protectedNode(node))return;
  if(node.nodeType===3){const value=node.nodeValue,old=bindings.get(node),source=old&&old.output===value?old.source:value,trimmed=source.trim();if(!trimmed)return;const output=source.replace(trimmed,t(trimmed));if(output!==source||old){bindings.set(node,{source,output});if(value!==output)node.nodeValue=output}}
  else if(node.nodeType===1){for(const attr of attrs){if(!node.hasAttribute(attr))continue;const key=attr,all=bindings.get(node)||{},old=all[key],value=node.getAttribute(attr),source=old&&old.output===value?old.source:value,output=t(source);if(output!==source||old){all[key]={source,output};bindings.set(node,all);if(value!==output)node.setAttribute(attr,output)}}}
 }
 function scan(root){translateNode(root);if(root.nodeType===3)return;const walk=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT|NodeFilter.SHOW_TEXT);while(walk.nextNode())translateNode(walk.currentNode)}
 function set(next){if(!supported.includes(next))return;locale=next;try{localStorage.setItem(storageKey,locale)}catch{}document.documentElement.lang=locale;scan(document.documentElement);for(const b of document.querySelectorAll('[data-ui-locale]')){const selected=String(b.dataset.uiLocale===locale);b.setAttribute('aria-pressed',selected);b.setAttribute('aria-checked',selected)}window.dispatchEvent(new CustomEvent('dubly:localechange',{detail:{locale}}))}
 window.DublyI18n={t,set,get locale(){return locale},get dateLocale(){return {ko:'ko-KR',en:'en-US',ja:'ja-JP'}[locale]}};
 const style=document.createElement('style');style.textContent=`
 .ui-locale{position:relative;display:flex;align-items:center;z-index:50}.ui-locale .locale-toggle{display:grid;place-items:center;width:34px;height:34px;padding:7px;border:1px solid #333;border-radius:8px;background:#151515;color:#aaa;cursor:pointer}.ui-locale .locale-toggle:hover,.ui-locale .locale-toggle[aria-expanded=true]{color:#fff;background:#242424;border-color:#555}.locale-toggle svg{display:block;width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}.ui-locale .locale-menu{position:absolute;right:0;top:calc(100% + 8px);width:160px;z-index:100;background:#1b1b1b;border:1px solid #3d3d3d;border-radius:10px;padding:5px;box-shadow:0 12px 32px #0009}.ui-locale .locale-menu{opacity:1;visibility:visible;transition:opacity .14s ease,visibility 0s}.ui-locale .locale-menu[hidden]{display:block!important;opacity:0;visibility:hidden;pointer-events:none;transition:opacity .14s ease,visibility 0s .14s}.ui-locale .locale-menu button{display:flex;align-items:center;justify-content:space-between;width:100%;height:36px;padding:0 11px;border:0;background:none;color:#aaa;border-radius:5px;font:500 12px Pretendard,sans-serif;cursor:pointer;text-align:left}.ui-locale .locale-menu button:hover,.ui-locale .locale-menu button:focus-visible{background:#303030;color:#fff}.locale-menu button[aria-checked=true]{color:#fff;background:#292929}.locale-menu button[aria-checked=true]:after{content:'✓';font-size:12px;color:#eee}.ui-locale button:focus-visible{outline:1px solid #eee;outline-offset:2px}.loading-locale{position:absolute;right:28px;top:26px}.header-locale{display:flex;align-items:center;gap:18px}@media(max-width:600px){.header-locale .header-note{display:none}.loading-locale{right:20px}}@media(prefers-reduced-motion:reduce){.ui-locale .locale-menu{transition:none!important}}`;document.head.append(style);
 let pickerCount=0;
 function picker(){
  const nav=document.createElement('nav');nav.className='ui-locale';nav.setAttribute('aria-label','화면 언어');
  const toggle=document.createElement('button');toggle.type='button';toggle.className='locale-toggle';toggle.setAttribute('aria-label','화면 언어');toggle.setAttribute('title','화면 언어');toggle.setAttribute('aria-haspopup','menu');toggle.setAttribute('aria-expanded','false');
  toggle.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h12M9 3v2M5 5c1 5 4 8 8 10M13 5c-1 5-4 8-9 11M13 21l4-10 4 10M14.5 17h5"/></svg>';
  const menu=document.createElement('div');menu.className='locale-menu';menu.id='locale-menu-'+(++pickerCount);menu.hidden=true;menu.setAttribute('role','menu');menu.setAttribute('aria-label','화면 언어');toggle.setAttribute('aria-controls',menu.id);
  function close(focus=false){menu.hidden=true;toggle.setAttribute('aria-expanded','false');if(focus)toggle.focus()}
  function open(){menu.hidden=false;toggle.setAttribute('aria-expanded','true')}
  toggle.onclick=()=>{if(menu.hidden)open();else close()};
  for(const code of supported){const button=document.createElement('button');button.type='button';button.textContent={ko:'한국어',en:'English',ja:'日本語'}[code];button.dataset.uiLocale=code;button.dataset.noI18n='';button.lang=code;button.setAttribute('role','menuitemradio');button.setAttribute('aria-label',{ko:'한국어로 보기',en:'View in English',ja:'日本語で表示'}[code]);button.onclick=()=>{set(code);close(true)};menu.append(button)}
  nav.append(toggle,menu);
  nav.onkeydown=e=>{const options=[...menu.querySelectorAll('button')],index=options.indexOf(document.activeElement);if(e.key==='Escape'&&!menu.hidden){e.preventDefault();close(true)}else if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){e.preventDefault();open();const next=e.key==='Home'?0:e.key==='End'?options.length-1:index<0?(e.key==='ArrowUp'?options.length-1:0):(index+(e.key==='ArrowDown'?1:-1)+options.length)%options.length;options[next].focus()}else if(e.key==='Tab')close()};
  document.addEventListener('pointerdown',e=>{if(!nav.contains(e.target))close()});
  nav.addEventListener('focusout',e=>{if(!nav.contains(e.relatedTarget))close()});
  return nav;
 }
 const header=document.querySelector('header');if(header){const box=document.createElement('div');box.className='header-locale';const note=header.querySelector('.header-note');if(note)box.append(note);const share=header.querySelector('#share-open');if(share)box.append(share);const reset=header.querySelector('#queue-reset');if(reset)box.append(reset);box.append(picker());header.append(box);const links=header.querySelector('.header-links');if(links&&!links.children.length)links.remove()}
 const loading=document.getElementById('loading-screen');if(loading){const box=picker();box.classList.add('loading-locale');loading.append(box)}
 const observer=new MutationObserver(records=>{for(const record of records){if(record.type==='childList')for(const node of record.addedNodes)scan(node);else translateNode(record.target)}});
 observer.observe(document.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attrs});set(locale);
})();
