const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const context=vm.createContext({window:{}});vm.runInContext(fs.readFileSync('static/project-share.js','utf8'),context);const Client=context.window.DublySharedProjectsClient;
const code='DUBLY-'+'A'.repeat(32),key='dubly-shared-projects-v1';
function browser(request,ids=[]){const data=new Map(),local=[...ids];return {local,data,client:new Client({storage:{getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v),removeItem:k=>data.delete(k)},history:{ids:()=>[...local],merge:ids=>{for(const id of ids)if(!local.includes(id))local.push(id)}},request})}}

test('a code connects two browsers; new additions sync both ways without publishing unrelated local history',async()=>{
 const shared=new Set(),calls=[];const request=async(url,body)=>{calls.push(body);if(url.endsWith('/create')){body.ids.forEach(id=>shared.add(id));return {code,ids:[...shared]}}assert.equal(body.code,code);body.ids.forEach(id=>shared.add(id));return {ids:[...shared]}};
 const a=browser(request,['first']),b=browser(request,['private']);await a.client.create();await b.client.connect(code);
 assert.deepEqual(b.local,['private','first']);assert(!shared.has('private'));
 a.client.remember('second');await a.client.sync();await b.client.sync();assert(b.local.includes('second'));
 b.client.remember('third');await b.client.sync();await a.client.sync();assert(a.local.includes('third'));assert(!shared.has('private'));
});

test('hiding stays local, survives later pulls, and explicit remembering restores that project',async()=>{
 const request=async()=>({ids:['first','second']});const b=browser(request);await b.client.connect(code);b.client.hide('first');b.local.splice(b.local.indexOf('first'),1);await b.client.sync();assert.deepEqual(b.local,['second']);
 b.client.remember('first');await b.client.sync();assert(b.local.includes('first'));
});

test('new additions during an in-flight sync are retained for the next request',async()=>{
 let release;const b=browser(async()=>({ids:[]}));await b.client.connect(code);b.client.remember('a');b.client.request=()=>new Promise(resolve=>release=resolve);const pending=b.client.sync();b.client.remember('b');release({ids:['a']});await pending;
 assert.deepEqual(Array.from(b.client.state.pending),['b']);
});

test('disconnect prevents an in-flight response from reconnecting or importing more videos',async()=>{
 let release;const b=browser(async()=>({ids:[]}));await b.client.connect(code);b.client.request=()=>new Promise(resolve=>release=resolve);const pending=b.client.sync();b.client.disconnect();release({ids:['late']});await pending;
 assert.equal(b.client.state,null);assert.deepEqual(b.local,[]);
});

test('network failure preserves unsent additions and failed import preserves the existing connection',async()=>{
 const b=browser(async()=>({ids:[]}));await b.client.connect(code);b.client.remember('new');b.client.request=async()=>{throw Error('offline')};await assert.rejects(b.client.sync(),/offline/);assert.deepEqual(Array.from(b.client.state.pending),['new']);
 await assert.rejects(b.client.connect('DUBLY-'+'B'.repeat(32)),/offline/);assert.equal(b.client.state.code,code);
});
