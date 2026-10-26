const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const script=fs.readFileSync('static/app.js','utf8');
const context={};vm.runInNewContext(script.slice(script.indexOf('function adoptUploadedParts('),script.indexOf('async function processBatch('))+';this.adopt=adoptUploadedParts;',context);
test('split uploads expand both lists in order and keep language/preset choices',()=>{
 const a={name:'Long.mp4',source:'ja',target:'ko',preset:'creative'},b={name:'Next.mp4'};
 const items=[a,b],queue=[a,b];
 const result={split:true,videos:[{id:'a',filename:'Long (1).mp4'},{id:'b',filename:'Long (2).mp4'},{id:'c',filename:'Long (3).mp4'}]};
 const added=context.adopt(a,result,items,queue);
 assert.equal(added.length,3);assert.deepEqual(items.map(x=>x.name),['Long (1).mp4','Long (2).mp4','Long (3).mp4','Next.mp4']);
 assert.deepEqual(queue.map(x=>x.id),['a','b','c',undefined]);
 assert.equal(items[1],queue[1]);assert.equal(items[1].source,'ja');assert.equal(items[1].preset,'creative');assert.equal(items[1].splitGroup,'a');
});
test('short upload stays a single item and cancelling parent marks all new children',()=>{
 const item={name:'Short.mp4',cancelled:true},items=[item];
 context.adopt(item,{id:'one',filename:'Short.mp4'},items,items);assert.equal(items.length,1);
 context.adopt(item,{split:true,videos:[{id:'x',filename:'A (1).mp4'},{id:'y',filename:'A (2).mp4'}]},items,items);
 assert.equal(items.length,2);assert.equal(items[1].cancelled,true);
});
