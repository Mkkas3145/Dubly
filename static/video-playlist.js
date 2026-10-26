/* Split provenance is authoritative; matching filenames never create a playlist. */
(()=>{
 function groupVideos(videos,descending=true){
  const groups=[],byId=new Map();
  for(const video of videos){
   const split=video.split,key=split?.group_id||video.id;
   let group=byId.get(key);
   if(!group){group={id:key,title:split?.title||video.filename,split:!!split,total:split?.parts||1,created:video.created,items:[]};byId.set(key,group);groups.push(group)}
   group.created=Math.max(group.created,video.created);group.items.push(video);
  }
  for(const group of groups)group.items.sort((a,b)=>(a.split?.part||1)-(b.split?.part||1));
  return groups.sort((a,b)=>(descending?-1:1)*(a.created-b.created));
 }
 function adjacent(items,id,direction){const i=items.findIndex(item=>item.id===id);return i<0?null:items[i+direction]||null}
 function nextPlayable(items,id){const next=adjacent(items,id,1);return next?.playable?next:null}
 const api={groupVideos,adjacent,nextPlayable};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 if(typeof window!=='undefined')window.DublyPlaylists=api;
})();
