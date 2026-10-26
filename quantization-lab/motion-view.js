(function(root){
'use strict';
const V=root.QuantHardwareView||(typeof require==='function'?require('./hardware-view.js'):null);
const E=root.QuantExecution||(typeof require==='function'?require('./execution-model.js'):null);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const drawings={
 ssd:'<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M7 7h10M7 11h10M7 17h4"/><circle cx="17" cy="17" r=".8"/>',
 memory:'<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 9v6m4-6v6m4-6v6m4-6v6M6 3v3m6-3v3m6-3v3M6 18v3m6-3v3m6-3v3"/>',
 cpu:'<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M8 2v3m8-3v3M8 19v3m8-3v3M2 8h3m-3 8h3m14-8h3m-3 8h3"/>',
 weights:'<path d="M3 7l9-4 9 4-9 4-9-4Zm0 5 9 4 9-4M3 17l9 4 9-4"/>',
 tensor:'<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 3v18m6-18v18M3 9h18M3 15h18"/>',
 cache:'<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0"/>',
 registers:'<rect x="3" y="3" width="18" height="5" rx="1"/><rect x="3" y="10" width="18" height="5" rx="1"/><path d="M3 18h18M7 5.5h0m0 7h0"/>',
 cuda:'<path d="m13 2-9 12h7l-1 8 10-13h-7l0-7Z"/>',
 load:'<path d="M3 7h13m-4-4 4 4-4 4M21 17H8m4-4-4 4 4 4"/>',
 convert:'<path d="M4 8a8 8 0 0 1 14-2l2 2M20 3v5h-5M20 16a8 8 0 0 1-14 2l-2-2M4 21v-5h5"/>',
 store:'<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
 play:'<path d="m8 4 12 8-12 8V4Z"/>',
 replay:'<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',
 input:'<path d="M14 3h6v18h-6M3 12h13m-4-4 4 4-4 4"/>',
 release:'<circle cx="12" cy="12" r="9"/><path d="M7 12h10"/>',
 control:'<path d="M7 3h10v6H7zM12 9v5M4 21v-7h16v7M2 21h5m10 0h5"/>',
 dot:'<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/>'
};
function icon(name,cls=''){return `<svg class="ui-icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${drawings[name]||drawings.tensor}</svg>`;}
const locationIcons={ssd:'ssd',host:'memory',cpu:'cpu',weights:'weights',activations:'tensor',kv:'cache',l2:'cache',shared:'memory',registers:'registers',cuda:'cuda',tensor:'tensor',attention:'cpu'};
function connections(frame){
 if(['Compute','Convert'].includes(frame.kind)&&frame.unit){
  const unit=frame.unit==='sm'?'attention':frame.unit;
  const input=frame.unit==='cpu'?(frame.from||Object.values(frame.before.cells).find(c=>frame.entities.includes(c.entity)&&c.loc==='cpu')?.loc||'host'):'registers';
  const out=frame.to||(frame.unit==='cpu'?'host':'registers');
  const dtype=Object.values(frame.before.cells).find(c=>frame.entities.includes(c.entity)&&c.loc===input)?.dtype||frame.dtype||'META';
  return [...(input!==unit?[{from:input,to:unit,dtype,kind:'operand',label:'Operand read',lane:-1}]:[]),...(out!==unit?[{from:unit,to:out,dtype:frame.dtype||dtype,kind:'result',label:'Result write',lane:1}]:[])];
 }
 if(frame.from&&frame.to&&frame.from!==frame.to)return[{from:frame.from,to:frame.to,dtype:frame.dtype||'META',kind:'transfer',label:frame.kind,lane:0}];
 return [];
}
function routes(stage){const map=new Map();for(const f of stage.frames)for(const c of connections(f)){const k=c.from+'|'+c.to; if(!map.has(k))map.set(k,c);}return [...map.values()];}
function phaseGroups(stage){
 const frames=stage.frames,last=frames.length-1,groups=[];
 const add=(label,kind,indices)=>groups.push({label,kind,indices});
 if(stage.id===18){
  const norm=frames.findIndex(f=>f.kind==='Compute'),head=frames.findIndex(f=>f.kind==='Compute'&&f.unit==='tensor'),sample=frames.findIndex(f=>f.title==='Promote logits for sampling');
  add('Inputs and loads','Load',Array.from({length:norm},(_,i)=>i));
  add('Final RMSNorm','Compute',Array.from({length:head-norm},(_,i)=>norm+i));
  add('LM head','Compute',Array.from({length:sample-head},(_,i)=>head+i));
  add('Sampling','Compute',Array.from({length:last-sample},(_,i)=>sample+i));add('Output','Output',[last]);return groups;
 }
 add('Inputs','Start',[0]);
 const lastCompute=frames.reduce((a,f,i)=>f.kind==='Compute'?i:a,-1);
 for(let i=1;i<last;i++){
  const f=frames[i];if(f.kind==='Release'){groups.at(-1).indices.push(i);continue;}
  const key=f.kind==='Store'||((f.kind==='Convert'||f.kind==='Load')&&lastCompute>=0&&i>lastCompute)?'Output':f.kind;
  const prev=groups.at(-1);if(prev.kind===key&&key!=='Compute')prev.indices.push(i);
  else add(key==='Compute'&&f.unit==='tensor'?'GEMM':key==='Output'?(stage.id===0?'Checkpoint':'Output'):key,key,[i]);
 }
 if(groups.at(-1).kind==='Output')groups.at(-1).indices.push(last);else add('Output','Output',[last]);
 return groups;
}
function formatList(cells){return [...new Set(cells.map(c=>c.dtype))].map(d=>V.badge(d)).join('');}
function overview(model,stage,cursor){
 const groups=phaseGroups(stage);let html='';
 groups.forEach((g,index)=>{
  const active=g.indices.includes(cursor),done=Math.max(...g.indices)<cursor,frames=g.indices.map(i=>stage.frames[i]),changed=frames.flatMap(f=>f.changes.filter(c=>c.after).map(c=>c.after));
  let lines=[],formats='';
  if(g.kind==='Start'||g.kind==='Output'){
   const ids=g.kind==='Start'?stage.inputs:stage.outputs,snapshot=g.kind==='Start'?stage.start:stage.end;
   const priority=['weights','activations','kv','registers','host','ssd','cpu','l2','shared'];const cells=ids.map(id=>priority.map(loc=>Object.values(snapshot.cells).find(c=>c.entity===id&&c.loc===loc)).find(Boolean)).filter(Boolean);
   lines=[ids.map(id=>model.entities[id].label).join(' / '),[...new Set(cells.map(c=>E.locations[c.loc]))].join(' · ')];formats=formatList(cells);
  }else if(g.kind==='Load'){
   const loads=frames.filter(f=>f.kind==='Load'),sources=[...new Set(loads.map(f=>f.from).filter(l=>l&&!['l2','shared','registers'].includes(l)))],targets=[...new Set(loads.map(f=>f.to).filter(Boolean))],onchip=targets.filter(l=>['l2','shared','registers'].includes(l));
   lines=['Read: '+sources.map(l=>E.locations[l]).join(' / '),(onchip.length?'Via: ':'To: ')+(onchip.length?onchip:targets).map(l=>E.locations[l]).join(' / ')];formats=formatList(changed);
  }else{
   const keyFrames=frames.filter(f=>!['Release','Start','End'].includes(f.kind));
   lines=[...new Set(keyFrames.map(f=>f.title))];
   if(lines.length>2)lines=[lines[0],`${keyFrames.length} ordered actions`];formats=formatList(changed.filter(c=>c.loc==='registers'));
  }
  const iconName={Start:'input',Load:'load',Convert:'convert',Compute:g.label==='GEMM'?'tensor':'cuda',Output:'store',Initialize:'cache',Control:'control',Repeat:'replay'}[g.kind]||'dot';
  if(index)html+=`<span class="overview-connector ${active?'is-current':''} ${done?'is-done':''}" aria-hidden="true"><svg viewBox="0 0 28 20"><path d="M2 10h22m-6-5 6 5-6 5"/><circle cx="7" cy="10" r="2"/></svg></span>`;
  html+=`<button class="overview-phase ${active?'is-current':''} ${done?'is-done':''}" data-action="${g.indices[0]}" data-phase-card="${index}" aria-pressed="${active}" title="${esc(frames.map(f=>f.title).join('; '))}"><span class="overview-phase-heading">${icon(iconName)}<strong>${esc(g.label)}</strong><span>${String(index+1).padStart(2,'0')}</span></span><span class="overview-phase-copy">${lines.map(l=>`<span>${esc(l)}</span>`).join('')}</span><span class="overview-formats">${formats}</span><small>${g.indices[0]===0&&g.indices.length===1?'Start state':g.indices.length===1&&g.indices[0]===stage.frames.length-1?'End state':Math.max(1,g.indices[0])===Math.min(stage.frames.length-2,g.indices.at(-1))?`Action ${Math.max(1,g.indices[0])}`:`Actions ${Math.max(1,g.indices[0])}–${Math.min(stage.frames.length-2,g.indices.at(-1))}`}</small></button>`;
 });
 return `<div class="dataflow-title"><strong>${icon('load')}Operator dataflow</strong><span>Select a stage to inspect its actions</span></div><div class="overview-scroll"><div class="overview-phases">${html}</div></div>`;
}
root.QuantMotion={icon,locationIcons,connections,routes,phaseGroups,overview};if(typeof module!=='undefined')module.exports=root.QuantMotion;
})(typeof globalThis!=='undefined'?globalThis:this);
