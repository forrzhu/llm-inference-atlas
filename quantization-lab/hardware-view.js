(function(root){
'use strict';
const H=root.QuantHardware||(typeof require==='function'?require('./hardware-model.js'):null);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const locNames={ssd:'Host / SSD',host:'Host / DRAM',cpu:'Host / CPU',weights:'GPU DRAM / Weights',activations:'GPU DRAM / Activations',kv:'GPU DRAM / KV Cache',l2:'GPU die / L2 Cache',shared:'SM / Shared Memory',registers:'SM / Registers',cuda:'SM / CUDA Cores',tensor:'SM / Tensor Cores',sm:'SM / Attention or layer kernel'};
const positions={ssd:[32,44,245,96,'SSD'],host:[318,44,330,96,'CPU DRAM'],cpu:[690,44,395,96,'CPU'],weights:[49,263,240,142,'Model weights'],activations:[49,432,240,145,'Activations / residual'],kv:[49,604,240,170,'KV cache · layer ℓ'],l2:[354,263,710,84,'L2 Cache'],shared:[372,415,262,119,'Shared Memory'],registers:[658,415,388,153,'Registers'],cuda:[372,629,315,118,'CUDA Cores'],tensor:[729,629,317,118,'Tensor Cores']};
const paths={
 'ssd-host':'M277 92 H314','host-ssd':'M318 92 H281','cpu-host':'M690 92 H652',
 'host-weights':'M483 140 V161 H312 V255 H169 V259','host-input':'M511 140 V165 H323 V424 H169 V428',
 'read-w':'M289 311 H316 V290 H350','read-a':'M289 481 H324 V308 H350','read-kv':'M289 664 H332 V326 H350',
 'l2-shared':'M610 347 V411','l2-reg':'M850 347 V411','shared-reg':'M634 469 H654',
 'reg-tensor':'M890 568 V625','reg-cuda':'M658 545 H645 V604 H530 V625',
 'write-a':'M1046 478 H1072 V785 H312 V532 H293',
 'write-kv':'M1046 505 H1078 V787 H318 V710 H293'
};
const modes={held:'Retained',idle:'Inactive',read:'Read',write:'Write',readwrite:'Read / Write','optional-write':'Read / Write*',append:'Append',compute:'Compute',control:'Control',staging:'Staging',dependent:'Kernel-dependent',attention:'Attention kernel'};
function badge(dtype,label){const d=H.DT[dtype]||H.DT.META;return`<span class="dtype dtype-${dtype.toLowerCase()}" style="--dtype-fill:${d.fill};--dtype-ink:${d.ink};--dtype-stroke:${d.stroke}">${esc(label||d.name)}</span>`;}
function rich(text){return esc(text).replace(/(?:packed )?INT4|INT32|INT8|FP16|FP32|BF16/g,x=>badge(x.includes('INT4')?'INT4':x,x));}
const short=(s,n)=>s.length>n?s.slice(0,n-1)+'…':s;
function svgChip(item,x,y,w){const d=H.DT[item.dtype]||H.DT.META,bw=item.dtype==='META'?75:item.dtype==='INT32'?62:54;const prefix=item.role==='write'?'↳ ':item.role==='read'?'↗ ':'';const max=Math.floor((w-bw-17)/6.7);return`<g class="hw-data ${item.role==='write'?'updated':''}" data-dtype="${item.dtype}"><title>${esc(item.label)} — ${esc(d.name)} (${esc(item.role)})</title><text x="${x}" y="${y+14}" class="hw-data-label">${esc(short(prefix+item.label,max))}</text><rect class="hw-dtype" x="${x+w-bw}" y="${y}" width="${bw}" height="20" rx="4" fill="${d.fill}" stroke="${d.stroke}"/><text x="${x+w-bw/2}" y="${y+14}" text-anchor="middle" fill="${d.ink}" class="hw-dtype-label">${esc(d.name)}</text></g>`;}
function node(id,s){const [x,y,w,h,title]=positions[id],active=!['idle','held'].includes(s.mode),isCompute=['cpu','cuda','tensor'].includes(id);let text=`<g class="hw-node ${active?'active':''} state-${s.mode} ${isCompute?'compute-node':''}" data-location="${id}" aria-label="${esc(locNames[id])}: ${modes[s.mode]}. ${esc(s.note)}"><title>${esc(locNames[id])}: ${esc(s.items.map(x=>x.label+' '+x.dtype).join('; '))}</title><rect class="node-frame" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text x="${x+12}" y="${y+22}" class="hw-node-title">${esc(title)}</text>`;
 const status=modes[s.mode],statusWidth=status.length*5.3+12;text+=`<rect class="node-status-bg" x="${x+w-statusWidth-10}" y="${y+8}" width="${statusWidth}" height="18" rx="3"/><text class="node-status" x="${x+w-16}" y="${y+21}" text-anchor="end">${esc(status)}</text>`;
 const horizontal=id==='l2';if(horizontal){s.items.forEach((a,i)=>{text+=svgChip(a,x+13+i*245,y+34,228);});}
 else s.items.forEach((a,i)=>{text+=svgChip(a,x+12,y+35+i*23,w-24);});
 const noteY=y+h-11,limit=Math.floor((w-24)/6);text+=`<text x="${x+12}" y="${noteY}" class="hw-node-note">${esc(short(s.note,limit))}</text></g>`;return text;
}
function renderSVG(t){let html=`<svg class="hardware-svg" viewBox="0 0 1120 836" role="img" aria-labelledby="hardware-svg-title hardware-svg-desc"><title id="hardware-svg-title">${esc(t.operator[0])}: tensor locations and dtypes</title><desc id="hardware-svg-desc">Operator-level reference model. Read, compute and write locations are highlighted. ${esc(t.events.map(e=>e.action+' '+locNames[e.loc]+': '+e.tensor+' '+e.dtype).join('. '))}. Not a cycle-accurate trace.</desc><defs>`;
 for(const [key,d]of Object.entries(H.DT))html+=`<marker id="hw-arrow-${key}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L8 5 L1 9" fill="none" stroke="${d.ink}" stroke-width="1.5"/></marker>`;
 html+='</defs><g class="hw-regions"><rect class="host-region" x="16" y="12" width="1088" height="139" rx="10"/><text x="33" y="32" class="region-title">Host</text><rect class="gpu-region" x="16" y="178" width="1088" height="636" rx="12"/><text x="35" y="203" class="region-title">GPU device</text><text x="1087" y="202" text-anchor="end" class="region-note">Logical hierarchy · not to scale</text><rect class="dram-region" x="35" y="226" width="270" height="568" rx="9"/><text x="49" y="248" class="region-title">GPU DRAM · HBM / GDDR</text><rect class="die-region" x="334" y="226" width="750" height="568" rx="9"/><text x="350" y="248" class="region-title">GPU die · on-chip</text>';
 html+=`<rect class="sm-region ${['idle','control'].includes(t.nodes.sm.mode)?'':'active'}" x="354" y="373" width="710" height="401" rx="10"/><text x="372" y="398" class="region-title">Streaming Multiprocessor (SM)</text><text x="1045" y="398" text-anchor="end" class="region-note">${esc(modes[t.nodes.sm.mode])}</text></g><g class="hw-connections" fill="none">`;
 const active=new Map(t.edges.map(e=>[e.id,e]));for(const [id,p]of Object.entries(paths)){const e=active.get(id),dtype=e?.dtype||'META',dt=H.DT[dtype]||H.DT.META;html+=`<path class="hw-connection ${e?'active':''} ${e?.write?'write-edge':''}" data-edge="${id}" d="${p}" fill="none" stroke="${e?dt.ink:'#a8b8c9'}" stroke-width="${e?2.5:1.2}" ${e?.write?'stroke-dasharray="7 4"':''} marker-end="url(#hw-arrow-${dtype})"/>`;}
 html+='</g>';for(const id of Object.keys(positions))html+=node(id,t.nodes[id]);
 html+='</svg>';return html;
}
function renderEvents(t){return `<div class="event-heading"><strong>Data movement</strong><span>Read → Compute / Convert → Write</span></div><div class="hardware-events">${t.events.map(e=>`<div class="hardware-event"><span class="event-action">${esc(e.action)}</span><span class="event-location">${esc(locNames[e.loc])}</span><span class="event-tensor">${esc(e.tensor)} ${badge(e.dtype)}${e.note?`<small>${esc(e.note)}</small>`:''}</span></div>`).join('')}</div>`;}
root.QuantHardwareView={badge,rich,renderSVG,renderEvents,locNames,positions,paths};
if(typeof module!=='undefined')module.exports=root.QuantHardwareView;
})(typeof globalThis!=='undefined'?globalThis:this);
