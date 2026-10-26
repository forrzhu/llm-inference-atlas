const assert=require('node:assert/strict');
const M=require('../model.js'),E=require('../execution-model.js'),A=require('../motion-view.js'),V=require('../execution-view.js');
let operators=0,arrows=0;
for(const r of Object.keys(M.recipes))for(const phase of ['prefill','decode']){
 const model=E.build(r,phase);
 for(const s of model.stages){
  const groups=A.phaseGroups(s);assert(groups.length>=3&&groups.length<=5,`${s.title}: ${groups.length} readable phases`);
  assert.deepEqual(groups.flatMap(g=>g.indices),s.frames.map((_,i)=>i),'Every action is represented exactly once and in order');
  for(const [i,f]of s.frames.entries()){
   const connections=A.connections(f);for(const c of connections){assert(c.from!==c.to);assert(E.locations[c.from]||c.from==='attention');assert(E.locations[c.to]||c.to==='attention');arrows++;}
   if(f.kind==='Convert'&&f.unit==='cuda'){assert.equal(connections.length,2);assert.equal(connections[0].from,'registers');assert.equal(connections[0].to,'cuda');assert.equal(connections[1].to,'registers');}
   if(f.kind==='Compute'&&f.unit==='tensor'){assert.equal(connections.length,2);assert.equal(connections[0].to,'tensor');assert.equal(connections[1].from,'tensor');}
   if(f.kind==='Compute'&&f.unit==='sm'){assert(connections.every(c=>c.from!=='tensor'&&c.to!=='tensor'),'Attention mapping must stay kernel-dependent');}
   if(['Start','End','Release','Initialize','Control','Repeat'].includes(f.kind))assert.equal(connections.length,0,'Do not invent transfers for boundaries or control');
   const before=JSON.stringify(f.snapshot);assert(A.overview(model,s,i).includes('overview-connector'));assert(V.diagram(model,f,f.snapshot).includes('ui-icon'));assert.equal(JSON.stringify(f.snapshot),before,'Rendering must not change live state');
  }
  assert(A.routes(s).length>0||[8,17,19].includes(s.id));operators++;
 }
}
console.log(`PASS: ${operators} operator overviews, 3–5 phases each; ${arrows} movement / operand / result arrows; complete action coverage, immutable snapshots and kernel-dependent attention.`);
