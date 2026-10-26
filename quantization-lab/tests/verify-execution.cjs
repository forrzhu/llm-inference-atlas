const assert=require('node:assert/strict');
const fs=require('node:fs');
const M=require('../model.js'),H=require('../hardware-model.js'),E=require('../execution-model.js'),X=require('../execution-view.js');
let actions=0,boundaries=0;
for(const r of Object.values(M.recipes))for(const phase of ['prefill','decode']){
 const m=E.build(r.id,phase);assert.equal(m.stages.length,20);
 for(const [i,s]of m.stages.entries()){
  assert.equal(s.id,i);assert.equal(s.frames[0].kind,'Start');assert.equal(s.frames.at(-1).kind,'End');
  if(i){assert.strictEqual(s.start,m.stages[i-1].end);boundaries++;}
  for(const [j,f]of s.frames.entries()){
   if(j)assert.strictEqual(f.before,s.frames[j-1].snapshot,'No discontinuity between actions');
   for(const c of Object.values(f.snapshot.cells)){assert(m.entities[c.entity]);assert(E.locations[c.loc]);assert(H.DT[c.dtype]);assert.equal(c.key,[c.entity,c.loc,c.dtype].join('|'));}
   if(f.kind==='Load'&&f.from)for(const id of f.entities){assert(Object.values(f.before.cells).some(c=>c.entity===id&&c.loc===f.from),`Every load needs a source: ${r.id}/${phase}/${i}/${j}/${id}`);}
   if(i>=2)for(const c of Object.values(m.stages[1].end.cells).filter(c=>c.loc==='weights'))assert.deepEqual(f.snapshot.cells[c.key],c,'Resident weights and parameters must remain immutable');
   const html=X.diagram(m,f,f.snapshot,null);assert(!html.includes('…</span>'),'No truncated tensor label');assert(!/[\p{Script=Han}]/u.test(html));
   assert(html.includes('data-memory="registers"'));assert(X.delta(m,f));actions++;
  }
 }
 for(const [id,input]of [[4,'u'],[10,'oh'],[13,'u2'],[15,'f']]){
  const s=m.stages[id],f=s.frames.find(f=>f.kind==='Compute'&&f.unit==='tensor');assert(f);
  const reg=Object.values(f.before.cells).filter(c=>c.loc==='registers');
  assert.equal(reg.find(c=>c.entity==='W').dtype,r.ab===8?'INT8':'FP16');assert.equal(reg.find(c=>c.entity===input).dtype,r.ab===8?'INT8':'FP16');
  for(const out of s.outputs)assert.equal(Object.values(f.snapshot.cells).find(c=>c.entity===out).dtype,r.acc);
  for(const out of s.outputs)assert(Object.values(s.end.cells).some(c=>c.entity===out&&c.loc==='activations'&&c.dtype==='FP16'));
  if(r.ab===8)assert(reg.some(c=>c.entity==='ascale'&&c.dtype==='META'));
 }
 for(const id of ['kc','vc']){const a=Object.values(m.stages[6].start.cells).find(c=>c.entity===id&&c.loc==='kv'),b=Object.values(m.stages[6].end.cells).find(c=>c.entity===id&&c.loc==='kv');assert(a.shape.includes(`, ${m.S}, `));assert(b.shape.includes(`, ${m.N}, `));assert.equal(b.dtype,m.KV);}
 assert(!Object.values(m.stages[6].end.cells).some(c=>['q','qr'].includes(c.entity)&&c.loc==='kv'));
 assert(Object.values(m.stages[5].end.cells).some(c=>c.entity==='v'&&c.loc==='activations'&&c.dtype==='FP16'),'RoPE retains V');
 for(const i of [3,4,5,6,7,8,9,10])assert(Object.values(m.stages[i].end.cells).some(c=>c.entity==='x'&&c.loc==='activations'),'First residual survives until residual add');
 for(const i of [12,13,14,15])assert(Object.values(m.stages[i].end.cells).some(c=>c.entity==='r'&&c.loc==='activations'),'Second residual survives until residual add');
 assert(Object.values(m.stages[7].end.cells).some(c=>c.entity==='score'&&c.loc==='registers'&&c.dtype==='FP32'));
 assert(!Object.values(m.stages[7].end.cells).some(c=>c.entity==='score'&&c.loc==='activations'));
 assert(Object.values(m.stages[8].end.cells).some(c=>c.entity==='prob'&&c.loc==='registers'&&c.dtype==='FP16'));
 assert(Object.values(m.stages[19].end.cells).some(c=>c.entity==='nextid'&&c.dtype==='INT32'));
 for(const id of ['W','kc','vc','q','prob','nextid']){const history=E.history(m,id);assert(history.length);for(const h of history)assert.strictEqual(m.stages[h.stage].frames[h.frame].snapshot,h.snapshot);assert(X.inspector(m,id,6,0).includes('Lifecycle in this forward'));}
}
for(const f of ['index.html','app.js','model.js','hardware-model.js','execution-model.js','execution-view.js','research.md','research.txt','hardware.html'])assert(!/[\p{Script=Han}]/u.test(fs.readFileSync(require('node:path').join(__dirname,'..',f),'utf8')),`${f} must be English`);
console.log(`PASS: ${actions} frames across all 8 configurations; ${boundaries} exact operator boundaries; continuous action states, valid inputs, immutable model weights, residual lifetimes, KV append, on-chip attention, GEMM precision, tensor history and English-only content.`);
