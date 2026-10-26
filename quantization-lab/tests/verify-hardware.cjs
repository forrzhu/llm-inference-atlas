const assert=require('node:assert/strict');
const fs=require('node:fs');
const M=require('../model.js');
const H=require('../hardware-model.js');
const V=require('../hardware-view.js');
const out='/tmp/quantization-research/hardware-checks';fs.mkdirSync(out,{recursive:true});
let traces=0;
for(const r of Object.values(M.recipes))for(const phase of ['prefill','decode'])for(let i=0;i<20;i++){
 const t=H.getTrace(r,phase,i),spec=H.getSpec(r,phase,i);traces++;
 assert(t.events.length>0);assert.equal(t.operator.length,2);
 for(const e of t.events){assert(t.nodes[e.loc]);assert(H.DT[e.dtype]);}
 for(const n of Object.values(t.nodes))for(const x of n.items)assert(H.DT[x.dtype]);
 for(const e of t.edges)assert(V.paths[e.id]);
 for(const key of ['input','storage','operands','accumulator','output'])assert(spec[key]);
 if(i>=2)assert(!t.events.some(e=>e.loc==='weights'&&e.action.startsWith('Write')),'Runtime weights must remain read-only');
 if(i!==6)assert(!t.events.some(e=>e.loc==='kv'&&e.action.startsWith('Write')),'KV append belongs to its own operator');
 if([4,10,13,15].includes(i)){
  const items=t.nodes.registers.items;
  assert.equal(items.find(x=>x.label==='Accumulator C').dtype,r.acc);
  assert.equal(items.find(x=>x.label==='W operand').dtype,r.ab===8?'INT8':'FP16');
  assert.equal(t.nodes.shared.items.find(x=>x.label==='W tile').dtype,r.wb===4?'INT4':r.wb===8?'INT8':'FP16');
 }
 if([7,9].includes(i)){
  assert.equal(t.nodes.tensor.mode,'dependent','Do not hardcode attention to Tensor Cores');
  assert.equal(t.nodes.registers.items.find(x=>x.label.endsWith('operand')).dtype,'FP16');
  assert.equal(t.nodes.registers.items.find(x=>x.label.endsWith('accumulator')).dtype,'FP32');
 }
 if(i===6){assert(t.nodes.kv.note.includes(phase==='decode'?'8 → 9':'0 → 8'));assert.equal(t.nodes.kv.items[0].dtype,r.kb===4?'INT4':r.kb===8?'INT8':'FP16');}
 if(i===18){assert.equal(t.nodes.weights.items[0].dtype,'FP16');assert.equal(t.nodes.tensor.items[1].dtype,'FP32');}
 const reads=t.events.filter(x=>x.action==='Read');assert.equal(new Set(reads.map(x=>x.loc)).size,reads.length,'No duplicate read events');
 if(phase==='decode'&&[0,4,6,7,8,18].includes(i))fs.writeFileSync(`${out}/${r.id}-${i}.html`,V.renderSVG(t));
}
console.log(`PASS: ${traces} hardware traces; immutable weights, KV append, attention mapping, dtype transitions, FP16 LM head, valid event locations.`);
