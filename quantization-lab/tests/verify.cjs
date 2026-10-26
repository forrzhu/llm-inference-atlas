const assert=require('node:assert/strict');
const M=require('../model.js');
assert.deepEqual([-2.5,-1.5,-.5,.5,1.5,2.5].map(M.roundEven),[-2,-2,0,0,2,2]);
for(const bits of [4,8]){
 const q=M.quantize([-4,-3,0,3,4],bits,3),max=2**(bits-1)-1;
 assert.deepEqual(q.q,[-max,-max,0,max,max]);
 assert.deepEqual(q.reconstructed,[-3,-3,0,3,3]);
 assert.equal(q.clipped,2);assert.equal(q.maxError,1);
 const noClip=M.quantize([-.9,-.21,.21,.9],bits,1);
 assert(noClip.maxError<=noClip.s/2+1e-12);
}
const d=M.dotExample();assert.equal(d.sum,16982);assert(Math.abs(d.original-1.01)<1e-12);assert(Math.abs(d.restored-d.original)<.001);
const v={p:7,b:1,l:32,h:8,d:128,s:2048};
const base=M.memory(v,M.recipes.fp16),q4=M.memory(v,M.recipes.qserve);
assert.equal(base.kv,268435456);assert.equal(q4.kv,67108864);assert.equal(q4.weight,3500000000);
for(const r of Object.values(M.recipes))for(const phase of ['prefill','decode']){
 const s=M.getSteps(r.id,phase);assert.equal(s.length,20);
 s.forEach((x,i)=>{assert.equal(x.id,i);for(const k of ['input','output','stored','multiply','accumulate','where','story','insight','equation'])assert(x[k],`${r.id}/${phase}/${i} ${k}`);for(const id of x.refs)assert(M.references.some(r=>r.id===id));});
 assert(s[4].output.includes('FP16'));assert.equal(s[4].accumulate.split(' ')[0],r.acc);
 assert(s[6].outputShape.includes(phase==='prefill'?'8, 128':'9, 128'));
 if(r.id==='qserve')assert(s[4].insight.includes('query'));
}
console.log('PASS: ties-to-even, clipping, error bound, INT8 dot product, memory formula, and all 8 recipe/phase traces.');
