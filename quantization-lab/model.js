(function(root){
'use strict';
const recipes={
 fp16:{id:'fp16',name:'FP16 reference',tag:'W16 A16 KV16',description:'Floating-point reference path',w:'FP16',a:'FP16',operand:'FP16 × FP16',acc:'FP32',kv:'FP16',wb:16,ab:16,kb:16,unpack:'Use FP16 tiles directly',epilogue:'FP32 accumulator → FP16',note:'Reference FP16 operands and FP32 accumulation; implementations can choose different internal precision.'},
 w4:{id:'w4',name:'W4A16 · Weight-only',tag:'W4 A16 KV16',description:'Compressed storage; FP16 compute operands',w:'packed INT4',a:'FP16',operand:'FP16 × FP16',acc:'FP32',kv:'FP16',wb:4,ab:16,kb:16,unpack:'INT4 unpacking + group scales → FP16',epilogue:'FP32 accumulator → FP16',note:'W4 is the storage format. Weight tiles are reconstructed in FP16 before GEMM; KV remains FP16.'},
 w8:{id:'w8',name:'W8A8 · Integer GEMM',tag:'W8 A8 KV8',description:'INT8 operands, INT32 accumulation',w:'INT8',a:'INT8',operand:'INT8 × INT8',acc:'INT32',kv:'INT8',wb:8,ab:8,kb:8,unpack:'INT8 tiles feed INT8 GEMM directly',epilogue:'INT32 × floating-point scales → FP16',note:'KV8 is an additional choice here. W8A8 alone does not specify KV precision; A8 refers to quantized linear inputs.'},
 qserve:{id:'qserve',name:'QServe · W4A8KV4',tag:'W4 A8 KV4',description:'Progressive W4-to-W8 tile reconstruction',w:'packed INT4',a:'INT8',operand:'INT8 × INT8',acc:'INT32',kv:'packed INT4',wb:4,ab:8,kb:4,unpack:'INT4 → INT8 (progressive group reconstruction)',epilogue:'INT32 × outer weight / token scales → FP16',note:'QServe precision path: W4 → INT8 tiles → INT8 GEMM. Attention query remains high-precision. This does not implement QoQ or SmoothAttention.'}
};
const references=[
 {id:'concepts',name:'Hugging Face · Quantization concepts',url:'https://huggingface.co/docs/transformers/en/quantization/concept_guide',what:'Scale, zero-point, granularity, PTQ and QAT'},
 {id:'schemes',name:'NVIDIA TensorRT · Quantization schemes',url:'https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html',what:'Integer ranges, dequantization and rounding'},
 {id:'cuda',name:'NVIDIA · CUDA Programming Guide',url:'https://docs.nvidia.com/cuda/cuda-programming-guide/',what:'Host / device, SM, global memory, Shared Memory and Registers'},
 {id:'qserve',name:'QServe · original paper (2024)',url:'https://arxiv.org/html/2405.04532v1',what:'Precision path, progressive quantization and high-precision query'},
 {id:'kv',name:'Hugging Face · KV cache quantization',url:'https://huggingface.co/blog/kv-cache-quantization',what:'KV capacity, cache quantization and high-precision residual cache'},
 {id:'flash',name:'FlashAttention · original paper (2022)',url:'https://arxiv.org/abs/2205.14135',what:'Tiled fused attention and reduced traffic between HBM and on-chip SRAM'},
 {id:'trt',name:'NVIDIA · TensorRT-LLM quantization',url:'https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md',what:'Quantization configurations and hardware dependencies'},
 {id:'llama',name:'Llama modeling · Hugging Face Transformers implementation',url:'https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py',what:'Operator order: RMSNorm, RoPE, GQA, SwiGLU and residual connections'}
];
function getSteps(id,phase){
 const E=root.QuantExecution||(typeof require==='function'?require('./execution-model.js'):null),H=root.QuantHardware||(typeof require==='function'?require('./hardware-model.js'):null),model=E.build(id,phase),r=recipes[id];
 return model.stages.map(s=>{const v=H.getSpec(r,phase,s.id);return{id:s.id,title:s.title,en:s.title,kind:[4,10,13,15].includes(s.id)?'weight':[6,7,9].includes(s.id)?'kv':[0,1,17,19].includes(s.id)?'control':'a',input:v.input,output:v.output,stored:v.storage,multiply:v.operands,accumulate:v.accumulator,where:'Logical reference execution: see hardware locations',story:s.explanation,insight:s.goal+(s.id===4?' Attention query remains high-precision.':''),equation:s.equation,inputShape:s.inputs.map(i=>model.entities[i].shape).join('; '),outputShape:s.id===6?`K/V [1, 8, ${model.N}, 128]`:s.outputs.map(i=>model.entities[i].shape).join('; '),refs:['llama','cuda']};});
}
function roundEven(x){const f=Math.floor(x),d=x-f;if(Math.abs(d-.5)<1e-12)return f%2===0?f:f+1;return Math.round(x);}
function quantize(values,bits,range){if(![4,8].includes(bits)||!(range>0))throw new Error('Invalid quantizer');const max=2**(bits-1)-1,s=range/max;const q=values.map(x=>Math.max(-max,Math.min(max,roundEven(x/s))));const reconstructed=q.map(x=>x*s),errors=values.map((x,i)=>reconstructed[i]-x);return {q,reconstructed,errors,s,max,mse:errors.reduce((a,e)=>a+e*e,0)/values.length,maxError:Math.max(...errors.map(Math.abs)),clipped:values.filter(x=>Math.abs(x)>range).length};}
function dot(a,b){return a.reduce((s,x,i)=>s+x*b[i],0);}
function dotExample(){const a=[.7,-.2,.1,1.2],w=[.3,.6,-.4,.8],qa=quantize(a,8,1.2),qw=quantize(w,8,.8),sum=dot(qa.q,qw.q);return{a,w,qa,qw,sum,original:dot(a,w),restored:sum*qa.s*qw.s};}
function memory({p,b,l,h,d,s},r){const weight=p*1e9*r.wb/8,kv=2*b*l*s*h*d*r.kb/8;return{weight,kv,total:weight+kv,weight16:p*1e9*2,kv16:2*b*l*s*h*d*2};}
root.QuantModel={recipes,references,getSteps,roundEven,quantize,dotExample,memory};
if(typeof module!=='undefined')module.exports=root.QuantModel;
})(typeof globalThis!=='undefined'?globalThis:this);
