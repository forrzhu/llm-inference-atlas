(function(root){
'use strict';
const DT={FP16:{name:'FP16',fill:'#e6efff',ink:'#2459aa',stroke:'#7fa4da'},FP32:{name:'FP32',fill:'#ddf2ec',ink:'#156553',stroke:'#77b7a5'},INT4:{name:'INT4',fill:'#eee4fb',ink:'#703da2',stroke:'#ad8bcf'},INT8:{name:'INT8',fill:'#fff0d2',ink:'#875200',stroke:'#cdab63'},INT32:{name:'INT32',fill:'#fae5de',ink:'#91492f',stroke:'#cf927e'},META:{name:'metadata',fill:'#e9edf2',ink:'#526176',stroke:'#b0bccb'}};
const LABELS=[
 ['Weight quantization & packing','权重量化与打包'],['Model loading','模型加载'],['Tokenization & embedding','分词与嵌入'],['Input RMSNorm','Attention 前的 RMSNorm'],['QKV projections','Q / K / V 投影'],['Rotary position embedding (RoPE)','位置编码'],['KV cache append','追加当前 K / V'],['Attention scores: QKᵀ','计算 attention scores'],['Softmax','归一化 attention scores'],['Attention output: PV','对 Value 加权求和'],['Attention output projection','Attention 输出投影'],['Residual add (attention)','Attention 残差连接'],['Post-attention RMSNorm','FFN 前的 RMSNorm'],['FFN gate / up projections','FFN 的两个输入投影'],['SiLU & elementwise product','FFN 门控'],['FFN down projection','FFN 输出投影'],['Residual add (FFN)','FFN 残差连接'],['Remaining Transformer layers','其余 Transformer 层'],['Final RMSNorm / LM head / sampling','输出 token'],['Autoregressive decode loop','下一轮 Decode']
];
function operator(id,r,phase){if(id===0&&r.wb===16)return['FP16 checkpoint preparation','准备 FP16 checkpoint'];if(id===2&&phase==='decode')return['Decode input & embedding','Decode token 的 embedding'];return LABELS[id];}
function getTrace(r,phase,id){
 const T=phase==='prefill'?8:1,S=phase==='prefill'?0:8,n=S+T,W=r.wb===4?'INT4':r.wb===8?'INT8':'FP16',KV=r.kb===4?'INT4':r.kb===8?'INT8':'FP16';
 const item=(label,dtype,role='held')=>({label,dtype,role});
 const nodes={
  ssd:{mode:'held',items:[item('Checkpoint W',W)],note:'Persistent checkpoint'},
  host:{mode:'idle',items:[],note:'No selected host transfer'},
  cpu:{mode:'idle',items:[],note:'Tokenizer / scheduler'},
  weights:{mode:id>=1?'held':'idle',items:id>=1?[item('Linear W',W),item('Embedding / γ / head','FP16'),...(r.wb<16?[item('Scales / zero-points','META')]:[])]:[],note:id>=1?'Resident across requests':'Not loaded'},
  activations:{mode:id>=2?'held':'idle',items:id>=2?[item('Residual stream X','FP16')]:[],note:id>=2?'Current forward buffers':'Not allocated'},
  kv:{mode:id>=2?'held':'idle',items:id>=2?[item('K cache',KV),item('V cache',KV),...(r.kb<16?[item('Scales / zero-points','META')]:[])]:[],note:id<2?'Not allocated':`Layer ℓ · S = ${id>=6?n:S}`},
  l2:{mode:'idle',items:[],note:'No selected tile'},shared:{mode:'idle',items:[],note:'Kernel-dependent staging'},registers:{mode:'idle',items:[],note:'Thread-local tiles / scalars'},cuda:{mode:'idle',items:[],note:'Elementwise / reductions / Q/DQ'},tensor:{mode:'idle',items:[],note:'Matrix multiply-accumulate'},sm:{mode:'idle',items:[],note:'One illustrative SM'}
 };
 const events=[],edges=[];
 const set=(loc,mode,items,note)=>Object.assign(nodes[loc],{mode,items,note});
 const event=(action,loc,tensor,dtype,note='')=>events.push({action,loc,tensor,dtype,note});
 const edge=(id,dtype,write=false)=>edges.push({id,dtype,write});
 const read=(loc,items)=>{set(loc,'read',items,nodes[loc].note);event('Read',loc,items.filter(x=>x.role==='read').map(x=>x.label).join(', '),(items.find(x=>x.role==='read')||items[0])?.dtype||'META');};
 const gpuInput=(label)=>{read('activations',[item(label,'FP16','read')]);set('l2','read',[item('Activation tile','FP16','read')],'Cache hit / fill depends on locality');edge('read-a','FP16');edge('l2-reg','FP16');};
 const output=(label)=>{const input=nodes.activations.items.filter(x=>x.role==='read').slice(0,1);set('activations','optional-write',[...input,item(label,'FP16','write')],'Read → write*; fusion may retain output on-chip');event('Write*','activations',label,'FP16','Optional materialization');edge('write-a','FP16',true);};
 const linear=(name,out)=>{
  read('weights',[item(name,W,'read'),item('Embedding / γ / head','FP16'),...(r.wb<16?[item('Scales / zero-points','META')]:[])]);
  read('activations',[item('GEMM input','FP16','read')]);
  set('l2','read',[item('W tile',W,'read'),item('A tile','FP16','read')],'Illustrative global-memory read path');
  set('shared','staging',[item('W tile',W),item('A tile','FP16')],'Example tiled GEMM staging');
  set('registers','compute',[item('A operand',r.ab===8?'INT8':'FP16','compute'),item('W operand',r.ab===8?'INT8':'FP16','compute'),item('Accumulator C',r.acc,'write'),item('Output tile','FP16','write')],'Operand and accumulator registers');
  set('cuda','compute',[item('Output tile','FP16','write')],r.id==='qserve'?'W4→W8; A quantization; epilogue':r.id==='w4'?'W4→FP16; epilogue':r.id==='w8'?'A quantization; output rescaling':'FP32→FP16 output conversion');
  set('tensor','compute',[item('A × W',r.ab===8?'INT8':'FP16','compute'),item('Accumulator',r.acc,'write')],'Matrix multiply-accumulate');
  set('sm','compute',[],'Tiled GEMM: Q/DQ → MMA → epilogue');
  event('Read','weights',name,W);event('Read','activations','GEMM input','FP16');
  if(r.wb===4)event('Convert','registers',r.id==='qserve'?'W4 → W8':'W4 → W16',r.id==='qserve'?'INT8':'FP16',r.id==='qserve'?'Progressive group dequantization':'Group scale / zero-point');
  if(r.ab===8)event('Convert','registers','A16 → A8','INT8','Per-token activation scale');
  event('Compute','tensor','A × W → C',r.acc);event('Convert','cuda',r.ab===8?'INT32 accumulator × scales → FP16':'FP32 accumulator → FP16','FP16','GEMM epilogue');output(out);
  edge('read-w',W);edge('read-a','FP16');edge('l2-shared',W);edge('shared-reg',W);edge('reg-tensor',r.ab===8?'INT8':'FP16');edge('reg-cuda','FP16');
 };
 switch(id){
 case 0:
  set('cpu','compute',[],r.wb===16?'Checkpoint serialization':'Weight quantization + packing');
  set('host','readwrite',[item('Source W','FP16','read'),item('Checkpoint W',W,'write')],'Offline example: FP16 source weights');
  set('ssd','write',[item('Checkpoint W',W,'write')],r.wb<16?'Packed values + quantization metadata':'FP16 checkpoint');
  event('Read','host','Source W','FP16');event('Compute','cpu',r.wb===16?'Prepare checkpoint':'Quantize / pack W',W);event('Write','host','Checkpoint W',W);event('Write','ssd','Checkpoint W',W);
  edge('cpu-host',W);edge('host-ssd',W,true);break;
 case 1:
  read('ssd',[item('Checkpoint W',W,'read')]);set('host','readwrite',[item('Staging buffer W',W)],'Common host-staging path');
  set('cpu','control',[],'Initialize runtime / transfer');set('weights','write',[item('Linear W',W,'write'),item('Embedding / γ / head','FP16','write'),...(r.wb<16?[item('Quantization parameters','META','write')]:[])],'Resident GPU parameters');
  event('Transfer','host','Checkpoint W',W);event('Write','weights','Linear W',W,'Initialization; not repeated per token');event('Write','weights','Embedding / norm / LM head','FP16');
  edge('ssd-host',W);edge('host-weights',W,true);break;
 case 2:
  if(phase==='prefill'){set('cpu','compute',[item('Token IDs','INT32','write')],'Tokenization');set('host','write',[item('Token IDs','INT32','write')],'INT32 IDs in this illustration');event('Compute','cpu','Tokenization','INT32');set('activations','read',[item('Token IDs','INT32','read')],'Input ID buffer');event('Transfer','host','Token IDs → GPU DRAM','INT32');event('Read','activations','Token IDs','INT32');edge('cpu-host','INT32');edge('host-input','INT32',true);}
  else{set('activations','read',[item('Previous sampled ID','INT32','read')],'Decode input; no tokenization needed');event('Read','activations','Sampled token ID','INT32');}
  read('weights',[item('Embedding table','FP16','read'),item('Linear W (retained)',W)]);
  set('l2','read',[item('Embedding rows','FP16','read')],'Cache-assisted lookup');
  set('registers','compute',[item('Token IDs','INT32'),item('Embedding tile','FP16','write')],'Gather selected embedding rows');
  set('cuda','compute',[item('Embedding output','FP16','write')],'Embedding gather / loads');set('sm','compute',[],'Embedding lookup');
  event('Read','weights','Embedding rows','FP16');event('Compute','cuda','Embedding gather','FP16');output('X = Embedding(IDs)');edge('read-w','FP16');edge('l2-reg','FP16');edge('reg-cuda','FP16');break;
 case 3:case 12:
  gpuInput(id===3?'Layer input X':'Attention residual R');read('weights',[item('RMSNorm γ','FP16','read'),item('Linear W (retained)',W)]);
  set('l2','read',[item('X tile','FP16'),item('γ','FP16')],'Activation / parameter tiles');
  set('registers','compute',[item('Input tile','FP16'),item('Mean(x²) / statistics','FP32','write'),item('Normalized tile','FP16','write')],'FP32 reduction; FP16 output boundary');
  set('cuda','compute',[item('Reduction statistics','FP32','write'),item('Normalized output','FP16','write')],'RMSNorm');set('sm','compute',[],'RMSNorm reduction and affine transform');
  event('Read','weights','γ','FP16');event('Compute','cuda','RMSNorm statistics','FP32');output('RMSNorm output');edge('read-w','FP16');edge('reg-cuda','FP32');break;
 case 4:linear('WQ / WK / WV','Q, K, V');break;
 case 5:
  gpuInput('Q, K, V');set('registers','compute',[item('Q, K input','FP16'),item('Q′, K′ output','FP16','write'),item('V (unchanged)','FP16')],'Position-dependent rotation');
  set('cuda','compute',[item('Q′, K′ output','FP16','write')],'RoPE on Q and K');set('sm','compute',[],'RoPE; V is unchanged');event('Compute','cuda','RoPE(Q, K)','FP16','Boundary dtype; internal precision varies');output('Q′, K′, V');edge('reg-cuda','FP16');break;
 case 6:
  gpuInput('Current K′, V');set('registers','compute',[item('Current K′, V','FP16'),item('Cache entries',KV,'write'),...(r.kb<16?[item('Quantization parameters','META','write')]:[])],'Current-token KV conversion');
  set('cuda','compute',[item('New cache entries',KV,'write')],r.kb===16?'KV append / copy':'KV quantization + packing');set('sm','compute',[],'Append current K/V to the layer cache');
  set('kv','append',[item('K cache',KV,'write'),item('V cache',KV,'write'),...(r.kb<16?[item('Quantization parameters','META','write')]:[])],`Layer ℓ · S: ${S} → ${n} tokens`);
  event('Retain','kv',`Past K/V (${S} tokens)`,KV);if(r.kb<16)event('Convert','registers',`K/V: FP16 → ${KV}`,KV);event('Write','kv',`Append ${T} token(s): S ${S} → ${n}`,KV);
  edge('reg-cuda',KV);edge('write-kv',KV,true);break;
 case 7:case 9:{
  const k=id===7,label=k?'K cache':'V cache';
  read('kv',[item(label,KV,'read'),item(k?'V cache (retained)':'K cache (retained)',KV)]);nodes.kv.note=`Layer ℓ · S = ${n} tokens`;
  set('l2','read',[item(k?'K tile':'V tile',KV,'read'),...(k?[item('Q tile','FP16','read')]:[])],'Cache traffic depends on kernel locality');
  if(k)read('activations',[item('Q′','FP16','read'),item('V (retained)','FP16')]);
  set('registers','compute',[item(k?'Q operand':'P operand','FP16'),...(KV==='FP16'?[]:[item(k?'K cache codes':'V cache codes',KV)]),item(k?'K compute tile':'V compute tile','FP16','write'),item(k?'Score accumulator':'Output accumulator','FP32','write')],'Floating-point attention example');
  set('cuda','compute',[item(k?'Score / mask':'Output conversion','FP32','write')],r.kb<16?'KV dequantization + scalar operations':k?'Score scaling / causal mask':'Output conversion');
  set('tensor','dependent',[item(k?'Q × Kᵀ':'P × V','FP16'),item('Accumulator','FP32')],'Dot-product unit is kernel-dependent');set('sm','attention',[],'Attention kernel: CUDA / Tensor mapping varies');
  event('Read','kv',label,KV);if(k)event('Read','activations','Q′','FP16');else event('Read','registers','Probability tile P','FP16','Retained from fused softmax');
  if(r.kb<16)event('Convert','registers',`${k?'K':'V'}: ${KV} → FP16`,'FP16');event('Compute','sm',k?'QKᵀ → FP32 scores; scale / mask':'P × V → FP32 accumulator','FP32','Execution-unit mapping depends on attention kernel');
  edge('read-kv',KV);if(k)edge('read-a','FP16');edge('l2-reg',KV);edge('reg-cuda','FP32');
  if(!k)output('Attention output O');break;
 }
 case 8:
  set('registers','readwrite',[item('Score tile','FP32','read'),item('Max / sum / exp','FP32','write'),item('P tile for PV','FP16','write')],'Online softmax can retain tiles on-chip');
  set('cuda','compute',[item('Softmax statistics','FP32','write'),item('P tile','FP16','write')],'Softmax reduction / normalization');set('sm','compute',[],'Softmax; no full score matrix required in DRAM');
  event('Read','registers','Score tile','FP32');event('Compute','cuda','Softmax / reduction','FP32');event('Write','registers','P tile for PV','FP16','On-chip tile; no required full DRAM matrix');edge('reg-cuda','FP32');break;
 case 10:linear('WO','Attention projection output');break;
 case 11:case 16:
  gpuInput(id===11?'X + attention projection':'R + FFN output');
  set('registers','readwrite',[item('Residual input','FP16'),item('Branch output','FP16'),item('Residual result','FP16','write')],'FP16 input / output boundaries');set('cuda','compute',[item('Residual result','FP16','write')],'Elementwise residual add');set('sm','compute',[],'Residual add; may fuse with adjacent kernels');
  event('Compute','cuda','Residual add','FP16','Boundary dtype; arithmetic can vary');output(id===11?'Residual R':'Layer output Xnext');edge('reg-cuda','FP16');break;
 case 13:linear('Wgate / Wup','Gate G / Up');break;
 case 14:
  gpuInput('Gate G, Up');set('registers','readwrite',[item('Gate G / Up','FP16'),item('FFN activation F','FP16','write')],'FP16 boundaries; intermediates depend on kernel');set('cuda','compute',[item('F = SiLU(G) × Up','FP16','write')],'SiLU + elementwise product');set('sm','compute',[],'SwiGLU activation');event('Compute','cuda','SiLU(G) ⊙ Up','FP16','Output boundary');output('FFN activation F');edge('reg-cuda','FP16');break;
 case 15:linear('Wdown','FFN output D');break;
 case 17:
  set('weights','held',[item('W for layers ℓ+1 … L',W),item('Norm parameters','FP16')],'Distinct parameters for each layer');set('kv','held',[item('K cache per layer',KV),item('V cache per layer',KV)],`Each layer: S ${S} → ${n}`);set('activations','held',[item('Layer output Xnext','FP16')],'Repeated operator sequence');set('sm','control',[],'Repeat RMSNorm → attention → FFN per layer');
  event('Repeat','sm','Remaining layer operators','META','Each layer has its own W and KV cache');break;
 case 18:
  read('weights',[item('Final norm γ / LM head','FP16','read'),item('Other linear W',W)]);read('activations',[item('Last-position hidden state','FP16','read')]);
  set('l2','read',[item('LM head tile','FP16'),item('Hidden-state tile','FP16')],'FP16 LM head in this model');set('shared','staging',[item('LM head / A tile','FP16')],'Example tiled GEMM');
  set('registers','compute',[item('LM-head accumulator','FP32','write'),item('Logit tile','FP16','write'),item('Sampling computation','FP32','write'),item('Sampled token ID','INT32','write')],'Final norm → LM head → sampling');
  set('cuda','compute',[item('Norm / sampling','FP32'),item('Token ID','INT32','write')],'Final RMSNorm / sampling');set('tensor','compute',[item('Head operands','FP16'),item('Accumulator','FP32','write')],'FP16 LM-head GEMM');set('sm','compute',[],'Final norm, LM head and sampling');
  set('activations','optional-write',[item('Last-position input','FP16','read'),item('Logits','FP16','write'),item('Next token ID','INT32','write')],'ID location / materialization depends on runtime');
  event('Read','weights','LM head / γ','FP16');event('Compute','tensor','LM head GEMM','FP32');event('Compute','cuda','Sampling transforms','FP32');event('Write','registers','Next token ID','INT32');event('Write*','activations','Logits','FP16','Optional materialization');event('Write*','activations','Sampled token ID','INT32','Location depends on runtime');
  edge('read-w','FP16');edge('read-a','FP16');edge('l2-shared','FP16');edge('shared-reg','FP16');edge('reg-tensor','FP16');edge('reg-cuda','FP32');edge('write-a','INT32',true);break;
 case 19:
  set('cpu','control',[],'EOS / length check; schedule next forward');set('activations','held',[item('Sampled token ID','INT32')],'Next forward: T=1');set('kv','held',[item('K cache',KV),item('V cache',KV)],`Retained · S=${n}; next append increases S`);set('weights','held',[item('Resident linear W',W),item('Embedding / norm / head','FP16')],'No checkpoint reload per token');
  event('Retain','weights','Model parameters',W);event('Retain','kv',`K/V (${n} tokens)`,KV);event('Control','cpu','Stop or dispatch next Decode','META');break;
 }
 if((id>=3&&id<=10)||(id>=12&&id<=15)){const residual=id<11?'Residual X (retained)':'Residual R (retained)';if(nodes.activations.items.length<3)nodes.activations.items.push(item(residual,'FP16'));}if(id===6&&nodes.activations.items.length<3)nodes.activations.items.push(item('Q′ (retained)','FP16'));
 const uniqueEvents=events.filter((e,i,a)=>e.action!=='Read'||a.findIndex(f=>f.action==='Read'&&f.loc===e.loc)===i);
 return{operator:operator(id,r,phase),nodes,events:uniqueEvents,edges,T,S,n,W,KV,id,phase,recipe:r.id};
}
function getSpec(r,phase,id){
 const T=phase==='prefill'?8:1,S=phase==='prefill'?0:8,W=r.wb===4?'packed INT4':r.wb===8?'INT8':'FP16',KV=r.kb===4?'packed INT4':r.kb===8?'INT8':'FP16';
 const v={input:'FP16 hidden states',storage:'FP16 activation boundaries',operands:'Elementwise floating-point operation',accumulator:'Kernel-dependent intermediates',output:'FP16'};
 if([4,10,13,15].includes(id)){const weight={4:'WQ / WK / WV',10:'WO',13:'Wgate / Wup',15:'Wdown'}[id];return{input:`FP16 A; ${W} ${weight}; quantization parameters where applicable`,storage:`GPU DRAM: ${weight} in ${W}; A boundary in FP16`,operands:r.operand,accumulator:r.acc,output:id===4?'Q / K / V: FP16':id===13?'Gate G / Up: FP16':'FP16 linear output'};}
 switch(id){
 case 0:return{input:'Original floating-point weights (FP16 example)',storage:`SSD checkpoint: ${W}; quantization metadata when applicable`,operands:r.wb===16?'Checkpoint preparation':'Offline quantization, rounding and packing',accumulator:'N/A — offline preprocessing',output:`Checkpoint: ${W}`};
 case 1:return{input:`Checkpoint payload: ${W}; retained FP16 parameters`,storage:`GPU DRAM: resident ${W} linear weights + FP16 embedding / norm / LM head`,operands:'Host-to-device transfer / runtime initialization',accumulator:'N/A — no GEMM',output:'Resident GPU model parameters'};
 case 2:return{input:`Token IDs: INT32 / INT64; FP16 embedding table (T=${T})`,storage:'Embedding table: GPU DRAM, FP16',operands:'Embedding gather / lookup',accumulator:'N/A — no GEMM',output:'FP16 hidden states'};
 case 3:case 12:return{input:'FP16 hidden states; FP16 RMSNorm γ',storage:'FP16 parameters and activation boundaries',operands:'RMSNorm reduction and elementwise transform',accumulator:'FP32 statistics / reduction (reference model)',output:'FP16 normalized hidden states'};
 case 5:return{...v,input:'FP16 Q, K; position indices',operands:'Floating-point RoPE on Q and K',accumulator:'No GEMM accumulator; internal precision is kernel-dependent',output:'Q′ / K′: FP16; V unchanged'};
 case 6:return{input:`Current K′ / V: FP16; existing ${S}-token KV cache`,storage:`GPU DRAM: K/V in ${KV}; quantization parameters when applicable`,operands:r.kb===16?'KV append / copy':'KV quantization and packing',accumulator:r.kb===16?'N/A — KV append / copy':'Quantization statistics are implementation-dependent',output:`K/V cache length: ${S} → ${S+T}; dtype: ${KV}`};
 case 7:return{input:`Q′: FP16; cached K: ${KV}`,storage:`K cache: GPU DRAM, ${KV}; score tile: on-chip FP32`,operands:'FP16 Q × FP16 K after KV dequantization if needed',accumulator:'FP32 (reference attention path)',output:'FP32 scaled / masked score tiles'};
 case 8:return{input:'FP32 score tile',storage:'On-chip score / probability tiles; no required full DRAM matrix',operands:'Max, exp, sum and normalization',accumulator:'FP32 max / sum reduction',output:'FP32 softmax statistics; FP16 P tile for PV'};
 case 9:return{input:`P tile: FP16; cached V: ${KV}`,storage:`V cache: GPU DRAM, ${KV}; P / V tiles: on-chip`,operands:'FP16 P × FP16 V after KV dequantization if needed',accumulator:'FP32 (reference attention path)',output:'FP16 attention output'};
 case 11:case 16:return{...v,input:'FP16 residual stream; FP16 branch output',operands:'Elementwise residual add',accumulator:'No GEMM accumulator; arithmetic precision is kernel-dependent',output:'FP16 residual stream'};
 case 14:return{...v,input:'Gate G and Up: FP16',operands:'SiLU(G) ⊙ Up',accumulator:'No GEMM reduction; intermediates are kernel-dependent',output:'FP16 FFN activation F'};
 case 17:return{input:'FP16 layer output',storage:`Per-layer ${W} parameters and ${KV} KV caches`,operands:'Repeat the Transformer layer operator sequence',accumulator:'Determined by each operator',output:'FP16 final-layer hidden states'};
 case 18:return{input:'Last-position hidden state: FP16; LM head / final norm: FP16',storage:'FP16 LM head and norm parameters; optional FP16 logits buffer',operands:'Final RMSNorm; FP16 LM-head GEMM; sampling transforms',accumulator:'FP32 norm statistics / GEMM accumulator / sampling transforms',output:'FP16 logits boundary → INT32 / INT64 sampled token ID'};
 case 19:return{input:'Sampled token ID; updated per-layer KV caches',storage:`Resident ${W} weights; retained ${KV} KV cache`,operands:'EOS / length check; schedule next forward',accumulator:'N/A — control flow',output:'Next Decode input (T=1), or stop'};
 }return v;
}
root.QuantHardware={DT,operator,getTrace,getSpec};
if(typeof module!=='undefined')module.exports=root.QuantHardware;
})(typeof globalThis!=='undefined'?globalThis:this);
