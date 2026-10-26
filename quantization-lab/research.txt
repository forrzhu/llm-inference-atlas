# Quantization Lab：Skill 调研、选择与技术依据

核查日期：2026-10-26。目标：给初学者解释一次 decoder-only Transformer 推理中，权重、激活、KV cache 在什么地方存储、以什么精度搬运和计算，并提供可分享的交互页面。

## 1. 比较方式与结论

先读取用户指定的 `skills-lock.json`，其中只有 `wheresryan22/anatomy`；再读取本地 Anatomy 的完整 SKILL.md、相关开源 skill 的实际指令、仓库说明与许可证。下表是按本次需求做的适配判断，**不是模型盲测或质量排行榜**。

| Skill | 公开来源与许可证 | 对本任务的价值 | 约束与取舍 | 选择 |
|---|---|---|---|---|
| anatomy | [wheresryan22/anatomy](https://github.com/wheresryan22/anatomy)，MIT；本地锁定 hash 为 b243118c08624f740a28dc667c99a3536ff7ab98ececc6aa81e2473aaefedd08 | 高细节等轴测实体装置；同一模型驱动数值与动作；适合解释真实结构 | 核心要求是“用实体装置作为解释”，标签受限，至少 40 个实体零件；本次需要大量 tensor 名称、dtype、缩放参数、软件/硬件边界，装置隐喻会增加读图成本 | 已认真检查，未用于主图 |
| dashmotion 2.2.4 | [csthink/dashmotion](https://github.com/csthink/dashmotion)，MIT | HTML/SVG 自包含；结构边界、正交数据流、连线方向、暂停/减少动态效果；附确定性布局与结构检查脚本 | 默认的流动动画不能代表真实硬件时序；生成器不理解量化语义；需要我们自己核对节点、精度和数据生命周期 | **采用**：生成独立硬件总图，并运行随附检查器 |
| animdiagram 1.1 | [praveenvijayan/animdiagram](https://github.com/praveenvijayan/animdiagram)，MIT | 一条共享故事时钟表达先后关系；静态帧也能读懂；可嵌入单一 SVG | 文档建议每图不超过 9 个节点、8 个事件；重点是循环 SVG，精度切换、逐步查看、数值实验仍需另写交互 | 候选；本次未采用 |
| frontend-design | [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/frontend-design)，已核对该目录 LICENSE.txt：Apache-2.0 | 从主题决定布局、字体和颜色；真实内容先行；强调响应式、可访问性和设计自检 | 不提供量化事实或数值正确性保证，也不提供部署 | **采用**：交互页面的信息层次与视觉规范 |

选型原因：本任务的核心可视对象是 **tensor 数据流 + 存储边界 + 格式变化**。Dashmotion 的节点/边界语法与它直接对应；Frontend Design 负责把复杂图变成可学习的界面。Anatomy 的完整工作流本次不启用，也没有把“参考过”写成“已安装”。

实际参考的源文件指纹（Git blob SHA）：

- dashmotion SKILL.md：d46b50ff9969b2d3e6d9339a22808381c6713815
- dashmotion layout.py：9e02f3d3e79c88912a49c4c1043dd20d6ec6f5bd
- dashmotion check_diagram.py：919101d6a8aa1498a6676b5b0cd98c05d5ba7613
- frontend-design SKILL.md：a5333457c414d20d625f307df945842c0952ecc3

不进行全局 skill 安装；只在本次构建中调用选中 skill 的必要脚本，保留适用许可证。

## 2. 视觉与教学方案（编码前确定）

- 主界面是教学实验台：浅蓝灰背景、深蓝文字，权重紫色、激活蓝色、KV 橙色，颜色只表示数据类别。
- 字体：系统中文无衬线作为正文，系统 monospace 只用于 dtype、公式和 tensor 标签，不依赖字体 CDN。
- 先看完整 Transformer 层的计算链，再用“上一步 / 下一步 / 播放”追踪一份数据；旁边同步显示输入、存储格式、乘法输入、累加格式、输出和生命周期。
- 支持 FP16、W4A16、W8A8，以及明确标为论文特例的 QServe W4A8KV4；prefill/decode 独立切换。
- 独立的数值实验只用基础对称均匀整数格式；不冒充 AWQ、GPTQ、NF4 或 QServe 的算法实现。
- 提供跨层/跨 token 的解释、逐算子矩阵乘法放大图、显存 payload 计算器、来源和边界说明。
- 不显示未经测量的延迟、带宽或性能提升倍数；动画速度不代表硬件实际耗时。

## 3. 技术核查依据与呈现边界

| 要解释的内容 | 一手资料 | 在演示中的处理 |
|---|---|---|
| 量化/反量化、scale、zero point、按组量化 | [Hugging Face Quantization concepts](https://huggingface.co/docs/transformers/en/quantization/concept_guide)；[TensorRT Quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) | 数值实验固定 z=0；明确 clipping 和 rounding；INT4 使用 [-7,7] 的教学对称范围，4 bit 物理编码有 16 种，教学规则只使用 15 个整数值 |
| Q/DQ 节点与实际 kernel 的关系 | [TensorRT Explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html) | 语义步骤不等于独立 kernel 或一次 HBM 写回；写出“融合时可能只存在于片上” |
| 全局内存、cache、shared memory、寄存器与 SM | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 单个离散 NVIDIA GPU；GPU DRAM 可为 HBM 或 GDDR；L2、shared memory、寄存器是示意数据路径，具体指令、旁路和缓存命中因 kernel 而异 |
| W4A16 的反量化；W8A8 与 INT32 累加；QServe 的 W4A8KV4 | [QServe 原论文](https://arxiv.org/html/2405.04532v1)，尤其背景、Fig. 5、Sec. IV 与 Fig. 7；[NVIDIA TensorRT-LLM Quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md) | 存储格式、乘法输入、累加、输出四栏分开；QServe 的 INT4→INT8 是两级方案中的组级恢复，不能等同普通浮点反量化 |
| KV cache 量化与 residual buffer | [Hugging Face KV-cache quantization](https://huggingface.co/blog/kv-cache-quantization) | 每层保存 K/V，不保存历史 Q；缓存格式和 attention 计算格式分别说明；注明某些实现保留最近 token 的高精度缓冲 |
| attention 不必物化完整分数矩阵 | [FlashAttention 原论文](https://arxiv.org/abs/2205.14135) | QK、softmax、PV 在逻辑图上分开，实际融合时按 tile 处理，避免画出必然写入 HBM 的全量矩阵 |

## 4. 明确采用的教学假设

1. 单 GPU、模型权重放得下显存，无 CPU/SSD offload、无张量并行、无 prefix cache 和 sliding window。
2. 使用 Llama 风格 pre-norm、RoPE、SwiGLU、GQA decoder；embedding 与 LM head 在教学路径中保留 FP16，线性投影采用所选方案。
3. 主要激活/残差边界保存 FP16；norm 统计、softmax 归约、浮点 GEMM 累加在示例中采用 FP32；不声称每个实现都采用同一 dtype。
4. W8A8/W4A8 的 A8 指进入量化线性矩阵乘法的激活；其输出可回到 FP16，attention 的 Q 不由标签自动决定。QServe 的 attention query 保留高精度。
5. 权重通常加载/整理后驻留 GPU DRAM，多次推理复用。kernel 按 tile 读入片上存储；下一 token 可能重新从 DRAM 读取权重，但无需重新从 SSD/CPU 加载整个模型。
6. KV 是请求状态；每层独立、随上下文增长。当前 token 的 K/V 追加到该层缓存；历史 Q 不缓存。
7. 源码演示中所用 rounding 明确定义；不把 JavaScript 默认 rounding 当成所有推理框架的统一规则。
8. 显存计算器计算权重与 KV 的理论 payload，另列简单 scale 元数据估算；不代表完整运行时峰值显存，也不等于实测速度。

## 5. 验证要求

- 对称量化的端点、零值、ties-to-even 舍入、重建误差与 INT8 点积缩放运行数值校验。
- 核对 KV payload 公式：2 × B × L × S × Hkv × d × bits/8。
- 对 skill 生成的 SVG 运行官方结构检查器，修复所有 violation。
- 在浏览器检查精度切换、prefill/decode、逐步导航、数值滑块、移动视图和来源链接。
- 公开发布后，以部署返回状态确认上线；若平台禁止公网访问，明确报告，不将私人预览冒充公开链接。


## 6. 已完成的验证

- 8 种 recipe/phase 组合的 20 个步骤完整性检查通过。
- ties-to-even、clipping、未截断误差上界、INT8 点积缩放和 KV payload 数值检查通过。
- Dashmotion 原版结构检查器：hardware.html 为 0 violations。
- 本地 HTTP 预览受运行环境权限限制。公开上线后，已通过浏览器核对方案切换、prefill/decode、KV 8→9 的 shape、逐步导航与播放/暂停、量化滑块、显存数值、分享链接刷新恢复与 skill 选型内容；390px 视图无整页横向溢出；没有浏览器 console error。
- 发布位置为现有公开 LLM 教学站点的独立 quantization-lab 路径。

## Revision 2 · Hardware model and terminology

- UI section titles and technical fields use standard English terms: Inference Overview, GEMM Precision, Storage, Compute operands, Accumulator / Reduction and Location. Operators have English names and Chinese descriptions.
- Hardware Model is embedded in Inference Overview. It separates Host, GPU DRAM, GPU die / L2 and one illustrative SM. Shared Memory, Registers, CUDA Cores and Tensor Cores are drawn inside the SM.
- Dtype colors are shared by hardware tensor labels, operator labels and detail tables: FP16, FP32, INT4, INT8 and INT32. Location is encoded by labeled regions; Read / Write and retained buffers are marked separately.
- Model weights remain resident during runtime. Activations, residuals, operand tiles, accumulators and KV append / read states change with the selected operator.
- Shared staging is an example, not a mandatory path. Attention dot-product mapping is marked kernel-dependent. Write* means optional DRAM materialization that kernel fusion may remove. This is not a profiler trace or physical chip floorplan.
- Added 160 hardware-state invariant checks covering every recipe / phase / operator. Representative SVGs pass the existing structural checker with 0 violations.
- GPU hierarchy checked against NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
