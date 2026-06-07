# 端侧 AI 开发板选型指南：Jetson / 地瓜 RDK / 鲁班猫

> 面向语音交互（ASR + LLM + TTS）、具身智能与边缘推理场景的硬件横向对比。
> 最后更新：2026-06-03

---

## 一、为什么需要开发板选型

端侧部署的核心矛盾始终是**算力、内存、功耗、成本**四者的权衡。不同开发板在架构设计上的根本差异，直接决定了：

1. **能跑多大的模型** -- 内存容量和带宽是硬上限
2. **跑起来有多快** -- 推理速度决定交互体验（首字延迟 < 300ms 是语音交互的及格线）
3. **能不能全端侧闭环** -- ASR + LLM + TTS 三模型并发时的资源争抢情况
4. **开发和维护成本** -- 软件生态成熟度、工具链完善程度、社区支持

以下从三个主流阵营逐层展开。

---

## 二、NVIDIA Jetson 系列

### 2.1 架构本质

Jetson 系列的核心优势来自 **GPU 统一内存架构（Unified Memory）**：CPU 和 GPU 共享同一片物理内存，通过 NVLink/CXL 技术实现零拷贝数据传输。这对语音 Pipeline 至关关键 -- ASR 的音频特征、LLM 的 Token 序列、TTS 的 Mel 频谱，全部在同一内存空间流转，省去了传统 x86 平台上 PCIe 拷贝的延迟开销。

### 2.2 型号矩阵

| 型号 | GPU 架构 | AI 算力 (INT8) | 内存 | 典型功耗 | 参考价格 (含底板) |
|------|----------|----------------|------|----------|-------------------|
| **Jetson Orin Nano Super** | Ampere, 1024 CUDA cores | 67 TOPS | 8GB | 7-15W | ~$250 (￥1,800) |
| **Jetson Orin NX 16GB** | Ampere, 1024 CUDA cores | 100 TOPS | 16GB (102 GB/s 带宽) | 10-25W | ~$600 (￥4,000-5,500) |
| **Jetson AGX Orin 64GB** | Ampere, 2048 CUDA cores | 275 TOPS | 64GB (205 GB/s 带宽) | 15-60W | ~$2,000 (￥14,000+) |
| **Jetson AGX Thor T4000** | Blackwell, FP4 原生加速 | 2070 TFLOPS (FP4) | 128GB | 未量产定价 | 待定 |

#### Orin Nano Super
- **定位**：入门级边缘计算，2025 年 NVIDIA 通过 OTA 固件将原始 Orin Nano 的算力从 40 TOPS 提升至 67 TOPS，因此命名 "Super"
- **适合场景**：轻量级视觉推理（YOLOv8 目标检测）、简单语音唤醒 + VAD、作为云端 LLM 的前端采集终端
- **语音交互局限**：8GB 内存扣除系统占用后约剩 5-6GB 可用，无法同时常驻 ASR + LLM(>1B) + TTS。只能走混合架构（端侧 ASR/TTS + 云端 LLM）
- **亮点**：CUDA 生态完整，开发者上手门槛最低；功耗极低，可用 USB-C 供电

#### Orin NX 16GB（推荐主力型号）
- **定位**：中小型移动机器人 / 桌面级 AI 终端的黄金平衡点
- **语音交互能力**：
  - LLM: Qwen2.5-1.5B/3B INT4 量化，20-30 Tokens/s，TTFT < 200ms
  - ASR: Whisper-Small TensorRT 加速版，实时流式识别
  - TTS: ChatTTS/VITS 轻量版，流式合成延迟 < 150ms
  - 三个模型可同时常驻内存并发运行，不 OOM
- **软件栈**：TensorRT-LLM + DeepStream/Holoscan 音频流水线，PyTorch/TensorFlow 全支持
- **短板**：价格偏高；16GB 内存跑 7B 模型仍然吃力（INT4 量化后约需 5-6GB 模型权重 + KV Cache 动态增长）

#### AGX Orin 64GB
- **定位**：高性能商用机器人、多模态 VLM + 语音融合终端
- **语音交互能力**：Qwen2.5-7B/Llama-3-8B INT4 全端侧运行，同时挂载 CosyVoice 高拟真 TTS 和大参数 ASR，KV Cache 可预留数十 GB 支持长上下文多轮对话
- **适合团队**：预算充足、追求极致性能的商业项目

#### AGX Thor（下一代）
- 基于 Blackwell 架构，原生支持 FP4/FP8 量化。如果路线图包含**端到端多模态大模型**（直接音频输入 -> 音频输出，跳过 ASR/TTS 级联），Thor 是唯一的选择。但截至 2026 年中尚未大规模量产落地。

### 2.4 Jetson 对 ASR / TTS 的原生支持（核心差异化优势）

这是 Jetson 相对于国产开发板**最大的、不可替代的优势所在**。理解这一点，需要从 GPU 的通用计算本质说起。

#### 为什么 Jetson 能"原生支持所有模型"

NVIDIA GPU 是**通用并行计算架构（GPGPU）**，其 CUDA 核心可以执行任意计算逻辑 -- 从矩阵乘法到 FFT，从卷积到循环神经网络，从 Attention 机制到音频信号处理中的 STFT/ISTFT。这意味着：

- **不存在"不支持的算子"**。任何 PyTorch / TensorFlow 模型定义的计算图，都可以被 CUDA 编译器翻译成 GPU 可执行的指令。TensorRT 作为编译优化层，做的事情是"如何跑得更快"，而不是"能不能跑"
- **ASR 和 TTS 模型中常见的特殊算子在 GPU 上全部原生可执行**：
  - ASR 类：Conformer / Transformer-Encoder 中的 Multi-Head Attention、Conv1d/2d Subsampling、SpecAugment 数据增强、CTC Loss / CTC Decode
  - TTS 类：VITS 中的流式生成对抗训练、基于 Fourier 的周期性建模、HiFi-GAN Vocoder 中的多尺度/多周期判别器、ChatTTS 中的扩散解码过程
  - 音频前处理：MFCC / FBank 特征提取、STFT / ISTFT、Mel 滤波器组、噪声抑制 / 回声消除中的频域运算
- **量化自由度高**：FP32 -> FP16 -> INT8 -> INT4（W4A16），全链路精度可选。TensorRT 会自动选择每层的最优精度

#### 主流语音模型在 Jetson 上的部署路径

| 语音任务 | 代表模型 | Jetson 部署方式 | 是否需要修改模型代码 | 社区成熟度 |
|----------|---------|----------------|---------------------|-----------|
| **语音识别 (ASR)** | Whisper (Base/Small/Medium) | `whisper.cpp` + TensorRT 加速，或官方 TensorRT demo | 否 | 极高，开箱即用 |
| **语音识别 (ASR)** | FunASR (Paraformer 系列) | ONNX -> TensorRT，或直接 PyTorch 推理 | 视版本而定 | 高，阿里有适配指南 |
| **语音识别 (ASR)** | SenseVoice (商汤) | PyTorch -> Torch-TensorRT | 否 | 中等，较新 |
| **语音识别 (ASR)** | WeNet / U2++ (流式/非流式) | Torch-TensorRT 或 ONNX Runtime | 需确认算子覆盖 | 中等 |
| **文字转语音 (TTS)** | ChatTTS | PyTorch 原生推理，可选 Torch-TensorRT 加速编码器 | 否 | 高，社区活跃 |
| **文字转语音 (TTS)** | CosyVoice (阿里) | PyTorch -> TorchScript -> TensorRT | 少量适配工作 | 中高，阿里提供参考 |
| **文字转语音 (TTS)** | VITS / VITS2 | ONNX -> TensorRT，或 whisper.cpp 集成的 TTS 引擎 | 否 | 高 |
| **文字转语音 (TTS)** | MeloTTS / Bark | PyTorch 原生推理 | 否 | 高 |
| **语音活动检测 (VAD)** | Silero VAD | ONNX Runtime 或 PyTorch | 否 | 极高，一键部署 |
| **语音唤醒 (KWS)** | OpenWakeWord / Porcupine | ONNX / 预编译库 | 否 | 高 |

**关键结论**：上表中**每一个模型都可以在 Jetson 上直接运行，不需要修改模型结构、不需要替换算子、不需要写自定义 C++ kernel**。你从 HuggingFace 下载权重，用社区提供的转换脚本一转，就能跑。

#### 这一点与地瓜、鲁班猫的本质差别

| 维度 | Jetson (GPU) | 地瓜 (BPU) | 鲁班猫 (RK NPU) |
|------|-------------|-----------|-----------------|
| **新模型上手时间** | 下载权重 -> 转 TensorRT -> 跑通，通常 < 半天 | HF -> ONNX -> BPU Bin，需验证算子支持，通常 1-3 天 | HF -> ONNX -> RKNN，大概率遇到不支持算子，通常 3-7 天 |
| **遇到新型算子时** | CUDA 自动 fallback 到通用实现，功能不受影响 | 可能需要切 CPU 执行，性能骤降；或需要向地平线提需求 | 必须手动拆图回退 CPU 或自行实现替代算子 |
| **模型选择自由度** | 可以随意尝试最新开源模型，试错成本极低 | 受限于 BPU 算子库覆盖范围，新模型可能无法转换 | 受限于 RKNN 算子库，覆盖面比 BPU 更窄 |
| **ASR+TTS 组合灵活性** | 任意 ASR + 任意 LLM + 任意 TTS 自由组合 | 需要每个模型都通过 HBBC 验证才能组合 | 需要每个模型都通过 RKNN 验证，组合难度最高 |

#### 实际影响：语音交互迭代速度

假设你要从 Whisper 切换到更新的 SenseVoice ASR，同时把 TTS 从 VITS 升级到 ChatTTS：

- **Jetson**：两个模型分别下载权重、分别转 TensorRT，各花 1-2 小时，当天完成切换验证
- **地瓜 S100P**：两个模型分别走 HF->ONNX->BPU Bin 流程。如果 SenseVoice 用了新的 Attention 变体而纳什架构暂不支持，需要等待地平线工具链更新或自己写 CPU fallback，预计 2-5 天
- **鲁班猫 RK3588**：SenseVoice 几乎必然包含 RKNN 不支持的算子（如 GLU 变体、复杂的位置编码），ChatTTS 的扩散解码机制也超出 RKLLM 能力范围。最终大概率要用 CPU + onnxruntime 跑，速度和功耗都不理想，预计 1-2 周

**这就是"原生支持"的真实商业价值 -- 不是"能跑"，而是"能快速迭代"。**

### 2.5 Jetson 核心优势总结

| 维度 | 评价 |
|------|------|
| **Transformer 硬解** | 原生 Tensor Cores 支持 Attention/FFN/MHA 算子，TensorRT-LLM 自动融合优化 |
| **统一内存带宽** | Orin NX 102 GB/s，AGX Orin 205 GB/s，KV Cache 读写吞吐充足 |
| **软件生态** | 业界最强。HuggingFace 模型转 TensorRT 教程满天飞，几乎无适配成本 |
| **流式推理成熟度** | CUDA Stream 并发调度成熟，ASR/LLM/TTS 多线程不互相阻塞 |
| **主要劣势** | 价格高（尤其国内市场）；部分型号供货周期长；国产化替代压力 |

---

## 三、地瓜机器人 RDK 系列（地平线 BPU）

### 3.1 架构本质

地瓜机器人（DigiBot/DiGa）基于**地平线（Horizon Robotics）** 的 BPU（Brain Processing Unit）架构。与 NVIDIA GPU 的通用并行计算思路不同，BPU 从设计之初就是为**深度学习推理**定制的专用 ASIC：

- **数据驱动架构**：针对 CNN（卷积网络）和 Transformer 的典型计算模式做了硬件级定制
- **纳什架构（Nash）**：RDK X5 / S100P 采用的新一代 BPU 架构，显著提升了对 Transformer Attention 的支持效率
- **大小脑协同**：SoC 集成 CPU + BPU + MCU，MCU 可独立处理实时控制任务（电机控制、声源定位），不抢占 BPU 推理资源
- **统一内存**：CPU/BPU/MCU 共享内存池，同样避免了数据拷贝开销

### 3.2 型号矩阵

| 型号 | BPU 算力 (INT8) | 内存 | NPU 核心 | MCU | 参考价格 |
|------|-------------------|------|----------|-----|----------|
| **RDK X3** (Ultra) | 5 TOPS | 4GB LPDDR4 | 双核 BPU | 有 | ~￥800-1,000 |
| **RDK X5** (Pro) | 30 TOPS | 8GB LPDDR5 | 新一代 BPU | 有 | ~￥1,500-2,000 |
| **RDK S100P** | 128 TOPS | 24GB LPDDR5/X | 新一代纳什 BPU | 有 | ~￥2,500-3,500 (套件) |

#### RDK X3 / X3 Ultra
- **定位**：轻量级边缘设备，对标 RK3588 的低端市场
- **能力**：5 TOPS 算力适合 YOLO 系列、轻量级语音前处理
- **局限**：4GB 内存极其有限，基本只适合纯视觉或纯语音前处理场景

#### RDK X5
- **定位**：中端升级款，LPDDR5 带宽提升明显
- **能力**：30 TOPS 可以跑 1B~3B 小模型，配合 8GB 内存勉强支撑小型 SLM + ASR
- **适用**：需要一定本地智能但不需要 7B 大模型的场景

#### RDK S100P（旗舰，当前重点）
- **定位**：具身智能主控板，Jetson AGX Orin 的国产替代候选
- **核心参数**：
  - 128 TOPS INT8 算力（超过 Orin NX 的 100 TOPS）
  - **24GB 统一内存**（超过 Orin NX 的 16GB，接近 AGX Orin 32GB 版）
  - 纳世 BPU 对 Transformer 的原生优化
  - 自带 MCU 实时控制（电机/传感器低延迟响应）
- **实测表现**（基于用户实际部署经验）：
  - Qwen2.5-7B INT4 量化后约占用 7GB 内存，TTFT 约 530ms
  - 24GB 内存意味着模型权重 + KV Cache + ASR/TTS 常驻后仍有余量
  - 正在研发 VLM（视觉语言模型）视觉问答功能
- **软件生态**：
  - **TogetheROS.Bot (TRB)**：基于 ROS 2 的机器人开发框架
  - 地平线 **HBBC（地平线二进制编译器）** 工具链：HuggingFace -> ONNX -> BPU Bin 模型转换
  - 对 Qwen 系列有官方转换支持和示例
  - 文档和本地技术支持在国内厂商中属于第一梯队

### 3.4 地瓜 BPU 上 ASR / TTS 的实际限制

BPU 的算子库是**白名单模式** -- 只有预定义的算子能映射到硬件加速单元，不在名单上的算子必须回退到 CPU 执行。这对 ASR 和 TTS 模型的影响是结构性的：

#### ASR 模型在 BPU 上的兼容性

| ASR 模型 | 核心架构 | BPU 兼容性 | 说明 |
|----------|---------|-----------|------|
| **Whisper (Encoder-only Transformer)** | Attention + FFN + 卷积子采样 | **较好** | 标准 Transformer 结构，纳什架构有原生 Attention 支持。但 Whisper 的 GELU 激活、LayerNorm 的 eps 参数变体可能需要验证 |
| **Paraformer (FunASR)** | Conformer + Predictor + CTC/Sampler | **中等** | Conformer 的卷积模块（Conv2d Subsampling）在 CNN 优化的 BPU 上表现好，但 Predictor 的自适应采样机制可能包含不常见算子 |
| **SenseVoice / 新模型** | 非 Transformer 或含新组件 | **需逐个验证** | 如果用了新的位置编码（RoPE 变体）、新型归一化（RMSNorm）、或非标准 Attention，转换可能失败 |
| **流式 ASR (WeNet/U2)** | Transducer / CTC attention hybrid | **较低** | Transducer 的联合网络（Joint Network）结构特殊，BPU 算子库大概率不支持 |

#### TTS 模型在 BPU 上的兼容性

TTS 比 ASR 更难部署到专用加速器，原因是 TTS 模型的架构多样性远高于 ASR：

| TTS 模型 | 核心架构 | BPU 兼容性 | 说明 |
|----------|---------|-----------|------|
| **VITS / VITS2** | 流式生成对抗 + HiFi-GAN Vocoder | **较差** | Vocoder 中的多周期判别器涉及复杂频域运算，生成器的流式随机采样依赖概率采样算子，BPU 对这类"非确定性计算"支持有限 |
| **ChatTTS** | 扩散概率模型 + 流式推理 | **很差** | 扩散解码过程需要数十到数百步迭代去噪，每步都涉及复杂的条件注入和随机采样。这本质上是迭代式生成，不是 BPU 擅长的单次前向传播 |
| **CosyVoice** | 流式因果卷积 + VAE | **中等偏差** | 因果卷积部分可加速，但语音编码/解码的频域变换和流式隐变量建模超出 BPU 算子覆盖 |
| **Edge-TTS / ESPnet-TTS** | 轻量级 LSTM / Tacotron2 变体 | **较好** | 传统序列到序列架构相对简单，算子覆盖率高，但音质远不及现代生成式 TTS |

#### 地瓜上 ASR/TTS 的务实方案

基于以上分析，地瓜 S100P 上的语音交互务实路线是：

1. **ASR**：优先使用 Whisper-Small 或 Paraformer-base 这类经过验证的模型，通过 HBBC 成功转换为 BPU Bin。避免使用最新发布的、架构创新的 ASR 模型
2. **TTS**：如果必须用高拟真 TTS（如 ChatTTS/CosyVoice），建议**放在 CPU 端用 PyTorch/onnxruntime 推理**，牺牲一些速度换取可用性。或者选择轻量化 TTS（如 edge-tts 的离线化版本）
3. **LLM**：这是 BPU 的主场，Qwen/Llama 系列 7B 以内的模型在纳什架构上运行效率高，应把主要算力预算分配给 LLM
4. **资源分配策略**：BPU 主要跑 LLM（占 70%+ 算力），ASR 和 TTS 用 CPU/MCU 分担，通过 24GB 大内存确保三者同时常驻

### 3.5 BPU vs GPU 的深层差异

| 维度 | NVIDIA GPU (CUDA) | 地平线 BPU |
|------|--------------------|------------|
| **计算范式** | 通用 SIMD/SIMT 并行，灵活度高 | 数据流驱动，针对 DL 算子模式硬化 |
| **Transformer 支持** | Tensor Cores 原生 MMA 指令 | 纳什架构新增 Attention 专用单元 |
| **量化精度** | FP16/INT8/INT4/W4A16 全覆盖 | 主要 INT8/W8A8，部分支持 W4A16 |
| **自定义算子** | CUDA C++ 编写 kernel 即可 | 需要通过 HBBC 工具链注册，灵活性较低 |
| **调试手段** | Nsight Systems/Compute 全套 profiler | 地平线提供的 hbm_profile 工具链 |
| **模型兼容性** | 几乎所有开源模型可直接跑 | 需要模型转换，部分新算子可能不支持需切 CPU |

---

## 四、鲁班猫系列（瑞芯微 RK 系列）

### 4.1 架构本质

鲁班猫（LubanCat）是基于**瑞芯微（Rockchip）** RK 系列 SoC 的开发板产品线（由野火电子出品）。RK3588 是目前该系列中性能最强的芯片：

- **CPU**: 八核 big.LITTLE (4×Cortex-A76 @ 2.4GHz + 4×Cortex-A55 @ 1.8GHz)
- **GPU**: ARM Mali-G610 MP4（支持 OpenGL ES 3.2 / Vulkan 1.2 / OpenCL 2.2）
- **NPU**: 6 TOPS，三核设计（可独立分配给不同任务），算子支持 **RKNN API**
- **内存**: LPDDR4X，通常配置 8GB 或 16GB
- **视频编解码**: 8K@60fps 解码 / 8K@30fps 编码（RV1109/1126 硬件编码器）

**关键认知**：RK3588 的 NPU 设计初衷是为**计算机视觉（CNN 类模型）**服务的。YOLO、SSD、MobileNet 这类 CNN 在 RKNN 上运行效率很高。但 Transformer（Attention + FFN 结构）并非其强项。

### 4.2 型号矩阵

| 型号 | SoC | NPU 算力 | 内存 | 参考价格 |
|------|-----|----------|------|----------|
| **LubanCat RK3588 (8GB)** | RK3588 | 6 TOPS | 8GB LPDDR4X | ~￥600-800 |
| **LubanCat RK3588 (16GB)** | RK3588 | 6 TOPS | 16GB LPDDR4X | ~￥900-1,300 |
| **LubanCat 1 (RK3568)** | RK3568 | 6 TOPS (单核 NPU) | 2-4GB | ~￥200-350 |
| **LubanCat Zero (RK3566)** | RK3566 | 6 TOPS (单核 NPU) | 1-2GB | ~￥120-180 |

#### LubanCat RK3588 主力型号
- **定位**：高性价比全能型开发板，接口丰富（双网口、HDMI IN/OUT、多路 USB、MIPI CSI/DSI）
- **语音交互真实表现**：
  - Qwen2-0.5B: ~15 Tokens/s（尚可）
  - TinyLlama-1.1B: ~10-12 Tokens/s
  - Qwen2.5-3B: ~7 Tokens/s（体验开始下降）
  - Qwen2.5-7B INT4: ~3-4 Tokens/s（**不可用于实时交互**，人类感知为"结巴"）
- **并发痛点**：6 TOPS 总算力被 ASR 占用后，LLM 推理速度骤降。常见妥协方案是把 ASR 或 TTS 卸载到 CPU 用 `whisper.cpp` / `onnxruntime` 硬扛，导致 CPU 占满发热严重
- **真正适合的场景**：
  - 视觉推理为主 + 轻量语音（YOLO 检测 + 语音播报）
  - 混合架构：端侧负责 VAD + 轻量 ASR + TTS，LLM 放云端
  - IoT 网关 + 边缘预处理节点
  - 学习嵌入式 Linux 开发的入门平台

### 4.3 RK3588 上 ASR / TTS 的实际限制（最严峻）

RK3588 的 NPU 是三个平台中**对语音模型支持最弱的**。这不仅是算力大小的问题，更是 NPU 架构设计目标与语音模型计算模式不匹配的结构性矛盾。

#### 为什么 RK3588 NPU 不适合跑语音模型

RK3588 的 NPU（瑞芯微自研）设计目标是**计算机视觉 CNN 推理**，其硬件特性包括：

- **优化的卷积引擎**：Winograd / Im2Col 加速，对 3x3/1x1 卷积极其高效
- **有限的片上 SRAM**：约几百 KB 级别，足够存卷积核和中间 Feature Map
- **定点运算为主**：INT8 量化是原生精度，浮点需要额外开销

而语音模型（ASR/TTS）的计算模式完全不同：

| 计算特征 | CNN (视觉) | Transformer (ASR) | 生成式模型 (TTS) |
|----------|-----------|-------------------|------------------|
| **核心操作** | Depthwise Conv2d | Multi-Head Attention + FFN | 自回归解码 / 扩散迭代 |
| **数据形状** | 固定尺寸的图像张量 | 变长序列（音频帧序列） | 动态生成的变长序列 |
| **内存访问模式** | 局部、规则的全局内存读写 | 全局性的 KV Cache 读写 | 依赖前一步输出的顺序依赖 |
| **算子复杂度** | 少量标准算子反复使用 | 多种 Attention 变体 + 复杂归一化 | 概率采样、随机数生成、频域变换 |
| **与 RK NPU 匹配度** | **极高（原生设计目标）** | **低** | **极低** |

#### 各语音任务在 RK3588 上的可行方案

| 任务 | NPU 方案 (理想) | 实际方案 (妥协) | 性能表现 |
|------|----------------|---------------|---------|
| **VAD (语音活动检测)** | Silero VAD -> RKNN 转换 | **可走 NPU**，模型小且结构简单 | 延迟 < 10ms，无压力 |
| **语音唤醒 (KWS)** | 开源唤醒模型 -> RKNN | **可走 NPU**，类似小型 CNN 分类 | 延迟 < 50ms，功耗极低 |
| **轻量 ASR** | Whisper-Tiny -> rkllm-toolkit | **勉强可转**，但速度慢（~5 tok/s 等效） | 仅适合短命令词识别 |
| **中等 ASR (Whisper-Base)** | Whisper-Base -> RKNN | **大概率需 CPU 回退**部分算子 | 用 `whisper.cpp` 跑 CPU，延迟 1-3s |
| **生产级 ASR** | Paraformer-Large / SenseVoice | **不建议在 RK3588 上部署** | 必须用 CPU，发热严重 |
| **轻量 TTS (VITS-small / edge-tts)** | ONNX -> RKNN | **可能转换成功但推理慢** | 音质一般，延迟 500ms+ |
| **高拟真 TTS (ChatTTS/CosyVoice)** | -- | **不可行**，扩散/流式生成超出 RKNN 能力范围 | 只能放云端或用极简替代品 |

#### RK3588 语音交互的唯一可行架构

基于上述现实约束，RK3588 上的语音交互只能采用以下混合架构：

```
┌─────────────────────────────────────────────────┐
│                  RK3588 本地                      │
│                                                   │
│  麦克风 → VAD(Silor, NPU) → 唤醒检测(NPU)       │
│                              ↓                   │
│                    轻量 ASR(whisper.cpp on CPU)   │
│                              ↓                   │
│                    TTS 播报(onnxruntime on CPU)    │
│                                                   │
│  ─ ─ ─ ─ ─ ─ ─ ─ 网络 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                    WebSocket / HTTP               │
│                              ↓                   │
│              ☁️ 云端 LLM API (GPT/Qwen-Cloud)     │
└─────────────────────────────────────────────────┘
```

在这个架构下，RK3588 的分工非常明确：
- **NPU（6 TOPS）**：只负责 VAD 和唤醒词检测，这两个任务都是类 CNN 的小模型，NPU 跑起来游刃有余
- **CPU（A76 x4）**：负责轻量 ASR 和简单 TTS，用 C++ 优化过的推理引擎（`whisper.cpp`、`onnxruntime`）硬扛
- **LLM**：完全外包给云端 API，本地不跑任何大语言模型

这个方案的**优点**是成本极低（板卡 ￥900 + 云端 API 按量付费），**缺点**是依赖网络、无法离线闭环、隐私敏感场景不适用。

### 4.4 RKNN / RKLLM 工具链现状

```
模型转换流程：
PyTorch/ONNX -> rknn-toolkit2 (CNN类) / rkllm-toolkit (Transformer类) -> RKNN 模型文件 -> NPU 运行
```

**实际开发中的典型坑点**：

1. **算子覆盖率不足**：新模型中出现的新型激活函数（SiLU/GELU 变体）、LayerNorm 变体、FlashAttention 变体等，RKNN 可能不支持，需要手动拆分图并回退到 CPU 执行
2. **量化限制**：RKLLM 主要支持 W8A8 量化，W4A16 支持不完善。这意味着同等精度的模型体积比 INT4 大一倍
3. **KV Cache 管理**：NPU 片上 SRAM 容量有限，长序列的 KV Cache 需要频繁在 DRAM 和 SRAM 之间搬运，成为性能瓶颈
4. **调试困难**：相比 CUDA 的 Nsight，RKNN 的 profiler 信息粒度粗，性能调优依赖经验试错
5. **社区资源**：相比 Jetson 的海量教程，RK3588 的 Transformer 部署经验散落在个人博客和论坛，系统性文档少

---

## 五、三方横评：语音交互视角

### 5.1 核心指标对比

| 指标 | Jetson Orin NX 16GB | 地瓜 RDK S100P | 鲁班猫 RK3588 16GB |
|------|---------------------|---------------|---------------------|
| **AI 算力** | 100 TOPS (INT8, GPU) | 128 TOPS (INT8, BPU) | 6 TOPS (NPU, 三核) |
| **内存容量** | 16 GB | **24 GB** | 16 GB |
| **内存带宽** | 102 GB/s | ~80-100 GB/s (估算) | ~34 GB/s (LPDDR4X) |
| **7B 模型 TTFT** | < 300ms (流畅) | ~530ms (实测值) | > 2s (不可用) |
| **3B 模型速度** | 25-30 tok/s | 15-20 tok/s (估算) | ~7 tok/s |
| **ASR+LLM+TTS 并发** | 流畅，CUDA 调度成熟 | 可行，24GB 余量大 | 勉强，需 CPU 协助 |
| **VAD/唤醒前置** | GPU 可轻松承载 | BPU 低功耗待机 | NPU 分核处理可行 |
| **参考价格** | ￥4,000-5,500 | ￥2,500-3,500 | ￥900-1,300 |
| **性价比得分** | 中等 | **高** | **极高**（但牺牲性能） |

### 5.2 全端侧闭环可行性评估

```
                    全端侧 ASR + LLM(>=3B) + TTS 闭环可行性

Jetson Orin NX 16GB    [███████████████████]  完全可行，商业级体验
Jetson AGX Orin 64GB   [████████████████████]  宽裕，可叠加 VLM
地瓜 RDK S100P         [████████████████░░░░]  可行，7B 模型可跑
地瓜 RDK X5            [█████████░░░░░░░░░░░]  勉强，建议 <= 3B
鲁班猫 RK3588 16GB     [██████░░░░░░░░░░░░░░]  不建议，<= 1B 尚可
鲁班猫 RK3588 8GB      [███░░░░░░░░░░░░░░░░░]  不现实，走混合架构
```

### 5.3 软件生态对比

| 维度 | Jetson (NVIDIA) | 地瓜 (地平线) | 鲁班猫 (瑞芯微) |
|------|-----------------|--------------|-----------------|
| **操作系统** | JetPack (Ubuntu-based L4T) | TogetheROS.Bot (Ubuntu + ROS 2) | Buildroot / Ubuntu / Android |
| **推理框架** | TensorRT / TensorRT-LLM | HBBC / Horizon Inference | RKNN / RKLLM / ONNX Runtime |
| **Python 支持** | 完善（nvidia 官方 wheel） | 支持（地平线提供 Python API） | rknn-toolkit2 (Python)，但底层依赖多 |
| **模型转换** | trtexec 一键转 | hb_mapper 工具链 | rknn-toolkit2 / rkllm-toolkit |
| **社区活跃度** | 极高（全球社区） | 国内活跃（中文社区） | 中等（散落在各论坛） |
| **文档质量** | 官方文档 + Developer forum | 中文文档 + 技术支持群 | Wiki + 个人博客为主 |
| **新模型跟进速度** | 数天到数周 | 数周到数月 | 数月到半年 |
| **调试工具** | Nsight Compute/Systems | hbm_profile | NPU profiler（基础） |

### 5.4 ASR / TTS 模型支持深度横评

以上各节已经分别阐述了三方在语音模型上的表现。此处做一次**结构化的汇总对比**，便于快速查阅。

#### ASR（自动语音识别）模型支持度

```
模型支持完整度（从"下载即用"到"不可行"）

Whisper-Small/Base:
  Jetson Orin NX     [████████████] TensorRT 一键转，实时流式
  地瓜 S100P         [████████░░░░] HBBC 可转，需验证 GELU/LayerNorm
  鲁班猫 RK3588      [███░░░░░░░░░] whisper.cpp 跑 CPU，延迟较高

Paraformer (FunASR):
  Jetson Orin NX     [██████████░░░] ONNX->TensorRT 或 PyTorch 原生
  地瓜 S100P         [███████░░░░░░] Conformer 卷积部分好，Predictor 需验证
  鲁班猫 RK3588      [█░░░░░░░░░░░░] 不建议，复杂度高

SenseVoice / 最新ASR:
  Jetson Orin NX     [████████████] PyTorch 原生跑，新架构无障碍
  地瓜 S100P         [████░░░░░░░░░] 新算子可能不兼容，等工具链更新
  鲁班猫 RK3588      [░░░░░░░░░░░░░] 不可行
```

#### TTS（文字转语音）模型支持度

```
ChatTTS / CosyVoice (高拟真生成式):
  Jetson Orin NX     [████████████░] PyTorch 原生推理，可选 Torch-TensorRT 加速
                      扩散解码可利用 GPU 并行加速去噪步骤
  地瓜 S100P         [██░░░░░░░░░░░] BPU 不适合迭代式生成，必须走 CPU
  鲁班猫 RK3588      [░░░░░░░░░░░░░] 完全不可行

VITS / VITS2 (中等音质):
  Jetson Orin NX     [████████████] ONNX->TensorRT，HiFi-GAN Vocoder 加速良好
  地瓜 S100P         [████░░░░░░░░░] Vocoder 的频域运算部分回退 CPU
  鲁班猫 RK3588      [██░░░░░░░░░░░] 可能转换但速度慢，实用价值有限

Edge-TTS / 离线轻量TTS:
  Jetson Orin NX     [████████████] 过于简单，大材小用
  地瓜 S100P         [████████░░░░] 可以跑 BPU，算力绰绰有余
  鲁班猫 RK3588      [███████░░░░░] NPU 可承担，CPU 也扛得住
```

#### "原生支持"的本质：从计算架构理解差异

| | **Jetson GPU** | **地瓜 BPU** | **鲁班猫 NPU** |
|---|---|---|---|
| **硬件本质** | 通用并行处理器（GPGPU） | 深度学习专用加速器（ASIC） | CNN 专用加速器（ASIC） |
| **执行模式** | 任意 CUDA kernel 都能跑 | 只有预注册算子能映射到硬件 | 只有预注册算子能映射到硬件 |
| **新算子出现时** | 编译器自动生成 GPU 代码，零适配 | 工具链不支持 -> 回退 CPU 或报错 | 工具链不支持 -> 必须手动拆图 |
| **对语音模型的友好度** | **最高 -- 语音模型只是另一种计算图** | **中等 -- Transformer 类尚可，生成式困难** | **最低 -- 仅类 CNN 结构可行** |
| **选型含义** | 语音模型选择不受硬件约束 | 语音模型必须在官方支持的列表内选 | 语音模型基本只能用最简单的方案 |

#### 一句话总结

> **Jetson 上你只需要问"这个模型好不好用"，不需要问"这块板能不能跑这个模型"。地瓜和鲁班猫上你必须先问"能不能跑"，然后才能讨论"好不好用"。对于快速迭代的语音交互项目，这种约束的差异是决定性的。**

---

## 六、选型决策树

```
你的需求是什么？

├─ 预算充裕，追求最低延迟 + 最快开发速度
│   └─> Jetson Orin NX 16GB （商业落地首选）
│       如需 7B+ 模型或多模态 → AGX Orin 64GB
│
├─ 做具身智能机器人（运动控制 + AI 推理），追求国产化
│   └─> 地瓜 RDK S100P
│       24GB 内存 + MCU 协同 = 语音 + 电机控制一体化
│       性价比高于 Jetson 同级别产品
│
├─ 成本敏感（千元级以内），可以接受混合架构（端侧+云）
│   └─> 鲁班猫 RK3588 16GB
│       端侧做 VAD + ASR + TTS，LLM 放云端 WebSocket 调用
│       也可作为学习嵌入式 Linux + NPU 开发的入门平台
│
├─ 纯视觉 / IoT 网关场景，不需要大模型
│   └─> 鲁班猫 RK3568 / LubanCat 1
│       6 TOPS NPU 跑 YOLO 绰绰有余，价格 ￥200-350
│
└─ 验证原型 / 快速 PoC，不想买硬件
    └─> 先用 Docker 模拟或云服务器验证模型尺寸
        再根据实测内存/延迟需求缩小范围
```

---

## 七、实战建议

### 7.1 无论选哪块板，必须做的优化

1. **量化是必修课**：FP16 -> INT8 -> INT4（如支持），每降一级模型体积减半、速度翻倍。优先选择原生支持 INT4 量化的平台（Jetson > 地瓜 > 瑞芯微）

2. **流式 Pipeline 是刚需**：
   ```
   音频输入 -> VAD(截断) -> 流式ASR(边说边识别) -> 分块文本 ->
   流式LLM(边收边推理) -> 句子级/短语级TTS(边生成边播放) -> 音频输出
   ```
   不要等上一模块完全结束才启动下一个，否则累积延迟必然超 1.2s

3. **VAD 前置节能**：最前端挂 Silero VAD 或 WebRTC VAD，无人说话时不唤醒后方推理模块。这对电池供电设备尤为重要

4. **模型蒸馏 / 剪枝**：如果平台算力不够跑目标模型，考虑用教师模型蒸馏一个小版本（如 Qwen2.5-1.5B 从 7B 蒸馏而来，保留 90% 能力）

### 7.2 各平台避坑要点

| 平台 | 最大坑点 | 应对策略 |
|------|---------|----------|
| **Jetson** | JetPack 版本与 CUDA/TensorRT 版本严格绑定，不能随意升级 | 锁定 JetPack 版本，所有包通过 apt 管理，不用 pip 装冲突版本 |
| **地瓜** | 模型转换链路长（HF->ONNX->BPU Bin），中间出错难排查 | 严格按官方文档步骤执行，先转小模型验证链路通畅再转大模型 |
| **鲁班猫** | 算子不支持时静默降速（回退 CPU 但不报错） | 每个 RKNN 模型加载后务必做 benchmark 对比，确认确实跑在了 NPU 上 |

---

## 八、未来趋势观察

1. **端侧 SLM（Small Language Model）爆发**：Qwen2.5-1.5B/3B、Phi-3、Gemma-2B 等小模型能力快速提升，正在降低对高端硬件的依赖。3B 级别模型在 RK3588 上虽然慢但已经"能用"，这会压缩中端开发板的生存空间

2. **国产 BPU/NPU 快速追赶**：地平线纳什架构、华为昇腾310/910、寒武纪 MLU 等，Transformer 支持力度持续加强。预计 2026-2027 年会出现更多 Jetson 的平价替代品

3. **端到端多模态大模型**：GPT-4o / Gemini 原生多模态能力的端侧化，可能改变 ASR -> LLM -> TTS 的级联范式。这类模型对算力和内存的要求更高，利好大内存平台（AGX Orin、Thor、S100P）

4. **SLP（Speech Language Processor）专用芯片**：面向语音交互的专用 SoC 开始出现（如 ESP32-S3 + 专用 DSP 方案），可能在超低成本语音交互场景中绕开通用 AI 加速器的竞争

---

## 九、参考资料

- [NVIDIA Jetson 官方文档](https://developer.nvidia.com/embedded/jetson)
- [地瓜机器人官网](https://www.d-robotics.cc/)
- [地平线机器人开发平台](https://horizonai.com/)
- [野火电子 - 鲁班猫](https://embedfire.com/lubancat)
- [RKNN Toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
