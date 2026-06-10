---
date: 2026-06-10
topic: 端侧部署 / 模型部署策略
problem: 如何在云端服务器、局域网服务器和端侧三种环境下正确部署 ASR/LLM/TTS 模型，以及多模型共用同一 GPU 对性能的影响
---

# 语音交互流水线（ASR / LLM / TTS）三类模型部署策略全解

## 摘要

本文系统梳理了语音交互流水线（ASR + LLM + TTS）在云端服务器、局域网服务器和端侧设备三种场景下的正确部署策略。核心结论：**多个重型模型共用同一块 GPU 会因算力和显存带宽争抢导致 TTS 首帧延迟大幅恶化**，正确做法是按模型大小和调用时序进行"分卡隔离"；端侧芯片（如 RK3588、S100P）受统一内存架构限制，需借助核心绑定（RK3588）或任务优先级调度（S100P）来缓解相同问题。

---

## 一、问题背景：共用一张显卡为什么会出问题？

### 1.1 实测数据说明问题

以阿里云 ECS（RTX 4090 46 GB）部署 Qwen3-TTS-0.6B 为例，同卡还部署了 Qwen3.5-35B LLM：

| 指标 | 理想（独占显卡） | 实测（与 LLM 共存） | 劣化幅度 |
|---|---|---|---|
| TTS HTTP 首帧延迟 | ~97 ms（官方参考） | ~280 ms | +183 ms |
| TTS WebSocket 首帧 | ~84 ms（本机回环） | ~259 ms | +175 ms |

直觉上 4090 算力远超本地 8 GB 小卡，TTS 应更快。实测恰恰相反——**算力过剩并不能免疫资源争抢。**

### 1.2 根本原因：GPU 内部的两场"厮杀"

**第一场：Prefill 阶段的算力霸占（Compute-Bound 冲突）**

当 LLM 收到 Prompt 时，Prefill 阶段是典型的计算密集型操作，会将 GPU 的所有流处理器（SM）几乎占满。此时 TTS 的计算 Kernel 在硬件层面抢不到 SM 核心，只能在 CUDA 工作队列中挂起等待，直接贡献 **80–120 ms 的调度等待**。

**第二场：Decoding 阶段的带宽绞杀（Memory-Bound 冲突）**

LLM 逐 Token 生成阶段（Decoding）是极度访存密集的：每生成一个 Token，GPU 都要把整个模型权重从显存搬运到片上缓存。Qwen3.5-35B 占用 ~29.6 GB 显存，4090 的 1 TB/s 带宽在高频搬运下几乎被塞满，TTS（0.6B）的权重读取请求在显存控制器前**排队挨饿**。

**两点额外加成：**

- **公网 RTT**：云端测试通过公网 IP 访问，单程 RTT 贡献额外 20–60 ms，应通过内网 IP 访问消除。
- **CUDA Context 内核串行化**：两个独立进程默认走串行排队（非 MPS 模式），跨模型的 Context Switch 带来额外的空转周期（气泡）。

### 1.3 与 LLM 自身高并发的对比

LLM 内部高并发时，vLLM 的 Continuous Batching 充当"交警"，可以把不同用户的 Prefill 和 Decoding 优雅地合批计算。而 LLM + TTS 双进程共存是"盲人互殴"——两套 CUDA Context 彼此不感知，驱动只能靠粗暴的时间片轮转来回切，没有任何跨模型的全局优化，比单模型高并发还要糟糕。

---

## 二、核心设计原则：按模型特征分配硬件

| 场景 | 最优策略 | 原理 |
|---|---|---|
| 单个小模型（0.6B TTS / ASR） | **单卡独占** | 无卡间通信开销，100% 独占带宽 |
| 超大模型（70B+ LLM，单卡装不下） | **单模型跨多卡（张量/流水线并行）** | 多卡协同提供成倍算力和显存 |
| 多模型流水线（LLM + TTS / ASR） | **不同模型分卡隔离** | 彻底消除资源争抢，各司其职 |

**误区提示**：把一个原本单卡能跑的小模型（如 0.6B）强行切分到多卡，卡间 PCIe 通信延迟（~5 ms）远大于模型计算耗时（~2 ms），多卡反而比单卡慢。

---

## 三、三种部署场景详解

### 3.1 云端服务器部署

**目标场景**：阿里云 ECS、腾讯云 GPU 实例等，算力充足、内网访问延迟低。

**推荐策略：双卡分离**

```
GPU 0（大卡，如 RTX 4090 / A100）：专职跑 LLM
GPU 1（小卡，如 RTX 4060 Ti 16G）：专职跑 ASR + TTS
```

通过 `CUDA_VISIBLE_DEVICES` 环境变量实现逻辑隔离，容器内部的模型自以为独占 GPU 0，实际物理上已分道扬镳：

**方案 A：宿主机直接启动**

```bash
# LLM 只看见卡 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-35B --port 8080

# TTS 只看见卡 1
CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.server \
  --model Qwen3-TTS-0.6B --port 8091

# ASR 也绑定卡 1
CUDA_VISIBLE_DEVICES=1 python -m funasr.server \
  --model SenseVoiceSmall --port 8090
```

**方案 B：Docker Compose（推荐生产环境）**

```yaml
version: '3.8'
services:
  llm-service:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  tts-service:
    image: vllm/vllm-omni:v0.20.0
    environment:
      - CUDA_VISIBLE_DEVICES=0          # 容器内视为 GPU 0，物理上是卡 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

**实测数据参考（阿里云 ECS RTX 4090，TTS 独占 10% 显存 + LLM 共存）**：

| TTS 并发数 | TTFA 均值 | 推荐上限 |
|---|---|---|
| 1 路 | ~280 ms（公网） | — |
| 4 路 | ~297–313 ms | 可接受 |
| 6 路 | ~327–349 ms | 推荐上限 |
| 8 路 | ~363–410 ms | 明显排队 |

> 注：以上公网延迟含客户端到服务器的 RTT（数十毫秒）。如果走内网 IP，延迟可显著降低。若 LLM 和 TTS 实现物理分卡，云端 TTS 回环延迟有望逼近本机独占的 ~84 ms 水平。

**显存与协议选型建议：**

- TTS 与 LLM 共存时，通过 `--gpu-memory-utilization` 限制 TTS 的 KV Cache 上限，让路给 LLM（如设置为 0.10，实测仍可支撑 5–6 路 TTS 并发）。
- 实时流式播放首选 **WebSocket 双向流**（比 HTTP 流式快 10–20 ms）；批量离线合成可用 **HTTP 流式**。
- 业务端和服务端同属内网时，务必通过**内网 IP**访问，彻底消除公网 RTT。

---

### 3.2 局域网服务器部署

**目标场景**：机器人机房、办公室内 GPU 工作站，服务 1–10 路机器人终端。

**推荐策略：双卡方案（最高性价比）**

```
GPU 0（RTX 4090 24G）：独占 LLM（Qwen3.5-14B 量化版 / 35B）
GPU 1（RTX 4060 Ti 16G）：合跑 ASR + TTS
```

**为什么 ASR 和 TTS 可以共用一张卡？**

1. **时序天然互补**：用户说话时 ASR 高负载，TTS 闲置；机器人回复时 TTS 高负载，ASR 仅做轻量唤醒检测。两者在时间轴上很少同时满负载。
2. **体量轻**：生产常用的 ASR（Whisper-Small / FunASR SenseVoice）+ TTS（0.6B 规模）总显存通常 4–6 GB，16G 显卡绰绰有余。
3. **脱离 LLM 绞杀**：从 LLM 那张卡上解放出来后，TTS 首帧随到随算，延迟可以稳定在 **84–120 ms**。

**实测数据（局域网主机，GeForce 8 GB，TTS 独占）**：

| 并发数 | TTFA | 说明 |
|---|---|---|
| 1 路 WebSocket | ~84 ms | 最优，接近官方参考值 |
| 1 路 HTTP 流式 | ~118 ms | 可接受 |
| 2 路 HTTP | ~142 ms/路 | 开始出现排队 |
| 推荐上限 | **1–2 路** | 8 GB 显存 KV Cache 仅 0.44 GiB |

**单卡极限（预算受限）**：

如果只有一张 RTX 4090 24G，必须将 LLM 压缩到 Qwen2.5-7B/14B INT4 量化，将总显存控制在 15 GB 内，剩余 9 GB 分给 ASR + TTS。此时建议开启 **NVIDIA MPS**，将 GPU 线程显式切片（给 ASR/TTS 预留 30% 算力），防止 LLM 长文本生成期间卡死语音通道。

**三卡方案（高并发中央服务器）**：

仅当局域网服务器需要同时支撑 50 路以上机器人并发时才有必要。配置如下：
- GPU 0：独占 ASR（批量流式音频解码）
- GPU 1：独占 LLM（Continuous Batching）
- GPU 2：独占 TTS（多路音频流并发合成）

---

### 3.3 端侧设备部署

**目标场景**：机器人本体或边缘节点，代表设备 RK3588 和 S100P。

端侧的挤兑问题比云端更严重，根本原因是**统一内存架构（Unified Memory）**：CPU、NPU/BPU、GPU 共享同一块 LPDDR 物理内存总线，带宽极其有限。

| 项目 | RK3588 | S100P |
|---|---|---|
| 算力 | 6 TOPS（NPU，INT8） | 128 TOPS（BPU，INT8） |
| 内存带宽 | ~34–42 GB/s（LPDDR4X） | 更高（LPDDR5） |
| 多模型隔离方式 | **物理核心绑定（3 个独立 NPU Core）** | **任务优先级调度（单 BPU 池）** |

#### RK3588 部署策略

RK3588 的 NPU 由 3 个物理独立的 Core（各 2 TOPS）组成，可以像"分卡"一样硬隔离：

```cpp
// ASR / TTS 绑定 Core 0（独占 2 TOPS，保证低延迟）
rknn_core_mask mask_tts = RKNN_NPU_CORE_0;
rknn_init(&ctx_tts, model_data, size, 0, &mask_tts);

// LLM 绑定 Core 1 + Core 2（合力 4 TOPS，保证推理吞吐）
rknn_core_mask mask_llm = RKNN_NPU_CORE_1_2;
rknn_init(&ctx_llm, model_data, size, 0, &mask_llm);
```

> **绝对不要使用 `RKNN_NPU_CORE_AUTO`**：自动分配会导致 TTS 和 LLM 在同一核心上排队，完全消除隔离收益。

**RK3588 完整资源分配建议：**

| 硬件单元 | 负责任务 |
|---|---|
| NPU Core 0 | ASR 识别 + TTS 首帧合成（保证响应速度） |
| NPU Core 1 + 2 | LLM 文本推理（保证吞吐） |
| Cortex-A76（高性能 CPU） | 主控逻辑、机器人运动规划、网络通信 |
| Mali-G610 GPU | 视觉预处理（摄像头/舌诊图像），不占用 NPU |

**额外优化：**
- LLM 必须用 **INT4/INT8 严格量化**（llama.cpp + RKNN 后端），每减少 1 GB 显存占用，对 LPDDR 带宽的挤压就成倍减轻。
- 使用 `rknn_create_mem` 的**零拷贝（Zero-Copy）API**，ASR 输出的特征向量和 LLM 输出的 Token ID 直接通过物理内存指针传给 TTS，严禁 `memcpy`。
- 所有模型在启动时**全量常驻内存**，禁止频繁动态加载/释放（每次 `rknn_init` 有数秒开销）。

#### S100P 部署策略

S100P 的 128 TOPS 算力是一个整体 BPU 池，没有像 RK3588 那样可手动绑定的物理核心标识。破局点在于软件层的**优先级抢占机制**：

```text
核心思路：
  LLM（Decoding）设为 Low Priority
  TTS（首帧 Prefill）设为 High Priority
  → BPU 完成 LLM 当前最小计算单元后，立刻插队执行 TTS Kernel
```

**具体实施：**

1. **任务优先级**：通过 `hb_dnn` API 为 TTS 推理任务声明 High Priority，LLM Decoding 降为 Normal/Low。TTS 请求一到，BPU 即在完成 LLM 当前计算单元后强行插队，保住首帧延迟。

2. **分句缓冲策略（Chunking）**：绝不在 LLM 吐出一个字时就立刻调用一次 TTS。在 LLM 和 TTS 之间加动态缓冲区：遇到句号、逗号等停顿标点，或累积到 6–10 个字，再整句推给 TTS。这样将 TTS 触发频率降低数倍，避免微秒级高频撞车。

3. **TROS 零拷贝通信**：整个 ASR → LLM → TTS 流水线基于**地瓜 TROS（TogetheROS.Bot）** 搭建，利用共享内存 Zero-Copy 机制传递数据，不走传统网络端口或高开销 IPC，把带宽 100% 留给模型推理。

4. **INT8 严格量化 + 模型常驻**：S100P 的 128 TOPS 算力建立在 INT8 之上，FP16 算子或不支持的算子会退化到 CPU 执行。必须用地瓜官方工具链将所有模型编译为 Nash 架构原生 `.bin` 文件，并全量常驻 24 GB 内存。

**S100P 资源分配建议：**

| 硬件单元 | 负责任务 |
|---|---|
| BPU（高优先级任务槽） | ASR 识别、TTS 首帧合成（High Priority） |
| BPU（低优先级任务槽） | LLM Decoding（Normal/Low Priority） |
| Cortex-A78AE CPU | 主控逻辑、TROS 通信、运动规划 |
| MCU 域 | 实时电机控制、底层总线通信 |

---

## 四、ASR / LLM / TTS 三类模型部署总结

### 4.1 各场景最优部署矩阵

| 部署环境 | ASR | LLM | TTS | 备注 |
|---|---|---|---|---|
| **云端服务器** | 与 TTS 共用小卡 | 独占大卡 | 与 ASR 共用小卡 | 优先内网访问；TTS 用 WS 流式 |
| **局域网服务器** | 与 TTS 共用 GPU 1 | 独占 GPU 0（大卡） | 与 ASR 共用 GPU 1 | 双卡方案首选 4060 Ti 16G |
| **RK3588 端侧** | NPU Core 0 | NPU Core 1+2 | NPU Core 0 | 必须手动 Core Binding |
| **S100P 端侧** | High Priority | Low Priority | High Priority | 分句缓冲 + TROS Zero-Copy |

### 4.2 延迟预期参考

| 模型类型 | 独占环境首帧延迟 | 共享环境首帧延迟 | 劣化根因 |
|---|---|---|---|
| TTS（0.6B） | ~84–120 ms | ~259–280 ms（公网）| SM 调度等待 + 带宽饥饿 |
| ASR（Whisper-Small） | ~50–100 ms | 受 LLM Prefill 影响 | 同上 |
| LLM（首 Token，35B） | ~200–500 ms | 受其他模型小幅影响 | LLM 自身计算为主 |

### 4.3 LLM 流式输出与 TTS 协同的注意点

LLM 流式输出时，TTS 首句出声需等待 LLM 凑齐第一个完整句子，总延迟通常在 **100–500 ms**（不含网络 RTT），主要取决于 LLM 吐字速度。优化方向：

1. **分句缓冲**：以语义标点（句号/逗号）或字数阈值（8–12 字）为 TTS 触发边界。
2. **流水线并行**：LLM 生成第二句时，TTS 同步合成第一句音频，两者并行运行。
3. **预热机制**：TTS 冷启动有 torch.compile/CUDA Graph 开销（数秒到数分钟），生产环境必须实现 warm-up，避免首次请求超时。

---

## 五、延伸思考

**当前方案的局限性：**

- 双卡方案对服务器机箱、主板 PCIe 通道数和电源功耗有要求（双 4090 峰值消耗 ~1200W），云端可选双卡 GPU 实例降低运维成本。
- RK3588 的 LPDDR4X 带宽（~40 GB/s）是硬天花板，即便分核隔离，LLM 参数量也需控制在 1.5B 以内，否则 Decoding 阶段仍会压垮整体带宽。
- S100P 缺乏物理核心绑定机制，优先级调度在极端高并发下仍可能出现抖动，长期看需要等待地瓜官方在 TROS 层提供更细粒度的算力隔离 API。

**横向对比：云端 vs 局域网 vs 端侧：**

- 云端：弹性算力、易于扩容，但公网 RTT 是延迟下限，适合多路并发业务；
- 局域网：延迟最低（回环 ~84 ms），数据不出内网，适合实时交互的单体机器人服务；
- 端侧：完全离线、零网络延迟，但算力和内存带宽受限，适合低并发的本地单机部署。

---

## 参考资料

- [Qwen3-TTS-0.6B 双环境部署基线报告](./2026-06-07_qwen3-tts-0.6b-部署基线报告.md)（本知识库）
- [Gemini 对话记录：多模型共享 GPU 的根因分析](C:/Users/wangp/Downloads/Gemini_2026-06-09.md)
- [Qwen3-TTS 模型卡片](https://huggingface.co/Qwen/Qwen3-TTS-0.6B)
- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm)
- [RKNN API 文档 — NPU Core Binding](https://github.com/rockchip-linux/rknn-toolkit2)
- [地瓜 RDK S100P 官方文档](https://developer.d-robotics.cc/rdk_doc/)
- [NVIDIA MPS 官方文档](https://docs.nvidia.com/deploy/mps/index.html)
