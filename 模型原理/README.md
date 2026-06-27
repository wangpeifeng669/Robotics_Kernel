# 模型原理

## 模块定位

本模块沉淀**大模型与深度学习**的核心原理，为语音、视觉、决策等上层应用提供理论支撑与推理流程理解。

## 核心主题

| 主题 | 说明 |
|------|------|
| 大模型构建 | Pretrain、SFT、RLHF/DPO 等训练阶段与数据要求 |
| Transformer | Self-Attention、位置编码、KV Cache 与上下文窗口 |
| 推理流程 | Prefill/Decode、批处理、量化（INT8/INT4）、投机解码 |

## 典型问题

- KV Cache 如何影响显存占用与吞吐？
- 端侧 7B 模型 INT4 量化后精度损失如何评估？
- 流式输出时 token 级延迟如何拆解与优化？

## 目录建议

```
模型原理/
├── transformer/      # 架构细节、变体（MoE、Mamba 等）
├── training/         # 训练流程、显存优化、分布式
├── inference/        # 推理引擎、vLLM/llama.cpp 等
├── quantization/     # 量化方法与精度对比
└── papers/           # 论文笔记与复现记录
```

## 关联模块

- `语音交互/` — 端到端语音模型、ASR/TTS 底层
- `端侧部署/` — 模型裁剪、算子适配与硬件加速
- `数据工程/` — RAG 检索与 Embedding 模型
