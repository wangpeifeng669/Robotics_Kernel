# Data Engineering · 数据工程与 RAG

## 模块定位

本模块负责**知识数据的隔离、清洗、检索增强（RAG）**以及意图识别，支撑业务问答与多轮对话的准确性。

## 核心主题

| 主题 | 说明 |
|------|------|
| 数据隔离 | 多租户/多业务线数据边界、权限与脱敏 |
| 意图识别 | 分类模型、槽位填充、与 LLM 的协同路由 |
| RAG 总方案 | 分块、Embedding、重排序、Hybrid Search |
| 测试案例 | 召回率/准确率评测集、Bad Case 回归 |

## 典型问题

- 如何避免 A 业务文档被 B 业务检索到？
- Chunk 大小与 overlap 如何根据文档类型调优？
- RAG 幻觉：引用溯源与「不知道」拒答策略如何设计？

## 目录建议

```
Data_Engineering/
├── rag/              # RAG 架构、Pipeline 与配置
├── intent/           # 意图识别方案与标注规范
├── isolation/        # 数据隔离与向量库分区策略
├── datasets/         # 评测集、Golden QA
└── eval/             # 自动化评测脚本与报告
```

## 关联模块

- `Model_Mechanics` — Embedding 与 Reranker 模型选型
- `Voice_Interaction` — 语音 query 进入 RAG 的前处理
- `Business_Multimodal` — 业务域知识与多模态检索
