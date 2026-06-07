# Edge Deployment · 端侧部署与硬件适配

## 模块定位

本模块覆盖**边缘设备上的模型部署、算子填坑与局域网私有化**，面向 RK3588、地瓜开发板等实际硬件平台。

## 核心主题

| 主题 | 说明 |
|------|------|
| RK3588 / 地瓜开发板 | NPU/GPU 能力、SDK 版本、交叉编译与环境 |
| 算子填坑 | ONNX/TensorRT/RKNN 不支持算子的替换、拆分与自定义实现 |
| 局域网私有化 | 离线推理、本地模型服务、无公网依赖的完整闭环 |

## 典型问题

- 某 LayerNorm/Softmax 变体在 RKNN 中报错，如何等价替换？
- 7B 模型在 8GB 内存板卡上如何分片或换小模型？
- 私有化部署下 OTA 更新与模型版本管理如何做？

## 目录建议

```
Edge_Deployment/
├── rk3588/           # RK3588 工具链、RKNN 转换与 benchmark
├── digua/            # 地瓜开发板专项笔记
├── operator_fix/     # 算子兼容性问题与解决方案
├── private_deploy/   # 局域网服务架构、Docker/ systemd
└── benchmarks/       # 端侧 FPS、延迟、功耗测试
```

## 关联模块

- `Model_Mechanics` — 量化、剪枝等模型侧优化
- `Comm_Architecture` — 端侧与云端/PC 的通信边界
- `Ops_Automation` — 环境脚本、驱动安装与排障
