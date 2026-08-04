# PyTorch-ONNX-RKNN模型格式对比与转换链路

PyTorch、ONNX、RKNN 三种格式分别对应模型生命周期中的三个阶段：训练产物、跨框架交换中间态、硬件绑定终态。核心规律是——每次格式转换都在**丢失灵活性、换取执行效率**：PyTorch 动态图可调试可修改但推理慢，ONNX 静态图可跨框架加载但只是"格式通用"而非"硬件通用"，RKNN 是 NPU 微码编译产物只能在 Rockchip 芯片上跑但推理极快。端侧部署的正确路径是 `.pt → .onnx → .rknn`，ONNX 是必经桥梁而非终点。

---

## 一、三种格式的本质定位

| 维度 | PyTorch (.pt/.pth) | ONNX (.onnx) | RKNN (.rknn) |
|------|---------------------|---------------|---------------|
| **本质** | Python 代码 + 权重快照 | 标准化静态计算图（protobuf） | NPU 微码编译产物 |
| **计算图** | 动态图（每轮实时构建） | 静态图（冻结拓扑） | 硬件指令序列 |
| **灵活性** | 最高——可改代码、加 hook、调试 | 中等——结构固定但可跨框架加载 | 最低——绑定特定芯片型号 |
| **推理速度** | 慢（Python 解释器开销） | 中（算子融合优化，但取决于 EP） | 极快（NPU 直接执行） |
| **体积** | 大（未压缩 FP32） | 中等 | 小（INT8 量化 + 图裁剪） |
| **适合场景** | 训练、微调、实验 | 跨框架交换、GPU 推理 | RK3588 NPU 端侧部署 |
| **硬件依赖** | PyTorch 库 | ONNX Runtime + EP | Rockchip NPU（型号绑定） |

### 1.1 PyTorch：训练的活代码

PyTorch 的 `.pt` / `.pth` 保存的是 `nn.Module` 的结构定义和参数 tensor。推理时仍依赖 PyTorch 运行时逐行执行 Python 逻辑，每轮前向传播实时构建计算图（动态图），可以按条件跳过分支、动态改变形状——这是训练时梯度反传的基础。

优点：训练/调试/修改全链路畅通，可以随时 print 中间 tensor、加 hook 观察梯度。

缺点：推理慢（Python 解释器开销 + 无算子融合），体积大，强依赖 PyTorch 运行时环境。

### 1.2 ONNX：跨框架搬运箱

ONNX 是一种标准化的静态计算图描述格式，用 protobuf 序列化。把 PyTorch 的动态图冻结成固定拓扑，算子类型和输入输出形状全部确定下来。

**ONNX 是格式层面的通用，不是硬件层面的通用。** 它解决的是"谁都能读"的问题，但"谁都能高效跑"还需要各硬件自己的编译层。

核心价值：跨框架交换——PyTorch 导出 → ONNX → TensorRT / OpenVINO / CoreML 各自加载，不需要为每个框架写一套模型代码。静态图也让后端引擎可以做 Conv+BN+ReLU 融合、常量折叠等优化，推理速度比 PyTorch 原生快 2-5 倍。

陷阱：不是所有 PyTorch 算子都能导出 ONNX（自定义算子、动态控制流如 `if`/`while`），导出时经常需要改模型代码适配。ONNX 版本碎片化（opset 11/13/17 算子集不同）也是痛点。

### 1.3 RKNN：NPU 的死命令

RKNN 是 Rockchip 专门为自家 NPU（RK3588/RK3568 等）设计的编译产物。ONNX 经过 rknn-toolkit2 被编译成 NPU 能直接执行的指令序列，INT8/INT16 量化在这一步完成。

极致优化：算子不只融合，而是被翻译成 NPU 硬件微码。推理跑在 NPU 上而非 CPU/GPU，延迟极低。INT8 量化把 FP32 权重压缩 4 倍，加上图优化裁剪冗余算子，模型体积通常缩小 3-4 倍。

代价：彻底绑定硬件，.rknn 只能在对应型号 Rockchip NPU 上跑。不是所有 ONNX 算子都被 NPU 支持（不支持的 fallback 到 CPU，性能打折）。调试手段有限——基本只能看输入输出，中间层 tensor 难窥探。

---

## 二、格式转换链路

端侧部署的完整路径：

```
训练/微调 → .pt (PyTorch)
    ↓ torch.onnx.export（注意 opset 版本、动态形状处理）
ONNX → .onnx
    ↓ rknn-toolkit2（量化 INT8、指定 target rk3588）
RKNN → .rknn → 推理 API 调用
```

### 2.1 PyTorch → ONNX 的关键注意点

```python
import torch

model = MyModel()
model.eval()

# 动态形状：ASR/TTS 输入长度不固定时必须设置
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,       # 算子集版本，常用 13/17
    dynamic_axes={
        "input": {0: "batch", 1: "seq_len"},  # 哪些维度可变
        "output": {0: "batch", 1: "seq_len"},
    },
    input_names=["input"],
    output_names=["output"],
)
```

导出后务必用 `onnxruntime` 验证推理结果与 PyTorch 一致，再进入下一步。

### 2.2 ONNX → RKNN 的关键注意点

```python
from rknn.api import RKNN

rknn = RKNN()

# 配置量化参数
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
    optimization_level=3,
)

# 加载 ONNX
rknn.load_onnx(model="model.onnx")

# 构建（含 INT8 量化）
rknn.build(
    do_quantization=True,
    dataset="dataset.txt",  # 量化校准数据集，影响精度
)

# 导出
rknn.export_rknn("model.rknn")
```

两个实战要点：
- **量化精度损失**：INT8 量化对分类/检测模型影响小，对 TTS/ASR 等浮点精度敏感的模型需要精细调校（per-channel 量化优于 per-tensor）
- **算子支持检查**：rknn 不支持的 ONNX 算子会自动 CPU fallback，那个层就慢了。先用 `onnxruntime` 验证正确性，再用 rknn-toolkit2 的 `hybrid_quantization_step` 查看哪些算子在 NPU、哪些在 CPU

---

## 三、ONNX Runtime 的执行机制

### 3.1 ONNX 不是"CPU 跑"，而是"按 EP 分发"

ONNX Runtime 默认确实用 CPU EP 跑，但核心设计是**按 Execution Provider 分发算子**——哪个 EP 支持某个算子，就把该算子分给它执行。

| 设备 | ONNX Runtime 执行方式 | 实际效果 |
|------|----------------------|----------|
| NVIDIA 4090 | CUDA EP / TensorRT EP | 跑在 GPU 上，充分利用 CUDA 核心 |
| RK3588 CPU | CPU EP（默认） | 跑在 ARM CPU 上，可用但 NPU 闲置 |
| RK3588 NPU | **不存在 NPU EP** | ONNX 无法直接驱动 NPU，必须转 RKNN |
| Intel GPU/NPU | OpenVINO EP | 跑在 Intel 硬件上 |
| Mac M 系列 | CoreML EP | 跑在 Apple Neural Engine/GPU 上 |

### 3.2 EP（Execution Provider）= 算子和硬件之间的翻译官

EP 是 ONNX Runtime 内部的概念，本质是"哪个硬件执行这个算子"的决策层。Runtime 加载模型后，分发器逐个扫描算子节点，按注册 EP 的优先级问："你支持这个算子吗？"

- CUDA EP 注册了 Conv、MatMul 等几十个算子的 GPU 实现 → 这些算子跑在 GPU 上
- CPU EP 支持全部 ONNX 标准算子 → 没被其他 EP 抢走的算子全部 CPU 兜底

**Rockchip 没有为 ONNX Runtime 写 RK3588 NPU EP，所以 NPU 完全闲置。** 正确路径是通过 rknn-toolkit2 编译成 .rknn，用 Rockchip 自己的 C API 调用 NPU。

### 3.3 ONNX 是"万能交换格式"，不是"万能运行格式"

类比：ONNX 像 Java 字节码——"一次编写到处加载"，但"到处高效运行"还得靠各平台的 JIT/硬件适配。每个硬件厂商还要做自己的编译层：

- NVIDIA → TensorRT（从 ONNX 进一步编译）
- Rockchip → rknn-toolkit2（从 ONNX 进一步编译）
- Intel → OpenVINO（从 ONNX 进一步编译）

---

## 四、算子与 MAC——为什么 GPU/NPU 跑模型更快

### 4.1 算子 = 计算的最小单元

神经网络的每一层"计算动作"就是一个算子。一个 ResNet50 的前向推理，本质上是几百个算子按顺序串起来执行：

```
输入 → Conv → BatchNorm → ReLU → MaxPool → Conv → ... → Softmax → 输出
```

ONNX 定义了约 200 个标准算子（opset），每个算子有固定名称、输入输出规格。ONNX 文件就是一张算子拓扑图。

### 4.2 MAC = 最原子级运算

**MAC（Multiply-Accumulate）= a × b + c**，一次乘加。这是硬件层面最基本的一步操作。

矩阵乘法的本质就是把海量 MAC 排成网格批量执行：

```
Y[j] = Σ X[i] × W[i,j]     ← 每个 Y[j] 是一组 MAC 的累加结果
```

3 维输入 × 384 维输出 → 一个 Linear 层 = 384×384 ≈ **14.7 万次 MAC**。

### 4.3 GPU 快不是因为"硬件内置了算子函数"

GPU 硬件层面只做一件事：**大量并行的乘法和加法（MAC）**。它没有 Conv 电路、没有 MatMul 电路。

GPU 快的根本原因：Conv/MatMul 的本质是海量乘加操作，天然适合并行。4090 有 16384 个 CUDA 核心，每个核心同时做几十个 MAC——86M 次乘加在一个时钟周期内铺开完成。

CPU 的 4-8 个大核心，单核确实快（频率高、流水线深），但串行做海量乘加需要循环几十万次——总量差距太大。

| | CPU | GPU | NPU |
|--|-----|-----|-----|
| 核心数 | 4-8 个 | 16384 个 | 几个专用单元 |
| 单核能力 | 很强（复杂逻辑） | 很弱（只会乘加） | 极专（只做 INT8 乘加） |
| 做复杂分支 | 快 | 慴（遇到 if/else 就卡） | 不支持 |
| 做海量乘加 | 慴（串行排队） | 极快（并行铺开） | 极快（专用电路） |

**模型推理 = 海量简单乘加 = 并行硬件的天堂。** CPU 是教授型核心，单兵强但不适合搬砖；GPU 是实习生型核心，16384 人每人搬一小块砖，总量碾压；NPU 是更极端的专用搬砖机器——核心更少更专、只做 INT8 乘加、不支持任何分支逻辑，但效率极高。

### 4.4 算子在 GPU 上的执行链路

CUDA EP 把抽象的 ONNX 算子翻译成 GPU 能执行的 **CUDA kernel**（并行程序），GPU 硬件执行 kernel 时 16384 个核心同时启动。不是 GPU 硬件里有 Conv 函数，而是 Conv 函数里全是 MAC，GPU 天生就是 MAC 并行机器。

---

## 五、端侧部署场景下的实际选型

### 5.1 场景对照表

| 场景 | 格式 | 执行方式 | 速度 |
|------|------|---------|------|
| 4090 云端推理 | ONNX + CUDA/TRT EP | GPU 直接跑 | 快 |
| 4090 云端极致优化 | TensorRT (.trt) | 从 ONNX 再编译 | 更快 |
| RK3588 CPU fallback | ONNX + CPU EP | CPU 跑，NPU 闲置 | 能用但慢 |
| RK3588 NPU 部署 | RKNN (.rknn) | NPU 直接跑 | 极快 |

### 5.2 动态形状问题

ASR/TTS 的输入音频长度不固定，ONNX 导出要设 `dynamic_axes`，但 RKNN 对动态形状支持有限——需要按固定 chunk 长度切片推理，或接受 padding 后裁剪。

### 5.3 量化精度策略

| 模型类型 | 量化影响 | 建议 |
|---------|---------|------|
| 分类/检测/Embedding | 小 | per-tensor INT8 即可 |
| TTS/ASR（浮点精度敏感） | 大 | per-channel INT8，精细调校 |
| LLM | 中 | 混合精度（关键层 FP16 + 其余 INT8） |

---

## 六、核心知识点速查

**一句话记忆链**：

```
MAC (a×b+c)              → 最原子级运算，一次乘加
向量乘矩阵                → 把 N 次 MAC 排成一组，算出一个输出数字
矩阵乘矩阵                → 把很多组 MAC 排成网格，一次算出整个输出向量
神经网络一层 = Y = W×X+b   → 一次大矩阵乘法 = 一层的全部推理计算
模型推理全过程             → 逐层做矩阵乘法 = 反复做 Y = W×X+b
格式转换链路               → .pt（训练）→ .onnx（交换）→ .rknn（NPU 执行）
每次转换的方向             → 丢失灵活性，换取执行效率
```

**三个核心区分**：

- **算子** ≈ 函数调用（`Conv()`、`MatMul()`）——描述"做什么计算"
- **EP** ≈ 函数的实现选择（同一 `Conv()`，CUDA EP 调 GPU 算，CPU EP 调 CPU 算）——描述"在哪里算"
- **MAC** ≈ 算子的原子操作（`a×b+c`）——硬件层面最基本的一步，GPU/NPU 用并行碾压

---

## 参考资料

- [ONNX 官方文档 - opset 版本与算子定义](https://onnx.ai/onnx/repo-docs/)
- [ONNX Runtime - Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [rknn-toolkit2 官方文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [PyTorch torch.onnx.export API](https://pytorch.org/docs/stable/onnx.html)
- [TensorRT ONNX Parser](https://docs.nvidia.com/deeplearning/tensorrt/archives/)
