# vits-zh-aishell3转RKNN实测

**核心摘要**：sherpa-onnx 官方包 vits-icefall-zh-aishell3 无法整图收成一个 `.rknn`。切分位置在潜变量按时长拉长之后：text_encoder、时长预测、长度扩展留在 CPU，flow 与 decoder 收成 `flow_decoder.rknn` 交给 NPU。为定死动态帧数，潜变量补零冻结到 160 帧。本机 simulator 上与官方全图相比，有效波形余弦相似度 0.999569，转换日志无算子回退。

> **环境与口径**
> 模型：sherpa-onnx 官方包 `vits-icefall-zh-aishell3`（AISHELL-3 中文多说话人 VITS）
> 目标：RK3588，RKNN Toolkit2 2.3.2，FP16，不量化
> 上板目录：`deploy/l2_vits_zh_aishell3/board_pkg`（源工程）
> 日期：2026-09-30（东八区）

整张图没有收成一个 `.rknn`。前半段留在 CPU 上用 ONNX Runtime 跑，后半段收成 `flow_decoder.rknn` 交给 NPU。切开的位置在「潜变量已经按音素时长拉长」之后：`text_encoder`、时长预测和长度扩展在 CPU，`flow` 和 `decoder` 在 NPU。

本机 simulator 对同一句「今天天气不错」、说话人 21，有效波形的余弦相似度是 0.999569，最大绝对误差 2.14×10⁻²。转换日志里没有算子回退。

---

## 一、这个模型做什么

它是一个中文语音合成模型：输入一句话和一个说话人编号，输出一段 8 kHz 单声道波形。训练数据来自 AISHELL-3，说话人 174 个，编号 0 到 173。音素表 219 个，覆盖声母、带调韵母，以及句首 `^`、句尾 `eos`。

采样率 8000 Hz，一帧对应 256 个采样点。三层转置卷积的步长是 8×8×4，乘起来正好是 256。所以「多少帧」和「多少采样点」是锁死的。

板上这次用的文本前端只查词典：一个汉字对一条音素，标点和空格跳过，数字和日期不展开。安装包里虽然带了 `date.fst`、`number.fst`，当前脚本没有走那条规则。

---

## 二、从字到声音

图外先把汉字变成音素编号，图内再走六步。官方 `model.onnx` 只有一个输出 `audio`，中间结果都在图里面。

```text
汉字
  → lexicon / tokens          图外，得到 tokens
  → text_encoder              每个音素一条 96 维特征，并拆成均值和方差
  → duration_predictor        每个音素要占几帧
  → 长度扩展 + 采样            按时长把特征拉长，再加一点噪声，得到潜变量
  → flow                      把潜变量变成声码器能用的特征
  → decoder                   特征变成波形
  → audio
```

旁边还有一条细支路：说话人编号经过 `global_emb`，变成 256 维向量，形状 `(1, 256, 1)`。时长预测、flow、decoder 都用它来换音色。

各步在图里的实际含义：

| 步骤 | 输入 | 输出 | 在图里对应什么 |
|---|---|---|---|
| tokens | 汉字 | `tokens` INT64 `[1, 音素数]`，另有长度、噪声、语速、说话人 | `lexicon.txt` + `tokens.txt`。句首补 `^`，句尾补 `eos` |
| text_encoder | 音素编号 | 每个音素的均值和对数方差 | 词嵌入 + 6 层 Conformer（有 macaron 前馈、自注意力、卷积模块）+ 投影 |
| duration_predictor | 编码器特征和说话人向量 | 每个音素的时长 | 随机时长预测器。内部有 3 段样条流，节点非常多 |
| 长度扩展 | 时长、编码器特征 | `latent` `(1, 96, 帧数)`，`mask` `(1, 1, 帧数)` | 没有模块前缀。`Exp → 乘语速 alpha → Ceil` 得到整数帧数，再用 `CumSum` / `Range` / `MatMul` 把音素特征按帧复制，并采样 `均值 + 噪声 × 标准差` |
| flow | 潜变量、mask、说话人向量 | 同形状特征 | 4 层卷积耦合（`flows.0/2/4/6`）和 4 次通道对调（`flows.1/3/5/7`）交替。导出图已经按推理方向接好 |
| decoder | flow 的输出（先乘 mask） | `audio` | HiFi-GAN 式声码器：输入卷积、3 次转置卷积上采样、9 个残差块、最后 `Tanh` |
| audio | 波形 | 8 kHz PCM | 只保留真实帧数 × 256 个采样点，补零的尾巴丢掉 |

时长预测器内部的 `duration_predictor/flows.*` 和后面的 `flow` 不是同一段。前者决定「这个字读多长」，后者决定「拉长之后的特征长什么样」。

---

## 三、节点和算子

官方 `model.onnx`：opset 13，**4896 个节点、51 种算子**、500 个初始权重张量，文件 29.1 MiB，权重约 24.4 MiB（float32）。

4896 是图上的执行步，不是 4896 层网络。其中 **1392 个是 `Constant`**，用来放形状、切片下标和折出来的小常量。去掉常量后还有 3504 个计算节点。

四个带名字的模块，加上一段没有模块前缀的图级计算：

| 模块 | 节点 | 去掉 Constant | 算子种类 | 权重大约 | 去向 |
|---|---:|---:|---:|---:|---|
| `text_encoder` | 1482 | 1178 | 31 | 算在 CPU 前段里 | CPU |
| `duration_predictor` | 2668 | 1853 | 40 | 算在 CPU 前段里 | CPU |
| 图级胶水（说话人、时长取整、长度扩展、采样） | 288 | 83 | 30 | 很少 | CPU |
| `flow` | 340 | 280 | 14 | 算在 NPU 子图里 | NPU |
| `decoder` | 114 | 108 | 7 | 算在 NPU 子图里 | NPU |
| NPU 侧胶水（mask 相乘、输出 Squeeze） | 4 | 2 | 3 | 可忽略 | NPU |
| 合计 | 4896 | 3504 | 51 | 24.4 MiB | |

CPU 前段 `front.onnx` 是前三行，4438 个节点、约 6.1 MiB 权重。NPU 子图 `flow_decoder.onnx` 是后三行，458 个节点、约 18.3 MiB 权重。节点多的是时长预测器，参数重的是声码器这一侧。

时长预测器为什么这么胀：`flows.3`、`flows.5`、`flows.7` 各有 828 个节点。它们是分段有理样条，导出 ONNX 时用 `NonZero`、`GatherND`、`Range`、`CumSum` 做「这个值落在哪一段」的查找。卷积本身只有旁边的 `dds`（3 层）和耦合网络，节点并不多。

全图 51 种算子里，Toolkit2 2.3.2 支持表标成 **Not Supported** 的有 8 种、112 个节点：

| 不支持的算子 | 节点数 | 落在哪 |
|---|---:|---|
| Range | 51 | 编码器 13，时长预测 36，长度扩展 2 |
| NonZero | 21 | 全在时长预测的样条里 |
| GatherND | 15 | 全在时长预测的样条里 |
| Neg | 10 | 时长预测 6，flow 4 |
| CumSum | 7 | 时长预测 6，长度扩展 1 |
| Not | 5 | 编码器 1，时长预测 3，长度扩展 1 |
| RandomNormalLike | 2 | 时长噪声 1，潜变量采样 1 |
| Ceil | 1 | 时长向上取整 |

其余 43 种在支持表里，包括卷积、矩阵乘、注意力用的 `Softmax`、归一化拆开的 `ReduceMean` / `Pow` / `Sqrt`。支持不等于能整图编译：编码器的时间长度随句子变，RKNN 这版要的是固定形状。

计算量比较集中的算子（含 Constant）：

| 算子 | 节点 | 主要出现的地方 |
|---|---:|---|
| Constant | 1392 | 各处的形状和下标 |
| Add | 406 | 残差、归一化 |
| Mul | 368 | mask、仿射、噪声缩放 |
| Shape / Unsqueeze / Reshape / Gather | 276 / 263 / 154 / 152 | 动态形状和样条查找 |
| Conv | 147 | 时长网络、flow、decoder |
| MatMul | 62 | 编码器自注意力 60，长度扩展 2 |
| Softmax | 12 | 编码器 6 层各 1，时长预测 6 |
| LeakyRelu | 34 | 全在 decoder |
| ConvTranspose | 3 | decoder 的三次上采样 |
| Tanh | 17 | flow 的门控 16，decoder 输出 1 |

---

## 四、哪些模块能上 NPU

判断标准就两条：算子在 Toolkit2 2.3.2 的支持表里，并且形状能在编译前定死。

**可以上 NPU 的是 `flow` 和 `decoder`。**

`decoder` 是一维卷积声码器，和已经在 RK3588 上跑通的 HiFi-GAN 同一类：`Conv`、`LeakyRelu`、`ConvTranspose`、`Add`、`Tanh`。114 个节点里没有不支持的算子。

`flow` 的主体也是卷积。每一层耦合把通道对半拆开，用一小段带说话人条件的卷积网络预测缩放和平移，再和另一半拼回去。中间夹着通道对调，对调在图里只是一次 `Slice`。整段 340 个节点里，唯一不支持的是 4 个 `Neg`（耦合层里的取负）。取负和乘常量 `-1` 相同，而 `Mul` 在支持表里，所以切图时把这 4 个 `Neg` 改成 `Mul`，并补一个名为 `neg_one` 的常量。改完之后这段子图没有不支持算子。

这两段还有一个共同条件：时间长度必须固定。句子有长有短，扩展之后的帧数不固定，RKNN 拒收这种动态轴。处理办法是在脚本里把潜变量和 mask 右侧补零到 **160 帧**。160 × 256 / 8000 = 5.12 秒。短句听的是前面的有效帧，补零只影响结尾几帧的卷积上下文。超过 160 帧，板上脚本直接退出，不截断。

**留在 CPU 的是 `text_encoder`、`duration_predictor` 和长度扩展。**

- 音素个数、扩展后的帧数都随句子变。这三段的时间维是动态的。
- 不支持算子几乎都堆在这里：样条查找（`NonZero`、`GatherND`、`Range`、`CumSum`）、时长取整（`Ceil`）、两处高斯采样（`RandomNormalLike`）。样条那三块就有 2484 个节点，不是改一两个算子名就能搬上 NPU。
- 编码器虽然主体是矩阵乘和卷积，但相对位置和 mask 用了 `Range`、`Not`，序列长度又不定。整段编码器没有单独送去转换。

说话人嵌入只有一次 `Gather`，计算量可以忽略，它和长度扩展写在同一段 CPU 图里，输出名叫 `speaker_embed`，交给后面的 NPU 子图当条件。

---

## 五、怎么切成 CPU 一段、NPU 一段

切图脚本是 `deploy/l2_vits_zh_aishell3/cut_graphs.py`（源工程）。它不重训、不从 checkpoint 重导，只在官方 `model.onnx` 上按张量名字往回收集节点。

三支边界张量：

| 图内原名 | 切出来之后的名字 | 形状 | 含义 |
|---|---|---|---|
| `/Add_output_0` | `latent` | `(1, 96, 帧数)` | 加过噪声的潜变量 |
| `/Cast_4_output_0` | `mask` | `(1, 1, 帧数)` | 有效帧为 1，填充为 0 |
| `/Unsqueeze_output_0` | `speaker_embed` | `(1, 256, 1)` | 说话人条件 |

从这三支往回走，得到 `front.onnx`：输入仍是官方那 6 个（`tokens`、`tokens_lens`、`noise_scale`、`alpha`、`noise_scale_dur`、`speaker`），输出就是上面三个。

从这三支往 `audio` 走，得到声码器子图。然后再做两件事：

1. 4 个 `Neg` 改成乘 `-1`。
2. 把动态帧数冻成 160。输入变成 `latent (1, 96, 160)`、`mask (1, 1, 160)`、`speaker_embed (1, 256, 1)`，输出 `audio (1, 40960)`。

冻形状之后，用「你好」对过一次：动态子图和固定 160 帧子图，主体最大绝对误差小于 1×10⁻⁴，含尾音的整段小于 1×10⁻²。补零会改变结尾几帧的卷积邻域，所以允许尾部稍大，主体必须贴住。

转换脚本是 `convert_rknn.py`：

```text
config(target_platform=rk3588, float_dtype=float16)
load_onnx(flow_decoder.onnx)
build(do_quantization=False)
export → models/flow_decoder_fp16.rknn
```

三个输入都是浮点特征，不按图像做减均值除标准差。编译日志里的融合主要是：常量折叠、把一维卷积垫成 NPU 要的四维布局、若干个无效的 Add/Mul 被收掉、`Div` 收成 `Mul`。日志里没有 `fallback` 或 `not support`。

装进 `board_pkg` 的是这两份模型，外加词典和未切分的官方全图：

| 文件 | 作用 |
|---|---|
| `front.onnx`（10.7 MiB） | CPU。文本编码、时长、长度扩展、采样 |
| `flow_decoder.rknn`（13.0 MiB） | NPU。flow + decoder，FP16 |
| `model.onnx`（29.1 MiB） | 未切分的官方全图，用来听感对照 |
| `infer_board.py` | 把两段串起来 |
| `lexicon.txt` / `tokens.txt` / `speakers.txt` | 汉字到音素、音素到编号、说话人列表 |

`flow_decoder.rknn` 只能在板上用 `librknnrt` 2.3.2 跑。本机对照走的是 Toolkit 的 simulator，每次对照都会重新 `build` 一次，不直接加载这块 rknn 文件。

---

## 六、怎么在板上跑完整链路

`board_pkg/infer_board.py` 把两段接上。两个模型不会自动相连。

```text
汉字
  → phoneme_ids()                 词典，得到 token 编号
  → onnxruntime 跑 front.onnx     CPU，得到 latent / mask / speaker_embed
  → 右侧补零到 160 帧              记下真实帧数
  → rknn-toolkit-lite2 跑
    flow_decoder.rknn             NPU，得到 40960 点
  → 只留 真实帧数 × 256 个点       写成 8 kHz、16 bit wav
```

噪声尺度 `noise_scale=0.667`、时长噪声 `noise_scale_dur=0.8`、语速 `alpha=1.0`，与 sherpa-onnx 离线 VITS 的默认值一致。换掉会改变音色和语速。

前段里有 `RandomNormalLike`，同一次前向的噪声不能重放。听感对照要先跑官方全图，把同一次的潜变量取出来再送进 RKNN，两段波形的差别才只来自 FP16。`export_wav.py` 就是这么写 `full.wav` 和 `rknn.wav` 的。

要看每一层落在 NPU 还是 CPU，板上可以：

```bash
RKNN_LOG_LEVEL=4 python3 infer_board.py --text "今天天气不错" --sid 21 --output out.wav
```

这句话的 NPU 子图在转换时没有算子回退，预期是计算层在 NPU，CPU 只负责这段子图的输入输出搬运。前段 `front.onnx` 整段在 CPU，不经过 RKNN。

---

## 参考资料

- `deploy/l2_vits_zh_aishell3/cut_graphs.py` — 切图脚本（源工程）
- `deploy/l2_vits_zh_aishell3/convert_rknn.py` — RKNN 转换脚本（源工程）
- `deploy/l2_vits_zh_aishell3/board_pkg/infer_board.py` — 板上两段串联推理脚本（源工程）
- `deploy/l2_vits_zh_aishell3/board_pkg/export_wav.py` — 听感对照波形导出脚本（源工程）
- sherpa-onnx 官方包 `vits-icefall-zh-aishell3`（AISHELL-3 中文多说话人 VITS）
