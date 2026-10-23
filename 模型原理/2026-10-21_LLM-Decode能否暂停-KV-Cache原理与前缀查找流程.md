# LLM Decode 能否暂停 - KV Cache 原理与前缀查找流程

**核心摘要**：从模型层面看，decode 一定可以暂停 —— 自回归生成是 "一次 forward 出一个 token" 的循环，token 与 token 之间是天然的暂停点，而生成的全部状态只有 KV cache。实际部署中 "能不能暂停" 的差异，来自推理框架是否暴露了暂停这个原语：RK3588 上的 RKLLM 每生成一个 token 就回调一次，回调里可以让推理挂起、KV 原样保留；4090 上的 vLLM 单请求只有 abort（取消），恢复方式是把 "prompt + 已生成 token" 重新发一次。两者数学上完全等价：KV cache 是每个 token 经过每一层时算出的 K/V，新 token 只读取旧 KV、不修改旧 KV，所以只要前缀 token 序列一致，旧 KV 就能复用。vLLM 启动时就把 KV 显存切成固定数量的 block，block 一旦写满（16 个 token）立刻按内容 hash 登记缓存，等显存紧张时再按 LRU 淘汰最久未用的块 —— 这是一套 "边生成边缓存、惰性释放" 的显存管理机制。



***

## 一、问题背景：两个平台给出了相反的答案

在实际部署中遇到了一个看似矛盾的现象：



* **4090 上部署 Qwen3-VL-30B-A3B-Instruct-FP8（vLLM 框架）**：生成过程中不能暂停，只能取消（abort），想继续时把已经生成的文本拼回输入，再调用一次生成；

* **RK3588 上部署 Qwen2.5-1.5B-Instruct W8A8（RKLLM 框架）**：每个 decode 步骤都有回调，可以在回调里暂停，不继续生成下一个 token。

到底哪个对？模型到底支不支持 decode 中途暂停？要回答这个问题，需要先把三件事讲清楚：decode 循环的本质、KV cache 是什么、缓存是怎么被查找和复用的。

结论先放在前面：**两个平台都没有错，也都不是模型能力的差异。模型层面 decode 随时可以暂停；差异在于框架把生成循环放在了谁手里、给了什么控制接口。**



***

## 二、Decode 的本质：一个每步都可以停下的循环

大模型的生成方式叫**自回归（autoregressive）**：模型一次只产出一个 token，把它接到输入末尾，再产出下一个，循环往复。



```
[Prefill：一次性处理整段 prompt]
        ↓
  forward 出 token 1   ← token 边界
        ↓
  forward 出 token 2   ← token 边界
        ↓
  forward 出 token 3   ← token 边界（可以在这里停）
        ↓
  forward 出 token 4
        ↓
       ……（直到 EOS 或达到最大长度）
```

这里有两个关键认知：



1. **不存在 "生成了一半的 token"**。一次 forward 完整地算出一个 token，暂停只可能发生在 token 与 token 之间。所以 token 边界就是天然的暂停点，想停就停，不存在 "这一步必须算完否则状态损坏" 的问题。

2. **生成到任意位置时，全部状态只有两样东西**：已经生成的 token 序列，以及它们对应的 KV cache。没有任何藏在别处的隐藏状态。

这就解释了为什么 "暂停" 在数学上永远可行：只要把这两样东西留住，下次就能从断点继续；即使 KV 丢了，只要 token 序列还在，KV 也可以重新算出来 —— 因为 KV 是 token 序列的确定性函数，给定相同输入，算出来的 KV 逐字节相同。

那 KV cache 到底是什么？为什么它能代表全部状态？



***

## 三、KV Cache 到底是什么

### 3.1 回顾注意力：Q、K、V

Transformer 每一层的核心是自注意力。输入序列中的每个 token 会经过三个线性投影，得到三个向量：



* **Q（Query，查询）**：我在找什么信息；

* **K（Key，键）**：我这里有什么信息；

* **V（Value，值）**：我真正要交付的信息内容。

注意力的计算是：

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

通俗讲：每个 token 用自己的 Q 去和所有 token 的 K 做匹配，得到一组权重，再按权重把所有 token 的 V 加权求和，就是融合了上下文的输出。

### 3.2 KV Cache 的定义

一个 token 在经过**每一层**时，都会被投影出该层的 K 和 V。这些 K、V 一旦算出来，后面每生成一个新 token 都要反复用到。于是把它们存起来，这份存储就是 **KV cache**。

生成新 token 时：



* 只需要为新 token 计算它自己的 Q、K、V；

* 新 token 的 Q 去和缓存里所有历史 K/V 做注意力；

* 新 token 的 K、V 追加进缓存，供后续 token 使用。

KV cache 的总大小随序列长度线性增长：

$\text{KVCacheBytes} = 2 \times L \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times \text{BytesPerElem}$

其中开头的 2 表示 K 和 V 两份，L 是序列长度。每多一个 token，就多存一份 K/V。

### 3.3 两个方向的依赖要分开

理解 KV 复用最容易混淆的地方，是把两个方向的依赖混为一谈：



* **深度方向（层与层之间）**：第 2 层的输入是第 1 层的输出，所以同一次 forward 必须逐层计算，不能跳层。这一层间依赖真实存在，也是 KV 与 token 内容强相关的原因。

* **序列方向（token 与 token 之间）**：新 token 计算时通过注意力**读取**历史 token 的 KV，但**不会回头修改**它们。

为什么新 token 不会改旧 KV？因为 Transformer 用的是**因果注意力（causal attention）**：当前 token 只能看见自己和之前的 token，看不见未来。旧 token 的视野里根本没有后面新来的 token，所以新 token 加不加入，对旧 token 的计算结果毫无影响。

这就引出一个直接结论：

> 「你好」中 "你"" 好 "两个字的 KV，和「你好啊」中" 你 ""好" 的 KV 逐字节相同。新增的 "啊" 只是把自己的 K/V 追加进缓存，绝不回头改旧账。

所以前缀相同的序列，旧 KV 一定可以复用 —— 这正是 "前缀缓存（prefix caching）" 名字的由来。



***

## 四、没有 KV Cache 会怎样

按最朴素的方式实现生成：每生成一个新 token，都把整个序列（prompt + 所有已生成 token）重新送入模型，所有 token 的 Q/K/V 全部重算一遍。



* 序列长度为 N 时，总计算量约为 O (N²)，N 越大生成越慢；

* 而实际上，历史 token 算过的 K/V 从头到尾不会变，重算它们纯属浪费。

有了 KV cache，每生成一个 token 只需要计算这一个 token，decode 阶段的单步成本与序列长度基本无关。

代价是显存：KV cache 随序列长度线性增长，长上下文会占用大量显存。现代推理引擎的大量工程（PagedAttention、显存调度、前缀缓存）都是围绕 "如何高效管理这块 KV 显存" 展开的。



***

## 五、KV Cache 的查找与复用流程（以 vLLM 为例）

vLLM 是 4090 上最常用的推理引擎，它的 KV 管理机制也是理解 "暂停 / 取消" 问题的关键。

### 5.1 把 KV 切成 Block：PagedAttention

vLLM 借鉴操作系统的分页内存，把 KV cache 切成固定大小的 **block**，每个 block 存放 block\_size 个 token 的 K/V，默认 block\_size = 16。

启动时，引擎按 `gpu_memory_utilization`（默认 0.9）确定显存预算，**一次性预分配全部 block，组成一个 block pool**，block 数量在整个服务生命周期内固定。

引擎内部维护三个核心数据结构：



| 数据结构             | 作用                                |
| ---------------- | --------------------------------- |
| Block Pool       | 全部 block 的集合，启动时预分配               |
| Free Block Queue | 空闲 / 可淘汰 block 组成的双向链表，O (1) 移动元素 |
| Cache Blocks     | hash key → block 的映射表，即前缀缓存目录     |

### 5.2 Block 的三个状态

每个 block 在生命周期中处于三种状态之一：



| 状态          | 含义                                          |
| ----------- | ------------------------------------------- |
| **RUNNING** | 正在被某个请求使用（引用计数 ref\_cnt > 0）                |
| **CACHED**  | 没有请求使用（ref\_cnt = 0），但登记在 hash 表中，可被复用、可被淘汰 |
| **FREE**    | 完全空闲，内容无意义，可直接覆写                            |

### 5.3 Block 的 hash 怎么算：链式 hash

vLLM 采用基于 hash 的方法标识每个 block。一个 block 的 hash 由三部分组成：



* **父 block 的 hash**：前缀部分的 hash；

* **本 block 内的 token 序列**：精确的 token ID；

* **额外 hash**：LoRA ID、多模态输入 hash、多租户隔离 salt 等。

因为 hash 包含父 block 的 hash，所以这是**链式 hash**：前缀任意一个 token 改变，从该位置开始后续所有 block 的 hash 全部不同。默认 hash 算法为 sha256，基本杜绝碰撞。

### 5.4 缓存是什么时候写入的：边生成边写

这是最反直觉、也最关键的一点：**缓存不是请求结束时写入的，也不是 abort 时写入的，而是在生成过程中，一个 block 被填满的瞬间就写入。**



* prefill 阶段，分配的 block 一旦被 16 个 token 填满，立刻登记进 cache 表；

* decode 阶段同理，每凑满 16 个 token 就登记一个 block；

* 官方文档对 block 数据结构的注释写得很直白：block 的 hash "在 block 填满时赋值，在 block 被淘汰时重置"。

注意：**只缓存满 block（We only cache full blocks）**。每个请求最后那个不满 16 的 partial block，不会登记进 hash 表。

### 5.5 新请求的查找流程

一个新请求进来，调度器的处理顺序是：



```
1. get_computed_blocks()
   把请求的 token 序列按 block 计算 hash，
   从第一个 block 开始逐块查 cache 表
        ↓
2. 命中的 block：touch（ref_cnt + 1），
   从 free 队列中移除，防止被淘汰
   —— 这部分 token 的 forward 全部跳过，零计算
        ↓
3. 未命中的部分：从 free 队列取新 block，
   只对这些 token 做 prefill
        ↓
4. 新算的 block 一旦填满，立刻登记进 cache 表
        ↓
5. 进入 decode，每步只算新 token
```

命中的 block 是直接挂上物理 block 的引用（copy-on-write，多请求共享同一份物理 KV），省掉的不只是 K/V 计算，而是整个 forward（attention + MLP 全部跳过）。

### 5.6 两个例子对照

**例 1：「你好」→「你好啊」（尾巴加长）**



* 前缀 "你好" 逐 token 一致，对应的 KV 与旧序列完全相同，可以复用；

* 只需为 "啊" 计算新的 K/V，成本极低；

* 这就是 "前缀" 的含义：后面接什么、接多少都不影响前缀复用。

**例 2：「你好」→「你坏啊」（中间改了）**



* 第 2 个 token 从 "好" 变成 "坏"，从这个分歧点开始，链式 hash 全部对不上；

* 只有 "你" 的 KV 能复用，"坏"" 啊 " 全部重算。

一个工程细节：跨请求的缓存只认满 block，2 个 token 的 "你好" 是 partial block，单独请求结束后不会跨请求保留。长前缀（几百 token 的 system prompt、长对话历史）必然包含大量满 block，跨请求复用实实在在生效；同一请求内的多轮续写则是单 token 粒度的增量，旧 KV 始终保留。

### 5.7 显存不够时：LRU 惰性淘汰

请求结束（正常结束或 abort）时，block 的处理：



* 满 block：ref\_cnt 降为 0，从 RUNNING 变为 CACHED，加入 free 队列尾部，**数据不擦除**；

* 尾部 partial block：没进过 hash 表，直接回 FREE。

当新请求需要 block、free 队列里没有纯 FREE 块、轮到 CACHED 块被分配时，才发生淘汰：



* 从 free 队列**头部**取出 block—— 这就是最久未被使用的 block（LRU）；

* 从 cache 表移除、清掉 hash，block 回到 FREE 状态可被覆写。

请求结束时 block 按**逆序**（尾部 block 先）入队，因为越靠后的 block 越不可能被别的请求复用，应该最先被淘汰。

所以这是一套 \*\* 惰性释放（lazy eviction）\*\* 机制：数据清除是 "被逼出来的"，而不是请求结束时顺手做的。

### 5.8 什么情况下才会从头全量 Prefill

三种情况会导致缓存完全不命中、只能全量重算：



1. **前缀缓存没开**：vLLM v1 默认开启 APC；老版本或个别模型需要显式加 `--enable-prefix-caching`；

2. **重发的 token 序列与原序列不一致**：哪怕中间只差一个 token，从分歧点开始的 block 全部失效，所以已生成的 token 必须原样拼回；

3. **缓存被 LRU 淘汰**：取消到重发之间如果有大量其他请求把显存占满，旧 block 会被驱逐 —— 单卡、低并发场景很少发生。

验证命中情况有两种方式：Prometheus 指标 `vllm:gpu_prefix_cache_hits` 与 `vllm:gpu_prefix_cache_queries` 的比值就是命中率；也可以看 TTFT（首 token 延迟），命中时即使输入上千 token，TTFT 也接近一次 decode 的延迟。



***

## 六、回到最初的问题：Decode 能否暂停

现在可以把两个平台的行为放在一起对照了。

### 6.1 RK3588・RKLLM：回调式，循环在你手里

RKLLM 是端侧单用户框架，生成循环跑在你的 `rkllm_run()` 调用中。每 decode 出一个 token，SDK 立刻回调你注册的函数。回调的返回值决定走向（官方头文件注释）：



```
// 回调函数：每生成一个 token 触发一次
int callback(result, userdata, state) {
    // return 0：正常继续推理
    // return 1：暂停推理（可编辑输出、注入新 prompt），
    //           之后再次调用 rkllm_run 即可恢复
    if (need_pause) return 1;
    return 0;
}
```



* 返回 1 时，生成循环退出，但**不销毁 handle、不释放 KV**，KV 原样留在内存中；

* 之后再次调用 `rkllm_run`（可以带上修改后的内容），从断点直接继续，**恢复成本为 0**。

### 6.2 4090・vLLM：调度引擎，循环在引擎手里

vLLM 是服务端框架，生成循环在引擎调度器中，通过 continuous batching 一次前向同时推进整批请求，单请求无法干预循环：



* 单请求只有两个选择：继续读流，或 abort；

* vLLM 确实有 `/pause` 接口，但那是**引擎级**的 —— 暂停后整个服务拒绝所有请求（用于异步强化学习时同步权重），不是 "暂停某一路生成"；

* abort 后，满 block 转入前缀缓存（CACHED，保留），尾部 partial block 丢弃；

* 恢复方式：把 "prompt + 已生成 token" 作为新请求重新提交，满 block 按 hash 命中、零计算复用，最多只重算最后 15 个 token。

### 6.3 两种方式的对照



| 维度               | RKLLM（暂停）   | vLLM（abort + 重发）              |
| ---------------- | ----------- | ----------------------------- |
| 满 block          | 原地保留        | 进缓存，hash 命中后零计算复用             |
| 尾部 partial block | 保留          | 丢弃，最多重算 15 个 token            |
| 恢复计算量            | 0，直接 decode | 1 次小 prefill（≤15 token）+ 调度开销 |
| 输出结果             | 一致          | 一致（相同 token + 相同采样参数）         |

所以 "暂停" 和 "取消再重发" 操作的是同一个状态（KV cache），区别只是：**等待期间 KV 留在显存里，还是先释放、再用 token 序列确定性地重建。**

如果在 4090 上确实需要 "挂起而不释放" 的原语（比如人工中途改稿、注入引导），可以改用进程内的推理库：HuggingFace transformers（用 `StoppingCriteria` 停下、拿 `past_key_values` 续写）或 llama.cpp（回调式），行为就和 RKLLM 一致。

代价其实是对称的：RKLLM 挂起期间，max\_context 对应的 KV 内存也一直被占着 —— 只是端侧没有别的请求争抢，所以感知不到这个代价。



***

## 七、生活化类比：一间固定工位的办公室

把 vLLM 的显存管理想象成一间办公室：



1. **办公室开业时工位就固定了**（block pool，启动时一次性划分），数量不随生意好坏变化；

2. **工人（请求）来了领工位**，每写满 16 行草稿，就把这份草稿按内容登记到办公室的共享目录（hash 表）里 —— 登记是边写边做的，不是下班时才做；

3. **工人下班（请求结束或取消），工位不立刻清空**：写满的草稿按目录留在工位上（CACHED），别人可以照着目录找到、接着用；只有最后没写满 16 行的那页纸直接扔掉；

4. **新工人来了，先查目录**：如果发现自己的开头和某份旧草稿一样，就直接把那份草稿拿过来用（引用计数 +1），只补写后面不一样的部分；

5. **只有当办公室真的没有空工位时，才清理最久没人碰过的工位**（LRU），把旧草稿销毁；

6. **"暂停" 就是工人把笔放下、工位占着去吃饭**，回来原样接着写；**"取消" 是工人离职**，但草稿留在目录里，新工人按目录找到它、补写最后几行即可。

两种情况下，工作成果都不会丢，区别只是工位占着等，还是让出来、之后再按目录找回。



***

## 八、知识点速查表



| 概念                      | 作用                                  |
| ----------------------- | ----------------------------------- |
| 自回归生成                   | 一次 forward 出一个 token，循环直到 EOS       |
| Token 边界                | 天然暂停点，不存在 "生成一半的 token"             |
| KV Cache                | 每个 token 在每层的 K/V 缓存，是生成的全部状态       |
| 因果注意力                   | 旧 token 看不见新 token，新 token 不改旧 KV   |
| Block（PagedAttention）   | KV 按 16 token 分页管理                  |
| 链式 hash                 | block hash 依赖父 hash + 本 block token |
| 缓存写入时机                  | block 填满即写，边生成边写，只缓存满 block         |
| 命中复用                    | 命中 block 直接引用，整个 forward 跳过，零计算     |
| RUNNING / CACHED / FREE | block 三状态：使用中 / 缓存可复用 / 空闲          |
| LRU 淘汰                  | 显存不足时淘汰队头最久未用的 CACHED block         |
| RKLLM 回调                | 返回 0 继续、1 挂起，KV 保留                  |
| vLLM abort              | 单请求取消；满 block 进缓存，partial 丢弃        |
| 全量重算条件                  | APC 未开、序列不一致、缓存被 LRU 淘汰             |



***

## 九、延伸思考



* **为什么服务端不做单请求暂停？** 云端显存是多用户共享的生命线，一个暂停的请求占着 KV 不放，会挤占其他请求的空间、拉低全局吞吐；而 HTTP 是请求 - 响应模型，"挂起一个请求" 语义上也不自然。端侧单用户场景下，KV 内存静态预留、没人争抢，暂停的代价可以忽略，所以 RKLLM 直接提供了挂起原语。

* **多轮对话为什么便宜？** 固定的 system prompt、重复的对话历史都由前缀缓存覆盖，整条前缀只算一次，后续每轮只处理新增 token。这也是 OpenAI、Anthropic 等公共端点普遍启用前缀缓存的原因。

* **多模态输入的缓存**：图片等非文本输入会被转成 placeholder token，缓存时把图片内容的 hash 纳入 block hash—— 只有同一张图片才能命中。

* **缓存的生命周期**：前缀缓存是进程级的，vLLM 服务重启即清空；多租户场景可用 per-request salt 隔离缓存，避免内容被侧信道探测。

* **与仓库内其他知识的关系**：本文是 `2026-06-22_大模型推理流程.md` 中 "KV Cache 工程延伸" 一节的展开；decode 阶段为什么受内存带宽限制，见 `2026-09-27_内存带宽为何决定解码速度-算术强度与带宽三因子.md`；vLLM 从 token 到 SSE 接口的完整链路，见 `智能体开发/2026-10-19_vLLM推理引擎-从token到SSE接口.md`。



***

## 参考资料



* vLLM 官方设计文档：Automatic Prefix Caching（block 数据结构、链式 hash、分配 / 释放 / 淘汰流程）

* vLLM 官方文档：Online Serving（/abort\_requests、/pause、Sleep Mode）

* RKLLM 官方头文件：LLMResultCallback 回调语义（0 继续 / 1 暂停）

* Vaswani et al. *Attention Is All You Need*, NeurIPS 2017

* 仓库内相关文章：`模型原理/2026-06-22_大模型推理流程.md`、`模型原理/2026-09-27_内存带宽为何决定解码速度-算术强度与带宽三因子.md`、`智能体开发/2026-10-19_vLLM推理引擎-从token到SSE接口.md`