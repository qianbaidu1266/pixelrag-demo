# PixelRAG 技术原理

> 仓库：https://github.com/StarTrail-org/PixelRAG （Apache 2.0，Berkeley SkyLab / BAIR）
> 本文基于 GitHub main 分支代码与 arXiv 论文核实，非 README 转述。

## 一句话概括

文档不做任何 OCR / 解析，直接渲染成图片切块，用图文统一向量模型（Qwen3-VL-Embedding-2B）向量化后建索引；查询时用文字直接命中图片，答案由 VLM 看着命中的那张图读出来。全程没有"图片翻译成文字"这一步。

## 三步流水线

**1. 渲染（Render）**
- 网页：无头 Chromium 截图；PDF：poppler 渲染成图。
- 长页面（网页）先按 8192px 高切成 tile，再横切成小条。**PDF 实测路径不同**（本机验证）：每页 PDF 渲染成一张 1700×2200 的图，一页就是一个 tile、一个 chunk，不再细切（35 页 PDF → 35 个 tile，y_offset 全 0）。
- 关键观念转变：这里没有"文本 chunk"，**一个 chunk 就是一张图片**。对 PDF 来说 tile 序号 ≈ 页码（tile_0000 是第 1 页）。
- 代码里 1024px 小条逻辑是针对超长网页截图的（把视觉 token 压到八分之一以下），PDF 页面高度在阈值内不触发。

**2. 向量化（Embed）**
- 每条小图送进 `Qwen3-VL-Embedding-2B`，输出 **2048 维单向量**，存 FAISS / Qdrant。
- 图片从头到尾没变成文字——VLM 视觉编码器的中间表示直接当向量用。
- 文字 query 用同一个模型编码，和图片向量在同一个空间里直接算余弦相似度。

**3. 读答案（Read）**
- 命中的小图当像素直接喂给 VLM（如 Qwen3-VL），让它看图回答。
- 论文明确写 "without intermediate text conversion"——检索端和阅读端都是视觉的，这才是完整的端到端视觉 RAG。

## 与 ColPali 的路线区别（重点）

两者都是"真多模态"（图片不转文字），但向量粒度设计完全不同：

| | ColPali / ColQwen | PixelRAG |
|---|---|---|
| 切分方式 | 模型**内部** patch 化，页面物理上是一整张图 | **外部物理切块**，1024px 一条 |
| 向量粒度 | 多向量（late interaction），一页一堆 patch 向量 | 单向量，一个 chunk 一个向量 |
| 匹配算法 | MaxSim：query 每个词向量去页面 patch 里找最高分再求和 | 标准向量检索（余弦 / 内积） |
| 索引要求 | 需要多向量检索器（fast-plaid 等） | 标准 FAISS / Qdrant，HNSW、量化、payload 过滤全是现成的 |
| 检索粒度上限 | patch 级，更细 | 单 chunk 一个向量，粗一些，但切块够小时差距不大 |
| 中文基座 | ColPali 基座 PaliGemma 中文弱，**必须选 ColQwen** | Qwen3-VL，中文天然好 |
| Mac 可跑性 | byaldi 官方承认 MPS 编码性能差，flash-attn 是 CUDA-only | `embed_cpu.py` 明确支持 MPS，一等公民 |

工程含义：PixelRAG 把"多模态"从检索算法问题（MaxSim + 专用索引）降维成了"切块策略 + 标准向量库"问题，这是它能做到三千万级截图索引的原因。

## 与"翻译型多模态 RAG"的本质区别

翻译型（WeKnora 默认路径、绝大多数 RAG 的图片处理）：图片 → VLM 生成文字描述 → 文字向量。
问题是**生成描述是有损压缩**：VLM 没写进描述里的信息在半路被永久扔掉，query 落在描述之外时检索彻底失效且无补救。

PixelRAG / ColPali 的损失只发生在"编码压缩"一步，没有"语言生成"瓶颈——图上任何视觉信息（表格线、柱状图、版式、公式排版）都保留在像素里。

## 论文实验亮点

- 在 NQ、SimpleQA 这类**纯文本问答**任务上也打赢了文本 RAG——因为传统解析管线会把表格拍平、丢版式，截图把这些全保住了。
- 降低截图分辨率做检索，token 成本降到约 1/3，精度不掉。

## 局限

- 只适合"版式即信息"的文档（网页、PDF、扫描件）。纯结构化文本（如 ISO 20022 报文 XML）用不上它。
- 截图模型是在英文维基上 LoRA 微调的，**中文网页/中文 PDF 的效果论文未覆盖**，需实测（本机实测中文查询命中正常，见快速启动文档）。
- 本地已验证的坑见《PixelRAG快速启动指南.md》附录。
