# pixelrag-demo

基于 [PixelRAG](https://github.com/StarTrail-org/PixelRAG) 验证多模态 RAG 能力：Qwen3-VL-Embedding 图文向量化 + DeepSeek 视觉问答，全流程本地跑通，中文行业报告 PDF 实测召回全对。

在一台 **没有 NVIDIA 显卡的 Mac（M1 Pro 32GB）** 上端到端验证：文字提问 → 直接命中 PDF 页面截图 → 视觉模型看图生成答案。全程不经过 OCR、不做任何"图片转文字"。

## 这是什么

传统 RAG 把文档解析成文本再检索——表格被拍平、版式信息丢失、图片彻底不可见。PixelRAG 的思路是**不解析，直接看图**：把 PDF 每页渲染成截图，用图文统一向量模型编码入库，检索命中的就是那页截图，再让视觉模型对着图回答问题。

```
docs/*.pdf ──渲染──> 每页一张截图(tile) ──Qwen3-VL-Embedding──> FAISS 向量索引
                                                                    │
用户提问(文字) ──同一模型编码──> 向量相似度检索 ──命中页截图(±邻居页)──> DeepSeek 看图作答
```

与"图片转文字再检索"的区别：信息不在半路被丢掉。图表里的数字、版式、配色全部保留在像素里，问题命中靠的是图文向量空间的相似度，答案靠 VLM 直接读图。

## 目录结构

```
pixelrag-demo/
├── build_index.sh            # 建索引脚本（渲染 → 切块 → 向量化）
├── query.py                  # 交互式问答（检索 + DeepSeek 看图生成答案）
├── pixelrag.yaml             # 配置文件（key 需自己填）
├── requirements.txt          # 依赖清单（锁实测通过版本）
├── docs/                     # 测试语料：4 份 PDF 共 259 页（CNNIC 公开报告 + PixelRAG 论文）
├── paper_index/              # 预建索引：已向量化，clone 后可直接查询
├── models/                   # Qwen3-VL-Embedding-2B 本地缓存（4GB，首次运行自动下载，不入库）
└── AIDocuments/              # 原理详解 / 快速启动 / 方案对比 / 召回测试记录
```

## 快速开始

```bash
# 1. 系统依赖（PDF 渲染需要 poppler）
brew install poppler

# 2. Python 环境
conda create -n pixelrag python=3.12 -y
conda activate pixelrag
pip install --no-compile -r requirements.txt

# 3. 配置：在 pixelrag.yaml 的 answer.api_key 里填入你的 DeepSeek API key（sk-xxx）

# 4. 直接提问（预建索引已包含 docs/ 里 4 份文档，无需重建）
python query.py "你的问题"     # 单次问答
python query.py               # 交互模式，模型只加载一次，q 退出
```

首次运行会自动下载 Qwen3-VL-Embedding-2B（约 4GB）。换自己的语料：把文档放进 `docs/`，
跑 `./build_index.sh` 增量重建（`--force` 全量重建）。

没填 API key 也能跑——只做检索，输出命中的图片块路径和相似度分数。
## 实测结果（M1 Pro 32GB，4 份 PDF 共 259 页混合索引）

| 问题 | 命中 | DeepSeek 答案 | 核验 |
|---|---|---|---|
| 截至2025年6月我国生成式人工智能用户规模是多少 | genai2025 第15页 | 5.15亿人 | ✅ 与官方发布一致 |
| 我国5G基站总数达到多少万个 | cnnic57 第14页 | 483.8万个（截至2025年12月） | ✅ 该报告 2026-02 才发布，晚于任何模型训练语料，答案只能来自读图 |
| 我国网民规模达多少 互联网普及率多少 | cnnic56 第23页 | 11.23亿人 / 79.7% | ✅ 与官方口径一致 |
| Overview of PixelRAG and the diagram（英文） | 论文第2页 Figure 1 | — | ✅ 纯文字提问直接命中架构图截图 |

关键观察：**2026 年 2 月发布的第 57 次报告，其数据不可能出现在任何模型的训练集里**，DeepSeek 依然答对——证明答案是看着图读出来的，不是模型记忆。

性能基线：259 页向量化约 11 分钟（MPS，2.5 秒/页）；模型加载 30-60 秒，之后每轮查询约 1 秒 + DeepSeek 生成几秒。邻居扩展多送 2 页约增加 2k 图像 token。

## 实现说明

本项目不改 PixelRAG 源码，只做三件事：

1. **`build_index.sh`** — 封装 `pixelrag index build`，内置代理清理、本地模型缓存路径、离线模式，增量/全量一键切换。
2. **`query.py`** — 替代官方 `pixelrag serve`（0.4.0 的 serve 在 macOS 上加载模型时段错误，见踩坑记录）。复用 embed 阶段同款的 MPS fp16 加载路径，指令格式与官方 `_encode_queries` 一致（system 角色放指令、last-token pooling、L2 归一化）；答案生成把命中页截图 base64 后与问题一起发给 DeepSeek 视觉模型。
3. **配置驱动** — 模型、索引目录、答案生成的 key 全部收敛在 `pixelrag.yaml`，支持 `PIXELRAG_CONFIG` 环境变量切换多套索引。

## 踩坑记录（Mac / Apple Silicon）

1. **PyPI 0.4.0 与 GitHub main 有偏差** — `LocalSource` 只支持目录不支持单文件（main 已修未发版）；`metadata.npz` 字段名与 main 不同。
2. **`pixelrag serve` 不可用** — CPU fp32 加载模型时段错误（exit 139），它只有 cpu/cuda 两档没有 MPS。用 `query.py` 代替。
3. **faiss 必须在 torch 之后 import** — faiss-cpu 自带 OpenMP 运行时，先导入会占符号位导致 torch/MPS 段错误；共存还需 `KMP_DUPLICATE_LIB_OK=TRUE`。
4. **Qwen3-VL 纯文本前向在 MPS 上段错误** — 必须 `PYTORCH_ENABLE_MPS_FALLBACK=1`（未实现的算子回落 CPU）。
5. **HuggingFace 下载** — 国内环境设 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`；模型全在本地后加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
6. **pip 加 `--no-compile`** — 避免 pip 清理 .pyc 时触发批量删除保护。

以上环境变量均已内置在脚本里，正常使用不需要手动传。

详细版见 [AIDocuments/](AIDocuments/)：《PixelRAG技术原理》《PixelRAG快速启动指南》《多模态RAG方案对比》《召回测试记录》。

## 原理延伸

- **为什么是"真多模态"**：图片从未被转成文字。Qwen3-VL-Embedding 是图文统一向量空间模型，文字 query 和图片 chunk 直接算相似度；没有 OCR、没有 caption 生成的信息瓶颈。
- **单向量 vs 多向量**：PixelRAG 先在外部把页面物理切小（每页一个 tile），每个 chunk 一个向量，所以能用标准 FAISS/HNSW；ColPali 走的是模型内部 patch 级多向量（late interaction / MaxSim），检索精度更细但要专门的多向量索引。两种路线对比见《多模态RAG方案对比》。
- **邻居扩展**：命中页前后各带 N 页一起送模型（`answer.neighbors`），等价于传统 RAG 的 chunk overlap，解决表格跨页、上下文被切断的问题。

## License

本项目代码 MIT。注意依赖的 [PixelRAG](https://github.com/StarTrail-org/PixelRAG) 与各模型有各自的开源协议，商用前请自行确认。
