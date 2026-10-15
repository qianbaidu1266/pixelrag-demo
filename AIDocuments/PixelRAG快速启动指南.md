# PixelRAG 快速启动指南（本机已验证）

> 验证环境：M1 Pro 32GB / macOS / Python 3.12.15 / pixelrag 0.4.0（PyPI）
> 本文档记录的是**实际跑通的过程**，不是 README 翻译。所有命令均在本机执行成功。

## 0. 不需要 clone 仓库

PixelRAG 发在 PyPI 上，`pip install` 即可。clone 源码只在改它代码或跑训练时才需要。

## 1. 环境准备

```bash
# poppler（PDF 渲染依赖）
brew install poppler

# 正式 conda 环境（已建好：/opt/anaconda3/envs/pixelrag）
conda create -n pixelrag python=3.12 -y
/opt/anaconda3/envs/pixelrag/bin/pip install --no-compile -r requirements.txt
```

`requirements.txt` 里锁了本机验证通过的版本（torch 2.14.1 / transformers 5.19.0 / faiss-cpu 1.15.1 / pixelrag 0.4.0 等）。
单独装等价写法：`pip install --no-compile 'pixelrag[index,pdf,serve]'`（记得加 `--no-compile`，
否则 pip 清理临时文件时会触发安全中心的批量删除保护）。

PyCharm 解释器选 `/opt/anaconda3/envs/pixelrag/bin/python`。

## 2. 模型位置（已迁移）

**Qwen3-VL-Embedding-2B（4.0GB）现在就放在项目目录里，随项目一起走：**

```
/Users/yangchao/Desktop/myHub/Multimodal/pixelrag-demo/models/models--Qwen--Qwen3-VL-Embedding-2B/
```

`models/` 是标准 HuggingFace 缓存目录结构（`models--<org>--<name>/snapshots/...`），
`query.py` 和 `build_index.sh` 都已内置指向它（用脚本自身位置解析，不写死绝对路径），
所以整个 `pixelrag-demo` 目录拷到别处也能直接用。

手动加载时指定：

```bash
export HF_HUB_CACHE=/Users/yangchao/Desktop/myHub/Multimodal/pixelrag-demo/models
export HF_HUB_OFFLINE=1        # 模型已在本地，跳过联网校验
export TRANSFORMERS_OFFLINE=1
```

首次下载（如需新模型）走国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
# 并临时清掉代理：env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy
# 本机代理 127.0.0.1:51035 对 hf.co 会报 502
```

## 3. 建索引（PDF → 图片块 → 向量）

目录结构（**PyPI 0.4.0 只支持目录，不支持单文件**）：

```
pixelrag-demo/
├── docs/              # 待索引的 PDF 放这里
│   └── paper.pdf
├── pixelrag.yaml
└── paper_index/       # 索引产物（自动生成，18MB / 35 块）
```

`pixelrag.yaml`：

```yaml
source:
  type: local
  path: ./docs
embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto        # Mac 上自动选 MPS
output: ./paper_index
```

```bash
cd /Users/yangchao/Desktop/myHub/Multimodal/pixelrag-demo
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /opt/anaconda3/envs/pixelrag/bin/pixelrag index build
```

模型已缓存后，换新材料只需：PDF 丢进 `docs/` → 重跑上面命令（渲染+向量化，几分钟）。

## 4. 查询（含 DeepSeek 生成答案）

**不要用 `pixelrag serve`**——0.4.0 的 serve 在 Mac 上 CPU fp32 加载模型时段错误（exit 139，它只有 cpu/cuda 两档没有 MPS）。用 `query.py`：

```bash
cd /Users/yangchao/Desktop/myHub/Multimodal/pixelrag-demo

# 交互模式（推荐）：模型只加载一次，持续问答，q 退出
/opt/anaconda3/envs/pixelrag/bin/python query.py

# 单次模式：答一次就退出
/opt/anaconda3/envs/pixelrag/bin/python query.py "你的问题"
```

所需环境变量（HF_HUB_CACHE / OFFLINE / MPS_FALLBACK / KMP_DUPLICATE_LIB_OK）**已全部内置在脚本里**，不用手动传。模型加载约 30-60 秒，之后每次查询约 1 秒。

**生成答案（RAG 的 "A"）走 DeepSeek 官方视觉模型**：DeepSeek 于 2026-08-21 上线 DeepSeek-V4-Flash-Vision-Exp，多模态能力现已并入正式模型 `deepseek-flash`（旧模型名仍可用但已标记下线）。OpenAI 兼容格式，图片 base64 内嵌，JPEG 原生支持，单图最多 1024 token。

key 配在 `pixelrag.yaml` 的 `answer` 段（这是首选位置）：

```yaml
answer:
  provider: deepseek
  model: deepseek-flash
  api_key: "sk-xxx"    # 填这里
  top_n: 1             # 送几张命中图给模型看
```

优先级：yaml 的 `api_key` > demo 目录 `.deepseek_key` 文件 > `DEEPSEEK_API_KEY` 环境变量。三处都不填则只做检索（显示命中图块路径）。配置后每轮查询自动变成：文字 → 命中图片 → 图片+问题发给 DeepSeek → 输出答案。

**邻居扩展**（传统 RAG chunk overlap 的图片版）：PDF 一页就是一个 chunk，跨页表格/上下文会被切断。`neighbors: 1` 表示命中页前后各带 1 页一起送（每次送出页数 = 命中页 + 2×neighbors，上限内去重）。运行时会打印 `-> sending N page(s) to DeepSeek: [2+3+4]` 告知实际送了哪些页。`top_n` 是另一个维度：按得分再送第 2、3 名（它们可能相距很远，与邻居无关）。计费参考：每张命中图最多 1024 token，3 页 ≈ 3k token/轮 + 文本，走 V4-Flash 价格（峰谷分时，谷时段半价）。

## 5. 已验证的结果

用 PixelRAG 论文自己的 PDF（35 块，2048 维 FAISS 索引）：

- 英文查询 `Overview of PixelRAG and the diagram` → **top1 命中第 2 页 Figure 1 架构对比图**（纯文字 query 直接命中图片截图，中间无任何 OCR/描述）。
- 中文查询 `PixelRAG 的架构和方法是什么` → top1 命中方法章节，Qwen3-VL 中文正常。

## 附录：踩坑记录（按发现顺序）

1. **PyPI 0.4.0 落后 GitHub main**：`LocalSource` 只 `rglob` 扫目录，传单文件得到 0 个文档 → PDF 必须放子目录。
2. **本机代理对 hf.co 报 502** → 清代理变量 + `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`。
3. **`pixelrag serve` 在 Mac 段错误**（CPU fp32，无 MPS 档）→ 用 `query.py` 替代。
4. **faiss 必须在 torch 之后 import**：faiss-cpu 自带 OpenMP 运行时，先 import 占符号位导致 torch/MPS 段错误；共存还需 `KMP_DUPLICATE_LIB_OK=TRUE`（否则 OMP Error #15）。
5. **Qwen3-VL 纯文本前向在 MPS 上段错误** → 必须 `PYTORCH_ENABLE_MPS_FALLBACK=1`（未实现的算子回落 CPU）。
6. 0.4.0 的 `serve` 默认从 cwd 找 `./articles.json`，实际产物在 index-dir 里（需 `--articles-json` 参数，但 serve 本身不可用，仅存档）。
7. 0.4.0 的 `metadata.npz` 字段：`article_ids / tile_indices / chunk_indices / y_offsets / tile_heights`（没有 main 分支的 `tile_paths`）。

## 环境变量速查（query.py 已内置 setdefault，命令行可不带）

| 变量 | 作用 |
|---|---|
| `HF_HUB_CACHE` | 模型缓存指向项目内 `pixelrag-demo/models` |
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | 跳过联网校验（模型已全在本地） |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | MPS 未实现算子回落 CPU，防段错误 |
| `KMP_DUPLICATE_LIB_OK=TRUE` | faiss 与 torch 双 OpenMP 共存 |
