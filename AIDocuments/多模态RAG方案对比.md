# 多模态 RAG 方案对比（2026-10 调研）

> 结论先行：**本地 Mac 上做真多模态 RAG，选 PixelRAG**（已跑通验证）。翻译型方案（WeKnora 默认路径）是"半个多模态"，ColPali 是研究级方案但 Mac 生态差，Morphik 有 License 限制。

## 什么叫"真正的多模态 RAG"

两道关：
1. **检索端**：图片本身被向量化（不是转成文字再向量化），文字 query 能直接命中图片。
2. **阅读端**：命中后 VLM 直接看图回答，而不是看 OCR 出来的文字。

只过第一道关的是"半个"；两道都过的（PixelRAG）才是端到端视觉 RAG。**任何"图片→VLM 描述→文字向量"的方案都是翻译型**——描述是有损的，VLM 没写进描述的信息永久丢失。

## 候选方案对比

| 方案 | 路线 | Stars | License | Mac 可跑 | 核心问题 |
|---|---|---|---|---|---|
| **PixelRAG** (StarTrail-org) | 物理切块 + 单向量（Qwen3-VL-Embedding-2B）+ VLM 看图读答案 | 10.3k | Apache 2.0 | ✅ MPS 一等公民，已验证 | 中文微调未覆盖（实测查询正常） |
| ColPali/ColQwen (illuin-tech) | 模型内部 patch 化 + 多向量 late interaction (MaxSim) | 2.8k | Apache 2.0 | ⚠️ byaldi 官方承认 MPS 性能差，flash-attn 装不上 | 只是模型库，检索引擎要自建；中文必须选 ColQwen |
| Morphik Core | ColPali 检索 + 产品化 Console | 3.7k | **BSL（非真开源）** | 未知 | 商用有营收门槛，四年后才转 Apache；主推托管服务 |
| WeKnora (腾讯微信AI) | 翻译型为主（VLM OCR + 描述）；可选 WeMM 图文向量但需自起 vLLM | — | NOASSERTION | 容器方案 | compose 无任何模型服务；图片向量是知识库级 opt-in，模型不支持时静默降级 |
| WeKnora-pro (xiaohuangpin) | WeKnora + MinerU 解析增强 | 138 | AGPL-v3 | ⚠️ torch.cuda 判断，MPS 不识别 | **比官方主线落后一年**（0.1.4 vs 0.8.2），无 ImageEmbedder 接口，compose 无 build 段（直接 up 拉的是官方镜像），多模态能力是倒退 |

## WeKnora 生态调研结论（防止以后再踩）

- GitHub 搜 `WeKnora` 164 个仓库，加 `multimodal` 后 **0 个**——社区没有现成的 WeKnora 多模态方案。
- WeKnora 代码里多模态第二层（图片向量）是真实存在的：`ImageEmbedder` 接口、modality gap 补偿（图片召回阈值 0.2 + 检索池扩大 50%）、图片 rerank，但需要自己部署 vLLM 跑 WeMM-Embedding（2B/4B/9B），compose 之外。
- WeKnora 的 docker-compose 服务：frontend / app / sandbox / docreader / postgres(paradedb, 自带 pgvector+BM25，默认向量库) / redis / minio / searxng / langfuse / mcp。模型全部启动后在 Web UI 配，或预置 `config/builtin_models.yaml`。
- 灌水仓库警示：`WeKnora-Optimization-Advisor` 全仓库只有 markdown，自称 "100% COMPLETE PRODUCTION READY"，MaxSim 只是 README 抄公式，零代码改动。

## 技术路线小结

- **要省事 + Mac + 中文**：PixelRAG，没有第二个选项。
- **要认真研究多向量 vs 单向量精度差异**：ColQwen + 一台 NVIDIA 机器。
- **WeKnora** 适合的场景是标准文本知识库 + 顺带处理扫描件（翻译型够用），要图文混合检索得自己补 vLLM + WeMM 整条链。
- **纯结构化文本**（如 ISO 20022 报文材料）任何视觉方案都用不上，老老实实文本 RAG。
