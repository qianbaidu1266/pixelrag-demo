#!/bin/bash
# PixelRAG 建索引：docs/ 下所有支持的文件 -> 渲染 -> 向量化 -> paper_index/
# 用法: ./build_index.sh            # 增量（已渲染且未变化的文档自动跳过）
#       ./build_index.sh --force    # 全量重建（清掉 paper_index 重来）
set -e
cd "$(dirname "$0")"

# 清代理 + 指向本地模型缓存（本目录下的 models/，模型已全部在本地，离线加载）
exec env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  HF_HUB_CACHE="$PWD/models" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/anaconda3/envs/pixelrag/bin/pixelrag index build "$@"
