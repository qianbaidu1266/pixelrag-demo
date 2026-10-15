"""PixelRAG interactive query tool (MPS / Apple Silicon).

Loads the Qwen3-VL-Embedding-2B model once, then answers queries in a loop
against a local FAISS index built by `pixelrag index build`.

Mirrors pixelrag_serve.api._encode_queries (instruction in system role, chat
template, last-token pooling, L2 norm) but loads the model the way
pixelrag_embed.embed_cpu does (transformers, MPS, fp16) -- the path already
proven to work on Apple Silicon by the index build.

Usage:
    python query.py                      # interactive REPL
    python query.py "your question"      # one-shot mode, then exit
"""
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Qwen3-VL text-only path has unimplemented MPS ops
os.environ.setdefault(
    "HF_HUB_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),  # model lives here, not in ~/.cache
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # model is fully cached; skip all network checks
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + torch bundle separate OpenMP runtimes

import sys
import numpy as np
import torch  # must come BEFORE faiss: faiss bundles its own OpenMP runtime, importing it first segfaults torch/MPS
import faiss
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
CONFIG_FILE = os.environ.get("PIXELRAG_CONFIG", "pixelrag.yaml")  # which config/index to use
INSTRUCTION = "Retrieve images or text relevant to the user's query."
TOP_K = 3
INDEX_DIR = "paper_index"  # overwritten from the config's output field in main()

# ---- DeepSeek answer step (RAG 的 "A")，配置在 pixelrag.yaml 的 answer 段 ----
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
_answer_cfg = {"model": "deepseek-flash", "api_key": None, "top_n": 1, "neighbors": 1}


def load_answer_config():
    """Read config from CONFIG_FILE: index dir from `output`, answer params from `answer`.
    api_key priority: yaml > .deepseek_key file > env var."""
    global _answer_cfg, INDEX_DIR
    import yaml
    if os.path.exists(CONFIG_FILE):
        cfg = yaml.safe_load(open(CONFIG_FILE).read()) or {}
        out = (cfg.get("output") or "").strip()
        if out:
            INDEX_DIR = out
        ans = cfg.get("answer") or {}
        if ans.get("model"):
            _answer_cfg["model"] = ans["model"]
        if ans.get("top_n"):
            _answer_cfg["top_n"] = int(ans["top_n"])
        if ans.get("neighbors") is not None:
            _answer_cfg["neighbors"] = int(ans["neighbors"])
        key = (ans.get("api_key") or "").strip()
        if key:
            _answer_cfg["api_key"] = key
    if not _answer_cfg["api_key"]:
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deepseek_key")
        if os.path.exists(key_file):
            _answer_cfg["api_key"] = open(key_file).read().strip()
    if not _answer_cfg["api_key"]:
        _answer_cfg["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "").strip() or None
    return _answer_cfg


def expand_hits(hits):
    """Neighbor expansion (传统 RAG overlap 的图片版):
    top-1 hit ± neighbors pages of the same article, then the remaining top-ranked hits.
    Dedup by image path; drop tiles that don't exist (e.g. prev of page 1)."""
    seen, out = set(), []

    def add(h):
        if h and os.path.exists(h["image"]) and h["image"] not in seen:
            seen.add(h["image"])
            out.append(h)

    n = _answer_cfg["neighbors"]
    top = hits[0]
    for d in range(-n, n + 1):
        if d == 0:
            add(top)
            continue
        tile = top["tile"] + d
        add({"rank": 0, "score": 0.0,
             "image": os.path.join(INDEX_DIR, "tiles", f"{top['article']}.png.tiles", f"tile_{tile:04d}.jpg"),
             "article": top["article"], "tile": tile, "chunk": -1})
    for h in hits[1:_answer_cfg["top_n"]]:
        add(h)
    return out


def answer_query(query: str, hits) -> str:
    """Send the expanded hit image(s) + question to DeepSeek (vision) and return its answer."""
    import base64
    import requests

    images = expand_hits(hits)
    pages = "+".join(str(h["tile"] + 1) for h in images)   # tile N = page N+1
    print(f"  -> sending {len(images)} page(s) to DeepSeek: [{pages}]", flush=True)

    content = [{"type": "text",
                "text": f"下面是同一份文档连续的若干页截图。请根据图片内容回答问题；图片里没有答案就直说，不要编。\n\n问题：{query}"}]
    for h in images:
        with open(h["image"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # trust_env=False: ignore HTTP(S)_PROXY env vars -- local proxy 502s on some hosts
    session = requests.Session()
    session.trust_env = False
    resp = session.post(
        DEEPSEEK_BASE_URL,
        headers={"Authorization": f"Bearer {_answer_cfg['api_key']}", "Content-Type": "application/json"},
        json={"model": _answer_cfg["model"], "messages": [{"role": "user", "content": content}],
              "temperature": 0.3},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_model():
    """Load embedding model on MPS (fp16). Loaded ONCE for the whole session."""
    print(f"loading {MODEL} on MPS (fp16) ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval().to("mps")
    return processor, model


def load_index():
    """Load FAISS index + metadata. Pure file reads, no model involved."""
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    meta = np.load(f"{INDEX_DIR}/metadata.npz", allow_pickle=True)
    return index, meta


def encode_query(processor, model, query: str) -> np.ndarray:
    """Text -> 2048-dim vector, same space as the image chunk vectors."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.model(**inputs)
    last_hidden = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1          # last non-pad token position
    pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()


def search(index, meta, qv: np.ndarray, k: int = TOP_K):
    scores, ids = index.search(qv, k)
    hits = []
    for rank, (score, i) in enumerate(zip(scores[0], ids[0]), 1):
        # reconstruct the tile image path on disk from metadata
        article = int(meta["article_ids"][i])
        tile = int(meta["tile_indices"][i])
        chunk = int(meta["chunk_indices"][i])
        img = os.path.join(INDEX_DIR, "tiles", f"{article}.png.tiles", f"tile_{tile:04d}.jpg")
        hits.append({"rank": rank, "score": float(score), "image": img,
                     "article": article, "tile": tile, "chunk": chunk})
    return hits


def print_hits(query: str, hits):
    print(f"\nQ: {query}")
    for h in hits:
        exists = "OK" if os.path.exists(h["image"]) else "MISSING"
        print(f"  #{h['rank']}  score={h['score']:.4f}  article={h['article']} "
              f"tile={h['tile']} chunk={h['chunk']}  [{exists}]")
        print(f"      {h['image']}")
    print()


def main():
    one_shot = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    ans = load_answer_config()  # must run first: sets INDEX_DIR from config

    processor, model = load_model()
    index, meta = load_index()
    print(f"index ready: {index.ntotal} chunks, dim={index.d} (dir={INDEX_DIR}, config={CONFIG_FILE})", flush=True)
    if ans["api_key"]:
        print(f"DeepSeek answer step: ON (model={ans['model']}, top-{ans['top_n']} image, config={CONFIG_FILE})", flush=True)
    else:
        print(f"DeepSeek answer step: OFF -- fill api_key in {CONFIG_FILE} answer section -- retrieval only", flush=True)

    if one_shot:
        hits = search(index, meta, encode_query(processor, model, one_shot))
        print_hits(one_shot, hits)
        if ans["api_key"]:
            print("A:", answer_query(one_shot, hits))
        return

    print("interactive mode -- type a query, 'q' to quit\n", flush=True)
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            break
        hits = search(index, meta, encode_query(processor, model, query))
        print_hits(query, hits)
        if ans["api_key"]:
            print("A:", answer_query(query, hits))

    print("bye")


if __name__ == "__main__":
    main()
