from fastapi.responses import StreamingResponse
import uvicorn
from fastapi import FastAPI
from typing import Dict
from .schemas import PromptRequest, ReplyResponse
import ollama
import os
import sys
import time
import json
from pathlib import Path

# rag モジュールを import できるようにパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag")))
from rag_demo3 import load_jsonl, build_chroma, rag_retrieve_extended, ask_ollama

app = FastAPI()

import chromadb

# ---- Global paths ----
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR.parent / "rag"
CHROMA_PATH = BASE_DIR / "chroma_db"

GOMI_JSONL = RAG_DIR / "rag_docs_merged.jsonl"
AREA_JSONL = RAG_DIR / "area.jsonl"

# ---- 1. Initialize ChromaDB client (only once) ----
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


# ---- 2. Utility: get existing collection or build if missing ----
def get_or_build_collection(
    client: chromadb.Client,
    name: str,
    docs: list[str] | None = None,
    meta: list[dict] | None = None,
):
    """
    Try to get an existing collection.
    If it does not exist and docs/meta are provided, build the collection.
    Otherwise, raise an error.
    """
    try:
        return client.get_collection(name)
    except Exception:
        if docs is None or meta is None:
            raise RuntimeError(
                f"Collection '{name}' not found and no data provided to build it."
            )
        return build_chroma(docs, meta, name=name)


# =========================
# gomi collection (garbage rules)
# =========================

# Load documents and metadata from JSONL
gomi_docs, gomi_meta = load_jsonl(
    os.path.abspath(GOMI_JSONL),
    key="品名",
)

# Get existing collection or build it if missing
gomi_collection = get_or_build_collection(
    client=chroma_client,
    name="gomi",
    docs=gomi_docs,
    meta=gomi_meta,
)

# Extract item names for exact / candidate matching
# ========== 注意：Hybrid Grounding システムでは不要 ==========
# known_items = [m.get("品名", "") for m in gomi_meta]
# Hybrid システムが直接 gomi_collection を使用するため、
# 事前の品名リスト構築は不要になりました


# =========================
# area collection (location / schedule)
# =========================

# Load documents and metadata from JSONL
area_docs, area_meta = load_jsonl(
    os.path.abspath(AREA_JSONL),
    key="町名",
)

# Get existing collection or build it if missing
area_collection = get_or_build_collection(
    client=chroma_client,
    name="area",
    docs=area_docs,
    meta=area_meta,
)


# =========================
# knowledge collection (user-provided knowledge)
# =========================

# Only try to load the collection; never rebuild automatically
try:
    knowledge_collection = chroma_client.get_collection("knowledge")
except Exception:
    knowledge_collection = None


# =========================
# Debug output (optional)
# =========================

print("gomi collection size:", gomi_collection.count())
print("area collection size:", area_collection.count())
if knowledge_collection:
    print("knowledge collection size:", knowledge_collection.count())
else:
    print("knowledge collection not found")

# # ==== DB 構築 ==== (gomi/area はそのまま)
# gomi_docs, gomi_meta = load_jsonl(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "rag_docs_merged.jsonl")),
#     key="品名"
# )
# area_docs, area_meta = load_jsonl(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "area.jsonl")),
#     key="町名"
# )

# gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
# area_collection = build_chroma(area_docs, area_meta, name="area")
# known_items = [m.get("品名", "") for m in gomi_meta]

# # ← ここを追加！
# import chromadb
# client = chromadb.PersistentClient(path="./chroma_db")
# try:
#     knowledge_collection = client.get_collection("knowledge")
# except:
#     knowledge_collection = None



# client = chromadb.PersistentClient(path="./chroma_db")
# col = client.get_or_create_collection("knowledge")

# print(col.count())   # ← チャンク数
# print(col.peek(3))   # ← 最初の3件を表示


# ==== ログ保存用ユーティリティ ====
LOG_FILE = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "logs.jsonl")))

def save_log(user_input: str, assistant_output: str, mode: str):
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user": user_input,
        "assistant": assistant_output,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")




# ==== Blocking モード ====
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    # ========== Hybrid Grounding システム対応 ==========
    # known_items パラメータは不要（後方互換性のため None として渡す）
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,  # ← Hybrid システムが直接使用
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=None,  # ← 不要（旧版との互換性のため残す）
        area_meta=area_meta,
        top_k=2
    )
    print("\n===== DEBUG: FULL PROMPT START =====\n")
    print(rag_prompt)
    print("\n===== DEBUG: FULL PROMPT END =====\n")

    reply = ask_ollama(rag_prompt)

    return {
        "reply": reply,
        "references": references  # ← LLMが使った or マッチしたchunk情報
    }



# ==== Streaming モード ====
from fastapi.responses import StreamingResponse
import json

@app.post("/api/bot/respond_stream")
async def rag_respond_stream(req: PromptRequest):
    # ========== Hybrid Grounding システム対応 + 性能監視 ==========
    import time
    retrieval_start = time.perf_counter()
    
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,  # ← Hybrid システムが直接使用
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=None,  # ← 不要（旧版との互換性のため残す）
        area_meta=area_meta,
        top_k=2
    )
    
    retrieval_time = (time.perf_counter() - retrieval_start) * 1000
    print(f"\n⏱️  RAG検索耗時: {retrieval_time:.2f}ms")
    print("\n===== DEBUG: FULL PROMPT START =====\n")
    print(rag_prompt)
    print("\n===== DEBUG: FULL PROMPT END =====\n")
    
    # 性能情報を references に追加
    if references and isinstance(references, list):
        references.append({
            "type": "performance",
            "retrieval_time_ms": retrieval_time
        })

    def stream_gen():
        collected = ""
        stream = ollama.chat(
            model="swallow:latest",
            messages=[{"role": "user", "content": rag_prompt}],
            stream=True
        )
        for event in stream:
            content = event.get("message", {}).get("content", "")
            if content:
                collected += content
                yield content
        if collected:
            save_log(req.prompt, collected, mode="Streaming(API)")

    # 📌 references を JSON にしてヘッダーに埋め込む
    return StreamingResponse(
        stream_gen(),
        media_type="text/plain",
        headers={"X-References": json.dumps(references, ensure_ascii=True)}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
