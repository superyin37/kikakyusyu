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

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("knowledge")

print(col.count())   # ← チャンク数
print(col.peek(3))   # ← 最初の3件を表示


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

# ==== DB 構築 ==== (gomi/area はそのまま)
gomi_docs, gomi_meta = load_jsonl(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "rag_docs_merged.jsonl")),
    key="品名"
)
area_docs, area_meta = load_jsonl(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "area.jsonl")),
    key="町名"
)

gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
area_collection = build_chroma(area_docs, area_meta, name="area")
known_items = [m.get("品名", "") for m in gomi_meta]

# ← ここを追加！
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
try:
    knowledge_collection = client.get_collection("knowledge")
except:
    knowledge_collection = None


# ==== Blocking モード ====
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2
    )

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
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2
    )

    def stream_gen():
        collected = ""
        stream = ollama.chat(
            model="hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:latest",
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
        headers={"X-References": json.dumps(references, ensure_ascii=False)}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
