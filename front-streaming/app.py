# app.py
import sys
import time
import os
import streamlit as st
from typing import Optional
import requests
import json
from pathlib import Path

from gpu_stats import init_nvml_once, shutdown_nvml, get_gpu_stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.user_knowledge import add_file_to_chroma

st.set_page_config(page_title="Llama Chat (Streaming+Metrics)", page_icon="⏱️")

LOG_FILE = Path("backend/logs.jsonl")
KNOWLEDGE_DIR = Path("knowledge_files")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

# ---- NVML 初期化 ----
init_nvml_once()


# test_file = Path("knowledge_files/IIAI_AAI_2025_paper_0381 (3).pdf")
# if test_file.exists():
#     add_file_to_chroma(test_file)
# else:
#     pass


def save_log(user_input: str, assistant_output: str, mode: str, total_sec: float):
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user": user_input,
        "assistant": assistant_output,
        "total_time": round(total_sec, 3),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

def load_logs(limit: int = 20):
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries[-limit:]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("⏱️ Llama Chat – Streaming & Metrics")

# ===== ログをチャット形式で表示 =====
logs = load_logs(limit=5)
if logs:
    st.subheader("🗂 過去のやり取り（最新5件）")
    for entry in logs:
        with st.chat_message("user"):
            st.markdown(entry.get("user", ""))
        with st.chat_message("assistant"):
            st.markdown(entry.get("assistant", ""))
    st.divider()

# ---- サイドバー ----
with st.sidebar:
    st.subheader("GPU / VRAM Monitor")
    vram_box = st.empty()
    util_box = st.empty()

    stats = get_gpu_stats()
    if stats:
        used_gb, total_gb, util_p, name = stats
        vram_box.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
        util_box.caption(f"{name} | Util {util_p}%")
    else:
        vram_box.metric("VRAM (GB)", "N/A")
        util_box.caption("GPU not detected")

    # 応答モード選択
    mode = st.radio("応答モードを選択", ["Blocking", "Streaming"], horizontal=True, key="response_mode")

    # ===== ナレッジファイル管理 =====
    st.subheader("ナレッジファイル管理")

    upload_file = st.file_uploader("ファイルをアップロード", type=["txt", "pdf", "csv", "json"])
    if upload_file is not None:
        save_path = KNOWLEDGE_DIR / upload_file.name
        with open(save_path, "wb") as f:
            f.write(upload_file.getbuffer())
        st.success(f"アップロードしました: {upload_file.name}")

        # ChromaDB に登録
        add_file_to_chroma(save_path)

    # アップロード済みファイル一覧
    files = list(KNOWLEDGE_DIR.glob("*"))
    if files:
        st.caption("検索対象ファイル一覧:")
        for f in files:
            st.text(f.name)
    else:
        st.caption("まだファイルはありません。")

# ===== 以下は通常のチャット処理 =====
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("メッセージを入力してください")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    col1, col2, col3, col4 = st.columns(4)
    ttfb_area   = col1.empty()
    total_area  = col2.empty()
    tokps_area  = col3.empty()
    outtok_area = col4.empty()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = ""
        t_start = time.perf_counter()
        total_sec = None

        if mode == "Blocking":
            try:
                api_url = "http://localhost:8000/api/bot/respond"
                res = requests.post(api_url, json={"prompt": user_input}, timeout=20)
                res.raise_for_status()
                data = res.json()
                reply = data.get("reply", "")
                references = data.get("references", [])
            except Exception as e:
                reply = "APIリクエストでエラー: " + str(e)
                references = []

            t_end = time.perf_counter()
            total_sec = t_end - t_start
            ttfb_area.metric("TTFB (s)", round(total_sec, 3))
            total_area.metric("Total (s)", round(total_sec, 3))
            tokps_area.metric("Tokens/sec", "-")
            outtok_area.metric("Output tokens", "-")

            collected = reply
            placeholder.markdown(collected)

            # 📑 参考情報を表示
            if references:
                st.markdown("### 📑 参考情報（上位チャンク）")
                for ref in references:
                    file = ref.get("file", "?")
                    page = ref.get("page", "?")
                    # chunk, chunk_id, id など候補キーを探す
                    chunk = ref.get("chunk") or ref.get("chunk_id") or ref.get("id", "?")
                    text = ref.get("text", "")[:200]

                    st.markdown(
                        f"- **{file} p.{page} (chunk {chunk})**\n"
                        f"  \n> {text}..."
                    )

        else:
            try:
                api_url = "http://localhost:8000/api/bot/respond_stream"
                with requests.post(api_url, json={"prompt": user_input}, stream=True, timeout=60) as res:
                    res.raise_for_status()
                    ttfb = None
                    collected = ""
                    for chunk in res.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        if ttfb is None:
                            ttfb = time.perf_counter()
                            ttfb_area.metric("TTFB (s)", round(ttfb - t_start, 3))
                        text = chunk.decode("utf-8")
                        collected += text
                        placeholder.markdown(collected)

                    t_end = time.perf_counter()
                    total_sec = t_end - t_start
                    total_area.metric("Total (s)", round(total_sec, 3))
                    tokps_area.metric("Tokens/sec", "-")
                    outtok_area.metric("Output tokens", "-")

                    # 📑 Streaming の場合はヘッダーから references を取得
                    references = []
                    if "X-References" in res.headers:
                        try:
                            references = json.loads(res.headers["X-References"])
                        except Exception:
                            references = []

                    if references:
                        st.markdown("#### 📑 参考情報")
                        for ref in references:
                            st.markdown(
                                f"- **{ref.get('file','?')} p.{ref.get('page','?')} (chunk {ref.get('chunk','?')})**\n"
                                f"  \n> {ref.get('text','')[:200]}..."
                            )

                    s1 = get_gpu_stats()
                    if s1:
                        used_gb, total_gb, util_p, name = s1
                        vram_box.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
                        util_box.caption(f"{name} | Util {util_p}%")
            except Exception as e:
                collected = "APIリクエストでエラー: " + str(e)
                placeholder.markdown(collected)


        st.session_state["messages"].append({"role": "assistant", "content": collected})

        if total_sec is not None:
            save_log(user_input, collected, mode, total_sec)
# ---- 終了処理 ----
import atexit
atexit.register(shutdown_nvml)