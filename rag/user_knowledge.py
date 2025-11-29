# user_knowledge.py
import json
import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ========== ファイルごとのチャンク戦略 ==========
def chunk_pdf(file_path: Path, chunk_size=500):
    """
    PDFを読み込み、テキストをchunk_size文字ごとに分割する
    """
    reader = PdfReader(str(file_path))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        # 500文字ずつに分割
        for j in range(0, len(text), chunk_size):
            chunk_text = text[j:j+chunk_size]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "file": file_path.name,
                    "page": i + 1,
                    "chunk": j // chunk_size + 1
                }
            })
    return chunks



def chunk_txt(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [
        {"text": chunk, "metadata": {"file": file_path.name, "chunk": i}}
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def chunk_csv(file_path: Path, batch_size=50):
    df = pd.read_csv(file_path)
    chunks = []
    for i in range(0, len(df), batch_size):
        part = df.iloc[i:i+batch_size]
        text = part.to_string()
        chunks.append({
            "text": text,
            "metadata": {"file": file_path.name, "row_start": i, "row_end": i+len(part)-1}
        })
    return chunks


def chunk_json(file_path: Path):
    data = json.load(open(file_path, encoding="utf-8"))
    chunks = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            text = json.dumps(item, ensure_ascii=False)
            chunks.append({
                "text": text,
                "metadata": {"file": file_path.name, "index": i}
            })
    elif isinstance(data, dict):
        for key, value in data.items():
            text = json.dumps(value, ensure_ascii=False)
            chunks.append({
                "text": text,
                "metadata": {"file": file_path.name, "key": key}
            })
    else:
        chunks.append({
            "text": json.dumps(data, ensure_ascii=False),
            "metadata": {"file": file_path.name}
        })
    return chunks


# ========== ChromaDB 保存処理 ==========

def add_file_to_chroma(file_path: Path, persist_dir="./chroma_db", collection_name="knowledge"):
    """
    ファイルを適切にチャンク化して ChromaDB に保存
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        chunks = chunk_pdf(file_path)
    elif ext == ".txt":
        chunks = chunk_txt(file_path)
    elif ext == ".csv":
        chunks = chunk_csv(file_path)
    elif ext == ".json":
        chunks = chunk_json(file_path)
    else:
        print(f"⚠️ 未対応の拡張子: {ext}")
        return None

    if not chunks:
        print(f"⚠️ {file_path} からテキストを抽出できませんでした")
        return None

    # DB 接続
    client = chromadb.PersistentClient(path=persist_dir)

    # コレクション取得 or 作成
    try:
        collection = client.get_collection(collection_name)
    except:
        embed = embedding_functions.OllamaEmbeddingFunction(
            model_name="kun432/cl-nagoya-ruri-large:337m"
        )
        collection = client.create_collection(collection_name, embedding_function=embed)

    # 追加
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"{file_path.stem}_{i}" for i in range(len(chunks))]
    )

    print(f"✅ {file_path.name} を {collection_name} に追加しました ({len(chunks)} チャンク)")
    return collection
