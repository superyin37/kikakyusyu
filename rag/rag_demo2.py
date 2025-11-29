#!/usr/bin/env python3
import argparse
import chromadb
from chromadb.utils import embedding_functions
import subprocess
import json
import re

# 例: 地区リスト
AREAS = ["青葉台", "大字伊川", "泉ケ丘", "稲積1～2丁目", "風師1～4丁目", "吉志4～7丁目", "北川町"]

def extract_keywords(user_input, known_items, known_areas=AREAS):
    """
    入力文から品名と地区を抽出する
    """
    keywords = {"品名": None, "地区": None}

    # 品名候補を検索（known_items は JSONL から抽出した品名リスト）
    for item in known_items:
        if item in user_input:
            keywords["品名"] = item
            break

    # 地区候補を検索
    for area in known_areas:
        if area in user_input:
            keywords["地区"] = area
            break

    return keywords



# ========== ステップ 1: JSONL を読み込む ==========
def load_jsonl(path):
    """
    JSONL ファイルからごみ分別データを読み込み、
    品名を embedding 用テキストとして返す。
    """
    docs = []
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # embedding 用には品名だけ
            docs.append(row.get("品名", ""))
            # そのほかの情報をメタデータとして保持
            meta.append(row)
    return docs, meta


# ========== ステップ 2: ベクトルデータベースを構築 ==========
def build_chroma(docs, meta, persist_dir="./chroma_db"):
    """
    ChromaDB コレクションを作成し、
    品名を embedding して保存する。
    """
    client = chromadb.PersistentClient(path=persist_dir)
    embed = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")

    # 既存のコレクションを削除
    try:
        client.delete_collection("gomi")
    except:
        pass

    # 新しいコレクションを作成
    collection = client.create_collection("gomi", embedding_function=embed)

    # ドキュメントを追加（品名のみ embedding, ただしメタに全体保持）
    collection.add(
        documents=docs,
        metadatas=meta,
        ids=[str(i) for i in range(len(docs))]
    )
    return collection


# ========== ステップ 3: クエリと生成 ==========
def query_chroma(collection, query, n=3):
    """
    品名ベースで検索し、関連するメタデータを返す。
    """
    results = collection.query(query_texts=[query], n_results=n)
    # metadatas に全情報が入っている
    return results["metadatas"][0]

def ask_ollama(prompt, model="llama3"):
    """
    Ollama モデルにプロンプトを渡して実行し、
    生成された出力を返す。
    """
    cmd = ["ollama", "run", model]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(prompt)
    return out


def rag_retrieve(user_input, collection, top_k=3):
    """
    ユーザー入力に基づいて RAG 用のプロンプトを生成する
    """
    hits = query_chroma(collection, user_input, n=top_k)
    # コンテキストを整形
    context = "\n\n".join(
        [f"品名: {h.get('品名','')}\n出し方: {h.get('出し方','')}\n備考: {h.get('備考','')}" for h in hits]
    )
    prompt = f"以下は北九州市のごみ分別ルールです：\n{context}\n\n質問: {user_input}\n日本語で答えてください:"
    return prompt


def rag_retrieve_extended(user_input, gomi_collection, area_collection, known_items):
    """
    品名と地区を両方扱う拡張版 RAG
    """
    keys = extract_keywords(user_input, known_items)

    context_parts = []

    # 品名があれば検索
    if keys["品名"]:
        gomi_hits = query_chroma(gomi_collection, keys["品名"], n=3)
        gomi_context = "\n\n".join(
            [f"品名: {h.get('品名','')}\n出し方: {h.get('出し方','')}\n備考: {h.get('備考','')}" for h in gomi_hits]
        )
        context_parts.append(f"【ごみ分別情報】\n{gomi_context}")

    # 地区があれば検索（仮に地区別ルール DB がある場合）
    if keys["地区"]:
        area_hits = query_chroma(area_collection, keys["地区"], n=1)
        area_context = "\n\n".join([str(h) for h in area_hits])
        context_parts.append(f"【地区情報】\n{area_context}")

    # コンテキストをまとめる
    context = "\n\n".join(context_parts) if context_parts else "該当情報が見つかりませんでした。"

    # プロンプト生成
    prompt = f"以下は北九州市のごみ分別ルールです：\n{context}\n\n質問: {user_input}\n日本語で答えてください:"
    return prompt


# ========== メイン ==========
def main():
    parser = argparse.ArgumentParser(description="Ollama + ChromaDB による RAG デモ")
    parser.add_argument("--jsonl", required=True, help="ごみ分別 JSONL ファイルのパス")
    parser.add_argument("--model", default="llama2", help="使用する Ollama モデル名 (デフォルト: llama2)")
    args = parser.parse_args()

    # データを読み込む
    docs, meta = load_jsonl(args.jsonl)
    print(f"{len(docs)} 件のデータを読み込みました")

    # ベクトルデータベースを構築
    collection = build_chroma(docs, meta)

    # 対話型 Q&A
    while True:
        q = input("\n質問を入力してください (終了するには 'exit' と入力): ")
        if q.lower() == "exit":
            break

        # RAG プロンプト生成
        prompt = rag_retrieve(q, collection, top_k=3)
        answer = ask_ollama(prompt, model=args.model)

        print("\nモデルの回答：\n", answer)


if __name__ == "__main__":
    main()