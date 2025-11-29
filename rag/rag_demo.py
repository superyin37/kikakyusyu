#!/usr/bin/env python3
import csv
import argparse
import chromadb
from chromadb.utils import embedding_functions
import subprocess

# ========== ステップ 1: CSV を読み込む ==========
def load_csv(path):
    """
    CSV ファイルからごみ分別データを読み込み、
    各行をテキストドキュメントに変換する。
    """
    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 各行をテキストスニペットに組み立てる
            text = f"品名: {row.get('品名','')}\n出し方: {row.get('出し方','')}\n備考: {row.get('備考','')}"
            docs.append(text)
    return docs


# ========== ステップ 2: ベクトルデータベースを構築 ==========
def build_chroma(docs, persist_dir="./chroma_db"):
    """
    ChromaDB コレクションを作成し、
    すべてのドキュメントに対して埋め込みを保存する。
    既に存在する場合は削除して再構築する。
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

    # ドキュメントを追加
    collection.add(documents=docs, ids=[str(i) for i in range(len(docs))])
    return collection


# ========== ステップ 3: クエリと生成 ==========
def query_chroma(collection, query, n=3):
    """
    ベクトルデータベースにクエリを投げ、
    最も関連性の高いドキュメントを取得する。
    """
    results = collection.query(query_texts=[query], n_results=n)
    return results["documents"][0]

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
    context = "\n\n".join(query_chroma(collection, user_input, n=top_k))
    prompt = f"以下は北九州市のごみ分別ルールです：\n{context}\n\n質問: {user_input}\n日本語で答えてください:"
    return prompt

# ========== メイン ==========
def main():
    parser = argparse.ArgumentParser(description="Ollama + ChromaDB による RAG デモ")
    parser.add_argument("--csv", required=True, help="ごみ分別 CSV ファイルのパス")
    parser.add_argument("--model", default="llama2", help="使用する Ollama モデル名 (デフォルト: llama2)")
    args = parser.parse_args()

    # データを読み込む
    docs = load_csv(args.csv)
    print(f"{len(docs)} 件のデータを読み込みました")

    # ベクトルデータベースを構築
    collection = build_chroma(docs)

    # 対話型 Q&A
    while True:
        q = input("\n質問を入力してください (終了するには 'exit' と入力): ")
        if q.lower() == "exit":
            break

        # 関連するコンテキストを取得
        context = "\n\n".join(query_chroma(collection, q, n=3))

        # 日本語のプロンプトを組み立てる
        prompt = f"以下は北九州市のごみ分別ルールです：\n{context}\n\n質問: {q}\n日本語で答えてください:"
        answer = ask_ollama(prompt, model=args.model)

        print("\nモデルの回答：\n", answer)


if __name__ == "__main__":
    main()