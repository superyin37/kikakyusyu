# メンバー
- 青木颯大
- Yin Hanyang
- 安田大朗

# プロジェクト概要
Kita は北九州市のごみ分別・収集日案内に特化した RAG（Retrieval-Augmented Generation）システムです。ユーザー入力から品名・町名を抽出し、ChromaDB に保存された分別ルール／収集日／ユーザー追加ナレッジを検索して、Ollama 上のローカル LLM が回答を生成します。Streamlit の WebUI から Blocking / Streaming の両モードで利用できます。

# システム設計（全体像）
詳細な構成図は system.md を参照してください。

## コンポーネント構成
- フロントエンド（Streamlit）
	- チャット UI（Blocking / Streaming）
	- GPU/VRAM モニタ
	- ナレッジファイル（PDF/TXT/CSV/JSON）アップロード
	- 会話ログ表示
- バックエンド API（FastAPI）
	- /api/bot/respond（Blocking）
	- /api/bot/respond_stream（Streaming）
	- RAG 実行と参照情報（references）返却
- RAG コア（rag）
	- MeCab による形態素解析（名詞抽出）
	- ChromaDB 検索（gomi / area / knowledge）
	- RAG プロンプト構築
- LLM 推論（Ollama）
	- ローカル推論（Streaming 対応）

## データフロー概要
1. ユーザー入力（品名・町名）を受け取る
2. **Hybrid Grounding システムで品名候補を生成**
   - 精密マッチング：入力が既知品名と完全一致 → 置信度1.0で即座に返却
   - 短入力（<20文字）：路径A（整体Embedding）のみ実行 → 高速応答
   - 長入力（≥20文字）：路径A + 路径B（LLM抽出）の双路径実行 → 高精度
   - フォールバック機能：Hybrid失敗時は従来のMeCab方式に自動降格
3. ChromaDB で検索（ごみ分別 / 町名収集日 / ユーザーナレッジ）
4. 取得コンテキストから RAG プロンプトを生成
5. Ollama で回答生成（Blocking / Streaming）
6. 参照情報（file/page/chunk + grounding情報）を WebUI に返却

# 実装されている主な機能
- **Hybrid品名指称システム（v2.0）**
  - 精密マッチング + 語義検索の二段階方式
  - 短入力高速路径（<300ms）/ 長入力LLM増強路径
  - 曖昧性検出と置信度評価
  - 自動フォールバック機能
- ごみ分別ルールの検索と回答生成
- 町名ごとの収集日（家庭ごみ/プラスチック/粗大）提示
- ユーザー追加ナレッジ（PDF/TXT/CSV/JSON）の検索・回答
- Streaming 応答と TTFB/総時間などのメトリクス表示
- GPU/VRAM モニタ（NVML / nvidia-smi / rocm-smi）
- ログ保存（API 側・WebUI 側）

# 技術スタック
- 言語/フレームワーク
	- Python 3.10+
	- FastAPI（API）
	- Streamlit（WebUI）
- RAG / 検索
	- ChromaDB（永続ベクトルDB）
	- MeCab（日本語形態素解析）
	- LangChain（テキスト分割）
- LLM / Embedding
	- Ollama（ローカル推論）
	- LLM: swallow:latest（Llama-3.1-Swallow-8B 系）
	- Embedding: kun432/cl-nagoya-ruri-large:337m
- 主要ライブラリ
	- PyPDF2, pandas, requests, uvicorn ほか

# モデルと利用箇所
- 生成モデル（LLM）
	- Ollama で `swallow:latest` を使用
	- API の Blocking / Streaming の両方で使用
- 埋め込みモデル（Embedding）
	- `kun432/cl-nagoya-ruri-large:337m`（Ollama Embedding）
	- ChromaDB の gomi / area / knowledge コレクション構築に使用

# ディレクトリ構成（主要）
- backend/
	- app.py: FastAPI エンドポイントと RAG 実行
	- schemas.py: リクエスト/レスポンス定義
- front-streaming/
	- app.py: Streamlit UI
	- gpu_stats.py: GPU/VRAM 取得
- rag/
	- rag_demo3.py: RAG コア（抽出/検索/プロンプト/推論）
	- hybrid_grounding.py: **Hybrid品名指称システム（v2.0新規追加）**
	- test_hybrid.py: Hybrid システムの単元テスト
	- benchmark_hybrid.py: 性能ベンチマークテスト
	- debug_hybrid.py: デバッグツール
	- user_knowledge.py: ナレッジファイルのチャンク化と登録
- knowledge_files/
	- ユーザーがアップロードしたナレッジ

## 実行方法  
1. uvのインストール, プロジェクトの作成  
```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv init
$ uv venv
```

2. uvでライブラリインストール (同期)  
pyproject.tomlに書かれているライブラリをuvにインストール
```
$ uv sync
```

3. uvを有効化  
```
source .venv/bin/activate
```

4. ollamaのインストール  
```
curl -fsSL https://ollama.com/install.sh | sh
```
起動
nohup ollama serve > ollama.log 2>&1 &

5. 実行方法
APIの起動  
```
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

WebUIの起動  
```
streamlit run front-streaming/app.py
```

## 基本的なgit操作  
git branch : 現在いるブランチを表示できる  
git switch -c sample : ブランチの作成  
git switch sample: sampleブランチへ移動  
git pull : リモートリポジトリの最新状態をローカルに反映  
git pull origin feature/create_api: 特定のブランチを最新状態にする  
