# メンバー
- 青木颯大
- Yin Hanyang
- 安田大朗

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

5. 実行方法
APIの起動  
```
$ uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

WebUIの起動  
```
$ streamlit run front-streaming/app.py
```

## 基本的なgit操作  
git branch : 現在いるブランチを表示できる  
git switch -c sample : ブランチの作成  
git switch sample: sampleブランチへ移動  
git pull : リモートリポジトリの最新状態をローカルに反映  
git pull origin feature/create_api: 特定のブランチを最新状態にする  
