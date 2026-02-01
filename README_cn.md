# 成员
- 青木颯大
- Yin Hanyang
- 安田大朗

# 项目概述
Kita 是面向北九州市垃圾分类与收集日查询的 RAG（Retrieval-Augmented Generation）系统。它从用户输入中抽取品名/町名，在 ChromaDB 中检索分类规则、收集日与用户追加的知识，并通过 Ollama 上的本地 LLM 生成回答。WebUI 使用 Streamlit，支持 Blocking 与 Streaming 两种模式。

# 系统设计（总览）
整体结构图请参考 system.md。

## 组件构成
- 前端（Streamlit）
  - 聊天 UI（Blocking / Streaming）
  - GPU/VRAM 监控
  - 知识文件上传（PDF/TXT/CSV/JSON）
  - 会话日志展示
- 后端 API（FastAPI）
  - /api/bot/respond（Blocking）
  - /api/bot/respond_stream（Streaming）
  - RAG 编排与 references 返回
- RAG 核心（rag）
  - MeCab 形态素分析（名词抽取）
  - ChromaDB 检索（gomi / area / knowledge）
  - RAG 提示词构建
- LLM 推理（Ollama）
  - 本地推理（支持 Streaming）

## 数据流概述
1. 接收用户输入（品名/町名）
2. MeCab 抽取名词 → 推断品名/町名候选
3. ChromaDB 检索（垃圾分类 / 町名收集日 / 用户知识）
4. 构建 RAG 提示词
5. Ollama 生成回答（Blocking / Streaming）
6. 将 references（file/page/chunk）返回给 WebUI

# 已实现的主要功能
- 垃圾分类规则检索与回答生成
- 町名对应的收集日（家庭垃圾/塑料/粗大）提示
- 用户追加知识（PDF/TXT/CSV/JSON）的检索与回答
- Streaming 输出与 TTFB/总耗时等指标展示
- GPU/VRAM 监控（NVML / nvidia-smi / rocm-smi）
- 日志保存（API 与 WebUI）

# 技术栈
- 语言/框架
  - Python 3.10+
  - FastAPI（API）
  - Streamlit（WebUI）
- RAG / 检索
  - ChromaDB（持久化向量数据库）
  - MeCab（日文形态素分析）
  - LangChain（文本切分）
- LLM / Embedding
  - Ollama（本地推理）
  - LLM: swallow:latest（Llama-3.1-Swallow-8B 系）
  - Embedding: kun432/cl-nagoya-ruri-large:337m
- 主要库
  - PyPDF2, pandas, requests, uvicorn 等

# 模型与使用位置
- 生成模型（LLM）
  - Ollama 上的 swallow:latest
  - Blocking / Streaming 两种模式均使用
- 向量模型（Embedding）
  - kun432/cl-nagoya-ruri-large:337m（Ollama Embedding）
  - 用于 gomi / area / knowledge 三类集合构建

# 目录结构（主要）
- backend/
  - app.py: FastAPI 端点与 RAG 执行
  - schemas.py: 请求/响应定义
- front-streaming/
  - app.py: Streamlit UI
  - gpu_stats.py: GPU/VRAM 获取
- rag/
  - rag_demo3.py: RAG 核心（抽取/检索/提示词/推理）
  - user_knowledge.py: 知识文件切分与入库
- knowledge_files/
  - 用户上传的知识文件

## 运行方法
1. 安装 uv 并初始化项目
```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv init
$ uv venv
```

2. 使用 uv 安装依赖（同步）
pyproject.toml 中声明的依赖将被安装
```
$ uv sync
```

3. 激活虚拟环境
```
source .venv/bin/activate
```

4. 安装并启动 Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```
启动：
nohup ollama serve > ollama.log 2>&1 &

5. 启动服务
API：
```
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

WebUI：
```
streamlit run front-streaming/app.py
```

## 常用 git 操作
- git branch：显示当前分支
- git switch -c sample：创建分支
- git switch sample：切换到分支
- git pull：拉取远程最新
- git pull origin feature/create_api：拉取指定分支
