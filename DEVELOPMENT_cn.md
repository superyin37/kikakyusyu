# Kita 开发文档（中文）

本项目是面向北九州市垃圾分类与收集日查询的 RAG 系统，前端基于 Streamlit，后端基于 FastAPI，检索依赖 ChromaDB，推理使用本地 Ollama。

## 1. 环境与快速启动
1) 安装与初始化
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv venv
source .venv/bin/activate
uv sync
```

2) 安装并启动 Ollama（需本机可用 GPU 或 CPU）
```
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
```

3) 启动服务
- 后端 API（FastAPI）：
```
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- 前端 WebUI（Streamlit）：
```
streamlit run front-streaming/app.py
```

4) 访问
- 默认 Streamlit: http://localhost:8501
- API: http://localhost:8000/api/bot/respond 或 /respond_stream

## 2. 目录与核心文件
- backend/
  - app.py：FastAPI 端点、RAG 调用与日志
  - schemas.py：Pydantic 请求/响应模型
- front-streaming/
  - app.py：Streamlit UI、模式切换（Blocking/Streaming）、GPU 监控
  - gpu_stats.py：NVML / nvidia-smi / rocm-smi 采集
- rag/
  - rag_demo3.py：RAG 主逻辑（关键词抽取、检索、提示词生成、调用 Ollama）
  - user_knowledge.py：用户知识文件切分与写入 ChromaDB
- knowledge_files/：用户上传的知识文件存放路径
- chroma_db/：ChromaDB 持久化目录
- system.md：系统架构图（ASCII）
- README.md：项目日文版说明
- README_cn.md：项目中文概要

## 3. 运行时流程（端到端）
1) 用户在 Streamlit 输入问题（包含品名/町名）
2) 前端根据模式调用 API：
   - Blocking: POST /api/bot/respond
   - Streaming: POST /api/bot/respond_stream（分块返回文本，参考信息置于响应头 X-References）
3) 后端执行 RAG：
   - MeCab 形态素分析，抽取名词 → 推断品名/町名
   - ChromaDB 检索：
     - gomi collection：垃圾分分类则与出し方
     - area collection：町名与收集日信息
     - knowledge collection：用户上传知识
   - 组装上下文与提示词，调用 Ollama（swallow:latest）生成
4) 后端返回答案与引用（references），前端显示并记录日志、更新 GPU 状态

## 4. API 说明（后端）
- POST /api/bot/respond
  - 请求体：{ "prompt": "..." }
  - 响应：{ "reply": "string", "references": [ {file,page,chunk,text} ] }

- POST /api/bot/respond_stream
  - 请求体：同上
  - 响应：Streaming 文本，HTTP 头 X-References 携带 JSON 串

参考文件：
- backend/app.py
- backend/schemas.py

## 5. RAG 细节与模型
- 关键词抽取：MeCab（/var/lib/mecab/dic/debian）在 rag/rag_demo3.py 中的 extract_nouns / extract_keywords
- 检索：ChromaDB PersistentClient（路径 chroma_db/），Ollama Embedding `kun432/cl-nagoya-ruri-large:337m`
- 生成：Ollama LLM `swallow:latest`，在 rag_demo3.py 的 ask_ollama 中调用
- 提示词构建：rag_retrieve_extended 生成多段上下文（垃圾分分类则、町名收集日、用户知识），并附加安全/一致性规则

## 6. 数据与知识库
- 垃圾规则：rag/rag_docs_merged.jsonl → gomi collection
- 町名收集日：rag/area.jsonl → area collection
- 用户知识：knowledge collection（用户上传 PDF/TXT/CSV/JSON）
- 持久化位置：kita/chroma_db/

### 添加用户知识
- 通过 Streamlit 侧边栏上传文件，自动切分并写入 ChromaDB（见 front-streaming/app.py → add_file_to_chroma）
- 或离线脚本：调用 rag/user_knowledge.py 中的 add_file_to_chroma(Path)

切分策略：
- PDF：按页面并每 500 字一段（含 page, chunk）
- TXT：递归分段（chunk_size=500, overlap=50）
- CSV：按 batch=50 行合并成文本
- JSON：按列表元素或键分块

## 7. 日志与监控
- API 侧日志：backend/logs.jsonl
- WebUI 侧日志：同文件（追加 total_time 等）
- GPU/VRAM：front-streaming/gpu_stats.py，优先 NVML，回退 nvidia-smi 或 rocm-smi
- Streamlit UI 展示 TTFB/Total 等指标（Blocking 与 Streaming 分别处理）

## 8. 开发与调试建议
- 首次运行前清理或备份 chroma_db/，确保与当前模型版本一致
- 修改 RAG 逻辑主要入口：rag/rag_demo3.py
  - 关键词抽取 → 检索 → 上下文组装 → 提示词 → ask_ollama
- 如需更换模型：
  - 生成模型：backend/app.py 与 rag/rag_demo3.py 的 ask_ollama
  - 向量模型：rag/rag_demo3.py 与 rag/user_knowledge.py 中的 embedding_functions.OllamaEmbeddingFunction
- 超时/错误排查：
  - Ollama 服务是否启动、模型是否已 pull
  - MeCab 字典路径是否存在
  - ChromaDB 目录权限
  - 端口冲突（8000/8501）

## 9. 常见问题
- Streaming 无输出：检查 /api/bot/respond_stream 是否可访问，或模型是否支持流式
- 参照信息为空：gomi/area/knowledge 是否构建成功；输入未命中品名/町名
- MeCab 报错：确认 debian 字典安装并可读
- GPU 统计为 N/A：环境缺少 NVML 或 nvidia-smi/rocm-smi

## 10. 测试建议
- 单元层：
  - 关键词抽取（rag_demo3.extract_keywords）
  - 知识切分与入库（user_knowledge.add_file_to_chroma）
- 集成层：
  - 调用 /api/bot/respond，校验 references 与回答格式
  - Streaming 路径，校验首包时间与头部 X-References
- 数据一致性：
  - gomi/area jsonl 变更后重建 chroma_db

## 11. 部署提示
- 持久化目录：chroma_db/、backend/logs.jsonl、knowledge_files/
- 推荐使用 systemd/supervisor 管理 Ollama 与 uvicorn
- 若容器化，需要在镜像中预装 MeCab 字典与 Ollama 模型，或在启动脚本中拉取

## 12. 参考文件
- 架构图：system.md
- 日文版说明：README.md
- 中文概要：README_cn.md
