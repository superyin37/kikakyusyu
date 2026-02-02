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
  - app.py：FastAPI 端点、RAG 调用与日志（v2.0已移除known_items依赖）
  - schemas.py：Pydantic 请求/响应模型
- front-streaming/
  - app.py：Streamlit UI、模式切换（Blocking/Streaming）、GPU 监控
  - gpu_stats.py：NVML / nvidia-smi / rocm-smi 采集
- rag/
  - rag_demo3.py：RAG 主逻辑（v2.0使用Hybrid系统，旧MeCab逻辑已注释）
  - **hybrid_grounding.py**：**Hybrid品名指称系统核心模块（v2.0新增）**
  - **test_hybrid.py**：单元测试套件（13个测试用例）
  - **benchmark_hybrid.py**：性能基准测试工具
  - **debug_hybrid.py**：调试工具（单输入详细分析）
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
   - **Hybrid Grounding 系统品名识别**（v2.0 核心升级）：
     - 第一步：精确匹配检查（输入 = 数据库品名 → 置信度1.0直接返回）
     - 第二步：路径选择
       * 短输入（<20字符）→ 路径A（整体Embedding）→ <300ms快速响应
       * 长输入（≥20字符）→ 路径A + 路径B（LLM提取）→ 高精度双路径
     - 第三步：候选合并与置信度评估
     - 降级保护：Hybrid失败自动回退MeCab方式
   - 町名提取：保持原有部分匹配逻辑（未改动）
   - ChromaDB 检索：
     - gomi collection：垃圾分类规则与出し方
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

### 5.1 品名识别（Hybrid Grounding v2.0）
**核心模块**：`rag/hybrid_grounding.py`

**三层识别策略**：
1. **精确匹配层**（优先级最高）
   - 输入与数据库品名完全一致 → 置信度1.0立即返回
   - 跳过向量检索，响应最快

2. **路径A：整体Embedding**
   - 对完整用户输入进行向量化
   - ChromaDB余弦相似度检索
   - 适用场景：短输入（<20字符）

3. **路径B：LLM辅助提取**
   - 使用LLM从长文本中提取候选品名
   - 对每个候选分别进行向量检索
   - 适用场景：长输入（≥20字符）

**配置参数**（`HybridConfig`类）：
```python
SHORT_INPUT_THRESHOLD = 20      # 快速路径阈值
CONFIDENCE_THRESHOLD_HIGH = 0.70  # 高置信度阈值（从0.45提升）
CONFIDENCE_THRESHOLD_LOW = 0.45   # 低置信度阈值（从0.30提升）
AMBIGUITY_THRESHOLD = 0.05       # 歧义判定阈值
```

**相似度计算**：
- 公式：`similarity = max(0.0, min(1.0, 1.0 - distance))`
- 双路径命中加成：`final_score = avg(scores) + 0.1`

**降级机制**：
```python
# Hybrid失败 → MeCab形态素分析 → 简单向量查询
```

### 5.2 检索与生成（保持不变）
- 关键词抽取（降级备用）：MeCab（/var/lib/mecab/dic/debian）在 rag/rag_demo3.py 中的 extract_nouns
- 检索：ChromaDB PersistentClient（路径 chroma_db/），Ollama Embedding `kun432/cl-nagoya-ruri-large:337m`
- 生成：Ollama LLM `swallow:latest`，在 rag_demo3.py 的 ask_ollama 中调用
- 提示词构建：rag_retrieve_extended 生成多段上下文（垃圾分类规则、町名收集日、用户知识），并附加安全/一致性规则

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

### Hybrid Grounding 系统监控
**性能指标**：
```python
# references中的grounding_info
{
  "type": "grounding_info",
  "confidence": "high|medium|low",
  "is_ambiguous": bool,
  "candidates": ["候选1", "候选2", "候选3"],
  "execution_time_ms": float
}

# references中的性能信息
{
  "type": "performance",
  "retrieval_time_ms": float  # 包含Hybrid耗时
}
```

**目标性能**（已验证）：
- 精确匹配：<5ms
- 短输入（路径A）：<300ms
- 长输入（双路径）：<600ms
- LLM超时保护：5秒

**控制台日志示例**：
```
✅ Hybrid Grounding: ノートパソコン
   置信度: high, 相似度: 1.000
   使用パス: path_a_exact, 耗時: 2.3ms
   
⚠️ Hybrid Grounding失敗、フォールバックモード: timeout
   形態素解析名詞: ['パソコン', 'プリンター']
   フォールバック成功: ノートパソコン
```

## 8. 开发与调试建议

### 8.1 Hybrid系统调试工具
```bash
# 单输入详细分析（推荐）
python rag/debug_hybrid.py "ノートパソコンを捨てたい"

# 运行测试套件
python -m pytest rag/test_hybrid.py -v -s

# 性能基准测试
python rag/benchmark_hybrid.py
```

### 8.2 配置调优
**修改文件**：`rag/hybrid_grounding.py` 中的 `HybridConfig` 类

```python
# 快速路径阈值（字符数）
SHORT_INPUT_THRESHOLD = 20  # 降低→更多使用快速路径

# 置信度阈值
CONFIDENCE_THRESHOLD_HIGH = 0.70  # 提高→更严格的高置信判定
CONFIDENCE_THRESHOLD_LOW = 0.45   # 调整low/medium分界线

# LLM参数
LLM_MODEL = "swallow:latest"      # 更换LLM模型
LLM_TEMPERATURE = 0.1              # 调整输出随机性
PATH_B_TIMEOUT = 5                 # LLM超时时间（秒）
```

### 8.3 常规调试
- 首次运行前清理或备份 chroma_db/，确保与当前模型版本一致
- 修改 RAG 逻辑主要入口：rag/rag_demo3.py
  - ~~关键词抽取~~（已替换为Hybrid系统）
  - 检索 → 上下文组装 → 提示词 → ask_ollama
- 如需更换模型：
  - 生成模型：backend/app.py 与 rag/rag_demo3.py 的 ask_ollama
  - 向量模型：rag/rag_demo3.py 与 rag/user_knowledge.py 中的 embedding_functions.OllamaEmbeddingFunction
  - Hybrid LLM：hybrid_grounding.py 中的 HybridConfig.LLM_MODEL
- 超时/错误排查：
  - Ollama 服务是否启动、模型是否已 pull
  - MeCab 字典路径是否存在
  - ChromaDB 目录权限
  - 端口冲突（8000/8501）

## 9. 常见问题

### Hybrid系统相关
- **品名识别不准确**：
  - 检查输入是否在数据库中（`python rag/debug_hybrid.py "<输入>"`）
  - 调整置信度阈值（HybridConfig.CONFIDENCE_THRESHOLD_HIGH）
  - 查看是否触发了降级逻辑（控制台日志）

- **Hybrid系统超时**：
  - 检查Ollama服务是否正常（`curl http://localhost:11434/api/tags`）
  - 增加PATH_B_TIMEOUT值（默认5秒）
  - 查看LLM模型是否已加载（`ollama list | grep swallow`）

- **相似度超过1.0错误**：
  - 已在v2.0修复（相似度裁剪）
  - 如仍出现，检查ChromaDB版本和embedding一致性

- **测试失败**：
  - 确保ChromaDB中有gomi collection数据
  - 检查rag_docs_merged.jsonl是否存在
  - 运行：`python -m pytest rag/test_hybrid.py -v -s --tb=short`

### 通用问题
- Streaming 无输出：检查 /api/bot/respond_stream 是否可访问，或模型是否支持流式
- 参照信息为空：gomi/area/knowledge 是否构建成功；输入未命中品名/町名
- MeCab 报错：确认 debian 字典安装并可读（仅降级时使用）
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
