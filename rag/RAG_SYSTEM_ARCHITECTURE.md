# RAG系统架构设计文档

## 文档概述
本文档详细描述了Kita（北九州市垃圾分类与收集日查询系统）的RAG（Retrieval-Augmented Generation）系统架构设计、技术实现和开发规范。

---

## 1. 系统概述

### 1.1 项目简介
Kita是一个专门为北九州市设计的智能垃圾分类与收集日查询系统，采用RAG架构实现知识增强的自然语言问答功能。系统能够：
- 准确回答垃圾分类规则
- 查询各町名的垃圾收集日期
- 支持用户自定义知识库（PDF/TXT/CSV/JSON）
- 提供流式和阻塞式两种响应模式

### 1.2 核心特性
- **多模态知识库支持**：预置垃圾分类规则 + 町名收集日 + 用户自定义知识
- **日语自然语言处理**：基于MeCab的形态素分析
- **向量化语义检索**：ChromaDB + Ollama Embedding
- **本地化LLM推理**：Ollama运行Llama-3.1-Swallow-8B模型
- **双模式响应**：支持Blocking和Streaming两种API模式
- **实时性能监控**：GPU/VRAM使用率监控、TTFB/响应时间追踪

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层 (Streamlit)                    │
│  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐   │
│  │ 聊天界面    │  │ 知识库管理     │  │ GPU/VRAM监控         │   │
│  └────────────┘  └───────────────┘  └──────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP Request/Response
┌─────────────────────▼───────────────────────────────────────────┐
│                      API层 (FastAPI)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  /api/bot/respond        (Blocking模式)                   │   │
│  │  /api/bot/respond_stream (Streaming模式)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                      RAG核心层 (rag/)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 关键词抽取    │  │ 向量检索      │  │ 提示词工程           │  │
│  │ (MeCab)      │  │ (ChromaDB)   │  │ (Prompt Engineering)│  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   数据存储层                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ChromaDB 持久化向量数据库 (chroma_db/)                   │   │
│  │   ├── gomi collection     (垃圾分类规则)                  │   │
│  │   ├── area collection     (町名收集日)                    │   │
│  │   └── knowledge collection (用户知识库)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  JSONL数据源                                               │   │
│  │   ├── rag_docs_merged.jsonl  (垃圾分类源数据)             │   │
│  │   └── area.jsonl             (町名收集日源数据)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  用户知识文件 (knowledge_files/)                           │   │
│  │   支持: PDF / TXT / CSV / JSON                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   LLM推理层 (Ollama)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LLM模型: swallow:latest (Llama-3.1-Swallow-8B)          │   │
│  │  Embedding: kun432/cl-nagoya-ruri-large:337m             │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 架构层次说明

#### 2.2.1 用户界面层 (front-streaming/)
- **技术栈**: Streamlit
- **主要文件**: `front-streaming/app.py`
- **功能**:
  - 聊天界面：支持Blocking/Streaming两种对话模式
  - 知识库管理：文件上传、列表展示
  - 性能监控：实时GPU/VRAM使用率、TTFB/Total Time显示
  - 会话历史：展示最近5条对话记录

#### 2.2.2 API层 (backend/)
- **技术栈**: FastAPI + Uvicorn
- **主要文件**: 
  - `backend/app.py`: API端点实现
  - `backend/schemas.py`: Pydantic数据模型
- **端点**:
  - `POST /api/bot/respond`: Blocking模式，返回完整JSON响应
  - `POST /api/bot/respond_stream`: Streaming模式，逐块返回文本，参考信息放在HTTP头
- **功能**:
  - 接收用户查询
  - 调用RAG核心进行检索和生成
  - 返回答案及参考信息（references）
  - 记录日志到`backend/logs.jsonl`

#### 2.2.3 RAG核心层 (rag/)
- **主要文件**:
  - `rag/rag_demo3.py`: RAG主逻辑
  - `rag/user_knowledge.py`: 用户知识库管理
- **核心模块**:
  1. **关键词抽取模块** (`extract_nouns`, `extract_keywords`)
     - 使用MeCab进行日语形态素分析
     - 提取名词并与已知品名/町名列表匹配
  2. **向量检索模块** (`query_chroma`, `rag_retrieve_extended`)
     - 在ChromaDB的多个collection中进行语义检索
     - 优先级：用户知识 > 垃圾分类规则 > 町名信息
  3. **提示词工程模块** (`rag_retrieve_extended`)
     - 构建包含上下文的RAG提示词
     - 集成安全规则和领域限制
  4. **LLM调用模块** (`ask_ollama`)
     - 通过Ollama Python SDK调用本地LLM
     - 支持流式和非流式输出

#### 2.2.4 数据存储层
- **ChromaDB** (持久化向量数据库)
  - 路径: `chroma_db/`
  - Collections:
    - `gomi`: 866条垃圾分类规则
    - `area`: 825个町名收集日信息
    - `knowledge`: 用户上传的知识文档
- **JSONL数据源**
  - `rag/rag_docs_merged.jsonl`: 垃圾分类源数据
  - `rag/area.jsonl`: 町名收集日源数据
- **用户知识文件**
  - 路径: `knowledge_files/`
  - 支持格式: PDF, TXT, CSV, JSON

#### 2.2.5 LLM推理层
- **Ollama服务**
  - 本地部署的LLM推理引擎
  - 模型管理和推理优化
- **使用的模型**:
  - 生成模型: `swallow:latest` (Llama-3.1-Swallow-8B-Instruct)
  - 向量模型: `kun432/cl-nagoya-ruri-large:337m`

---

## 3. 数据流详解

### 3.1 用户查询流程 (端到端)

```
1. 用户输入
   ↓
2. Streamlit前端接收
   ↓
3. 发送HTTP请求到FastAPI
   ├─ Blocking: POST /api/bot/respond
   └─ Streaming: POST /api/bot/respond_stream
   ↓
4. RAG核心处理
   ├─ 4.1 关键词抽取 (MeCab)
   │     - 形态素分析
   │     - 名词提取
   │     - 品名/町名匹配
   ├─ 4.2 向量检索 (ChromaDB)
   │     - gomi collection查询 (top_k=2)
   │     - area collection查询 (完全匹配)
   │     - knowledge collection查询 (top_k=2)
   ├─ 4.3 上下文组装
   │     - 合并检索结果
   │     - 构建结构化上下文
   ├─ 4.4 提示词生成
   │     - 插入系统指令
   │     - 添加安全规则
   │     - 格式化用户问题
   └─ 4.5 LLM推理 (Ollama)
         - 调用swallow模型
         - 生成日语回答
   ↓
5. 返回响应
   ├─ Blocking: 完整JSON (reply + references)
   └─ Streaming: 文本流 + HTTP头中的references
   ↓
6. 前端展示
   ├─ 显示生成的回答
   ├─ 展示参考信息 (文件、页码、片段)
   └─ 更新性能指标 (TTFB, Total Time)
```

### 3.2 知识库更新流程

```
1. 用户在Streamlit上传文件
   ↓
2. 文件保存到 knowledge_files/
   ↓
3. 调用 add_file_to_chroma()
   ├─ 3.1 根据文件类型选择分块策略
   │     ├─ PDF: 按页分块 (500字/块)
   │     ├─ TXT: 递归分块 (chunk_size=500, overlap=50)
   │     ├─ CSV: 批量合并 (50行/块)
   │     └─ JSON: 按元素或键分块
   ├─ 3.2 生成元数据
   │     - file: 文件名
   │     - page: 页码 (仅PDF)
   │     - chunk: 片段编号
   ├─ 3.3 向量化 (Ollama Embedding)
   │     - 调用kun432/cl-nagoya-ruri-large:337m
   └─ 3.4 写入ChromaDB
         - collection: knowledge
         - 持久化到 chroma_db/
   ↓
4. 即时生效，可在后续查询中检索
```

---

## 4. 核心技术组件

### 4.1 MeCab形态素分析

**作用**: 日语文本的词法分析，提取名词作为检索关键词

**配置**:
```python
dic_dir = "/var/lib/mecab/dic/debian"
tagger = MeCab.Tagger(f"-Ochasen -r /etc/mecabrc -d {dic_dir}")
```

**使用场景**:
- 从用户输入中提取品名候选（如"ノートPC" → ["ノート", "PC"]）
- 与预置品名列表进行匹配
- 提高关键词抽取准确率

**关键函数**: `extract_nouns()`, `extract_keywords()`

### 4.2 ChromaDB向量数据库

**作用**: 高性能的向量存储和语义检索

**特点**:
- 持久化存储（路径: `./chroma_db`）
- 支持多collection隔离
- 内置向量相似度搜索
- 自动管理索引

**Collection设计**:

| Collection | 数据量 | 主键字段 | Embedding字段 | 用途 |
|-----------|-------|---------|--------------|-----|
| gomi | 866条 | 品名 | 品名 | 垃圾分类规则检索 |
| area | 825条 | 町名 | 町名 | 收集日期查询 |
| knowledge | 动态 | file+chunk | 文本内容 | 用户知识库检索 |

**检索策略**:
- `top_k=2`: 返回最相关的2个结果
- 余弦相似度排序
- 支持metadata过滤

### 4.3 Ollama本地LLM

**架构优势**:
- 本地部署，数据不出网
- 支持多种开源模型
- 统一的API接口
- 流式输出优化

**使用的模型**:

1. **生成模型: swallow:latest**
   - 基础模型: Llama-3.1-Swallow-8B-Instruct
   - 针对日语优化
   - 8B参数量，平衡性能与资源
   - 支持指令遵循

2. **向量模型: kun432/cl-nagoya-ruri-large:337m**
   - 专为日语优化的Embedding模型
   - 337M参数
   - 高质量语义向量
   - 适合知识检索场景

**调用方式**:
```python
# 生成回答
res = ollama.chat(
    model="swallow:latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    stream=True  # 流式输出
)

# 生成向量
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    model_name="kun432/cl-nagoya-ruri-large:337m"
)
```

### 4.4 LangChain文本分块

**作用**: 将长文本切分为适合Embedding的片段

**策略**:
- RecursiveCharacterTextSplitter
- chunk_size=500 字符
- chunk_overlap=50 字符（保持上下文连贯）

**应用**:
- TXT文件处理
- 长PDF页面分块
- 保留语义完整性

---

## 5. RAG流程深度解析

### 5.1 关键词抽取与匹配

**目标**: 从自然语言输入中识别垃圾品名和町名

**算法流程**:
```python
def extract_keywords(user_input, known_items, known_areas):
    # 1. 形态素分析
    nouns = extract_nouns(user_input)  # MeCab提取名词
    
    # 2. 品名匹配
    for noun in nouns:
        if noun in known_items:  # 与866个品名匹配
            keywords["品名"] = noun
            break
    
    # 3. 町名匹配
    for area in known_areas:  # 与825个町名匹配
        if area in user_input:  # 部分匹配
            keywords["町名"] = area
            break
    
    return keywords  # {"品名": "...", "町名": "..."}
```

**优化点**:
- 优先完全匹配
- 支持部分匹配（町名）
- 名词候选回退机制

### 5.2 多源检索与上下文融合

**检索策略**:
```python
def rag_retrieve_extended(user_input, ...):
    # 1. 品名检索
    if keywords["品名"]:
        gomi_hits = query_chroma(gomi_collection, keywords["品名"], n=2)
        knowledge_hits = query_chroma(knowledge_collection, keywords["品名"], n=2)
    
    # 2. 町名检索
    if keywords["町名"]:
        # 完全匹配area_meta
        matched = [h for h in area_meta if h["町名"] == keywords["町名"]]
    
    # 3. 上下文组装
    context_parts = [
        "【垃圾分类信息】\n" + format_gomi(gomi_hits),
        "【用户知识库】\n" + format_knowledge(knowledge_hits),
        "【町名信息】\n" + format_area(matched)
    ]
    
    # 4. 生成最终上下文
    context = "\n\n".join(context_parts)
    return context, references
```

**优先级规则**:
1. 用户知识库（最高优先级）
2. 垃圾分类规则
3. 町名收集日

### 5.3 提示词工程

**系统提示词** (System Prompt):
```
あなたは北九州市のごみ分別・町名収集情報、さらにユーザが追加したナレッジ（PDF文書など）に基づいて回答するアシスタントです。

【優先度ルール】
1. ユーザナレッジベースに情報がある場合 → その情報を根拠に回答
2. ごみ分別・町名収集情報に該当する場合 → ごみ分別ルールに従って回答
3. 上記どちらにも該当しない場合のみ → 拒否メッセージを返す
```

**用户提示词构建**:
```python
prompt = f"""
【重要ルール】
1. 回答で使用できる品名は【ごみ分別情報】に記載された品名のみです。
2. 【ごみ分別情報】に記載されていない品名を新たに作ったり、置き換えたりしてはいけません。
3. 質問内容と【ごみ分別情報】の品名が一致しない場合は、注意書きを付けてください。

【ごみ分別情報】
{context}

【質問】
{user_input}

【出力形式】
- 品名
- 品名の出し方
- 備考
- 該当町名の収集日
"""
```

**安全性设计**:
- 领域限制：仅回答垃圾分类相关问题
- 事实性约束：禁止推测未知信息
- 引用溯源：要求说明信息来源

### 5.4 流式与阻塞式响应

**Blocking模式**:
```python
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    rag_prompt, references = rag_retrieve_extended(...)
    reply = ask_ollama(rag_prompt)
    return {
        "reply": reply,
        "references": references
    }
```

**Streaming模式**:
```python
@app.post("/api/bot/respond_stream")
async def rag_respond_stream(req: PromptRequest):
    rag_prompt, references = rag_retrieve_extended(...)
    
    def stream_gen():
        stream = ollama.chat(..., stream=True)
        for event in stream:
            content = event.get("message", {}).get("content", "")
            if content:
                yield content
    
    return StreamingResponse(
        stream_gen(),
        headers={"X-References": json.dumps(references)}
    )
```

**差异对比**:
| 特性 | Blocking | Streaming |
|-----|----------|-----------|
| 响应方式 | 等待完整生成后一次性返回 | 逐块返回，实时展示 |
| TTFB | 较高 | 较低 |
| 用户体验 | 等待时间长 | 渐进式展示 |
| 错误处理 | 简单 | 复杂 |
| References位置 | JSON body | HTTP Header |

---

## 6. 性能监控与日志

### 6.1 GPU/VRAM监控

**实现**: `front-streaming/gpu_stats.py`

**监控指标**:
- VRAM使用量 (GB)
- GPU利用率 (%)
- GPU型号

**采集方式** (优先级递减):
1. NVML (NVIDIA Management Library)
2. nvidia-smi 命令
3. rocm-smi 命令 (AMD GPU)

**Streamlit展示**:
```python
stats = get_gpu_stats()
if stats:
    used_gb, total_gb, util_p, name = stats
    st.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
    st.caption(f"{name} | Util {util_p}%")
```

### 6.2 响应时间追踪

**监控指标**:
- **TTFB** (Time To First Byte): 首字节返回时间
- **Total Time**: 完整响应时间
- **Tokens/sec**: 生成速度（预留）

**实现**:
```python
t_start = time.perf_counter()
# ... 请求处理 ...
t_end = time.perf_counter()
total_sec = t_end - t_start
```

### 6.3 日志系统

**日志文件**: `backend/logs.jsonl`

**日志格式**:
```json
{
  "timestamp": "2026-02-01 10:30:45",
  "mode": "Streaming(API)",
  "user": "ノートPCの捨て方を教えて",
  "assistant": "ノートPC（パソコン本体）は...",
  "total_time": 2.341
}
```

**记录时机**:
- API层: 每次请求完成后
- WebUI层: 用户查看响应后

**用途**:
- 问题溯源
- 性能分析
- 用户行为分析

---

## 7. 部署与运维

### 7.1 环境要求

**硬件**:
- CPU: 4核以上
- RAM: 16GB以上
- GPU: 推荐NVIDIA GPU (8GB+ VRAM) 或 AMD GPU
- 存储: 20GB以上可用空间

**软件**:
- 操作系统: Linux (推荐Ubuntu 20.04+)
- Python: 3.10+
- MeCab: debian字典
- Ollama: 最新版

### 7.2 安装步骤

详见项目根目录的README.md，核心步骤：
1. 安装uv包管理器
2. 创建虚拟环境并安装依赖
3. 安装并启动Ollama服务
4. 拉取所需模型
5. 启动后端API和前端WebUI

### 7.3 持久化数据

**需备份的目录**:
- `chroma_db/`: 向量数据库
- `knowledge_files/`: 用户上传的文件
- `backend/logs.jsonl`: 操作日志

**备份建议**:
- 定期备份chroma_db（每周）
- 自动备份logs.jsonl（每日）
- 可选备份knowledge_files

### 7.4 常见运维问题

**问题1**: MeCab初始化失败
- **原因**: 字典路径不正确
- **解决**: 确认 `/var/lib/mecab/dic/debian` 存在

**问题2**: Ollama模型未找到
- **原因**: 模型未拉取
- **解决**: 执行 `ollama pull swallow:latest`

**问题3**: ChromaDB权限错误
- **原因**: chroma_db目录权限不足
- **解决**: `chmod -R 755 chroma_db/`

**问题4**: Streaming无输出
- **原因**: 网络超时或模型响应慢
- **解决**: 增加timeout参数，检查GPU性能

---

## 8. 安全性与合规性

### 8.1 数据安全

- **本地化部署**: 所有数据和模型在本地，无外网依赖
- **无数据泄露**: 不向第三方API发送用户查询
- **持久化加密**: 可选择对chroma_db进行磁盘加密

### 8.2 输入验证

- **长度限制**: 限制输入文本长度（建议<1000字符）
- **类型检查**: Pydantic模型验证
- **注入防护**: 提示词中避免直接拼接用户输入

### 8.3 领域限制

通过系统提示词和规则约束，确保：
- 仅回答垃圾分类相关问题
- 拒绝无关查询（元指令、编码请求等）
- 不泄露系统内部信息

---

## 9. 扩展性设计

### 9.1 支持新数据源

**添加新的垃圾品名**:
1. 编辑 `rag/rag_docs_merged.jsonl`
2. 添加新的JSON行（品名、出し方、备考）
3. 重启后端API，自动重建collection

**添加新町名**:
1. 编辑 `rag/area.jsonl`
2. 添加新的收集日信息
3. 重启后端API

### 9.2 支持新LLM模型

**更换生成模型**:
1. 在Ollama中拉取新模型: `ollama pull <model_name>`
2. 修改 `backend/app.py` 和 `rag/rag_demo3.py` 中的 `model` 参数
3. 调整提示词以适配新模型

**更换Embedding模型**:
1. 拉取新的Embedding模型
2. 修改 `embedding_functions.OllamaEmbeddingFunction` 的 `model_name`
3. **重要**: 重建ChromaDB（删除旧的chroma_db/）

### 9.3 多语言支持

**实现思路**:
- 添加英语/中文等多语言提示词
- 使用多语言Embedding模型
- 数据源翻译或多语言标注
- UI国际化（i18n）

---

## 10. 测试策略

### 10.1 单元测试

**关键函数测试**:
- `extract_nouns()`: 测试MeCab分词准确性
- `extract_keywords()`: 测试关键词匹配逻辑
- `chunk_pdf()`: 测试PDF分块正确性
- `query_chroma()`: 测试向量检索结果

**测试框架**: pytest

### 10.2 集成测试

**API端点测试**:
```python
def test_blocking_api():
    response = requests.post(
        "http://localhost:8000/api/bot/respond",
        json={"prompt": "ノートPCの捨て方"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "references" in data
```

**RAG流程测试**:
- 已知品名查询 → 验证返回正确分类规则
- 町名查询 → 验证返回正确收集日
- 用户知识库查询 → 验证检索自定义文档

### 10.3 性能测试

**负载测试**:
- 并发请求数: 10/50/100
- 平均响应时间
- GPU/内存使用峰值

**优化目标**:
- Blocking TTFB < 3s
- Streaming TTFB < 1s
- GPU利用率 < 80%

---

## 11. 开发规范

### 11.1 代码风格

- 遵循PEP 8
- 使用类型提示 (Type Hints)
- 函数/变量命名：英文小写+下划线
- 类命名：驼峰命名法

### 11.2 提交规范

**Commit Message格式**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type类型**:
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- refactor: 代码重构
- test: 测试相关
- chore: 构建/工具变动

### 11.3 分支策略

- `main`: 稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复

---

## 12. 参考资料

### 12.1 技术文档
- [ChromaDB官方文档](https://docs.trychroma.com/)
- [Ollama文档](https://github.com/ollama/ollama)
- [LangChain文档](https://python.langchain.com/)
- [MeCab官网](https://taku910.github.io/mecab/)

### 12.2 项目文档
- README.md: 项目概述和快速启动
- README_cn.md: 中文版概述
- DEVELOPMENT_cn.md: 开发者指南

### 12.3 相关论文
- RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Llama 3: Scaling Language Models to 8B Parameters
- Swallow: Japanese Language Model Fine-tuning

---

## 附录A: 数据模型定义

### A.1 垃圾分类规则 (gomi)
```json
{
  "品名": "ノートパソコン",
  "出し方": "粗大ごみ",
  "備考": "小型電子機器回収ボックスへ"
}
```

### A.2 町名收集日 (area)
```json
{
  "町名": "荒手1～2丁目",
  "家庭ごみの収集日": "火曜日・金曜日",
  "プラスチックの収集日": "木曜日",
  "粗大ごみの収集日（事前申込制）": "第3木曜日"
}
```

### A.3 用户知识库 (knowledge)
```json
{
  "file": "manual.pdf",
  "page": 5,
  "chunk": 2,
  "text": "..."
}
```

---

## 附录B: API接口规范

### B.1 Blocking模式

**请求**:
```http
POST /api/bot/respond
Content-Type: application/json

{
  "prompt": "ノートPCの捨て方を教えて"
}
```

**响应**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "reply": "ノートPC（パソコン本体）は粗大ごみとして出すことができます...",
  "references": [
    {
      "file": "manual.pdf",
      "page": 3,
      "chunk": 1,
      "text": "パソコン本体は..."
    }
  ]
}
```

### B.2 Streaming模式

**请求**:
```http
POST /api/bot/respond_stream
Content-Type: application/json

{
  "prompt": "ノートPCの捨て方を教えて"
}
```

**响应**:
```http
HTTP/1.1 200 OK
Content-Type: text/plain
X-References: [{"file":"manual.pdf","page":3,"chunk":1,"text":"..."}]

ノートPC（パソコン本体）は粗大ごみとして...
```

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队
