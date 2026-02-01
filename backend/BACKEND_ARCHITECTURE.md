# 后端API架构与实现文档

## 文档概述
本文档详细描述Kita系统后端API的架构设计、实现细节、数据流程和最佳实践。后端基于FastAPI框架，提供RESTful API接口，连接前端WebUI和RAG核心系统。

---

## 目录
1. [后端概述](#1-后端概述)
2. [架构设计](#2-架构设计)
3. [API端点详解](#3-api端点详解)
4. [数据模型](#4-数据模型)
5. [RAG集成](#5-rag集成)
6. [ChromaDB管理](#6-chromadb管理)
7. [日志系统](#7-日志系统)
8. [错误处理](#8-错误处理)
9. [性能优化](#9-性能优化)
10. [安全性](#10-安全性)
11. [部署配置](#11-部署配置)
12. [测试策略](#12-测试策略)

---

## 1. 后端概述

### 1.1 技术栈

**核心框架**:
- **FastAPI**: 现代化的Web框架，支持异步、自动文档生成
- **Uvicorn**: ASGI服务器，高性能异步支持
- **Pydantic**: 数据验证和序列化

**依赖库**:
- **Ollama Python SDK**: LLM调用
- **ChromaDB**: 向量数据库客户端
- **Python 3.10+**: 类型提示、异步支持

### 1.2 核心职责

```
┌─────────────────────────────────────────────────┐
│              后端API核心职责                      │
├─────────────────────────────────────────────────┤
│ 1. 接收前端HTTP请求                              │
│ 2. 请求验证和数据转换                            │
│ 3. 调用RAG核心进行检索和生成                      │
│ 4. 管理ChromaDB连接和集合                        │
│ 5. 返回结构化响应（JSON/Stream）                 │
│ 6. 记录操作日志                                  │
│ 7. 错误处理和异常捕获                            │
└─────────────────────────────────────────────────┘
```

### 1.3 文件结构

```
backend/
├── app.py              # FastAPI应用主文件
├── schemas.py          # Pydantic数据模型
├── logs.jsonl          # 运行时生成的日志文件
└── chroma_db/          # ChromaDB持久化目录（运行时创建）
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端层 (Streamlit)                        │
│                 HTTP/JSON 请求/响应                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI 应用层                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  路由层 (Endpoints)                                   │   │
│  │   - POST /api/bot/respond        (Blocking)         │   │
│  │   - POST /api/bot/respond_stream (Streaming)        │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  请求处理层                                           │   │
│  │   - 数据验证 (Pydantic)                              │   │
│  │   - 参数解析                                         │   │
│  │   - 错误捕获                                         │   │
│  └─────────────────────┬────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    业务逻辑层                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ChromaDB 管理                                        │   │
│  │   - Collection 初始化                                │   │
│  │   - 连接池管理                                       │   │
│  │   - 数据加载                                         │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  RAG 集成                                             │   │
│  │   - 调用 rag_retrieve_extended()                     │   │
│  │   - 调用 ask_ollama()                                │   │
│  │   - 结果处理                                         │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  日志系统                                             │   │
│  │   - 请求/响应记录                                    │   │
│  │   - 性能指标                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 请求处理流程

#### 2.2.1 Blocking模式流程

```python
用户请求 (POST /api/bot/respond)
    ↓
1. FastAPI 接收请求
    ↓
2. Pydantic 验证数据 (PromptRequest)
    ↓
3. 调用 rag_retrieve_extended()
    │
    ├─ 关键词抽取
    ├─ ChromaDB 检索
    └─ 构建 RAG 提示词
    ↓
4. 调用 ask_ollama()
    │
    ├─ 发送到 Ollama 服务
    └─ 等待完整响应
    ↓
5. 构建响应对象 (ReplyResponse)
    │
    ├─ reply: 生成的回答
    └─ references: 参考信息列表
    ↓
6. 返回 JSON 响应
    ↓
7. 记录日志 (logs.jsonl)
```

#### 2.2.2 Streaming模式流程

```python
用户请求 (POST /api/bot/respond_stream)
    ↓
1. FastAPI 接收请求
    ↓
2. Pydantic 验证数据
    ↓
3. 调用 rag_retrieve_extended()
    ↓
4. 创建 Stream Generator
    │
    └─ def stream_gen():
           for chunk in ollama.chat(..., stream=True):
               yield chunk
    ↓
5. 返回 StreamingResponse
    │
    ├─ Body: 逐块文本流
    └─ Headers: X-References (JSON)
    ↓
6. 流式传输完成后记录日志
```

### 2.3 关键设计模式

#### 2.3.1 单例模式（ChromaDB客户端）

```python
# 全局单例客户端
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# 优点：
# - 避免重复连接
# - 共享连接池
# - 减少资源消耗
```

#### 2.3.2 工厂模式（Collection管理）

```python
def get_or_build_collection(client, name, docs=None, meta=None):
    """
    获取已存在的collection，如果不存在则构建
    """
    try:
        return client.get_collection(name)
    except Exception:
        if docs is None or meta is None:
            raise RuntimeError(f"Collection '{name}' not found")
        return build_chroma(docs, meta, name=name)

# 优点：
# - 自动化初始化
# - 容错处理
# - 代码复用
```

#### 2.3.3 生成器模式（Streaming响应）

```python
def stream_gen():
    collected = ""
    stream = ollama.chat(model="swallow:latest", messages=[...], stream=True)
    for event in stream:
        content = event.get("message", {}).get("content", "")
        if content:
            collected += content
            yield content
    save_log(req.prompt, collected, mode="Streaming(API)")

# 优点：
# - 内存高效
# - 实时响应
# - 自动清理
```

---

## 3. API端点详解

### 3.1 POST /api/bot/respond (Blocking模式)

#### 3.1.1 端点定义

```python
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    """
    Blocking模式RAG问答端点
    
    特点:
    - 等待完整生成后一次性返回
    - 包含完整的references
    - 适合短查询、批处理场景
    """
```

#### 3.1.2 请求格式

**HTTP Method**: POST

**URL**: `http://localhost:8000/api/bot/respond`

**Headers**:
```
Content-Type: application/json
```

**Body** (JSON):
```json
{
  "prompt": "ノートPCの捨て方を教えて"
}
```

**字段说明**:
- `prompt` (string, required): 用户查询文本，长度建议<1000字符

#### 3.1.3 响应格式

**Success (200 OK)**:
```json
{
  "reply": "ノートPC（パソコン本体）は粗大ごみとして出すことができます。小型のものは小型電子機器回収ボックスへ入れることもできます。",
  "references": [
    {
      "file": "manual.pdf",
      "page": 3,
      "chunk": 1,
      "text": "パソコン本体（デスクトップ型・ノート型）は粗大ごみとして..."
    },
    {
      "file": "guide.txt",
      "page": "?",
      "chunk": 5,
      "text": "小型電子機器は回収ボックスへ..."
    }
  ]
}
```

**字段说明**:
- `reply` (string): LLM生成的完整回答
- `references` (array): 参考信息列表
  - `file` (string): 来源文件名
  - `page` (number|string): 页码（PDF）或"?"（非PDF）
  - `chunk` (number|string): 片段编号
  - `text` (string): 片段文本（截取前300字符）

**Error (4xx/5xx)**:
```json
{
  "detail": "错误描述"
}
```

#### 3.1.4 实现代码解析

```python
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    # 1. RAG检索和提示词生成
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2
    )
    
    # 2. Debug输出（可选）
    print("\n===== DEBUG: FULL PROMPT START =====\n")
    print(rag_prompt)
    print("\n===== DEBUG: FULL PROMPT END =====\n")
    
    # 3. LLM生成
    reply = ask_ollama(rag_prompt)
    
    # 4. 返回响应
    return {
        "reply": reply,
        "references": references
    }
```

**关键点**:
- `top_k=2`: 控制检索结果数量
- `rag_prompt`: 包含上下文的完整提示词
- `references`: 用于前端显示参考信息
- Debug输出便于排查提示词问题

#### 3.1.5 性能特征

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| TTFB | 2-5s | 首字节返回时间 |
| 总耗时 | 3-8s | 取决于回答长度 |
| 内存占用 | ~100MB | 主要是模型加载 |
| 并发支持 | 1-5 | 受GPU限制 |

---

### 3.2 POST /api/bot/respond_stream (Streaming模式)

#### 3.2.1 端点定义

```python
@app.post("/api/bot/respond_stream")
async def rag_respond_stream(req: PromptRequest):
    """
    Streaming模式RAG问答端点
    
    特点:
    - 逐块返回文本，实时展示
    - References放在HTTP Header中
    - 适合长回答、交互式场景
    """
```

#### 3.2.2 请求格式

与Blocking模式相同。

#### 3.2.3 响应格式

**Success (200 OK)**:

**Headers**:
```
Content-Type: text/plain
X-References: [{"file":"manual.pdf","page":3,"chunk":1,"text":"..."}]
```

**Body** (Text Stream):
```
ノート
PC
（
パソコン
本体
）
は
粗大
ごみ
として
出す
こと
が
できます
...
```

**注意**:
- Body是流式文本，非JSON
- `X-References`是JSON编码的字符串，需前端解析

#### 3.2.4 实现代码解析

```python
@app.post("/api/bot/respond_stream")
async def rag_respond_stream(req: PromptRequest):
    # 1. RAG检索（与Blocking相同）
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2
    )
    
    # 2. Debug输出
    print("\n===== DEBUG: FULL PROMPT START =====\n")
    print(rag_prompt)
    print("\n===== DEBUG: FULL PROMPT END =====\n")
    
    # 3. 定义流生成器
    def stream_gen():
        collected = ""
        stream = ollama.chat(
            model="swallow:latest",
            messages=[{"role": "user", "content": rag_prompt}],
            stream=True  # 关键: 启用流式
        )
        for event in stream:
            content = event.get("message", {}).get("content", "")
            if content:
                collected += content
                yield content  # 逐块yield
        
        # 流结束后记录日志
        if collected:
            save_log(req.prompt, collected, mode="Streaming(API)")
    
    # 4. 返回StreamingResponse
    return StreamingResponse(
        stream_gen(),
        media_type="text/plain",
        headers={"X-References": json.dumps(references, ensure_ascii=True)}
    )
```

**关键技术点**:

1. **生成器函数**: `def stream_gen()` 使用 `yield` 逐块返回
2. **流式调用**: `ollama.chat(..., stream=True)`
3. **累积文本**: `collected` 用于最终日志记录
4. **Header传递**: `X-References` 通过HTTP头传递（Body已用于流）
5. **ensure_ascii=True**: 确保JSON在HTTP Header中安全传输

#### 3.2.5 性能特征

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| TTFB | 0.5-2s | 首字节返回时间（快） |
| 总耗时 | 3-8s | 与Blocking类似 |
| 用户感知 | 显著更好 | 实时看到生成过程 |
| 并发支持 | 1-5 | 同Blocking |

---

## 4. 数据模型

### 4.1 Pydantic模型定义

**文件**: `backend/schemas.py`

```python
from pydantic import BaseModel

class PromptRequest(BaseModel):
    """
    用户查询请求模型
    """
    prompt: str

class ReplyResponse(BaseModel):
    """
    Blocking模式响应模型
    """
    reply: str
```

### 4.2 模型扩展示例

#### 4.2.1 添加字段验证

```python
from pydantic import BaseModel, Field, validator

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("查询不能为空")
        # 禁止某些关键词
        forbidden = ["prompt", "system", "ignore"]
        if any(word in v.lower() for word in forbidden):
            raise ValueError("包含禁止的关键词")
        return v
```

#### 4.2.2 添加可选参数

```python
class PromptRequest(BaseModel):
    prompt: str
    top_k: int = Field(default=2, ge=1, le=10)  # 1-10之间
    mode: str = Field(default="auto", regex="^(auto|gomi|area|knowledge)$")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
```

#### 4.2.3 扩展响应模型

```python
class Reference(BaseModel):
    file: str
    page: Union[int, str]
    chunk: Union[int, str]
    text: str

class ReplyResponse(BaseModel):
    reply: str
    references: List[Reference] = []
    metadata: dict = {
        "model": "swallow:latest",
        "processing_time": 0.0,
        "tokens_used": 0
    }
```

---

## 5. RAG集成

### 5.1 RAG模块导入

```python
import sys
import os

# 添加rag目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag")))

# 导入RAG核心函数
from rag_demo3 import (
    load_jsonl,
    build_chroma,
    rag_retrieve_extended,
    ask_ollama
)
```

### 5.2 数据加载

```python
# 文件路径定义
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR.parent / "rag"
CHROMA_PATH = BASE_DIR / "chroma_db"

GOMI_JSONL = RAG_DIR / "rag_docs_merged.jsonl"
AREA_JSONL = RAG_DIR / "area.jsonl"

# 加载垃圾分类数据
gomi_docs, gomi_meta = load_jsonl(
    os.path.abspath(GOMI_JSONL),
    key="品名"
)

# 加载町名数据
area_docs, area_meta = load_jsonl(
    os.path.abspath(AREA_JSONL),
    key="町名"
)

# 提取品名列表（用于关键词匹配）
known_items = [m.get("品名", "") for m in gomi_meta]
```

### 5.3 Collection初始化策略

```python
def get_or_build_collection(client, name, docs=None, meta=None):
    """
    获取已存在的collection，如果不存在则构建
    
    优点:
    1. 避免每次启动都重建（耗时）
    2. 持久化数据，重启后保留
    3. 自动容错处理
    """
    try:
        # 尝试获取已存在的collection
        return client.get_collection(name)
    except Exception:
        # 不存在时构建（需要提供docs和meta）
        if docs is None or meta is None:
            raise RuntimeError(
                f"Collection '{name}' not found and no data provided"
            )
        return build_chroma(docs, meta, name=name)
```

**初始化流程**:
```python
# 1. 初始化ChromaDB客户端（持久化）
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# 2. 初始化gomi collection
gomi_collection = get_or_build_collection(
    client=chroma_client,
    name="gomi",
    docs=gomi_docs,
    meta=gomi_meta
)

# 3. 初始化area collection
area_collection = get_or_build_collection(
    client=chroma_client,
    name="area",
    docs=area_docs,
    meta=area_meta
)

# 4. 初始化knowledge collection（可选）
try:
    knowledge_collection = chroma_client.get_collection("knowledge")
except Exception:
    knowledge_collection = None
```

### 5.4 RAG调用封装

```python
def perform_rag(user_input: str) -> tuple[str, list]:
    """
    执行完整的RAG流程
    
    返回:
        reply: 生成的回答
        references: 参考信息列表
    """
    try:
        # 1. 检索和提示词生成
        rag_prompt, references = rag_retrieve_extended(
            user_input,
            gomi_collection,
            knowledge_collection=knowledge_collection,
            area_collection=area_collection,
            known_items=known_items,
            area_meta=area_meta,
            top_k=2
        )
        
        # 2. LLM生成
        reply = ask_ollama(rag_prompt)
        
        return reply, references
    
    except Exception as e:
        # 错误处理
        print(f"RAG执行失败: {e}")
        return "システムエラーが発生しました。", []
```

---

## 6. ChromaDB管理

### 6.1 持久化配置

```python
import chromadb

# 持久化客户端（数据保存到磁盘）
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# 非持久化客户端（仅内存，重启丢失）
# chroma_client = chromadb.Client()
```

**持久化路径**: `backend/chroma_db/`

**数据结构**:
```
chroma_db/
├── chroma.sqlite3           # SQLite数据库（元数据）
├── <collection_id>/         # 每个collection一个目录
│   ├── data_level0.bin      # 向量数据
│   ├── header.bin           # 索引头
│   └── ...
└── ...
```

### 6.2 Collection生命周期

#### 6.2.1 创建Collection

```python
from chromadb.utils import embedding_functions

# 定义Embedding函数
embed = embedding_functions.OllamaEmbeddingFunction(
    model_name="kun432/cl-nagoya-ruri-large:337m"
)

# 创建Collection
collection = client.create_collection(
    name="gomi",
    embedding_function=embed,
    metadata={"description": "垃圾分类规则"}
)
```

#### 6.2.2 获取Collection

```python
# 获取已存在的collection
collection = client.get_collection("gomi")

# 获取所有collections
all_collections = client.list_collections()
print(f"共有 {len(all_collections)} 个collections")
```

#### 6.2.3 删除Collection

```python
# 删除collection（慎用！）
client.delete_collection("gomi")
```

#### 6.2.4 重建Collection

```python
def rebuild_collection(client, name, docs, meta):
    """
    完全重建collection
    """
    try:
        client.delete_collection(name)
        print(f"已删除旧的 {name} collection")
    except:
        pass
    
    return build_chroma(docs, meta, name=name)
```

### 6.3 数据操作

#### 6.3.1 添加数据

```python
collection.add(
    documents=["文本1", "文本2"],
    metadatas=[{"key": "value1"}, {"key": "value2"}],
    ids=["id1", "id2"]
)
```

#### 6.3.2 查询数据

```python
results = collection.query(
    query_texts=["查询文本"],
    n_results=5,
    where={"key": "value"},  # 可选：元数据过滤
    include=["documents", "metadatas", "distances"]
)
```

#### 6.3.3 获取统计信息

```python
# 获取collection中的文档数量
count = collection.count()
print(f"{collection.name}: {count} 件")

# 查看前N条数据
preview = collection.peek(5)
print(preview)
```

### 6.4 错误处理

```python
def safe_get_collection(client, name):
    """
    安全获取collection，带重试和降级
    """
    max_retries = 3
    for i in range(max_retries):
        try:
            return client.get_collection(name)
        except Exception as e:
            if i == max_retries - 1:
                print(f"❌ 无法获取collection {name}: {e}")
                return None
            print(f"⚠️ 重试 {i+1}/{max_retries}...")
            time.sleep(1)
    return None
```

---

## 7. 日志系统

### 7.1 日志格式

**文件**: `backend/logs.jsonl`

**格式**: JSON Lines（每行一个JSON对象）

**字段定义**:
```json
{
  "timestamp": "2026-02-01 10:30:45",
  "mode": "Streaming(API)",
  "user": "ノートPCの捨て方を教えて",
  "assistant": "ノートPC（パソコン本体）は粗大ごみとして...",
  "total_time": 2.341
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| timestamp | string | 时间戳（YYYY-MM-DD HH:MM:SS） |
| mode | string | 模式（"Streaming(API)" / "Blocking(API)"） |
| user | string | 用户输入 |
| assistant | string | 系统回答 |
| total_time | float | 总耗时（秒，可选） |

### 7.2 日志记录实现

```python
import json
import time
from pathlib import Path

LOG_FILE = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "logs.jsonl")))

def save_log(user_input: str, assistant_output: str, mode: str):
    """
    记录一次对话到日志文件
    
    参数:
        user_input: 用户输入
        assistant_output: 系统回答
        mode: 运行模式
    """
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user": user_input,
        "assistant": assistant_output,
    }
    
    # 确保目录存在
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 追加写入
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
```

### 7.3 日志分析

#### 7.3.1 统计查询次数

```python
def count_queries(log_file: Path) -> dict:
    """
    统计各模式的查询次数
    """
    counts = {"Blocking(API)": 0, "Streaming(API)": 0}
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                mode = log.get("mode", "Unknown")
                counts[mode] = counts.get(mode, 0) + 1
            except:
                continue
    
    return counts
```

#### 7.3.2 计算平均响应时间

```python
def average_response_time(log_file: Path) -> float:
    """
    计算平均响应时间
    """
    times = []
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                if "total_time" in log:
                    times.append(log["total_time"])
            except:
                continue
    
    return sum(times) / len(times) if times else 0.0
```

#### 7.3.3 热门查询统计

```python
from collections import Counter

def top_queries(log_file: Path, top_n: int = 10) -> list:
    """
    统计最常见的查询
    """
    queries = []
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                queries.append(log.get("user", ""))
            except:
                continue
    
    return Counter(queries).most_common(top_n)
```

### 7.4 日志轮转

```python
import shutil
from datetime import datetime

def rotate_logs(log_file: Path, max_size_mb: int = 100):
    """
    日志文件大小超过限制时轮转
    """
    if not log_file.exists():
        return
    
    size_mb = log_file.stat().st_size / (1024 * 1024)
    
    if size_mb > max_size_mb:
        # 重命名为带时间戳的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = log_file.with_name(f"logs_{timestamp}.jsonl")
        shutil.move(log_file, backup_file)
        print(f"📦 日志已轮转: {backup_file}")
```

---

## 8. 错误处理

### 8.1 异常类型

| 异常 | 原因 | HTTP状态码 |
|-----|------|-----------|
| ValidationError | 请求数据验证失败 | 422 |
| ValueError | 参数值不合法 | 400 |
| RuntimeError | Collection不存在 | 500 |
| ConnectionError | Ollama连接失败 | 503 |
| TimeoutError | 请求超时 | 504 |

### 8.2 全局异常处理

```python
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    全局异常捕获
    """
    print(f"❌ 全局异常: {type(exc).__name__}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "内部服务器错误",
            "error_type": type(exc).__name__
        }
    )
```

### 8.3 端点级错误处理

```python
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    try:
        # 业务逻辑
        rag_prompt, references = rag_retrieve_extended(...)
        reply = ask_ollama(rag_prompt)
        return {"reply": reply, "references": references}
    
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama服务不可用: {e}"
        )
    
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"请求超时: {e}"
        )
    
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="处理请求时发生错误"
        )
```

### 8.4 降级策略

```python
def rag_respond_with_fallback(req: PromptRequest):
    """
    带降级的RAG响应
    """
    try:
        # 尝试完整RAG
        return perform_rag(req.prompt)
    except Exception as e:
        print(f"⚠️ RAG失败: {e}")
        
        try:
            # 降级1: 跳过检索，直接LLM
            simple_prompt = f"質問: {req.prompt}\n回答:"
            reply = ask_ollama(simple_prompt)
            return {"reply": reply, "references": []}
        except Exception as e2:
            print(f"❌ LLM也失败: {e2}")
            
            # 降级2: 返回预定义消息
            return {
                "reply": "申し訳ございません。システムエラーが発生しました。",
                "references": []
            }
```

---

## 9. 性能优化

### 9.1 启动优化

#### 9.1.1 延迟加载

```python
# 不推荐: 启动时立即加载所有数据
gomi_docs, gomi_meta = load_jsonl(GOMI_JSONL, key="品名")
gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")

# 推荐: 使用持久化 + 懒加载
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
try:
    gomi_collection = chroma_client.get_collection("gomi")
except:
    # 仅在不存在时构建
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
```

**效果**: 启动时间从30s → 2s

#### 9.1.2 预热模型

```python
@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行
    """
    print("🚀 正在启动...")
    
    # 预热Ollama模型
    try:
        ollama.chat(
            model="swallow:latest",
            messages=[{"role": "user", "content": "こんにちは"}]
        )
        print("✅ Ollama模型已预热")
    except Exception as e:
        print(f"⚠️ 模型预热失败: {e}")
```

### 9.2 请求优化

#### 9.2.1 连接池

```python
# ChromaDB默认使用连接池
# 确保使用单例客户端
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# 而非每次请求都创建新客户端
# ❌ client = chromadb.PersistentClient(path=str(CHROMA_PATH))
```

#### 9.2.2 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(user_input: str) -> str:
    """
    缓存检索结果
    """
    rag_prompt, _ = rag_retrieve_extended(
        user_input,
        gomi_collection,
        ...
    )
    return rag_prompt
```

### 9.3 并发优化

```python
# uvicorn启动参数
# --workers: 进程数（建议 = CPU核心数）
# --limit-concurrency: 并发连接数
uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --limit-concurrency 100
```

---

## 10. 安全性

### 10.1 输入验证

```python
from pydantic import BaseModel, validator

class PromptRequest(BaseModel):
    prompt: str
    
    @validator('prompt')
    def validate_prompt(cls, v):
        # 长度限制
        if len(v) > 1000:
            raise ValueError("查询过长（最多1000字符）")
        
        # 非空验证
        if not v.strip():
            raise ValueError("查询不能为空")
        
        # 黑名单关键词
        forbidden = ["prompt", "system", "ignore", "<script>"]
        if any(word in v.lower() for word in forbidden):
            raise ValueError("包含禁止的内容")
        
        return v
```

### 10.2 CORS配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit地址
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

### 10.3 Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/api/bot/respond")
@limiter.limit("10/minute")  # 每分钟10次
async def rag_respond(request: Request, req: PromptRequest):
    ...
```

---

## 11. 部署配置

### 11.1 环境变量

```bash
# .env文件
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_DB_PATH=./chroma_db
LOG_LEVEL=INFO
MAX_WORKERS=4
```

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    chroma_db_path: str = "./chroma_db"
    log_level: str = "INFO"
    max_workers: int = 4
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 11.2 Docker化

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装MeCab
RUN apt-get update && apt-get install -y mecab mecab-ipadic-utf8

# 安装Python依赖
COPY pyproject.toml .
RUN pip install uv && uv sync

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.3 Systemd服务

```ini
# /etc/systemd/system/kita-backend.service
[Unit]
Description=Kita Backend API
After=network.target

[Service]
Type=simple
User=kita
WorkingDirectory=/opt/kita
Environment="PATH=/opt/kita/.venv/bin"
ExecStart=/opt/kita/.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 12. 测试策略

### 12.1 单元测试

```python
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_blocking_api():
    response = client.post(
        "/api/bot/respond",
        json={"prompt": "ノートPCの捨て方"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "references" in data

def test_invalid_input():
    response = client.post(
        "/api/bot/respond",
        json={"prompt": ""}
    )
    assert response.status_code == 422
```

### 12.2 集成测试

```python
def test_end_to_end():
    # 测试完整流程
    response = client.post(
        "/api/bot/respond",
        json={"prompt": "八幡東区でノートPCを捨てたい"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 验证回答包含关键信息
    assert "ノートPC" in data["reply"] or "パソコン" in data["reply"]
    assert "粗大ごみ" in data["reply"]
```

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队
