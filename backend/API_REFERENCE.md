# 后端API快速参考手册

## 文档概述
本文档提供Kita后端API的快速参考，包括端点说明、请求/响应格式、错误码、配置参数和常用代码示例。

---

## 目录
1. [API概览](#1-api概览)
2. [端点详细参考](#2-端点详细参考)
3. [数据模型](#3-数据模型)
4. [错误码](#4-错误码)
5. [配置参数](#5-配置参数)
6. [代码示例](#6-代码示例)
7. [性能指标](#7-性能指标)
8. [常见问题](#8-常见问题)

---

## 1. API概览

### 1.1 基本信息

| 项目 | 值 |
|-----|---|
| Base URL | `http://localhost:8000` |
| 协议 | HTTP/1.1 |
| 数据格式 | JSON (Blocking) / Text Stream (Streaming) |
| 字符编码 | UTF-8 |
| 认证 | 无（可选添加） |

### 1.2 端点列表

| 端点 | 方法 | 描述 | 响应模式 |
|-----|------|------|---------|
| `/api/bot/respond` | POST | Blocking模式问答 | JSON |
| `/api/bot/respond_stream` | POST | Streaming模式问答 | Text Stream |
| `/docs` | GET | 交互式API文档（Swagger UI） | HTML |
| `/redoc` | GET | API文档（ReDoc） | HTML |

### 1.3 快速测试

```bash
# 测试Blocking API
curl -X POST http://localhost:8000/api/bot/respond \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}'

# 测试Streaming API
curl -X POST http://localhost:8000/api/bot/respond_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}' \
  -N
```

---

## 2. 端点详细参考

### 2.1 POST /api/bot/respond

#### 基本信息
- **URL**: `/api/bot/respond`
- **方法**: `POST`
- **Content-Type**: `application/json`
- **响应格式**: JSON

#### 请求

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "prompt": "用户查询文本"
}
```

**字段说明**:
| 字段 | 类型 | 必填 | 长度限制 | 说明 |
|-----|------|------|---------|------|
| prompt | string | 是 | 1-1000 | 用户查询文本 |

**请求示例**:
```json
{
  "prompt": "ノートPCの捨て方を教えて"
}
```

#### 响应

**成功响应 (200 OK)**:
```json
{
  "reply": "string",
  "references": [
    {
      "file": "string",
      "page": "number|string",
      "chunk": "number|string",
      "text": "string"
    }
  ]
}
```

**字段说明**:
| 字段 | 类型 | 说明 |
|-----|------|------|
| reply | string | LLM生成的完整回答 |
| references | array | 参考信息数组 |
| references[].file | string | 来源文件名 |
| references[].page | number\|string | 页码（PDF）或"?" |
| references[].chunk | number\|string | 片段编号 |
| references[].text | string | 片段文本（最多300字符） |

**响应示例**:
```json
{
  "reply": "ノートPC（パソコン本体）は粗大ごみとして出すことができます。小型のものは小型電子機器回収ボックスへ入れることもできます。",
  "references": [
    {
      "file": "manual.pdf",
      "page": 3,
      "chunk": 1,
      "text": "パソコン本体（デスクトップ型・ノート型）は粗大ごみとして出してください。小型のものは小型電子機器回収ボックスへ入れることもできます。"
    }
  ]
}
```

#### 性能指标

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| TTFB | 2-5秒 | 首字节返回时间 |
| 响应时间 | 3-8秒 | 完整响应时间 |
| 并发支持 | 1-5个 | 取决于GPU资源 |

---

### 2.2 POST /api/bot/respond_stream

#### 基本信息
- **URL**: `/api/bot/respond_stream`
- **方法**: `POST`
- **Content-Type**: `application/json`
- **响应格式**: Text Stream

#### 请求

**格式与 `/api/bot/respond` 相同**

#### 响应

**成功响应 (200 OK)**:

**Headers**:
```
Content-Type: text/plain; charset=utf-8
X-References: [{"file":"manual.pdf","page":3,"chunk":1,"text":"..."}]
```

**Body** (流式文本):
```
ノートPC（パソコン本体）は粗大ごみとして出すことができます。小型のものは小型電子機器回収ボックスへ入れることもできます。
```

**注意事项**:
1. Body是**纯文本流**，不是JSON
2. References放在HTTP Header `X-References` 中
3. `X-References` 是JSON编码的字符串，需要解析
4. 文本逐块返回，适合实时展示

#### X-References格式

```json
[
  {
    "file": "manual.pdf",
    "page": 3,
    "chunk": 1,
    "text": "パソコン本体（デスクトップ型・ノート型）は..."
  }
]
```

#### 性能指标

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| TTFB | 0.5-2秒 | 首字节返回时间 |
| 响应时间 | 3-8秒 | 完整流传输时间 |
| 用户体验 | ⭐⭐⭐⭐⭐ | 实时看到生成 |

---

## 3. 数据模型

### 3.1 PromptRequest

**用途**: 用户查询请求

**Schema**:
```json
{
  "prompt": "string"
}
```

**Pydantic定义**:
```python
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
```

**验证规则** (可扩展):
- `prompt`: 非空字符串
- 长度: 1-1000字符（推荐）
- 禁止包含: `<script>`, `prompt`, `system`等

---

### 3.2 ReplyResponse

**用途**: Blocking模式响应

**Schema**:
```json
{
  "reply": "string",
  "references": [
    {
      "file": "string",
      "page": "number|string",
      "chunk": "number|string",
      "text": "string"
    }
  ]
}
```

**Pydantic定义**:
```python
class ReplyResponse(BaseModel):
    reply: str
```

**注意**: `references` 在当前实现中直接返回，未定义为Pydantic模型。

---

### 3.3 Reference (非正式)

**用途**: 参考信息对象

**Schema**:
```json
{
  "file": "string",
  "page": "number|string",
  "chunk": "number|string",
  "text": "string"
}
```

**字段说明**:
- `file`: 文件名（如 `"manual.pdf"`）
- `page`: PDF页码（数字）或 `"?"` (非PDF)
- `chunk`: 片段编号（数字）或 `"?"`
- `text`: 片段文本，最多300字符

---

## 4. 错误码

### 4.1 HTTP状态码

| 状态码 | 含义 | 说明 |
|-------|------|------|
| 200 | OK | 请求成功 |
| 400 | Bad Request | 请求参数错误 |
| 422 | Unprocessable Entity | 数据验证失败（Pydantic） |
| 500 | Internal Server Error | 服务器内部错误 |
| 503 | Service Unavailable | Ollama服务不可用 |
| 504 | Gateway Timeout | 请求超时 |

### 4.2 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

**示例**:
```json
{
  "detail": "查询不能为空"
}
```

### 4.3 常见错误

#### 4.3.1 验证错误 (422)

**原因**: Prompt为空或格式错误

**请求**:
```json
{
  "prompt": ""
}
```

**响应**:
```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 4.3.2 服务不可用 (503)

**原因**: Ollama未启动或连接失败

**响应**:
```json
{
  "detail": "Ollama服务不可用: Connection refused"
}
```

**解决方案**:
```bash
# 检查Ollama是否运行
ps aux | grep ollama

# 启动Ollama
nohup ollama serve > ollama.log 2>&1 &
```

#### 4.3.3 超时 (504)

**原因**: LLM生成时间过长

**响应**:
```json
{
  "detail": "请求超时"
}
```

**解决方案**:
- 使用量化模型（更快）
- 减少上下文长度
- 增加timeout配置

---

## 5. 配置参数

### 5.1 环境变量

| 变量名 | 默认值 | 说明 |
|-------|-------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama服务地址 |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB持久化路径 |
| `LOG_FILE` | `./logs.jsonl` | 日志文件路径 |

**设置方式**:
```bash
export OLLAMA_BASE_URL=http://192.168.1.100:11434
```

### 5.2 Uvicorn启动参数

```bash
uvicorn backend.app:app \
  --host 0.0.0.0 \        # 监听地址
  --port 8000 \           # 端口
  --reload \              # 热重载（开发模式）
  --workers 4 \           # 进程数（生产模式）
  --limit-concurrency 100 # 最大并发连接
```

**参数说明**:
- `--reload`: 代码修改后自动重启（**仅开发**）
- `--workers`: 多进程，提高并发（**生产推荐**）
- `--limit-concurrency`: 限制并发，防止过载

### 5.3 RAG参数

**文件**: `backend/app.py`

```python
rag_prompt, references = rag_retrieve_extended(
    req.prompt,
    gomi_collection,
    knowledge_collection=knowledge_collection,
    area_collection=area_collection,
    known_items=known_items,
    area_meta=area_meta,
    top_k=2  # ← 修改这里
)
```

**可调参数**:
| 参数 | 默认值 | 说明 | 推荐范围 |
|-----|-------|------|---------|
| `top_k` | 2 | 检索返回的结果数 | 1-5 |

---

## 6. 代码示例

### 6.1 Python客户端

#### 6.1.1 Blocking模式

```python
import requests

def query_blocking(prompt: str) -> dict:
    """
    Blocking模式查询
    """
    url = "http://localhost:8000/api/bot/respond"
    response = requests.post(url, json={"prompt": prompt}, timeout=30)
    response.raise_for_status()
    return response.json()

# 使用
result = query_blocking("ノートPCの捨て方")
print(result["reply"])

for ref in result["references"]:
    print(f"参考: {ref['file']} p.{ref['page']}")
```

#### 6.1.2 Streaming模式

```python
import requests
import json

def query_streaming(prompt: str):
    """
    Streaming模式查询
    """
    url = "http://localhost:8000/api/bot/respond_stream"
    
    with requests.post(
        url,
        json={"prompt": prompt},
        stream=True,
        timeout=60
    ) as response:
        response.raise_for_status()
        
        # 逐块打印
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        
        # 获取参考信息
        refs_json = response.headers.get("X-References", "[]")
        references = json.loads(refs_json)
        
        print(f"\n\n参考数: {len(references)}")

# 使用
query_streaming("ノートPCの捨て方")
```

---

### 6.2 JavaScript客户端

#### 6.2.1 Blocking模式 (Fetch API)

```javascript
async function queryBlocking(prompt) {
  const response = await fetch('http://localhost:8000/api/bot/respond', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ prompt })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const data = await response.json();
  console.log('回答:', data.reply);
  console.log('参考:', data.references);
  return data;
}

// 使用
queryBlocking('ノートPCの捨て方');
```

#### 6.2.2 Streaming模式

```javascript
async function queryStreaming(prompt) {
  const response = await fetch('http://localhost:8000/api/bot/respond_stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ prompt })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  
  // 获取References
  const refsHeader = response.headers.get('X-References');
  const references = JSON.parse(refsHeader || '[]');
  
  // 读取流
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value, { stream: true });
    console.log(text); // 实时打印
  }
  
  console.log('参考:', references);
}

// 使用
queryStreaming('ノートPCの捨て方');
```

---

### 6.3 cURL命令

#### 6.3.1 Blocking模式

```bash
curl -X POST http://localhost:8000/api/bot/respond \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ノートPCの捨て方を教えて"
  }' | jq .
```

**带管道美化输出**:
```bash
curl -X POST http://localhost:8000/api/bot/respond \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}' \
  | jq -r '.reply'
```

#### 6.3.2 Streaming模式

```bash
curl -N -X POST http://localhost:8000/api/bot/respond_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}'
```

**提取References**:
```bash
curl -i -X POST http://localhost:8000/api/bot/respond_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}' \
  | grep "X-References"
```

---

### 6.4 性能测试

#### 6.4.1 单次请求测试

```python
import time
import requests

def benchmark_single():
    start = time.perf_counter()
    
    response = requests.post(
        "http://localhost:8000/api/bot/respond",
        json={"prompt": "ノートPCの捨て方"}
    )
    
    elapsed = time.perf_counter() - start
    print(f"响应时间: {elapsed:.2f}s")
    print(f"状态码: {response.status_code}")

benchmark_single()
```

#### 6.4.2 批量测试

```python
import time
import requests

queries = [
    "ノートPCの捨て方",
    "八幡東区の収集日",
    "プラスチックの分別方法"
]

for query in queries:
    start = time.perf_counter()
    
    response = requests.post(
        "http://localhost:8000/api/bot/respond",
        json={"prompt": query},
        timeout=30
    )
    
    elapsed = time.perf_counter() - start
    
    if response.ok:
        print(f"✓ {query}: {elapsed:.2f}s")
    else:
        print(f"✗ {query}: {response.status_code}")
```

#### 6.4.3 并发测试

```bash
# 使用Apache Bench
ab -n 10 -c 2 -p request.json -T application/json \
   http://localhost:8000/api/bot/respond

# request.json:
# {"prompt": "ノートPCの捨て方"}
```

**结果解读**:
```
Requests per second: 0.50 [#/sec]  # QPS
Time per request: 2000 [ms]        # 平均响应时间
```

---

## 7. 性能指标

### 7.1 响应时间基准

| 场景 | Blocking | Streaming |
|-----|----------|-----------|
| 简单查询（<10字） | 2-3s | TTFB: 0.5-1s |
| 中等查询（10-30字） | 3-5s | TTFB: 1-2s |
| 复杂查询（>30字） | 5-8s | TTFB: 1.5-2.5s |

### 7.2 吞吐量

| 配置 | QPS | 说明 |
|-----|-----|------|
| 单进程 | 0.2-0.5 | 受GPU限制 |
| 4进程 | 0.5-1.0 | 多进程并发 |
| 量化模型 | 0.5-0.8 | 速度提升30% |

### 7.3 资源占用

| 资源 | 占用 | 说明 |
|-----|------|------|
| CPU | 10-30% | 主要在检索阶段 |
| 内存 | 500MB-1GB | 包括模型加载 |
| GPU显存 | 4-8GB | 取决于模型大小 |
| 磁盘I/O | 低 | ChromaDB已缓存 |

---

## 8. 常见问题

### 8.1 API无法访问

**症状**: `Connection refused`

**原因**: 后端未启动

**解决**:
```bash
# 检查进程
ps aux | grep uvicorn

# 启动后端
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

### 8.2 响应很慢

**症状**: 响应时间>10s

**原因**: 
1. Ollama模型未预热
2. GPU性能不足
3. top_k设置过大

**解决**:
```python
# 1. 使用量化模型
ollama pull swallow:latest-q4_k_m

# 2. 减少top_k
top_k=1  # 在backend/app.py中修改

# 3. 限制max_tokens
ollama.chat(..., options={"num_predict": 300})
```

---

### 8.3 Streaming无输出

**症状**: 请求成功但无文本流

**原因**: 
1. Ollama不支持该模型的流式输出
2. 网络缓冲问题

**解决**:
```python
# Python客户端添加
response = requests.post(..., stream=True)
for chunk in response.iter_content(chunk_size=1):  # 减小chunk_size
    ...

# cURL添加-N参数
curl -N -X POST ...
```

---

### 8.4 References为空

**症状**: 返回的`references`数组为空

**原因**: 
1. knowledge collection未构建
2. 查询未命中任何文档

**解决**:
```bash
# 1. 检查knowledge collection
curl http://localhost:8000/debug/collections  # 需自行添加端点

# 2. 查看日志
tail -f backend/logs.jsonl

# 3. 手动查询ChromaDB
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_collection('knowledge')
print(f'知识库文档数: {col.count()}')
"
```

---

### 8.5 如何添加认证

**需求**: API需要Token认证

**实现**:
```python
from fastapi import Header, HTTPException

async def verify_token(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/api/bot/respond", dependencies=[Depends(verify_token)])
async def rag_respond(req: PromptRequest):
    ...
```

**使用**:
```bash
curl -X POST http://localhost:8000/api/bot/respond \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"prompt": "ノートPCの捨て方"}'
```

---

### 8.6 如何启用HTTPS

**需求**: 生产环境需要HTTPS

**方案1: Nginx反向代理**
```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**方案2: Uvicorn直接支持**
```bash
uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

---

## 9. 调试技巧

### 9.1 查看实时日志

```bash
# 后端日志
tail -f backend/logs.jsonl | jq .

# Uvicorn日志（终端输出）
# 包含请求信息、错误堆栈
```

### 9.2 启用Debug模式

```python
# backend/app.py
import logging

logging.basicConfig(level=logging.DEBUG)

@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    logging.debug(f"收到请求: {req.prompt}")
    # ...
    logging.debug(f"RAG prompt长度: {len(rag_prompt)}")
    # ...
```

### 9.3 使用FastAPI自动文档

访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**功能**:
- 交互式测试API
- 查看请求/响应Schema
- 自动生成cURL命令

---

## 10. 进阶用法

### 10.1 自定义响应格式

```python
from fastapi.responses import JSONResponse

@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    reply, references = perform_rag(req.prompt)
    
    # 自定义响应头
    return JSONResponse(
        content={"reply": reply, "references": references},
        headers={"X-Processing-Time": "2.5s"}
    )
```

### 10.2 添加中间件

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 10.3 WebSocket支持（未来）

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        reply, refs = perform_rag(data)
        await websocket.send_text(reply)
```

---

## 附录

### A. 完整请求/响应示例

#### Blocking模式

**请求**:
```http
POST /api/bot/respond HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "prompt": "八幡東区でノートPCを捨てたい"
}
```

**响应**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "reply": "ノートPC（パソコン本体）は粗大ごみとして出すことができます。八幡東区の家庭ごみ収集日は火曜日・金曜日です。",
  "references": [
    {
      "file": "rag_docs_merged.jsonl",
      "page": "?",
      "chunk": "?",
      "text": "品名: パソコン本体（ノート型）\n出し方: 粗大ごみ\n備考: 小型のものは小型電子機器回収ボックスへ"
    }
  ]
}
```

#### Streaming模式

**请求**: 同上

**响应**:
```http
HTTP/1.1 200 OK
Content-Type: text/plain; charset=utf-8
X-References: [{"file":"rag_docs_merged.jsonl","page":"?","chunk":"?","text":"..."}]

ノートPC（パソコン本体）は粗大ごみとして出すことができます。八幡東区の家庭ごみ収集日は火曜日・金曜日です。
```

---

### B. 性能调优检查清单

- [ ] 使用量化模型（q4_k_m）
- [ ] 调整top_k=1或2
- [ ] 启用ChromaDB HNSW索引
- [ ] 增加Uvicorn workers数量
- [ ] 限制max_tokens<500
- [ ] 使用连接池（默认已启用）
- [ ] 添加请求缓存（可选）
- [ ] 监控GPU使用率

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队

**相关文档**:
- 详细架构: `BACKEND_ARCHITECTURE.md`
- RAG系统: `../rag/RAG_DOCUMENTATION_INDEX.md`
- 前端文档: `../front-streaming/FRONTEND_GUIDE.md`（待创建）
