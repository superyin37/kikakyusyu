# RAG系统快速参考手册

## 文档概述
本文档提供Kita RAG系统的快速参考，包括常用命令、API参考、配置选项和故障排查。适合日常开发和运维使用。

---

## 目录
1. [快速启动](#1-快速启动)
2. [核心API参考](#2-核心api参考)
3. [配置参数](#3-配置参数)
4. [常用命令](#4-常用命令)
5. [故障排查速查表](#5-故障排查速查表)
6. [性能调优速查](#6-性能调优速查)

---

## 1. 快速启动

### 1.1 环境准备

```bash
# 安装uv包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
cd /path/to/project

# 创建虚拟环境
uv venv

# 激活虚拟环境（Linux/Mac）
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 安装依赖
uv sync
```

### 1.2 Ollama配置

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
nohup ollama serve > ollama.log 2>&1 &

# 拉取所需模型
ollama pull swallow:latest
ollama pull kun432/cl-nagoya-ruri-large:337m

# 验证模型
ollama list
```

### 1.3 启动服务

```bash
# 终端1: 启动后端API
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 终端2: 启动前端WebUI
streamlit run front-streaming/app.py

# 访问
# WebUI: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

---

## 2. 核心API参考

### 2.1 RAG核心函数

#### extract_keywords()

**功能**: 从用户输入中抽取品名和町名

**签名**:
```python
def extract_keywords(user_input, known_items=ITEMS, known_areas=AREAS) -> dict
```

**参数**:
- `user_input` (str): 用户输入文本
- `known_items` (list): 已知品名列表（默认866个）
- `known_areas` (list): 已知町名列表（默认825个）

**返回**:
```python
{
    "品名": "ノートPC" or None,
    "町名": "八幡東区" or None
}
```

**示例**:
```python
result = extract_keywords("八幡東区でノートPCを捨てたい")
# {"品名": "ノートPC", "町名": "八幡東区"}
```

---

#### query_chroma()

**功能**: 在ChromaDB collection中进行语义检索

**签名**:
```python
def query_chroma(collection, query: str, n: int = 3) -> list[dict]
```

**参数**:
- `collection`: ChromaDB collection对象
- `query` (str): 查询文本
- `n` (int): 返回结果数量

**返回**:
```python
[
    {
        "品名": "パソコン本体（ノート型）",
        "出し方": "粗大ごみ",
        "備考": "小型のものは小型電子機器回収ボックスへ",
        "text": "パソコン本体（ノート型）"
    },
    ...
]
```

**示例**:
```python
results = query_chroma(gomi_collection, "ノートPC", n=2)
for r in results:
    print(f"{r['品名']}: {r['出し方']}")
```

---

#### rag_retrieve_extended()

**功能**: 完整的RAG检索和提示词生成

**签名**:
```python
def rag_retrieve_extended(
    user_input: str,
    gomi_collection,
    area_collection,
    known_items: list,
    area_meta: list,
    knowledge_collection=None,
    known_areas: list = AREAS,
    top_k: int = 3
) -> tuple[str, list]
```

**返回**:
```python
(
    prompt,      # 完整的RAG提示词
    references   # 参考信息列表
)
```

**示例**:
```python
prompt, refs = rag_retrieve_extended(
    "ノートPCの捨て方",
    gomi_collection,
    area_collection,
    known_items,
    area_meta,
    top_k=2
)
print(prompt)
print(f"参考数: {len(refs)}")
```

---

#### ask_ollama()

**功能**: 调用Ollama LLM生成回答

**签名**:
```python
def ask_ollama(rag_prompt: str, model: str = "swallow:latest") -> str
```

**参数**:
- `rag_prompt` (str): RAG提示词
- `model` (str): 使用的模型名称

**返回**: 生成的回答文本

**示例**:
```python
response = ask_ollama(rag_prompt)
print(response)
```

---

### 2.2 知识库管理函数

#### add_file_to_chroma()

**功能**: 将文件添加到知识库

**签名**:
```python
def add_file_to_chroma(
    file_path: Path,
    persist_dir: str = "./chroma_db",
    collection_name: str = "knowledge"
) -> Optional[Collection]
```

**支持格式**: PDF, TXT, CSV, JSON

**示例**:
```python
from pathlib import Path
file_path = Path("knowledge_files/manual.pdf")
collection = add_file_to_chroma(file_path)
print(f"已添加 {collection.count()} 个chunks")
```

---

#### chunk_pdf()

**功能**: PDF文件分块

**签名**:
```python
def chunk_pdf(file_path: Path, chunk_size: int = 500) -> list[dict]
```

**返回**:
```python
[
    {
        "text": "...",
        "metadata": {
            "file": "manual.pdf",
            "page": 1,
            "chunk": 1
        }
    },
    ...
]
```

---

### 2.3 FastAPI端点

#### POST /api/bot/respond (Blocking模式)

**请求**:
```json
{
  "prompt": "ノートPCの捨て方を教えて"
}
```

**响应**:
```json
{
  "reply": "ノートPC（パソコン本体）は粗大ごみとして...",
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

**Python示例**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/bot/respond",
    json={"prompt": "ノートPCの捨て方"}
)
data = response.json()
print(data["reply"])
```

**cURL示例**:
```bash
curl -X POST http://localhost:8000/api/bot/respond \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ノートPCの捨て方"}'
```

---

#### POST /api/bot/respond_stream (Streaming模式)

**请求**: 同上

**响应**: 
- Body: 流式文本
- Header: `X-References` (JSON编码的参考信息)

**Python示例**:
```python
import requests
import json

with requests.post(
    "http://localhost:8000/api/bot/respond_stream",
    json={"prompt": "ノートPCの捨て方"},
    stream=True
) as response:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode(), end='', flush=True)
    
    # 获取参考信息
    refs = json.loads(response.headers.get("X-References", "[]"))
    print(f"\n\n参考: {len(refs)}件")
```

---

## 3. 配置参数

### 3.1 RAG参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|-----|-------|------|---------|
| `top_k` | 2 | 检索返回的结果数 | 1-5 |
| `chunk_size` | 500 | 文本分块大小（字符） | 300-800 |
| `chunk_overlap` | 50 | 块之间的重叠（字符） | 20-100 |

**修改方式**:
```python
# rag/rag_demo3.py
prompt, refs = rag_retrieve_extended(
    user_input,
    ...,
    top_k=3  # 修改这里
)
```

---

### 3.2 ChromaDB参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `persist_dir` | `./chroma_db` | 数据库持久化路径 |
| `hnsw:M` | 16 | HNSW图连接数 |
| `hnsw:ef_construction` | 200 | 构建时搜索深度 |

**修改方式**:
```python
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(
    name="knowledge",
    metadata={
        "hnsw:M": 32,  # 增加精度
        "hnsw:ef_construction": 400
    }
)
```

---

### 3.3 Ollama参数

**环境变量**:
```bash
# GPU内存限制（80%）
export OLLAMA_MAX_VRAM=0.8

# 启用Flash Attention
export OLLAMA_FLASH_ATTENTION=1

# 并发数
export OLLAMA_NUM_PARALLEL=1

# 超时时间（秒）
export OLLAMA_REQUEST_TIMEOUT=60
```

**模型参数**（在ask_ollama中）:
```python
response = ollama.chat(
    model="swallow:latest",
    messages=[...],
    options={
        "temperature": 0.3,      # 随机性（0=确定，1=随机）
        "top_p": 0.9,           # 核采样
        "num_predict": 500,     # 最大生成token数
        "stop": ["</s>", "\n\n\n"]  # 停止词
    }
)
```

---

### 3.4 MeCab配置

**字典路径**:
```python
dic_dir = "/var/lib/mecab/dic/debian"  # Debian/Ubuntu
# dic_dir = "/usr/local/lib/mecab/dic/ipadic"  # CentOS/RedHat
# dic_dir = "C:\\Program Files\\MeCab\\dic\\ipadic"  # Windows
```

**配置文件**:
```python
tagger = MeCab.Tagger(f"-Ochasen -r /etc/mecabrc -d {dic_dir}")
```

---

## 4. 常用命令

### 4.1 数据库管理

```bash
# 查看ChromaDB大小
du -sh chroma_db/

# 备份ChromaDB
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/

# 恢复备份
tar -xzf chroma_db_backup_20260201.tar.gz

# 清空知识库（保留gomi和area）
rm -rf chroma_db/*knowledge*

# 完全重建数据库
rm -rf chroma_db/
# 然后重启后端，自动重建
```

---

### 4.2 模型管理

```bash
# 查看已安装模型
ollama list

# 拉取新模型
ollama pull swallow:latest

# 删除模型
ollama rm swallow:latest

# 查看模型详情
ollama show swallow:latest

# 测试模型
echo "こんにちは" | ollama run swallow:latest
```

---

### 4.3 日志查看

```bash
# 查看后端日志
tail -f backend/logs.jsonl

# 统计查询次数
cat backend/logs.jsonl | wc -l

# 查看最近10条查询
tail -n 10 backend/logs.jsonl | jq .

# 统计Blocking vs Streaming使用率
cat backend/logs.jsonl | jq -r '.mode' | sort | uniq -c

# 查看平均响应时间
cat backend/logs.jsonl | jq -r '.total_time' | awk '{sum+=$1; count++} END {print sum/count}'
```

---

### 4.4 性能监控

```bash
# 查看GPU使用率
watch -n 1 nvidia-smi

# 查看内存使用
free -h

# 查看进程资源占用
top -p $(pgrep -f "uvicorn|streamlit|ollama")

# 查看端口占用
netstat -tuln | grep -E "8000|8501|11434"
```

---

## 5. 故障排查速查表

| 错误信息 | 可能原因 | 快速解决方案 |
|---------|---------|------------|
| `MeCab.Tagger: cannot open dictionary` | 字典路径错误 | `sudo apt-get install mecab mecab-ipadic-utf8` |
| `Connection refused [localhost:11434]` | Ollama未启动 | `nohup ollama serve > ollama.log 2>&1 &` |
| `sqlite3.OperationalError: database is locked` | 多进程访问冲突 | 重启后端，删除`chroma_db/*.lock` |
| `chromadb.errors.NoIndexException` | Collection不存在 | 删除`chroma_db/`并重启后端 |
| `TimeoutError: Request timeout` | LLM响应慢 | 增加timeout，检查GPU，使用量化模型 |
| `KeyError: 'message'` | Ollama返回格式错误 | 检查模型是否正确拉取 |
| `streamlit: command not found` | 虚拟环境未激活 | `source .venv/bin/activate` |
| `Port 8000 already in use` | 端口被占用 | `lsof -ti:8000 | xargs kill -9` |

---

## 6. 性能调优速查

### 6.1 常见性能问题

| 问题 | 症状 | 解决方案 |
|-----|------|---------|
| 首次响应慢 | TTFB > 5s | 预加载模型，使用keep-alive |
| 检索慢 | 检索耗时 > 500ms | 启用HNSW索引，减少top_k |
| 生成慢 | 响应慢但TTFB快 | 使用量化模型，减少max_tokens |
| GPU OOM | Out of memory错误 | 减少batch size，限制VRAM |
| 高并发慢 | 多用户同时查询 | 增加Ollama并发数，负载均衡 |

---

### 6.2 快速优化清单

**Level 1: 无需修改代码**
```bash
# 1. 使用量化模型
ollama pull swallow:latest-q4_k_m

# 2. 设置GPU内存限制
export OLLAMA_MAX_VRAM=0.8

# 3. 增加系统swap（如果RAM不足）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Level 2: 配置调整**
```python
# rag/rag_demo3.py
top_k = 2  # 减少到1可提速30%

# backend/app.py - Ollama调用
ollama.chat(
    model="swallow:latest-q4_k_m",  # 使用量化模型
    options={
        "num_predict": 300,  # 限制生成长度
        "temperature": 0.1   # 降低随机性，提速
    }
)
```

**Level 3: 架构优化**
- 实现Embedding缓存
- 使用异步I/O
- 添加Redis缓存层
- 多GPU负载均衡

---

### 6.3 性能基准参考

**理想性能指标**:
```
关键词抽取: < 50ms
向量检索:   < 200ms
提示词构建: < 10ms
LLM生成:    < 3s (Blocking) / TTFB < 1s (Streaming)
总耗时:     < 5s
```

**测试命令**:
```bash
# 使用Apache Bench测试API
ab -n 10 -c 1 -p request.json -T application/json \
   http://localhost:8000/api/bot/respond

# request.json内容:
# {"prompt": "ノートPCの捨て方"}
```

---

## 7. 开发工作流

### 7.1 添加新品名

1. 编辑 `rag/rag_docs_merged.jsonl`
2. 添加新行:
   ```json
   {"品名": "新品名", "出し方": "分別方法", "備考": "注意事項"}
   ```
3. 重启后端: `Ctrl+C` 然后重新运行 `uvicorn ...`
4. 验证: 在WebUI中查询新品名

---

### 7.2 添加新町名

1. 编辑 `rag/area.jsonl`
2. 添加新行:
   ```json
   {"町名": "新町名", "家庭ごみの収集日": "...", ...}
   ```
3. 重启后端
4. 验证

---

### 7.3 更新Embedding模型

1. 拉取新模型:
   ```bash
   ollama pull <new_embedding_model>
   ```

2. 修改代码:
   ```python
   # rag/rag_demo3.py 和 rag/user_knowledge.py
   embed = embedding_functions.OllamaEmbeddingFunction(
       model_name="<new_embedding_model>"
   )
   ```

3. **重要**: 删除旧数据库
   ```bash
   rm -rf chroma_db/
   ```

4. 重启后端（自动重建）

---

### 7.4 更新LLM模型

1. 拉取新模型:
   ```bash
   ollama pull <new_llm_model>
   ```

2. 修改代码:
   ```python
   # backend/app.py 和 rag/rag_demo3.py
   model = "<new_llm_model>"
   ```

3. 重启后端
4. 测试并调整提示词（不同模型可能需要不同格式）

---

## 8. 常用代码片段

### 8.1 手动查询ChromaDB

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 查看所有collections
print(client.list_collections())

# 查询gomi collection
gomi = client.get_collection("gomi")
print(f"gomi: {gomi.count()} 件")

# 查看前5条
print(gomi.peek(5))

# 搜索
results = gomi.query(query_texts=["ノートPC"], n_results=3)
print(results)
```

---

### 8.2 批量添加知识文件

```python
from pathlib import Path
from rag.user_knowledge import add_file_to_chroma

knowledge_dir = Path("knowledge_files")

for file in knowledge_dir.glob("*"):
    if file.suffix in [".pdf", ".txt", ".csv", ".json"]:
        print(f"处理: {file.name}")
        add_file_to_chroma(file)
```

---

### 8.3 性能测试脚本

```python
import time
import requests

queries = [
    "ノートPCの捨て方",
    "八幡東区の収集日",
    "プラスチックの分別"
]

for query in queries:
    start = time.perf_counter()
    
    response = requests.post(
        "http://localhost:8000/api/bot/respond",
        json={"prompt": query}
    )
    
    elapsed = time.perf_counter() - start
    print(f"{query}: {elapsed:.2f}s")
```

---

## 9. 安全注意事项

### 9.1 生产环境建议

- [ ] 使用HTTPS（通过Nginx反向代理）
- [ ] 添加API认证（Bearer Token）
- [ ] 限制请求频率（Rate Limiting）
- [ ] 输入长度验证（<1000字符）
- [ ] 日志脱敏（移除个人信息）
- [ ] 定期备份数据库
- [ ] 监控异常请求

---

### 9.2 输入验证示例

```python
from pydantic import BaseModel, validator

class PromptRequest(BaseModel):
    prompt: str
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 1000:
            raise ValueError("查询过长")
        if not v.strip():
            raise ValueError("查询为空")
        return v
```

---

## 10. 资源链接

### 官方文档
- [ChromaDB文档](https://docs.trychroma.com/)
- [Ollama文档](https://github.com/ollama/ollama)
- [MeCab官网](https://taku910.github.io/mecab/)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [Streamlit文档](https://docs.streamlit.io/)

### 项目仓库
- [LangChain](https://github.com/langchain-ai/langchain)
- [Swallow Model](https://huggingface.co/tokyotech-llm)

### 社区
- [Stack Overflow - RAG Tag](https://stackoverflow.com/questions/tagged/retrieval-augmented-generation)
- [Reddit - r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)

---

## 附录: 快捷键

### Streamlit
- `Ctrl+C`: 停止服务
- `Ctrl+R`: 重新运行（在浏览器中）
- `Ctrl+Shift+R`: 清除缓存并重新运行

### VSCode（推荐）
- `Ctrl+Shift+P`: 命令面板
- `Ctrl+` `: 打开终端
- `F5`: 启动调试

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队

**快速联系**:
- 技术问题: 查看`rag/RAG_SYSTEM_ARCHITECTURE.md`
- 实现细节: 查看`rag/RAG_IMPLEMENTATION_GUIDE.md`
- 日常使用: 本文档
