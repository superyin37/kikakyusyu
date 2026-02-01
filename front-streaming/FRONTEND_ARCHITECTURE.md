# 前端架构与实现文档

## 文档概述
本文档详细描述Kita系统前端WebUI的架构设计、实现细节、功能模块和最佳实践。前端基于Streamlit框架，提供交互式对话界面、实时性能监控和知识文件管理功能。

---

## 目录
1. [前端概述](#1-前端概述)
2. [架构设计](#2-架构设计)
3. [核心功能模块](#3-核心功能模块)
4. [UI组件详解](#4-ui组件详解)
5. [状态管理](#5-状态管理)
6. [API集成](#6-api集成)
7. [GPU监控系统](#7-gpu监控系统)
8. [知识文件管理](#8-知识文件管理)
9. [日志系统](#9-日志系统)
10. [性能优化](#10-性能优化)
11. [用户体验](#11-用户体验)
12. [部署配置](#12-部署配置)

---

## 1. 前端概述

### 1.1 技术栈

**核心框架**:
- **Streamlit**: Python Web应用框架，支持快速构建数据应用
- **Python 3.10+**: 类型提示、异步支持

**依赖库**:
- **requests**: HTTP客户端，调用后端API
- **pynvml**: NVIDIA GPU监控（可选）
- **json/pathlib**: 数据处理和文件管理

### 1.2 核心职责

```
┌─────────────────────────────────────────────────┐
│              前端WebUI核心职责                    │
├─────────────────────────────────────────────────┤
│ 1. 提供用户交互界面（对话框、按钮）              │
│ 2. 实时展示对话历史                              │
│ 3. 调用后端API获取回答                          │
│ 4. 支持Blocking/Streaming两种模式               │
│ 5. 监控GPU/VRAM使用情况                         │
│ 6. 管理知识文件上传和检索                        │
│ 7. 展示性能指标（TTFB、Total Time）             │
│ 8. 显示参考信息（References）                   │
└─────────────────────────────────────────────────┘
```

### 1.3 文件结构

```
front-streaming/
├── app.py              # Streamlit主应用
├── gpu_stats.py        # GPU监控模块
├── area.jsonl          # 町名数据（后端同步）
├── rag_docs_merged.jsonl  # 垃圾分类数据（后端同步）
└── __pycache__/        # Python缓存
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户浏览器                              │
│                    (http://localhost:8501)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                  Streamlit 应用层                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  UI 渲染层                                            │   │
│  │   - 标题、侧边栏、主内容区                           │   │
│  │   - Chat组件（用户/助手消息）                        │   │
│  │   - 指标展示（TTFB、Total Time）                     │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  交互逻辑层                                           │   │
│  │   - 用户输入处理                                     │   │
│  │   - 模式切换（Blocking/Streaming）                   │   │
│  │   - 文件上传处理                                     │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  状态管理层                                           │   │
│  │   - st.session_state（对话历史）                     │   │
│  │   - 临时变量（性能指标）                             │   │
│  └─────────────────────┬────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                  后端API层 (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - POST /api/bot/respond (Blocking)                  │   │
│  │  - POST /api/bot/respond_stream (Streaming)          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  RAG 核心 + ChromaDB                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

#### 2.2.1 Blocking模式数据流

```
用户输入
    ↓
1. Streamlit 接收输入
    ↓
2. 构建请求 JSON
    ↓
3. POST /api/bot/respond
    ↓
4. 等待完整响应（阻塞）
    ↓
5. 解析响应 JSON
    │
    ├─ reply: 显示回答
    └─ references: 显示参考信息
    ↓
6. 更新 session_state
    ↓
7. 记录日志
```

#### 2.2.2 Streaming模式数据流

```
用户输入
    ↓
1. Streamlit 接收输入
    ↓
2. 构建请求 JSON
    ↓
3. POST /api/bot/respond_stream
    ↓
4. 建立流连接（stream=True）
    ↓
5. 逐块接收文本
    │
    ├─ 第1块: 记录TTFB，实时显示
    ├─ 第2块: 追加显示
    ├─ 第3块: 追加显示
    └─ ...
    ↓
6. 流结束，解析 X-References
    ↓
7. 显示参考信息
    ↓
8. 更新 session_state
    ↓
9. 记录日志
```

### 2.3 关键设计模式

#### 2.3.1 单页应用（SPA）模式

Streamlit本质是单页应用，每次交互会重新运行整个脚本：

```python
# 脚本从头开始执行
st.set_page_config(...)

# 初始化（仅首次）
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# UI渲染
st.title("...")
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# 用户输入处理
user_input = st.chat_input("...")
if user_input:
    # 处理逻辑
    ...
```

**特点**:
- 每次交互都重新运行脚本
- 使用 `st.session_state` 保持状态
- 简单直观，无需复杂路由

#### 2.3.2 状态管理模式

```python
# 全局状态：对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 添加消息
st.session_state["messages"].append({
    "role": "user",
    "content": user_input
})

# 读取消息
for msg in st.session_state["messages"]:
    display_message(msg)
```

#### 2.3.3 响应式UI模式

```python
# 创建占位符
placeholder = st.empty()

# 实时更新
for chunk in stream:
    collected += chunk
    placeholder.markdown(collected)  # 实时刷新
```

---

## 3. 核心功能模块

### 3.1 功能概览

| 模块 | 功能 | 文件 |
|-----|------|------|
| 对话界面 | 用户输入、消息展示 | app.py |
| 模式切换 | Blocking/Streaming | app.py |
| GPU监控 | VRAM/利用率 | gpu_stats.py |
| 文件管理 | 上传、列表 | app.py |
| 日志查看 | 历史对话 | app.py |
| 性能指标 | TTFB、Total Time | app.py |
| 参考信息 | References展示 | app.py |

### 3.2 模块依赖关系

```
app.py (主应用)
    │
    ├─ import gpu_stats
    │      └─ get_gpu_stats()
    │
    ├─ import rag.user_knowledge
    │      └─ add_file_to_chroma()
    │
    └─ import requests
           └─ 调用后端API
```

---

## 4. UI组件详解

### 4.1 页面布局

```python
st.set_page_config(
    page_title="Llama Chat (Streaming+Metrics)",
    page_icon="⏱️"
)

st.title("⏱️ Llama Chat – Streaming & Metrics")

# 侧边栏
with st.sidebar:
    # GPU监控
    st.subheader("GPU / VRAM Monitor")
    vram_box = st.empty()
    util_box = st.empty()
    
    # 模式选择
    mode = st.radio("応答モードを選択", ["Blocking", "Streaming"])
    
    # 文件管理
    st.subheader("ナレッジファイル管理")
    upload_file = st.file_uploader("...")

# 主内容区
# - 历史对话展示
# - 实时对话区
# - 性能指标
```

**布局特点**:
- 左侧边栏：控制和监控
- 主内容区：对话历史和实时交互
- 顶部标题：品牌和功能提示

### 4.2 对话组件

#### 4.2.1 历史对话展示

```python
logs = load_logs(limit=5)
if logs:
    st.subheader("🗂 過去のやり取り（最新5件）")
    for entry in logs:
        with st.chat_message("user"):
            st.markdown(entry.get("user", ""))
        with st.chat_message("assistant"):
            st.markdown(entry.get("assistant", ""))
    st.divider()
```

**功能**:
- 从 `backend/logs.jsonl` 加载最新5条
- 使用 `st.chat_message` 组件
- 提供历史上下文

#### 4.2.2 实时对话区

```python
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("メッセージを入力してください")
if user_input:
    # 显示用户消息
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 显示助手消息
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # 实时更新 placeholder
```

**特点**:
- `st.chat_message`: 自动添加头像和样式
- `st.chat_input`: 固定底部输入框
- `st.empty()`: 占位符，支持实时更新

### 4.3 性能指标组件

```python
col1, col2, col3, col4 = st.columns(4)
ttfb_area   = col1.empty()
total_area  = col2.empty()
tokps_area  = col3.empty()
outtok_area = col4.empty()

# 实时更新
ttfb_area.metric("TTFB (s)", round(ttfb, 3))
total_area.metric("Total (s)", round(total_sec, 3))
tokps_area.metric("Tokens/sec", "-")
outtok_area.metric("Output tokens", "-")
```

**指标说明**:
- **TTFB** (Time To First Byte): 首字节返回时间
- **Total**: 总响应时间
- **Tokens/sec**: 每秒生成token数（暂未实现）
- **Output tokens**: 输出token总数（暂未实现）

### 4.4 参考信息组件

```python
if references:
    st.markdown("### 📑 参考情報（上位チャンク）")
    for ref in references:
        file = ref.get("file", "?")
        page = ref.get("page", "?")
        chunk = ref.get("chunk") or ref.get("chunk_id") or ref.get("id", "?")
        text = ref.get("text", "")[:200]
        
        st.markdown(
            f"- **{file} p.{page} (chunk {chunk})**\n"
            f"  \n> {text}..."
        )
```

**显示效果**:
```
### 📑 参考情報（上位チャンク）
- **manual.pdf p.3 (chunk 1)**
  > パソコン本体（デスクトップ型・ノート型）は粗大ごみとして...
```

### 4.5 侧边栏组件

#### 4.5.1 GPU监控

```python
with st.sidebar:
    st.subheader("GPU / VRAM Monitor")
    vram_box = st.empty()
    util_box = st.empty()
    
    stats = get_gpu_stats()
    if stats:
        used_gb, total_gb, util_p, name = stats
        vram_box.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
        util_box.caption(f"{name} | Util {util_p}%")
    else:
        vram_box.metric("VRAM (GB)", "N/A")
        util_box.caption("GPU not detected")
```

**显示效果**:
```
GPU / VRAM Monitor
┌──────────────────┐
│ VRAM (GB)        │
│ 4.23/8.00        │
└──────────────────┘
NVIDIA GeForce RTX 3060 | Util 85%
```

#### 4.5.2 模式选择

```python
mode = st.radio(
    "応答モードを選択",
    ["Blocking", "Streaming"],
    horizontal=True,
    key="response_mode"
)
```

**UI**:
```
応答モードを選択
◉ Blocking    ○ Streaming
```

#### 4.5.3 文件管理

```python
st.subheader("ナレッジファイル管理")

upload_file = st.file_uploader(
    "ファイルをアップロード",
    type=["txt", "pdf", "csv", "json"]
)

if upload_file is not None:
    save_path = KNOWLEDGE_DIR / upload_file.name
    with open(save_path, "wb") as f:
        f.write(upload_file.getbuffer())
    st.success(f"アップロードしました: {upload_file.name}")
    
    # 添加到ChromaDB
    add_file_to_chroma(save_path)

# 显示已上传文件
files = list(KNOWLEDGE_DIR.glob("*"))
if files:
    st.caption("検索対象ファイル一覧:")
    for f in files:
        st.text(f.name)
```

---

## 5. 状态管理

### 5.1 Session State

Streamlit的 `st.session_state` 是跨脚本运行保持状态的机制。

#### 5.1.1 对话历史

```python
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 添加消息
st.session_state["messages"].append({
    "role": "user",
    "content": user_input
})

st.session_state["messages"].append({
    "role": "assistant",
    "content": collected
})

# 读取消息
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

**数据结构**:
```python
st.session_state["messages"] = [
    {"role": "user", "content": "ノートPCの捨て方"},
    {"role": "assistant", "content": "ノートPCは粗大ごみとして..."},
    {"role": "user", "content": "八幡東区の収集日は？"},
    {"role": "assistant", "content": "火曜日と金曜日です。"},
]
```

#### 5.1.2 清空历史

```python
# 添加清空按钮
if st.sidebar.button("チャット履歴をクリア"):
    st.session_state["messages"] = []
    st.rerun()
```

### 5.2 临时变量

非持久化变量，每次脚本运行时重新创建：

```python
# 性能指标
t_start = time.perf_counter()
ttfb = None
total_sec = None

# 响应内容
collected = ""
references = []
```

---

## 6. API集成

### 6.1 Blocking模式集成

```python
if mode == "Blocking":
    try:
        api_url = "http://localhost:8000/api/bot/respond"
        res = requests.post(
            api_url,
            json={"prompt": user_input},
            timeout=20
        )
        res.raise_for_status()
        
        # 解析响应
        data = res.json()
        reply = data.get("reply", "")
        references = data.get("references", [])
        
    except Exception as e:
        reply = "APIリクエストでエラー: " + str(e)
        references = []
    
    # 计算总时间
    t_end = time.perf_counter()
    total_sec = t_end - t_start
    
    # 更新指标
    ttfb_area.metric("TTFB (s)", round(total_sec, 3))
    total_area.metric("Total (s)", round(total_sec, 3))
    
    # 显示回答
    collected = reply
    placeholder.markdown(collected)
```

**关键点**:
- `timeout=20`: 防止长时间等待
- `res.raise_for_status()`: 自动抛出HTTP错误
- `data.get()`: 安全获取字段，避免KeyError

### 6.2 Streaming模式集成

```python
if mode == "Streaming":
    try:
        api_url = "http://localhost:8000/api/bot/respond_stream"
        with requests.post(
            api_url,
            json={"prompt": user_input},
            stream=True,
            timeout=60
        ) as res:
            res.raise_for_status()
            
            ttfb = None
            collected = ""
            
            # 逐块接收
            for chunk in res.iter_content(chunk_size=None):
                if not chunk:
                    continue
                
                # 记录TTFB
                if ttfb is None:
                    ttfb = time.perf_counter()
                    ttfb_area.metric("TTFB (s)", round(ttfb - t_start, 3))
                
                # 实时显示
                text = chunk.decode("utf-8")
                collected += text
                placeholder.markdown(collected)
            
            # 计算总时间
            t_end = time.perf_counter()
            total_sec = t_end - t_start
            total_area.metric("Total (s)", round(total_sec, 3))
            
            # 解析References（在HTTP Header中）
            references = []
            if "X-References" in res.headers:
                try:
                    references = json.loads(res.headers["X-References"])
                except Exception:
                    references = []
            
            # 显示参考信息
            if references:
                st.markdown("#### 📑 参考情報")
                for ref in references:
                    st.markdown(
                        f"- **{ref.get('file','?')} p.{ref.get('page','?')} "
                        f"(chunk {ref.get('chunk','?')})**\n"
                        f"  \n> {ref.get('text','')[:200]}..."
                    )
    
    except Exception as e:
        collected = "APIリクエストでエラー: " + str(e)
        placeholder.markdown(collected)
```

**关键点**:
- `stream=True`: 启用流式接收
- `with ... as res`: 自动关闭连接
- `iter_content(chunk_size=None)`: 逐块迭代
- `X-References`: 从Header获取参考信息

### 6.3 错误处理

```python
try:
    res = requests.post(api_url, json={"prompt": user_input}, timeout=20)
    res.raise_for_status()
    # ...
except requests.exceptions.Timeout:
    collected = "⏱️ リクエストがタイムアウトしました。"
except requests.exceptions.ConnectionError:
    collected = "❌ サーバーに接続できません。バックエンドが起動しているか確認してください。"
except requests.exceptions.HTTPError as e:
    collected = f"❌ HTTPエラー: {e.response.status_code}"
except Exception as e:
    collected = f"❌ エラー: {str(e)}"

placeholder.markdown(collected)
```

---

## 7. GPU监控系统

### 7.1 gpu_stats.py 模块

#### 7.1.1 NVML方式（首选）

```python
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetName, nvmlDeviceGetCount
    )
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

def _nvml_get() -> Optional[Tuple[float, float, int, str]]:
    """返回 (used_GB, total_GB, util_percent, gpu_name)"""
    if not _NVML_AVAILABLE:
        return None
    try:
        count = nvmlDeviceGetCount()
        if count == 0:
            return None
        
        handle = nvmlDeviceGetHandleByIndex(0)  # 使用第一个GPU
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_rates = nvmlDeviceGetUtilizationRates(handle)
        name = nvmlDeviceGetName(handle)
        
        used_gb = mem_info.used / (1024 ** 3)
        total_gb = mem_info.total / (1024 ** 3)
        util_percent = util_rates.gpu
        
        return (used_gb, total_gb, util_percent, name)
    except Exception:
        return None
```

**优点**:
- 速度快（~1ms）
- 准确
- 低CPU占用

#### 7.1.2 nvidia-smi方式（备用）

```python
def _nvidia_smi_get() -> Optional[Tuple[float, float, int, str]]:
    """使用nvidia-smi命令获取GPU信息"""
    if not shutil.which("nvidia-smi"):
        return None
    
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,name",
             "--format=csv,noheader,nounits"],
            text=True
        )
        parts = output.strip().split(",")
        used_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        util_p = int(parts[2].strip())
        name = parts[3].strip()
        
        return (used_mb / 1024, total_mb / 1024, util_p, name)
    except Exception:
        return None
```

**缺点**:
- 速度慢（~100ms）
- CPU占用高

#### 7.1.3 rocm-smi方式（AMD GPU）

```python
def _rocm_smi_get() -> Optional[Tuple[float, float, int, str]]:
    """支持AMD GPU"""
    if not shutil.which("rocm-smi"):
        return None
    # 实现细节...
```

### 7.2 初始化和清理

```python
# 应用启动时
init_nvml_once()

# 应用结束时
import atexit
atexit.register(shutdown_nvml)
```

### 7.3 实时更新

```python
# 在侧边栏创建占位符
vram_box = st.empty()
util_box = st.empty()

# 查询GPU状态
stats = get_gpu_stats()
if stats:
    used_gb, total_gb, util_p, name = stats
    vram_box.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
    util_box.caption(f"{name} | Util {util_p}%")
else:
    vram_box.metric("VRAM (GB)", "N/A")
    util_box.caption("GPU not detected")
```

---

## 8. 知识文件管理

### 8.1 文件上传

```python
KNOWLEDGE_DIR = Path("knowledge_files")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

upload_file = st.file_uploader(
    "ファイルをアップロード",
    type=["txt", "pdf", "csv", "json"]
)

if upload_file is not None:
    # 保存文件
    save_path = KNOWLEDGE_DIR / upload_file.name
    with open(save_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    st.success(f"アップロードしました: {upload_file.name}")
    
    # 添加到ChromaDB
    add_file_to_chroma(save_path)
```

**流程**:
1. 用户选择文件
2. 保存到 `knowledge_files/` 目录
3. 调用 `add_file_to_chroma()` 添加到向量数据库
4. 显示成功消息

### 8.2 文件列表

```python
files = list(KNOWLEDGE_DIR.glob("*"))
if files:
    st.caption("検索対象ファイル一覧:")
    for f in files:
        st.text(f.name)
else:
    st.caption("まだファイルはありません。")
```

**显示效果**:
```
検索対象ファイル一覧:
manual.pdf
guide.txt
rules.csv
```

### 8.3 集成 user_knowledge 模块

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.user_knowledge import add_file_to_chroma

# 使用
add_file_to_chroma(save_path)
```

**功能**:
- 自动分块（PDF/TXT/CSV/JSON）
- 添加到 `knowledge` collection
- 支持后续RAG检索

---

## 9. 日志系统

### 9.1 日志记录

```python
LOG_FILE = Path("backend/logs.jsonl")

def save_log(user_input: str, assistant_output: str, mode: str, total_sec: float):
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user": user_input,
        "assistant": assistant_output,
        "total_time": round(total_sec, 3),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
```

**调用时机**:
```python
if total_sec is not None:
    save_log(user_input, collected, mode, total_sec)
```

### 9.2 日志加载

```python
def load_logs(limit: int = 20):
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries[-limit:]  # 返回最新N条
```

### 9.3 历史对话展示

```python
logs = load_logs(limit=5)
if logs:
    st.subheader("🗂 過去のやり取り（最新5件）")
    for entry in logs:
        with st.chat_message("user"):
            st.markdown(entry.get("user", ""))
        with st.chat_message("assistant"):
            st.markdown(entry.get("assistant", ""))
    st.divider()
```

---

## 10. 性能优化

### 10.1 避免重复计算

```python
# ❌ 每次都重新加载
logs = load_logs(limit=5)

# ✅ 使用缓存（如果日志不常更新）
@st.cache_data(ttl=60)  # 缓存60秒
def load_logs_cached(limit: int = 20):
    return load_logs(limit)
```

### 10.2 延迟加载

```python
# ❌ 启动时加载所有文件
files = list(KNOWLEDGE_DIR.glob("*"))

# ✅ 仅在展开时加载
with st.expander("検索対象ファイル一覧"):
    files = list(KNOWLEDGE_DIR.glob("*"))
    for f in files:
        st.text(f.name)
```

### 10.3 流式显示优化

```python
# 减少刷新频率
buffer = ""
for i, chunk in enumerate(res.iter_content(chunk_size=None)):
    buffer += chunk.decode("utf-8")
    
    # 每5块更新一次（减少渲染开销）
    if i % 5 == 0:
        placeholder.markdown(buffer)

# 最终更新
placeholder.markdown(buffer)
```

### 10.4 GPU监控频率

```python
# 仅在对话后更新GPU状态（避免频繁查询）
if user_input:
    # ... 处理对话 ...
    
    # 更新GPU状态
    stats = get_gpu_stats()
    if stats:
        vram_box.metric(...)
```

---

## 11. 用户体验

### 11.1 加载状态

```python
with st.spinner("生成中..."):
    # API调用
    res = requests.post(...)
```

### 11.2 错误提示

```python
if error:
    st.error("❌ エラーが発生しました: " + str(error))

# 或使用emoji增强视觉效果
st.warning("⚠️ バックエンドに接続できません。")
st.success("✅ ファイルをアップロードしました。")
st.info("ℹ️ Streaming モードは実時間で応答を表示します。")
```

### 11.3 进度反馈

```python
# Streaming模式自动提供实时反馈
# 可添加字符计数
st.caption(f"生成済み: {len(collected)} 文字")
```

### 11.4 快捷操作

```python
# 示例查询按钮
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("ノートPCの捨て方"):
        st.session_state["quick_query"] = "ノートPCの捨て方"
with col2:
    if st.button("八幡東区の収集日"):
        st.session_state["quick_query"] = "八幡東区の収集日"

# 处理快捷查询
if "quick_query" in st.session_state:
    user_input = st.session_state["quick_query"]
    del st.session_state["quick_query"]
    # 处理查询...
```

---

## 12. 部署配置

### 12.1 Streamlit配置

创建 `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200  # MB

[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false
```

### 12.2 启动命令

```bash
# 开发模式
streamlit run front-streaming/app.py

# 指定端口
streamlit run front-streaming/app.py --server.port 8501

# 生产模式（禁用文件监控）
streamlit run front-streaming/app.py --server.fileWatcherType none
```

### 12.3 Docker部署

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY pyproject.toml .
RUN pip install uv && uv sync

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "front-streaming/app.py", "--server.address", "0.0.0.0"]
```

### 12.4 Nginx反向代理

```nginx
server {
    listen 80;
    server_name kita.example.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 13. 调试技巧

### 13.1 显示调试信息

```python
# 显示session_state
with st.expander("Debug: Session State"):
    st.json(st.session_state)

# 显示请求详情
with st.expander("Debug: API Request"):
    st.code(f"URL: {api_url}\nPayload: {json.dumps({'prompt': user_input})}")
```

### 13.2 日志输出

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"用户输入: {user_input}")
logger.debug(f"API响应: {res.status_code}")
```

### 13.3 性能分析

```python
import time

start = time.perf_counter()
# ... 操作 ...
elapsed = time.perf_counter() - start
st.caption(f"⏱️ 处理耗时: {elapsed:.3f}s")
```

---

## 14. 扩展功能

### 14.1 对话导出

```python
if st.sidebar.button("チャット履歴をエクスポート"):
    chat_json = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2)
    st.download_button(
        "ダウンロード",
        chat_json,
        file_name="chat_history.json",
        mime="application/json"
    )
```

### 14.2 主题切换

```python
theme = st.sidebar.selectbox("テーマ", ["ライト", "ダーク"])
if theme == "ダーク":
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)
```

### 14.3 多语言支持

```python
lang = st.sidebar.selectbox("Language", ["日本語", "English", "中文"])

TRANSLATIONS = {
    "日本語": {"title": "Llama チャット", "input": "メッセージを入力"},
    "English": {"title": "Llama Chat", "input": "Enter message"},
    "中文": {"title": "Llama 对话", "input": "输入消息"},
}

st.title(TRANSLATIONS[lang]["title"])
user_input = st.chat_input(TRANSLATIONS[lang]["input"])
```

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队

**相关文档**:
- 快速参考: `FRONTEND_REFERENCE.md`
- 后端文档: `../backend/BACKEND_ARCHITECTURE.md`
- RAG系统: `../rag/RAG_DOCUMENTATION_INDEX.md`
