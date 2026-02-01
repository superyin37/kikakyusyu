# 前端快速参考手册

## 文档概述
本文档提供Kita前端WebUI的快速参考，包括组件使用、配置参数、常见问题和代码示例。

---

## 目录
1. [快速开始](#1-快速开始)
2. [UI组件参考](#2-ui组件参考)
3. [功能模块](#3-功能模块)
4. [配置参数](#4-配置参数)
5. [代码示例](#5-代码示例)
6. [常见问题](#6-常见问题)
7. [性能优化](#7-性能优化)
8. [调试技巧](#8-调试技巧)

---

## 1. 快速开始

### 1.1 启动前端

```bash
# 进入项目目录
cd d:\projects\kitakyusyu\gmo_intern-main\gmo_intern-main

# 激活虚拟环境（如果使用）
.\.venv\Scripts\Activate.ps1

# 启动Streamlit
streamlit run front-streaming/app.py
```

**默认访问地址**: http://localhost:8501

### 1.2 前提条件

- Python 3.10+
- 后端API运行在 http://localhost:8000
- 已安装依赖：
  ```bash
  pip install streamlit requests pynvml
  ```

### 1.3 目录结构

```
front-streaming/
├── app.py              # 主应用
├── gpu_stats.py        # GPU监控模块
└── knowledge_files/    # 上传的文件（运行时创建）
```

---

## 2. UI组件参考

### 2.1 对话组件

#### st.chat_message

显示对话消息，自动添加头像和样式。

**用法**:
```python
with st.chat_message("user"):
    st.markdown("用户的消息内容")

with st.chat_message("assistant"):
    st.markdown("助手的回答内容")
```

**角色类型**:
- `"user"`: 用户消息（右侧，蓝色）
- `"assistant"`: 助手消息（左侧，灰色）

#### st.chat_input

固定在底部的输入框。

**用法**:
```python
user_input = st.chat_input("メッセージを入力してください")
if user_input:
    # 处理用户输入
    print(user_input)
```

### 2.2 侧边栏组件

#### st.sidebar

创建侧边栏。

**用法**:
```python
with st.sidebar:
    st.subheader("设置")
    mode = st.radio("模式", ["Blocking", "Streaming"])
```

#### st.radio

单选按钮。

**用法**:
```python
mode = st.radio(
    "応答モードを選択",
    ["Blocking", "Streaming"],
    horizontal=True,  # 水平排列
    key="response_mode"
)
```

#### st.file_uploader

文件上传组件。

**用法**:
```python
upload_file = st.file_uploader(
    "ファイルをアップロード",
    type=["txt", "pdf", "csv", "json"],
    accept_multiple_files=False
)

if upload_file is not None:
    # 获取文件内容
    content = upload_file.read()
    # 获取文件名
    filename = upload_file.name
```

### 2.3 指标组件

#### st.metric

显示指标卡片。

**用法**:
```python
st.metric(
    label="TTFB (s)",
    value="2.34",
    delta="-0.5"  # 可选：变化值
)
```

**示例**:
```python
col1, col2, col3 = st.columns(3)
col1.metric("TTFB", "2.34s")
col2.metric("Total", "5.67s")
col3.metric("Tokens", "1250")
```

### 2.4 布局组件

#### st.columns

创建列布局。

**用法**:
```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("指标1", "100")
with col2:
    st.metric("指标2", "200")
# ...
```

#### st.empty

创建占位符，可后续更新。

**用法**:
```python
placeholder = st.empty()

# 初始显示
placeholder.text("加载中...")

# 更新内容
placeholder.markdown("完成！")
```

#### st.expander

可展开/折叠的容器。

**用法**:
```python
with st.expander("查看详情"):
    st.write("这里是详细内容")
```

### 2.5 消息组件

#### st.success / st.error / st.warning / st.info

显示状态消息。

**用法**:
```python
st.success("✅ アップロード成功")
st.error("❌ エラーが発生しました")
st.warning("⚠️ 警告メッセージ")
st.info("ℹ️ 情報メッセージ")
```

---

## 3. 功能模块

### 3.1 对话功能

#### 发送消息

```python
user_input = st.chat_input("メッセージを入力してください")
if user_input:
    # 1. 添加到历史
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })
    
    # 2. 调用API
    response = call_api(user_input)
    
    # 3. 显示回答
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response
    })
```

#### 显示历史

```python
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

### 3.2 模式切换

```python
mode = st.radio(
    "応答モードを選択",
    ["Blocking", "Streaming"],
    horizontal=True
)

if mode == "Blocking":
    # Blocking模式处理
    response = requests.post("http://localhost:8000/api/bot/respond", ...)
else:
    # Streaming模式处理
    with requests.post("http://localhost:8000/api/bot/respond_stream", stream=True) as res:
        for chunk in res.iter_content():
            # 实时显示
            ...
```

### 3.3 文件上传

```python
upload_file = st.file_uploader("ファイルをアップロード", type=["pdf", "txt"])

if upload_file is not None:
    # 保存文件
    save_path = KNOWLEDGE_DIR / upload_file.name
    with open(save_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    # 添加到ChromaDB
    from rag.user_knowledge import add_file_to_chroma
    add_file_to_chroma(save_path)
    
    st.success(f"✅ {upload_file.name} をアップロードしました")
```

### 3.4 GPU监控

```python
from gpu_stats import init_nvml_once, get_gpu_stats

# 初始化
init_nvml_once()

# 获取状态
stats = get_gpu_stats()
if stats:
    used_gb, total_gb, util_p, name = stats
    st.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
    st.caption(f"{name} | Util {util_p}%")
else:
    st.metric("VRAM (GB)", "N/A")
```

---

## 4. 配置参数

### 4.1 Streamlit配置

创建 `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 200  # MB

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
serverAddress = "localhost"
```

### 4.2 应用配置

**文件**: `front-streaming/app.py`

```python
# 页面配置
st.set_page_config(
    page_title="Llama Chat",
    page_icon="⏱️",
    layout="centered",  # "centered" | "wide"
    initial_sidebar_state="auto"  # "auto" | "expanded" | "collapsed"
)

# 日志文件
LOG_FILE = Path("backend/logs.jsonl")

# 知识文件目录
KNOWLEDGE_DIR = Path("knowledge_files")

# API地址
API_BASE_URL = "http://localhost:8000"
```

### 4.3 API端点

| 端点 | 用途 | URL |
|-----|------|-----|
| Blocking | 一次性返回完整回答 | `http://localhost:8000/api/bot/respond` |
| Streaming | 流式返回回答 | `http://localhost:8000/api/bot/respond_stream` |

### 4.4 超时设置

```python
# Blocking模式
response = requests.post(url, json=payload, timeout=20)  # 20秒

# Streaming模式
with requests.post(url, json=payload, stream=True, timeout=60) as res:  # 60秒
    ...
```

---

## 5. 代码示例

### 5.1 完整的Blocking请求

```python
import streamlit as st
import requests
import time

st.title("RAG チャット")

user_input = st.chat_input("質問を入力")

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 显示助手消息
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        try:
            # 调用API
            start = time.perf_counter()
            response = requests.post(
                "http://localhost:8000/api/bot/respond",
                json={"prompt": user_input},
                timeout=20
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            
            # 解析响应
            data = response.json()
            reply = data.get("reply", "")
            references = data.get("references", [])
            
            # 显示回答
            placeholder.markdown(reply)
            
            # 显示性能指标
            st.caption(f"⏱️ 応答時間: {elapsed:.2f}s")
            
            # 显示参考信息
            if references:
                with st.expander("📑 参考情報"):
                    for ref in references:
                        st.markdown(f"- **{ref['file']}** p.{ref['page']}")
                        st.markdown(f"> {ref['text'][:200]}...")
        
        except Exception as e:
            placeholder.error(f"❌ エラー: {e}")
```

### 5.2 完整的Streaming请求

```python
import streamlit as st
import requests
import time
import json

st.title("RAG チャット (Streaming)")

user_input = st.chat_input("質問を入力")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        try:
            start = time.perf_counter()
            ttfb = None
            collected = ""
            
            with requests.post(
                "http://localhost:8000/api/bot/respond_stream",
                json={"prompt": user_input},
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                
                # 逐块接收
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    
                    # 记录TTFB
                    if ttfb is None:
                        ttfb = time.perf_counter() - start
                    
                    # 实时显示
                    text = chunk.decode("utf-8")
                    collected += text
                    placeholder.markdown(collected)
                
                elapsed = time.perf_counter() - start
                
                # 显示指标
                col1, col2 = st.columns(2)
                col1.metric("TTFB", f"{ttfb:.2f}s")
                col2.metric("Total", f"{elapsed:.2f}s")
                
                # 解析References
                refs_header = response.headers.get("X-References", "[]")
                references = json.loads(refs_header)
                
                if references:
                    with st.expander("📑 参考情報"):
                        for ref in references:
                            st.markdown(f"- **{ref['file']}** p.{ref['page']}")
        
        except Exception as e:
            placeholder.error(f"❌ エラー: {e}")
```

### 5.3 GPU监控集成

```python
import streamlit as st
from gpu_stats import init_nvml_once, get_gpu_stats, shutdown_nvml
import atexit

# 初始化
init_nvml_once()

# 注册清理函数
atexit.register(shutdown_nvml)

# 侧边栏监控
with st.sidebar:
    st.subheader("GPU / VRAM Monitor")
    vram_box = st.empty()
    util_box = st.empty()
    
    # 获取GPU状态
    stats = get_gpu_stats()
    if stats:
        used_gb, total_gb, util_p, name = stats
        vram_box.metric("VRAM (GB)", f"{used_gb:.2f}/{total_gb:.2f}")
        util_box.caption(f"{name} | Util {util_p}%")
    else:
        vram_box.metric("VRAM (GB)", "N/A")
        util_box.caption("GPU not detected")
```

### 5.4 会话状态管理

```python
import streamlit as st

# 初始化
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Guest"

# 添加消息
def add_message(role: str, content: str):
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

# 清空历史
if st.sidebar.button("履歴をクリア"):
    st.session_state["messages"] = []
    st.rerun()

# 显示历史
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

### 5.5 日志管理

```python
import json
from pathlib import Path

LOG_FILE = Path("backend/logs.jsonl")

def save_log(user_input: str, assistant_output: str, mode: str, total_sec: float):
    """保存对话日志"""
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user": user_input,
        "assistant": assistant_output,
        "total_time": round(total_sec, 3),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

def load_logs(limit: int = 20):
    """加载最新N条日志"""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                continue
    return entries[-limit:]

# 显示历史对话
logs = load_logs(limit=5)
if logs:
    st.subheader("🗂 過去のやり取り")
    for entry in logs:
        with st.chat_message("user"):
            st.markdown(entry.get("user", ""))
        with st.chat_message("assistant"):
            st.markdown(entry.get("assistant", ""))
```

---

## 6. 常见问题

### 6.1 前端无法访问

**症状**: 浏览器显示 "无法访问此网站"

**原因**: Streamlit未启动

**解决**:
```bash
# 检查进程
ps aux | grep streamlit

# 启动Streamlit
streamlit run front-streaming/app.py
```

---

### 6.2 无法连接后端

**症状**: 显示 "APIリクエストでエラー: Connection refused"

**原因**: 后端API未启动

**解决**:
```bash
# 检查后端
curl http://localhost:8000/api/bot/respond

# 启动后端
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

### 6.3 Streaming无实时显示

**症状**: Streaming模式像Blocking一样等待全部完成

**原因**: 未使用 `stream=True`

**解决**:
```python
# ❌ 错误
response = requests.post(url, json=payload)

# ✅ 正确
with requests.post(url, json=payload, stream=True) as response:
    for chunk in response.iter_content():
        ...
```

---

### 6.4 GPU监控显示N/A

**症状**: VRAM显示 "N/A"

**原因**: 
1. 未安装pynvml
2. 不是NVIDIA GPU
3. NVML初始化失败

**解决**:
```bash
# 安装pynvml
pip install pynvml

# 检查GPU
nvidia-smi

# 检查Python
python -c "from pynvml import nvmlInit; nvmlInit(); print('OK')"
```

---

### 6.5 文件上传失败

**症状**: 上传后无反应或报错

**原因**: 
1. 文件过大（超过maxUploadSize）
2. 文件类型不支持
3. ChromaDB连接失败

**解决**:
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200  # 增加到200MB
```

```python
# 检查文件类型
upload_file = st.file_uploader(
    "ファイル",
    type=["txt", "pdf", "csv", "json"]  # 确保类型匹配
)
```

---

### 6.6 会话状态丢失

**症状**: 刷新页面后对话历史消失

**原因**: Streamlit的session_state仅在会话内保持

**解决**:
```python
# 保存到文件
if st.sidebar.button("保存历史"):
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state["messages"], f)

# 加载历史
if st.sidebar.button("加载历史"):
    with open("chat_history.json", "r") as f:
        st.session_state["messages"] = json.load(f)
```

---

### 6.7 页面卡顿

**症状**: 输入后页面卡住

**原因**: 
1. API响应慢
2. 渲染大量数据
3. 未设置timeout

**解决**:
```python
# 1. 添加timeout
response = requests.post(url, json=payload, timeout=20)

# 2. 使用spinner
with st.spinner("処理中..."):
    response = requests.post(...)

# 3. 限制显示数量
logs = load_logs(limit=5)  # 仅显示5条
```

---

## 7. 性能优化

### 7.1 缓存数据

```python
@st.cache_data(ttl=60)  # 缓存60秒
def load_logs_cached(limit: int = 20):
    return load_logs(limit)

# 使用缓存版本
logs = load_logs_cached(limit=5)
```

### 7.2 延迟加载

```python
# 仅在展开时加载
with st.expander("查看文件列表"):
    files = list(KNOWLEDGE_DIR.glob("*"))
    for f in files:
        st.text(f.name)
```

### 7.3 减少重新运行

```python
# 使用form避免每次输入都重新运行
with st.form("query_form"):
    user_input = st.text_input("質問")
    submitted = st.form_submit_button("送信")
    
    if submitted:
        # 处理查询
        ...
```

### 7.4 分批显示

```python
# 分批显示长回答
MAX_PREVIEW_LENGTH = 500

if len(reply) > MAX_PREVIEW_LENGTH:
    st.markdown(reply[:MAX_PREVIEW_LENGTH] + "...")
    with st.expander("全文を表示"):
        st.markdown(reply)
else:
    st.markdown(reply)
```

---

## 8. 调试技巧

### 8.1 显示Session State

```python
with st.sidebar.expander("Debug: Session State"):
    st.json(st.session_state)
```

### 8.2 显示API请求详情

```python
with st.expander("Debug: API Request"):
    st.code(f"""
URL: {api_url}
Payload: {json.dumps({"prompt": user_input}, indent=2)}
Timeout: 20s
    """)
```

### 8.3 显示响应详情

```python
with st.expander("Debug: API Response"):
    st.code(f"""
Status: {response.status_code}
Headers: {dict(response.headers)}
Body: {response.text[:500]}
    """)
```

### 8.4 性能分析

```python
import time

# 计时器
start = time.perf_counter()

# ... 操作 ...

elapsed = time.perf_counter() - start
st.caption(f"⏱️ 処理時間: {elapsed:.3f}s")
```

### 8.5 错误追踪

```python
try:
    response = requests.post(...)
except Exception as e:
    st.error(f"❌ エラー: {type(e).__name__}")
    with st.expander("詳細"):
        st.exception(e)  # 显示完整堆栈
```

---

## 9. 高级用法

### 9.1 多页面应用

创建 `pages/` 目录：

```
front-streaming/
├── app.py              # 主页
└── pages/
    ├── 1_📊_Analytics.py
    ├── 2_⚙️_Settings.py
    └── 3_📁_Files.py
```

Streamlit会自动识别为多页面应用。

### 9.2 自定义主题

```python
# 在代码中设置
st.markdown("""
    <style>
    .stChatMessage {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)
```

### 9.3 WebSocket支持（实验性）

```python
# 注意：Streamlit原生不支持WebSocket
# 可使用streamlit-javascript-callback库
```

### 9.4 数据导出

```python
# 导出对话历史为CSV
import pandas as pd

if st.sidebar.button("Export CSV"):
    df = pd.DataFrame(st.session_state["messages"])
    csv = df.to_csv(index=False)
    st.download_button(
        "Download",
        csv,
        file_name="chat_history.csv",
        mime="text/csv"
    )
```

---

## 10. 快速查询表

### 10.1 常用组件

| 组件 | 用途 | 示例 |
|-----|------|------|
| `st.title()` | 标题 | `st.title("标题")` |
| `st.markdown()` | Markdown文本 | `st.markdown("**粗体**")` |
| `st.button()` | 按钮 | `if st.button("点击"):` |
| `st.text_input()` | 文本输入 | `text = st.text_input("标签")` |
| `st.selectbox()` | 下拉选择 | `option = st.selectbox("选择", [...])` |
| `st.slider()` | 滑块 | `val = st.slider("标签", 0, 100)` |
| `st.checkbox()` | 复选框 | `if st.checkbox("选项"):` |

### 10.2 布局组件

| 组件 | 用途 | 示例 |
|-----|------|------|
| `st.sidebar` | 侧边栏 | `with st.sidebar:` |
| `st.columns()` | 列布局 | `col1, col2 = st.columns(2)` |
| `st.expander()` | 可折叠 | `with st.expander("标题"):` |
| `st.container()` | 容器 | `with st.container():` |
| `st.empty()` | 占位符 | `placeholder = st.empty()` |

### 10.3 显示组件

| 组件 | 用途 | 示例 |
|-----|------|------|
| `st.success()` | 成功消息 | `st.success("成功")` |
| `st.error()` | 错误消息 | `st.error("错误")` |
| `st.warning()` | 警告消息 | `st.warning("警告")` |
| `st.info()` | 信息消息 | `st.info("信息")` |
| `st.spinner()` | 加载动画 | `with st.spinner("加载中"):` |

### 10.4 数据组件

| 组件 | 用途 | 示例 |
|-----|------|------|
| `st.dataframe()` | 数据表 | `st.dataframe(df)` |
| `st.table()` | 静态表格 | `st.table(df)` |
| `st.metric()` | 指标卡片 | `st.metric("标签", "100")` |
| `st.json()` | JSON展示 | `st.json({"key": "value"})` |
| `st.code()` | 代码块 | `st.code("print('hello')")` |

---

## 附录

### A. Streamlit命令行参数

```bash
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.fileWatcherType none  # 生产模式
```

### B. 环境变量

```bash
# 设置端口
export STREAMLIT_SERVER_PORT=8501

# 禁用遥测
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### C. 性能基准

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| 页面加载 | <1s | 首次访问 |
| 交互响应 | <0.5s | 点击按钮 |
| Streaming TTFB | 0.5-2s | 首字节 |
| Blocking Total | 3-8s | 完整响应 |

### D. 快捷键

| 快捷键 | 功能 |
|-------|------|
| `R` | 重新运行应用 |
| `C` | 清除缓存 |
| `?` | 显示快捷键帮助 |

---

**文档版本**: v1.0  
**最后更新**: 2026-02-01  
**维护者**: Kita开发团队

**相关文档**:
- 详细架构: `FRONTEND_ARCHITECTURE.md`
- 后端API: `../backend/API_REFERENCE.md`
- RAG系统: `../rag/RAG_DOCUMENTATION_INDEX.md`
