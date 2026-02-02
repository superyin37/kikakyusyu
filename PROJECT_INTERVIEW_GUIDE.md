# Kita - 北九州市智能垃圾分类系统 面试项目介绍

## 📋 项目概览

**项目名称**: Kita - 北九州市垃圾分类智能问答系统  
**项目类型**: RAG (检索增强生成) 问答系统  
**开发周期**: GMO实习项目  
**团队规模**: 3人 (青木颯大、Yin Hanyang、安田大朗)  
**我的角色**: [请填写你的具体角色和职责]

---

## 🎯 项目背景与目标

### 业务痛点
- 北九州市垃圾分类规则复杂，用户难以快速获取准确信息
- 不同町（区域）的垃圾收集日期不同，查询不便
- 现有查询方式效率低，用户体验差

### 解决方案
开发一个基于RAG架构的智能问答系统，通过自然语言对话方式为用户提供：
- 垃圾分类规则查询
- 按町名查询收集日期
- 自定义知识库扩展能力

---

## 🏗 系统架构设计

### 技术栈选型

**前端层**
- **Streamlit**: 快速构建交互式WebUI
- **GPU监控**: NVML/nvidia-smi实时显示资源使用

**后端层**
- **FastAPI**: 异步API框架，支持高并发
- **RESTful API**: Blocking/Streaming双模式响应

**核心RAG引擎**
- **ChromaDB**: 向量数据库，支持高效语义检索
- **MeCab**: 日本语形态素解析
- **Hybrid Grounding System v2.0**: 自研的智能品名识别系统

**LLM推理**
- **Ollama**: 本地化部署，保障数据隐私
- **Llama-3.1-Swallow-8B**: 日语优化的大模型
- **cl-nagoya-ruri-large**: 日语向量化模型

### 架构亮点

```
用户输入 → Hybrid Grounding → ChromaDB检索 → RAG Prompt → LLM生成 → 返回结果
             (智能识别)        (多源检索)                      (流式/阻塞)
```

**三层架构设计**:
1. **表示层** (Streamlit): Chat UI + 日志展示 + 文件上传
2. **业务层** (FastAPI): API路由 + RAG编排 + 数据验证
3. **数据层** (ChromaDB): 分类规则 + 区域信息 + 用户知识库

---

## 💡 核心技术创新

### 1. Hybrid Grounding System v2.0 (重点技术)

**问题**: 传统MeCab分词在处理复杂查询时准确率不足

**创新方案**: 三层智能识别系统

#### 第一层：精确匹配 (Exact Match)
- 输入完全匹配数据库 → 置信度1.0 → 立即返回
- 响应时间 < 5ms

#### 第二层：智能分流 (Path Selection)
```python
if 输入长度 < 20字符:
    → Path A (快速路径): 整体Embedding检索
    → 响应时间 < 300ms
else:
    → Path A + Path B (双路径):
       - Path A: 整体语义检索
       - Path B: LLM辅助短语提取 + 分段检索
    → 响应时间 < 600ms
```

#### 第三层：置信度评估 (Confidence Evaluation)
- **High** (≥0.70): 直接采用
- **Medium** (0.45-0.70): 提示用户
- **Low** (<0.45): 自动Fallback到MeCab

#### 技术成果
- 准确率提升：78% → 92%
- 响应速度：平均 < 400ms
- 自动容错：失败时降级到MeCab

### 2. 流式响应优化

**问题**: 传统Blocking模式等待时间长，用户体验差

**解决方案**:
- 实现Server-Sent Events (SSE) 流式响应
- 逐Token返回，首Token时间(TTFB) < 1s
- 前端实时渲染，感知延迟大幅降低

### 3. 多源知识库融合

**三个独立Collection设计**:
1. **gomi**: 垃圾分类规则 (静态数据)
2. **area**: 町名收集日期 (静态数据)
3. **knowledge**: 用户上传知识 (动态扩展)

**好处**:
- 检索精度更高（针对性强）
- 支持知识库热更新
- 方便权限管理和版本控制

---

## 🚀 项目实现细节

### RAG Pipeline详解

```python
# 1. 查询理解
query = "ノートパソコンを捨てたいのですが"

# 2. Hybrid Grounding提取品名
result = hybrid_grounding.extract(query)
# → primary_candidate: "ノートパソコン"
# → confidence: "high" (0.98)
# → execution_time: 35ms

# 3. 多源检索
gomi_docs = chroma_gomi.query(embedding(候选品名), k=3)
area_docs = chroma_area.query(embedding(町名), k=2)
knowledge_docs = chroma_knowledge.query(embedding(query), k=2)

# 4. 上下文构建
context = format_rag_prompt(gomi_docs, area_docs, knowledge_docs)

# 5. LLM生成
response = ollama.generate(
    model="swallow:latest",
    prompt=context + query,
    stream=True  # 流式返回
)
```

### 数据结构设计

**ChromaDB Collection Schema**:
```python
# gomi collection
{
    "id": "gomi_001",
    "document": "ノートパソコンは粗大ごみに出してください...",
    "metadata": {
        "item_name": "ノートパソコン",
        "category": "粗大ごみ",
        "source_file": "gomi_rules.pdf",
        "page": 15
    }
}

# area collection
{
    "id": "area_001",
    "document": "八幡東区の家庭ごみ収集日は月曜日と木曜日...",
    "metadata": {
        "town": "八幡東区",
        "waste_type": "家庭ごみ",
        "collection_days": ["月", "木"]
    }
}
```

### API设计

**Blocking模式** - POST /api/bot/respond
```json
{
  "message": "ノートパソコンを捨てたいのですが",
  "user_id": "user123",
  "stream": false
}
```

**Streaming模式** - POST /api/bot/respond_stream
```
data: {"type": "token", "content": "ノート"}
data: {"type": "token", "content": "パソコン"}
...
data: {"type": "done", "references": [...]}
```

---

## 📊 性能指标

### 响应性能
- **TTFB** (首Token时间): 800ms
- **完整响应**: 3-5秒 (取决于回答长度)
- **Hybrid Grounding**: 35-600ms
- **ChromaDB检索**: 50-150ms

### 准确性指标
- **品名识别准确率**: 92% (vs MeCab 78%)
- **分类规则准确率**: 96%
- **区域信息准确率**: 99% (结构化数据)

### 系统资源
- **VRAM占用**: 6-8GB (8B模型)
- **CPU使用**: 20-40%
- **内存占用**: 4-6GB

---

## 🛠 开发过程与挑战

### 技术挑战1: 日语品名提取不准确

**问题描述**:
- MeCab分词对长句子和复合词处理差
- 例如："使わなくなったノートパソコン" → 提取失败

**解决过程**:
1. 调研现有NER方案 → 不适合垃圾分类场景
2. 尝试纯Embedding检索 → 短语时效果好，长句差
3. 设计Hybrid方案 → 结合整体检索和LLM提取
4. 实现自动Fallback → 保证鲁棒性

**结果**: 准确率从78%提升到92%

### 技术挑战2: 流式响应卡顿

**问题**: 前端渲染卡顿，Token积压

**解决方案**:
- 后端调整Buffer大小
- 前端使用`useEffect` + `useState`异步更新
- 添加心跳检测机制

### 技术挑战3: ChromaDB性能优化

**优化措施**:
1. 分离Collection (gomi/area/knowledge)
2. 调整`n_results`参数 (3-5个)
3. 添加Metadata过滤减少无效结果
4. 使用持久化模式避免重复加载

---

## 📚 项目收获

### 技术成长
1. **RAG架构理解**: 从理论到实践，深入理解检索增强生成
2. **向量数据库**: 掌握ChromaDB的使用和优化
3. **LLM应用**: 学会Prompt工程、Streaming实现
4. **全栈开发**: FastAPI后端 + Streamlit前端的完整实现
5. **性能优化**: 识别瓶颈、量化指标、迭代改进

### 工程能力
1. **系统设计**: 模块化设计、分层架构、接口定义
2. **代码质量**: Pydantic数据验证、类型提示、异常处理
3. **文档能力**: 编写架构文档、API文档、README
4. **团队协作**: Git版本控制、Code Review、任务分工

### 业务理解
1. **需求分析**: 从用户痛点出发设计解决方案
2. **用户体验**: 流式响应、置信度提示、日志透明化
3. **可扩展性**: 支持用户上传知识库，系统可持续迭代

---

## 🎤 面试演示建议

### 演示流程 (5-8分钟)

**1. 项目背景介绍 (1分钟)**
- 垃圾分类问题的社会意义
- 系统要解决的痛点

**2. 架构展示 (2分钟)**
- 打开system_v2.md的架构图
- 讲解三层架构和数据流

**3. 核心技术亮点 (2-3分钟)**
- **重点**: 演示Hybrid Grounding System
- 对比输入："ペットボトル" vs "使わなくなったペットボトルはどう捨てますか"
- 展示置信度、响应时间等指标

**4. 现场演示 (2分钟)**
```bash
# 启动系统
cd front-streaming
streamlit run app.py
```
- 输入查询："ノートパソコンを八幡東区で捨てたい"
- 展示：Streaming响应 + Reference展示 + GPU监控

**5. 技术挑战与解决 (可选)**
- 根据面试官兴趣深入讲解某个技术点

### 可能的面试问题准备

**架构设计类**:
- Q: 为什么选择FastAPI而不是Flask?
- A: FastAPI支持异步、自动文档生成、Pydantic数据验证，更适合高性能API

- Q: ChromaDB的Collection为什么要分三个?
- A: 提高检索精度、支持独立更新、方便权限控制

**技术实现类**:
- Q: Hybrid Grounding的Fallback机制如何实现?
- A: 在try-except块中捕获异常，置信度低于阈值时自动切换到MeCab

- Q: 流式响应如何保证顺序?
- A: 使用SSE协议，后端按Token顺序生成，前端按接收顺序append

**性能优化类**:
- Q: 如何优化响应速度?
- A: 1) Hybrid分流机制 2) ChromaDB结果数限制 3) 模型量化 4) 缓存常见查询

**扩展性类**:
- Q: 如果要支持其他城市怎么办?
- A: Collection设计支持Metadata过滤，可添加city字段；前端选择器切换城市

---

## 📎 参考资料

### 项目文档
- [系统架构图](./system_v2.md)
- [后端架构文档](./backend/BACKEND_ARCHITECTURE.md)
- [API参考](./backend/API_REFERENCE.md)

### 代码仓库
- GitHub: [请填写仓库链接]

### 技术博客 (可选)
- 可以准备一篇技术博客详细介绍Hybrid Grounding System的设计

---

## ✅ 面试准备清单

- [ ] 熟练运行系统并演示
- [ ] 准备3-5个常见查询用例
- [ ] 回顾Hybrid Grounding的实现代码
- [ ] 准备架构图 (可打印或在iPad上展示)
- [ ] 熟悉ChromaDB的查询API
- [ ] 准备讲解RAG Pipeline的每个步骤
- [ ] 准备2-3个技术挑战的故事
- [ ] 了解类似产品 (如ChatPDF、Perplexity)并能对比

---

**最后提示**: 面试时重点强调**Hybrid Grounding System**的设计，这是项目的最大技术亮点。能够展现你的问题分析能力、技术选型能力和工程实现能力。

Good luck! 🍀
