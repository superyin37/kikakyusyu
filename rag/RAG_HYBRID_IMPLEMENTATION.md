# Hybrid品名指称系统 - 详细开发实施文档

## 文档信息
- **版本**: v1.0
- **日期**: 2026-02-01
- **状态**: 开发指南
- **负责人**: 开发团队

---

## 目录
1. [实施概述](#1-实施概述)
2. [开发环境准备](#2-开发环境准备)
3. [核心模块实现](#3-核心模块实现)
4. [集成与测试](#4-集成与测试)
5. [性能优化](#5-性能优化)
6. [部署与监控](#6-部署与监控)
7. [故障排查](#7-故障排查)

---

## 1. 实施概述

### 1.1 实施目标
在现有RAG系统中，将基于MeCab的品名抽取逻辑替换为Hybrid双路径方案，提升品名识别准确率从70-80%至85-95%。

### 1.2 实施范围

**修改的文件**：
- `rag/rag_demo3.py` - 添加Hybrid品名识别模块
- `backend/app.py` - 可选：添加性能监控

**新增的文件**：
- `rag/hybrid_grounding.py` - 核心实现
- `rag/test_hybrid.py` - 单元测试
- `rag/benchmark_hybrid.py` - 性能基准测试

**不修改的部分**：
- ChromaDB collection结构
- API接口定义（schemas.py）
- 前端WebUI
- 町名处理逻辑

### 1.3 实施时间表

| 阶段 | 任务 | 预计时间 | 交付物 |
|-----|------|---------|--------|
| Phase 1 | 路径A实现与验证 | 2天 | `path_a_baseline()` |
| Phase 2 | 路径B实现与集成 | 3天 | `path_b_llm_filter()` |
| Phase 3 | 合并逻辑与优化 | 2天 | `merge_candidates()` |
| Phase 4 | 集成测试与调优 | 2天 | 测试报告 |
| Phase 5 | 生产部署与监控 | 1天 | 部署文档 |

**总计**: 10工作日（2周）

---

## 2. 开发环境准备

### 2.1 依赖检查

确认以下组件已安装并运行：

```bash
# 检查Ollama服务
ollama list

# 应包含以下模型：
# - swallow:latest (LLM)
# - kun432/cl-nagoya-ruri-large:337m (Embedding)

# 检查ChromaDB
python -c "import chromadb; print('ChromaDB OK')"

# 检查MeCab（保留用于降级）
python -c "import MeCab; print('MeCab OK')"
```

### 2.2 测试数据准备

创建测试数据集 `rag/test_inputs.json`：

```json
{
  "short_inputs": [
    "ノートパソコンを捨てたい",
    "冷蔵庫",
    "プラスチック製の収納箱"
  ],
  "long_inputs": [
    "引っ越しで使わなくなったノートパソコンと古いプリンター、それから壊れた電子レンジを処分したいのですが、どうすればいいですか？",
    "子供が大きくなったので、ベビーカーやベビーベッド、チャイルドシートなどの育児用品をまとめて捨てたいです。"
  ],
  "ambiguous_inputs": [
    "パソコン関連の機器",
    "家電製品"
  ]
}
```

---

## 3. 核心模块实现

### 3.1 创建核心模块文件

**文件路径**: `rag/hybrid_grounding.py`

```python
#!/usr/bin/env python3
"""
Hybrid品名指称模块
实现双路径品名候选生成：路径A（整体Embedding）+ 路径B（LLM过滤）
"""

import json
from typing import List, Dict, Tuple, Optional
import ollama
import chromadb
from dataclasses import dataclass
import time


# ========== 数据结构定义 ==========

@dataclass
class Candidate:
    """品名候选结构"""
    item_name: str          # 品名
    similarity: float       # 相似度分数
    source: str            # 来源路径 ("path_a" | "path_b")
    metadata: Dict         # 原始metadata（出し方、備考等）
    

@dataclass
class GroundingResult:
    """指称结果"""
    candidates: List[Candidate]     # 候选列表（按置信度排序）
    primary_candidate: Optional[Candidate]  # 主候选（置信度最高）
    is_ambiguous: bool             # 是否存在歧义
    confidence_level: str          # 置信度级别 ("high" | "medium" | "low")
    execution_time_ms: float       # 执行耗时
    path_used: str                # 使用的路径 ("both" | "path_a_only" | "degraded")


# ========== 配置常量 ==========

class HybridConfig:
    """Hybrid系统配置"""
    
    # 路径A参数
    PATH_A_TOP_K = 3
    
    # 路径B参数
    PATH_B_MAX_CANDIDATES = 5
    PATH_B_TOP_K = 3
    PATH_B_TIMEOUT = 5  # 秒
    
    # 快速路径阈值
    SHORT_INPUT_THRESHOLD = 20  # 字符数
    
    # 置信度阈值
    CONFIDENCE_THRESHOLD_HIGH = 0.45   # 高置信度阈值
    CONFIDENCE_THRESHOLD_LOW = 0.30    # 低置信度阈值
    AMBIGUITY_THRESHOLD = 0.05         # 歧义判定阈值（Top1与Top2差值）
    
    # LLM配置
    LLM_MODEL = "swallow:latest"
    LLM_TEMPERATURE = 0.1  # 低温度以提高稳定性


# ========== 路径A：整体Embedding指称 ==========

def path_a_global_embedding(
    user_input: str,
    gomi_collection: chromadb.Collection,
    top_k: int = HybridConfig.PATH_A_TOP_K
) -> List[Candidate]:
    """
    路径A：对用户输入全文进行Embedding，直接匹配垃圾品名
    
    Args:
        user_input: 用户输入文本
        gomi_collection: ChromaDB垃圾分类collection
        top_k: 返回的候选数量
        
    Returns:
        候选列表（Candidate对象）
    """
    try:
        results = gomi_collection.query(
            query_texts=[user_input],
            n_results=top_k
        )
        
        candidates = []
        if results and results["metadatas"] and results["distances"]:
            for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
                # ChromaDB返回的是距离，转换为相似度 (similarity = 1 - distance)
                similarity = 1.0 - distance
                
                candidates.append(Candidate(
                    item_name=meta.get("品名", ""),
                    similarity=similarity,
                    source="path_a",
                    metadata=meta
                ))
        
        return candidates
        
    except Exception as e:
        print(f"⚠️ 路径A执行失败: {e}")
        return []


# ========== 路径B：LLM候选过滤 ==========

def path_b_llm_filter(
    user_input: str,
    gomi_collection: chromadb.Collection,
    max_candidates: int = HybridConfig.PATH_B_MAX_CANDIDATES,
    top_k: int = HybridConfig.PATH_B_TOP_K
) -> List[Candidate]:
    """
    路径B：使用LLM从输入中提取可能的垃圾品名短语，然后分别Embedding匹配
    
    Args:
        user_input: 用户输入文本
        gomi_collection: ChromaDB垃圾分类collection
        max_candidates: LLM最多提取的候选短语数
        top_k: 每个候选短语匹配的Top-K数量
        
    Returns:
        候选列表（Candidate对象）
    """
    try:
        # Step 1: LLM提取候选短语
        extracted_phrases = _extract_phrases_with_llm(user_input, max_candidates)
        
        if not extracted_phrases:
            return []
        
        # Step 2: 对每个候选短语进行Embedding匹配
        all_candidates = []
        
        for phrase in extracted_phrases:
            results = gomi_collection.query(
                query_texts=[phrase],
                n_results=top_k
            )
            
            if results and results["metadatas"] and results["distances"]:
                for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
                    similarity = 1.0 - distance
                    
                    all_candidates.append(Candidate(
                        item_name=meta.get("品名", ""),
                        similarity=similarity,
                        source=f"path_b:{phrase}",  # 记录来源短语
                        metadata=meta
                    ))
        
        # Step 3: 去重并排序
        unique_candidates = _deduplicate_candidates(all_candidates)
        unique_candidates.sort(key=lambda c: c.similarity, reverse=True)
        
        return unique_candidates[:top_k]
        
    except Exception as e:
        print(f"⚠️ 路径B执行失败: {e}")
        return []


def _extract_phrases_with_llm(
    user_input: str,
    max_candidates: int
) -> List[str]:
    """
    使用LLM从用户输入中提取可能的垃圾品名短语
    
    Returns:
        候选短语列表（最多max_candidates个）
    """
    prompt = f"""あなたは北九州市のごみ分類システムです。以下のユーザー入力から、捨てたい物品の名称を抽出してください。

【重要ルール】
1. 物品名のみを抽出（説明文や動詞は含めない）
2. 最大{max_candidates}個まで
3. JSON形式で出力: {{"candidates": ["物品1", "物品2"]}}
4. 候補がない場合: {{"candidates": []}}

【入力】
{user_input}

【出力】（JSON形式のみ、説明不要）
"""

    try:
        response = ollama.chat(
            model=HybridConfig.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは物品名抽出の専門システムです。JSON形式で回答してください。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": HybridConfig.LLM_TEMPERATURE
            }
        )
        
        content = response["message"]["content"]
        
        # JSON解析（複数の形式に対応）
        # Case 1: 直接JSON
        if content.strip().startswith("{"):
            data = json.loads(content)
            return data.get("candidates", [])[:max_candidates]
        
        # Case 2: Markdown code block内のJSON
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
            return data.get("candidates", [])[:max_candidates]
        
        # Case 3: 解析失敗
        print(f"⚠️ LLM出力の解析失敗: {content[:100]}")
        return []
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析エラー: {e}")
        return []
    except Exception as e:
        print(f"⚠️ LLM呼び出しエラー: {e}")
        return []


def _deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """
    候補の重複除去（同じ品名の場合、最高スコアを保持）
    """
    seen = {}
    for c in candidates:
        if c.item_name not in seen or c.similarity > seen[c.item_name].similarity:
            seen[c.item_name] = c
    
    return list(seen.values())


# ========== 候補合併と決策 ==========

def merge_candidates(
    candidates_a: List[Candidate],
    candidates_b: List[Candidate]
) -> List[Candidate]:
    """
    路径A和路径B的候选进行合并、去重、重排序
    
    策略：
    1. 合并两个列表
    2. 相同品名的候选，权重提升（取平均分+0.1）
    3. 按最终分数排序
    """
    all_candidates = {}
    
    # 处理路径A候选
    for c in candidates_a:
        all_candidates[c.item_name] = {
            "candidate": c,
            "sources": ["path_a"],
            "scores": [c.similarity]
        }
    
    # 处理路径B候选（如果品名已存在，标记为双路径命中）
    for c in candidates_b:
        if c.item_name in all_candidates:
            # 双路径命中 - 提升置信度
            all_candidates[c.item_name]["sources"].append("path_b")
            all_candidates[c.item_name]["scores"].append(c.similarity)
        else:
            all_candidates[c.item_name] = {
                "candidate": c,
                "sources": ["path_b"],
                "scores": [c.similarity]
            }
    
    # 计算最终分数
    merged = []
    for item_name, data in all_candidates.items():
        candidate = data["candidate"]
        
        # 双路径命中：取平均分+0.1 bonus
        if len(data["sources"]) > 1:
            final_score = sum(data["scores"]) / len(data["scores"]) + 0.1
            candidate.source = "both"
        else:
            final_score = data["scores"][0]
        
        # 更新分数
        candidate.similarity = min(final_score, 1.0)  # 确保不超过1.0
        merged.append(candidate)
    
    # 按分数排序
    merged.sort(key=lambda c: c.similarity, reverse=True)
    
    return merged


# ========== 主函数：Hybrid品名指称 ==========

def hybrid_grounding(
    user_input: str,
    gomi_collection: chromadb.Collection,
    force_full_path: bool = False
) -> GroundingResult:
    """
    Hybrid品名指称主函数
    
    Args:
        user_input: 用户输入
        gomi_collection: ChromaDB垃圾分类collection
        force_full_path: 强制执行完整双路径（用于测试）
        
    Returns:
        GroundingResult对象
    """
    start_time = time.perf_counter()
    
    # 决定是否使用快速路径
    use_fast_path = (
        not force_full_path and 
        len(user_input) < HybridConfig.SHORT_INPUT_THRESHOLD
    )
    
    if use_fast_path:
        # 快速路径：仅路径A
        print(f"🚀 快速路径（输入长度: {len(user_input)}字符）")
        candidates_a = path_a_global_embedding(user_input, gomi_collection)
        final_candidates = candidates_a
        path_used = "path_a_only"
    else:
        # 完整路径：双路径并行（简化版串行）
        print(f"🔍 完整路径（输入长度: {len(user_input)}字符）")
        
        # 路径A
        candidates_a = path_a_global_embedding(user_input, gomi_collection)
        
        # 路径B（带降级）
        try:
            candidates_b = path_b_llm_filter(user_input, gomi_collection)
        except Exception as e:
            print(f"⚠️ 路径B失败，降级到路径A: {e}")
            candidates_b = []
        
        # 合并候选
        if candidates_b:
            final_candidates = merge_candidates(candidates_a, candidates_b)
            path_used = "both"
        else:
            final_candidates = candidates_a
            path_used = "degraded"
    
    # 置信度评估
    confidence_level, is_ambiguous = _evaluate_confidence(final_candidates)
    
    # 构建结果
    end_time = time.perf_counter()
    
    result = GroundingResult(
        candidates=final_candidates,
        primary_candidate=final_candidates[0] if final_candidates else None,
        is_ambiguous=is_ambiguous,
        confidence_level=confidence_level,
        execution_time_ms=(end_time - start_time) * 1000,
        path_used=path_used
    )
    
    return result


def _evaluate_confidence(candidates: List[Candidate]) -> Tuple[str, bool]:
    """
    评估置信度级别
    
    Returns:
        (confidence_level, is_ambiguous)
    """
    if not candidates:
        return "low", False
    
    top_score = candidates[0].similarity
    is_ambiguous = False
    
    # 检查歧义（Top1和Top2差值小）
    if len(candidates) >= 2:
        score_diff = top_score - candidates[1].similarity
        is_ambiguous = score_diff < HybridConfig.AMBIGUITY_THRESHOLD
    
    # 置信度分级
    if top_score >= HybridConfig.CONFIDENCE_THRESHOLD_HIGH:
        confidence_level = "high"
    elif top_score >= HybridConfig.CONFIDENCE_THRESHOLD_LOW:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    return confidence_level, is_ambiguous


# ========== 工具函数 ==========

def format_grounding_result(result: GroundingResult) -> str:
    """
    格式化指称结果用于日志输出
    """
    lines = [
        f"\n{'='*60}",
        f"Hybrid Grounding 结果",
        f"{'='*60}",
        f"执行耗时: {result.execution_time_ms:.2f}ms",
        f"路径使用: {result.path_used}",
        f"置信度: {result.confidence_level}",
        f"是否歧义: {result.is_ambiguous}",
        f"\n候选列表 (共{len(result.candidates)}个):"
    ]
    
    for i, c in enumerate(result.candidates, 1):
        lines.append(
            f"  {i}. {c.item_name} (score: {c.similarity:.3f}, source: {c.source})"
        )
    
    if result.primary_candidate:
        lines.append(f"\n主候选: {result.primary_candidate.item_name}")
    
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)
```

---

### 3.2 集成到现有RAG流程

**修改文件**: `rag/rag_demo3.py`

在文件开头添加导入：

```python
# 在现有import之后添加
from hybrid_grounding import (
    hybrid_grounding,
    format_grounding_result,
    HybridConfig
)
```

修改 `rag_retrieve_extended` 函数：

```python
def rag_retrieve_extended(
    user_input,
    gomi_collection,
    area_collection,
    known_items,  # 保留但不再使用
    area_meta,
    knowledge_collection=None,
    known_areas=AREAS,
    top_k=3,
    use_hybrid=True  # 新增开关
):
    context_parts = []
    references = []

    # ========= 新：使用Hybrid品名识别 =========
    if use_hybrid:
        grounding_result = hybrid_grounding(user_input, gomi_collection)
        
        # 调试输出
        print(format_grounding_result(grounding_result))
        
        # 根据置信度决定是否使用候选
        if grounding_result.confidence_level in ["high", "medium"]:
            primary = grounding_result.primary_candidate
            
            # 如果存在歧义，在context中添加其他候选
            if grounding_result.is_ambiguous and len(grounding_result.candidates) > 1:
                items_text = " / ".join([
                    c.item_name for c in grounding_result.candidates[:3]
                ])
                context_parts.append(
                    f"【注意】以下の品名候補が検出されました: {items_text}\n"
                    f"最も関連性が高いのは「{primary.item_name}」です。\n"
                )
            
            # 构建垃圾分类信息
            gomi_context = (
                f"品名: {primary.metadata.get('品名','')}\n"
                f"出し方: {primary.metadata.get('出し方','')}\n"
                f"備考: {primary.metadata.get('備考','')}"
            )
            context_parts.append(f"【ごみ分別情報】\n{gomi_context}")
            
        else:
            # 低置信度 - 降级到知识库搜索
            context_parts.append(
                "【注意】特定のごみ品名が識別できませんでした。"
                "一般的な情報で回答します。\n"
            )
    else:
        # 原有MeCab逻辑（降级路径）
        keys = extract_keywords(user_input, known_items, known_areas)
        if keys["品名"]:
            gomi_hits = query_chroma(gomi_collection, keys["品名"], n=top_k)
            # ... 原有逻辑
    
    # ========= 町名检索（保持不变） =========
    keys = extract_keywords(user_input, known_items=[], known_areas=known_areas)
    if keys["町名"] and area_meta:
        matched = [h for h in area_meta if h.get("町名") == keys["町名"]]
        if matched:
            formatted = []
            for h in matched:
                formatted.append(
                    f"{h.get('町名','不明')} の収集情報:\n"
                    f"- 家庭ごみ: {h.get('家庭ごみの収集日','不明')}\n"
                    f"- プラスチック: {h.get('プラスチックの収集日','不明')}\n"
                    f"- 粗大ごみ: {h.get('粗大ごみの収集日（事前申込制）','不明')}"
                )
            context_parts.append("【町名情報】\n" + "\n\n".join(formatted))
    
    # ========= 知识库检索（保持不变） =========
    if knowledge_collection:
        knowledge_hits = query_chroma(knowledge_collection, user_input, n=top_k)
        if knowledge_hits:
            for h in knowledge_hits[:2]:
                references.append({
                    "file": h.get("file", "?"),
                    "page": h.get("page", "?"),
                    "chunk": h.get("chunk", "?"),
                    "text": h.get("text", "")[:300]
                })
            knowledge_context = "\n\n".join([
                f"ファイル: {h.get('file','')}, p.{h.get('page','?')}, chunk {h.get('chunk','?')}"
                for h in knowledge_hits[:2]
            ])
            context_parts.append(f"【ユーザナレッジ情報】\n{knowledge_context}")
    
    # ========= 构建最终prompt（保持不变） =========
    context = "\n\n".join(context_parts) if context_parts else "該当情報が見つかりませんでした。"
    
    prompt = f"""
あなたは北九州市のごみ分別案内システムです。
以下に示す【ごみ分別情報】のみを唯一の事実情報として使用してください。

【ごみ分別情報】
{context}

【質問】
{user_input}

【出力形式】
- 品名
- 品名の出し方
- 備考
- 該当町名の収集日（不明な場合は「不明」と記載）
"""
    return prompt, references
```

---

### 3.3 向后兼容性开关

在 `backend/app.py` 中添加配置开关：

```python
# 在全局变量区域添加
ENABLE_HYBRID_GROUNDING = os.getenv("ENABLE_HYBRID_GROUNDING", "true").lower() == "true"

# 在API端点中传递参数
@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2,
        use_hybrid=ENABLE_HYBRID_GROUNDING  # 添加开关
    )
    # ... 其余代码不变
```

---

## 4. 集成与测试

### 4.1 单元测试

**文件路径**: `rag/test_hybrid.py`

```python
#!/usr/bin/env python3
"""
Hybrid品名指称系统单元测试
"""

import pytest
import chromadb
from hybrid_grounding import (
    hybrid_grounding,
    path_a_global_embedding,
    path_b_llm_filter,
    merge_candidates,
    HybridConfig
)
from rag_demo3 import load_jsonl, build_chroma


# ========== Fixtures ==========

@pytest.fixture(scope="module")
def gomi_collection():
    """测试用垃圾分类collection"""
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    collection = build_chroma(gomi_docs, gomi_meta, name="gomi_test")
    yield collection
    # Cleanup
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("gomi_test")
    except:
        pass


# ========== 路径A测试 ==========

def test_path_a_short_input(gomi_collection):
    """测试路径A - 短输入"""
    user_input = "ノートパソコン"
    candidates = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    
    assert len(candidates) > 0, "应返回至少1个候选"
    assert candidates[0].item_name is not None, "应返回有效品名"
    assert 0 <= candidates[0].similarity <= 1, "相似度应在0-1之间"
    
    # 验证返回的品名相关性
    top_name = candidates[0].item_name
    assert "パソコン" in top_name or "ノート" in top_name, f"品名应包含相关词汇，实际: {top_name}"


def test_path_a_long_input(gomi_collection):
    """测试路径A - 长输入"""
    user_input = "引っ越しで使わなくなったノートパソコンを処分したいです"
    candidates = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    
    assert len(candidates) > 0, "长输入也应返回候选"
    print(f"\n长输入Top-3候选: {[c.item_name for c in candidates]}")


# ========== 路径B测试 ==========

def test_path_b_extraction(gomi_collection):
    """测试路径B - LLM候选提取"""
    user_input = "古いノートパソコンとプリンターを捨てたい"
    candidates = path_b_llm_filter(user_input, gomi_collection)
    
    assert len(candidates) > 0, "应提取至少1个候选"
    
    # 检查是否包含预期品名
    item_names = [c.item_name for c in candidates]
    print(f"\n路径B提取的品名: {item_names}")
    
    has_pc = any("パソコン" in name for name in item_names)
    has_printer = any("プリンター" in name or "プリンタ" in name for name in item_names)
    
    assert has_pc or has_printer, f"应包含パソコンまたはプリンター，实际: {item_names}"


def test_path_b_timeout_handling(gomi_collection):
    """测试路径B - 超时处理"""
    # 模拟超长输入
    user_input = "x" * 1000
    
    try:
        candidates = path_b_llm_filter(user_input, gomi_collection)
        # 应能正常返回（即使为空）
        assert isinstance(candidates, list)
    except Exception as e:
        pytest.fail(f"路径B应捕获异常而非抛出: {e}")


# ========== 合并逻辑测试 ==========

def test_merge_candidates(gomi_collection):
    """测试候选合并逻辑"""
    from hybrid_grounding import Candidate
    
    # 模拟路径A候选
    candidates_a = [
        Candidate("ノートパソコン", 0.7, "path_a", {}),
        Candidate("デスクトップパソコン", 0.5, "path_a", {})
    ]
    
    # 模拟路径B候选（有重叠）
    candidates_b = [
        Candidate("ノートパソコン", 0.65, "path_b", {}),  # 重叠
        Candidate("プリンター", 0.6, "path_b", {})
    ]
    
    merged = merge_candidates(candidates_a, candidates_b)
    
    # 验证去重
    item_names = [c.item_name for c in merged]
    assert len(item_names) == len(set(item_names)), "应无重复品名"
    
    # 验证双路径命中的boost
    notebook = next((c for c in merged if c.item_name == "ノートパソコン"), None)
    assert notebook is not None
    assert notebook.source == "both", "应标记为双路径命中"
    assert notebook.similarity > 0.7, f"双路径命中应提升分数，实际: {notebook.similarity}"


# ========== 完整流程测试 ==========

def test_hybrid_full_pipeline_short(gomi_collection):
    """测试完整流程 - 短输入（快速路径）"""
    user_input = "冷蔵庫"
    result = hybrid_grounding(user_input, gomi_collection)
    
    assert result.primary_candidate is not None, "应返回主候选"
    assert result.path_used == "path_a_only", "短输入应使用快速路径"
    assert result.execution_time_ms < 200, f"快速路径应<200ms，实际: {result.execution_time_ms}ms"
    assert "冷蔵" in result.primary_candidate.item_name


def test_hybrid_full_pipeline_long(gomi_collection):
    """测试完整流程 - 长输入（双路径）"""
    user_input = "引っ越しで使わなくなったノートパソコンと古いプリンター、それから壊れた電子レンジを処分したいです"
    result = hybrid_grounding(user_input, gomi_collection, force_full_path=True)
    
    assert result.primary_candidate is not None, "应返回主候选"
    assert result.path_used in ["both", "degraded"], "长输入应尝试双路径"
    assert len(result.candidates) > 0, "应返回候选列表"
    
    print(f"\n长输入结果:")
    print(f"  路径: {result.path_used}")
    print(f"  耗时: {result.execution_time_ms:.2f}ms")
    print(f"  置信度: {result.confidence_level}")
    print(f"  主候选: {result.primary_candidate.item_name}")


def test_hybrid_ambiguity_detection(gomi_collection):
    """测试歧义检测"""
    # 构造一个可能产生歧义的输入
    user_input = "家電製品"  # 模糊输入
    result = hybrid_grounding(user_input, gomi_collection)
    
    assert len(result.candidates) > 0
    print(f"\n歧义检测:")
    print(f"  是否歧义: {result.is_ambiguous}")
    print(f"  候选: {[c.item_name for c in result.candidates[:3]]}")


def test_hybrid_low_confidence(gomi_collection):
    """测试低置信度场景"""
    user_input = "よくわからないもの"  # 无效输入
    result = hybrid_grounding(user_input, gomi_collection)
    
    if result.primary_candidate:
        assert result.confidence_level == "low", "无效输入应返回低置信度"
        print(f"\n低置信度场景: {result.primary_candidate.item_name} (score: {result.primary_candidate.similarity:.3f})")


# ========== 性能测试 ==========

def test_performance_benchmark(gomi_collection):
    """性能基准测试"""
    test_cases = [
        ("ノートパソコン", "short"),
        ("引っ越しで使わなくなったノートパソコンを処分したい", "long")
    ]
    
    results = []
    for user_input, case_type in test_cases:
        result = hybrid_grounding(user_input, gomi_collection)
        results.append({
            "type": case_type,
            "time_ms": result.execution_time_ms,
            "path": result.path_used
        })
    
    print("\n=== 性能基准测试 ===")
    for r in results:
        print(f"{r['type']:6s} | {r['time_ms']:6.2f}ms | {r['path']}")
    
    # 验证性能要求
    short_time = next(r["time_ms"] for r in results if r["type"] == "short")
    assert short_time < 200, f"短输入应<200ms，实际: {short_time}ms"


# ========== 运行测试 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

运行测试：

```bash
cd rag
python -m pytest test_hybrid.py -v -s
```

---

### 4.2 集成测试脚本

**文件路径**: `rag/benchmark_hybrid.py`

```python
#!/usr/bin/env python3
"""
Hybrid品名指称系统 - 端到端基准测试
"""

import json
import time
from pathlib import Path
from hybrid_grounding import hybrid_grounding
from rag_demo3 import load_jsonl, build_chroma
import chromadb


def load_test_cases():
    """加载测试用例"""
    test_file = Path("test_inputs.json")
    
    if not test_file.exists():
        # 创建默认测试用例
        test_data = {
            "test_cases": [
                {
                    "input": "ノートパソコンを捨てたい",
                    "expected_keywords": ["パソコン", "ノートパソコン"],
                    "type": "short"
                },
                {
                    "input": "冷蔵庫",
                    "expected_keywords": ["冷蔵庫"],
                    "type": "short"
                },
                {
                    "input": "引っ越しで使わなくなったノートパソコンと古いプリンター、それから壊れた電子レンジを処分したいです",
                    "expected_keywords": ["パソコン", "プリンター", "電子レンジ"],
                    "type": "long"
                },
                {
                    "input": "子供が大きくなったので、ベビーカーやベビーベッド、チャイルドシートを捨てたい",
                    "expected_keywords": ["ベビーカー", "ベビーベッド", "チャイルドシート"],
                    "type": "long"
                },
                {
                    "input": "プラスチック製の収納箱",
                    "expected_keywords": ["プラスチック", "収納", "箱"],
                    "type": "medium"
                }
            ]
        }
        test_file.write_text(json.dumps(test_data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return json.loads(test_file.read_text(encoding="utf-8"))


def evaluate_result(result, expected_keywords):
    """评估结果准确性"""
    if not result.primary_candidate:
        return False, "无主候选"
    
    item_name = result.primary_candidate.item_name
    
    # 检查是否包含任一预期关键词
    for keyword in expected_keywords:
        if keyword in item_name:
            return True, f"匹配: {keyword}"
    
    return False, f"未匹配: {item_name}"


def run_benchmark():
    """运行基准测试"""
    print("=" * 80)
    print("Hybrid品名指称系统 - 基准测试")
    print("=" * 80)
    
    # 初始化collection
    print("\n初始化ChromaDB...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    
    # 加载测试用例
    test_data = load_test_cases()
    test_cases = test_data["test_cases"]
    
    print(f"\n测试用例数: {len(test_cases)}")
    print("-" * 80)
    
    # 执行测试
    results = []
    
    for i, case in enumerate(test_cases, 1):
        user_input = case["input"]
        expected = case["expected_keywords"]
        case_type = case.get("type", "unknown")
        
        print(f"\n[{i}/{len(test_cases)}] {case_type.upper()}")
        print(f"输入: {user_input[:60]}...")
        
        # 执行Hybrid指称
        result = hybrid_grounding(user_input, gomi_collection)
        
        # 评估结果
        is_correct, reason = evaluate_result(result, expected)
        
        # 记录结果
        results.append({
            "input": user_input,
            "type": case_type,
            "primary_candidate": result.primary_candidate.item_name if result.primary_candidate else None,
            "similarity": result.primary_candidate.similarity if result.primary_candidate else 0,
            "confidence": result.confidence_level,
            "is_ambiguous": result.is_ambiguous,
            "execution_time_ms": result.execution_time_ms,
            "path_used": result.path_used,
            "is_correct": is_correct,
            "reason": reason
        })
        
        # 打印结果
        status = "✅" if is_correct else "❌"
        print(f"{status} 主候选: {result.primary_candidate.item_name if result.primary_candidate else 'None'}")
        print(f"   相似度: {result.primary_candidate.similarity:.3f if result.primary_candidate else 0}")
        print(f"   置信度: {result.confidence_level}")
        print(f"   路径: {result.path_used}")
        print(f"   耗时: {result.execution_time_ms:.2f}ms")
        print(f"   评估: {reason}")
    
    # 统计总结
    print("\n" + "=" * 80)
    print("统计总结")
    print("=" * 80)
    
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total * 100 if total > 0 else 0
    
    avg_time = sum(r["execution_time_ms"] for r in results) / total
    
    # 按类型统计
    type_stats = {}
    for r in results:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "correct": 0, "time": []}
        type_stats[t]["total"] += 1
        if r["is_correct"]:
            type_stats[t]["correct"] += 1
        type_stats[t]["time"].append(r["execution_time_ms"])
    
    print(f"\n总体准确率: {accuracy:.1f}% ({correct}/{total})")
    print(f"平均耗时: {avg_time:.2f}ms")
    
    print("\n按类型统计:")
    for t, stats in type_stats.items():
        t_accuracy = stats["correct"] / stats["total"] * 100
        t_avg_time = sum(stats["time"]) / len(stats["time"])
        print(f"  {t:8s}: {t_accuracy:5.1f}% ({stats['correct']}/{stats['total']}) | "
              f"平均耗时: {t_avg_time:6.2f}ms")
    
    # 路径使用统计
    path_counts = {}
    for r in results:
        path = r["path_used"]
        path_counts[path] = path_counts.get(path, 0) + 1
    
    print("\n路径使用统计:")
    for path, count in path_counts.items():
        print(f"  {path:15s}: {count:2d} ({count/total*100:.1f}%)")
    
    # 置信度分布
    confidence_counts = {}
    for r in results:
        conf = r["confidence"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    print("\n置信度分布:")
    for conf in ["high", "medium", "low"]:
        count = confidence_counts.get(conf, 0)
        print(f"  {conf:8s}: {count:2d} ({count/total*100:.1f}%)")
    
    # 保存详细结果
    output_file = Path("benchmark_results.json")
    output_file.write_text(
        json.dumps({
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "avg_time_ms": avg_time
            },
            "results": results
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n详细结果已保存到: {output_file}")
    print("=" * 80)
    
    return accuracy >= 85  # 目标准确率85%


if __name__ == "__main__":
    success = run_benchmark()
    exit(0 if success else 1)
```

运行基准测试：

```bash
cd rag
python benchmark_hybrid.py
```

---

## 5. 性能优化

### 5.1 并行执行优化

修改 `hybrid_grounding.py` 中的 `hybrid_grounding` 函数，实现路径A和路径B的并行执行：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def hybrid_grounding(
    user_input: str,
    gomi_collection: chromadb.Collection,
    force_full_path: bool = False,
    enable_parallel: bool = True  # 新增开关
) -> GroundingResult:
    """
    Hybrid品名指称主函数（优化版：支持并行执行）
    """
    start_time = time.perf_counter()
    
    use_fast_path = (
        not force_full_path and 
        len(user_input) < HybridConfig.SHORT_INPUT_THRESHOLD
    )
    
    if use_fast_path:
        # 快速路径（不变）
        candidates_a = path_a_global_embedding(user_input, gomi_collection)
        final_candidates = candidates_a
        path_used = "path_a_only"
    else:
        # 完整路径：并行执行
        if enable_parallel:
            candidates_a = []
            candidates_b = []
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(
                    path_a_global_embedding, user_input, gomi_collection
                )
                future_b = executor.submit(
                    path_b_llm_filter, user_input, gomi_collection
                )
                
                # 收集结果
                for future in as_completed([future_a, future_b]):
                    try:
                        if future == future_a:
                            candidates_a = future.result()
                        else:
                            candidates_b = future.result()
                    except Exception as e:
                        print(f"⚠️ 并行执行错误: {e}")
                        if future == future_b:
                            candidates_b = []
        else:
            # 串行执行（原逻辑）
            candidates_a = path_a_global_embedding(user_input, gomi_collection)
            try:
                candidates_b = path_b_llm_filter(user_input, gomi_collection)
            except:
                candidates_b = []
        
        # 合并候选
        if candidates_b:
            final_candidates = merge_candidates(candidates_a, candidates_b)
            path_used = "both"
        else:
            final_candidates = candidates_a
            path_used = "degraded"
    
    # ... 其余代码不变
```

**预期性能提升**：并行执行可将完整路径的延迟从~450ms降低至~300ms（约33%提升）。

---

### 5.2 缓存优化

添加LRU缓存以避免重复计算：

```python
from functools import lru_cache
import hashlib

def _make_cache_key(user_input: str) -> str:
    """生成缓存键"""
    return hashlib.md5(user_input.encode()).hexdigest()

# 添加缓存装饰器（仅用于路径A，路径B因LLM不稳定性不建议缓存）
@lru_cache(maxsize=128)
def _cached_path_a(input_hash: str, collection_name: str, top_k: int):
    """缓存路径A的结果（内部函数）"""
    # 实际调用通过外部函数传入collection
    pass

# 在path_a_global_embedding中集成缓存
# （详细实现略，需考虑collection对象无法直接缓存的问题）
```

---

## 6. 部署与监控

### 6.1 部署检查清单

```bash
# 1. 确认Ollama服务运行
ollama list | grep swallow
ollama list | grep kun432

# 2. 确认ChromaDB数据完整
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
print('gomi:', client.get_collection('gomi').count())
"

# 3. 运行单元测试
cd rag && python -m pytest test_hybrid.py -v

# 4. 运行基准测试
python benchmark_hybrid.py

# 5. 环境变量配置
export ENABLE_HYBRID_GROUNDING=true

# 6. 启动后端服务
cd backend && python -m uvicorn app:app --reload --port 8000

# 7. 启动前端
cd front-streaming && streamlit run app.py
```

### 6.2 性能监控指标

在 `backend/app.py` 中添加监控日志：

```python
import time

@app.post("/api/bot/respond")
async def rag_respond(req: PromptRequest):
    t_start = time.perf_counter()
    
    rag_prompt, references = rag_retrieve_extended(
        req.prompt,
        gomi_collection,
        knowledge_collection=knowledge_collection,
        area_collection=area_collection,
        known_items=known_items,
        area_meta=area_meta,
        top_k=2,
        use_hybrid=ENABLE_HYBRID_GROUNDING
    )
    
    t_rag_end = time.perf_counter()
    
    reply = ask_ollama(rag_prompt)
    
    t_end = time.perf_counter()
    
    # 记录性能指标
    metrics = {
        "rag_time_ms": (t_rag_end - t_start) * 1000,
        "llm_time_ms": (t_end - t_rag_end) * 1000,
        "total_time_ms": (t_end - t_start) * 1000,
        "hybrid_enabled": ENABLE_HYBRID_GROUNDING
    }
    
    print(f"⏱️ Metrics: {metrics}")
    
    return {
        "reply": reply,
        "references": references,
        "metrics": metrics  # 可选：返回给前端
    }
```

### 6.3 降级策略配置

创建配置文件 `backend/hybrid_config.yaml`：

```yaml
hybrid_grounding:
  enabled: true
  
  # 快速路径阈值
  short_input_threshold: 20
  
  # 并行执行
  enable_parallel: true
  
  # 超时配置
  path_b_timeout_seconds: 5
  
  # 置信度阈值
  confidence:
    high: 0.45
    low: 0.30
    ambiguity: 0.05
  
  # 降级策略
  degradation:
    # 路径B失败N次后自动禁用（重启恢复）
    max_path_b_failures: 10
    
    # 平均延迟超过阈值时禁用路径B
    max_avg_latency_ms: 600
```

---

## 7. 故障排查

### 7.1 常见问题与解决方案

| 问题 | 症状 | 排查步骤 | 解决方案 |
|-----|------|---------|---------|
| 路径B返回空列表 | LLM未提取到候选 | 1. 检查Ollama服务<br>2. 查看LLM输出日志 | 1. 调整LLM prompt<br>2. 降低temperature |
| 置信度普遍偏低 | 所有输入confidence=low | 检查Embedding模型加载 | 1. 重新加载模型<br>2. 调整阈值 |
| 延迟过高 | 响应时间>1s | 1. 检查并行执行开关<br>2. 查看GPU占用 | 1. 启用并行<br>2. 优化batch size |
| 路径A和路径B结果差异大 | merge后结果混乱 | 打印两个路径的候选 | 调整merge权重策略 |

### 7.2 调试工具

创建调试脚本 `rag/debug_hybrid.py`：

```python
#!/usr/bin/env python3
"""
Hybrid系统调试工具
"""

import sys
from hybrid_grounding import hybrid_grounding, format_grounding_result
from rag_demo3 import load_jsonl, build_chroma

def debug_input(user_input: str):
    """调试单个输入"""
    print(f"\n{'='*80}")
    print(f"调试输入: {user_input}")
    print(f"{'='*80}")
    
    # 加载collection
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    
    # 执行Hybrid指称（强制完整路径）
    result = hybrid_grounding(
        user_input,
        gomi_collection,
        force_full_path=True
    )
    
    # 打印详细结果
    print(format_grounding_result(result))
    
    # 返回候选详情
    print("\n详细候选信息:")
    for i, c in enumerate(result.candidates, 1):
        print(f"\n  候选 {i}:")
        print(f"    品名: {c.item_name}")
        print(f"    相似度: {c.similarity:.4f}")
        print(f"    来源: {c.source}")
        print(f"    出し方: {c.metadata.get('出し方', 'N/A')[:50]}...")
        print(f"    備考: {c.metadata.get('備考', 'N/A')[:50]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python debug_hybrid.py <用户输入>")
        print('示例: python debug_hybrid.py "ノートパソコンを捨てたい"')
        sys.exit(1)
    
    user_input = " ".join(sys.argv[1:])
    debug_input(user_input)
```

使用方法：

```bash
cd rag
python debug_hybrid.py "引っ越しで使わなくなったノートパソコンを処分したい"
```

---

## 8. 附录

### 8.1 性能基准参考

| 场景 | 输入类型 | 预期延迟 | 预期准确率 |
|-----|---------|---------|-----------|
| 短指令 | <20字符 | <150ms | >90% |
| 中等输入 | 20-50字符 | <300ms | >85% |
| 长文本 | >50字符 | <500ms | >80% |

### 8.2 相关文档

- [RAG_IMPROVEMENT_0201.md](RAG_IMPROVEMENT_0201.md) - 设计方案
- [RAG_SYSTEM_ARCHITECTURE.md](RAG_SYSTEM_ARCHITECTURE.md) - 系统架构
- [BACKEND_ARCHITECTURE.md](../backend/BACKEND_ARCHITECTURE.md) - 后端设计

### 8.3 变更日志

| 日期 | 版本 | 变更内容 |
|-----|------|---------|
| 2026-02-01 | v1.0 | 初始版本 |

---

**文档维护**: 开发团队  
**最后更新**: 2026-02-01
