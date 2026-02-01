#!/usr/bin/env python3
"""
Hybrid品名指称模块
实现双路径品名候选生成：路径A（整体Embedding）+ 路径B（LLM过滤）

本模块独立于现有RAG系统，可单独测试和验证
"""

import json
from typing import List, Dict, Tuple, Optional
import ollama
import chromadb
from dataclasses import dataclass, asdict
import time


# ========== 数据结构定义 ==========

@dataclass
class Candidate:
    """品名候选结构"""
    item_name: str          # 品名
    similarity: float       # 相似度分数
    source: str            # 来源路径 ("path_a" | "path_b" | "both")
    metadata: Dict         # 原始metadata（出し方、備考等）
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GroundingResult:
    """指称结果"""
    candidates: List[Candidate]     # 候选列表（按置信度排序）
    primary_candidate: Optional[Candidate]  # 主候选（置信度最高）
    is_ambiguous: bool             # 是否存在歧义
    confidence_level: str          # 置信度级别 ("high" | "medium" | "low")
    execution_time_ms: float       # 执行耗时
    path_used: str                # 使用的路径 ("both" | "path_a_only" | "degraded")
    
    def to_dict(self):
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "primary_candidate": self.primary_candidate.to_dict() if self.primary_candidate else None,
            "is_ambiguous": self.is_ambiguous,
            "confidence_level": self.confidence_level,
            "execution_time_ms": self.execution_time_ms,
            "path_used": self.path_used
        }


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
        
        print(f"  路径A: 返回{len(candidates)}个候选")
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
            print(f"  路径B: LLM未提取到候选短语")
            return []
        
        print(f"  路径B: LLM提取到{len(extracted_phrases)}个短语: {extracted_phrases}")
        
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
        
        result_candidates = unique_candidates[:top_k]
        print(f"  路径B: 返回{len(result_candidates)}个候选（去重后）")
        
        return result_candidates
        
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
        
        # Case 3: 単に```で囲まれている場合
        if "```" in content:
            json_str = content.split("```")[1].strip()
            data = json.loads(json_str)
            return data.get("candidates", [])[:max_candidates]
        
        # Case 4: 解析失敗
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
            print(f"  合并: {item_name} 双路径命中 -> 提升分数至 {final_score:.3f}")
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
        # 完整路径：双路径串行执行
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
            print(f"\n合并候选:")
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
        lines.append(f"\n✅ 主候选: {result.primary_candidate.item_name}")
        lines.append(f"   出し方: {result.primary_candidate.metadata.get('出し方', 'N/A')}")
    
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)


# ========== 命令行接口 ==========

if __name__ == "__main__":
    import sys
    import os
    
    # 添加rag目录到路径
    sys.path.append(os.path.dirname(__file__))
    
    if len(sys.argv) < 2:
        print("使用方法: python hybrid_grounding.py <用户输入>")
        print('示例: python hybrid_grounding.py "ノートパソコンを捨てたい"')
        sys.exit(1)
    
    user_input = " ".join(sys.argv[1:])
    
    # 加载collection
    from rag_demo3 import load_jsonl, build_chroma
    
    print("初始化ChromaDB...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    
    print(f"\n处理输入: {user_input}\n")
    
    # 执行Hybrid指称
    result = hybrid_grounding(user_input, gomi_collection, force_full_path=True)
    
    # 打印结果
    print(format_grounding_result(result))
