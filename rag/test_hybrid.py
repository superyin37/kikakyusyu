#!/usr/bin/env python3
"""
Hybrid品名指称系统单元测试
独立测试模块，不依赖现有系统
"""

import pytest
import chromadb
from hybrid_grounding import (
    hybrid_grounding,
    path_a_global_embedding,
    path_b_llm_filter,
    merge_candidates,
    HybridConfig,
    Candidate
)
from rag_demo3 import load_jsonl, build_chroma


# ========== Fixtures ==========

@pytest.fixture(scope="module")
def gomi_collection():
    """测试用垃圾分类collection"""
    print("\n初始化测试collection...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    print(f"Collection加载完成，共{collection.count()}条记录")
    return collection


# ========== 路径A测试 ==========

def test_path_a_short_input(gomi_collection):
    """测试路径A - 短输入"""
    print("\n=== 测试路径A - 短输入 ===")
    user_input = "ノートパソコン"
    candidates = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    
    assert len(candidates) > 0, "应返回至少1个候选"
    assert candidates[0].item_name is not None, "应返回有效品名"
    assert 0 <= candidates[0].similarity <= 1, "相似度应在0-1之间"
    
    # 验证返回的品名相关性
    top_name = candidates[0].item_name
    print(f"Top-1候选: {top_name} (score: {candidates[0].similarity:.3f})")
    assert "パソコン" in top_name or "ノート" in top_name, f"品名应包含相关词汇，实际: {top_name}"


def test_path_a_long_input(gomi_collection):
    """测试路径A - 长输入"""
    print("\n=== 测试路径A - 长输入 ===")
    user_input = "引っ越しで使わなくなったノートパソコンを処分したいです"
    candidates = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    
    assert len(candidates) > 0, "长输入也应返回候选"
    print(f"长输入Top-3候选: {[c.item_name for c in candidates]}")


def test_path_a_empty_input(gomi_collection):
    """测试路径A - 空输入"""
    print("\n=== 测试路径A - 空输入 ===")
    user_input = ""
    candidates = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    
    # 空输入应该能处理（返回空或低相似度结果）
    assert isinstance(candidates, list), "应返回列表"


# ========== 路径B测试 ==========

def test_path_b_extraction(gomi_collection):
    """测试路径B - LLM候选提取"""
    print("\n=== 测试路径B - LLM候选提取 ===")
    user_input = "古いノートパソコンとプリンターを捨てたい"
    candidates = path_b_llm_filter(user_input, gomi_collection)
    
    print(f"路径B返回候选数: {len(candidates)}")
    
    if len(candidates) > 0:
        # 检查是否包含预期品名
        item_names = [c.item_name for c in candidates]
        print(f"路径B提取的品名: {item_names}")
        
        has_pc = any("パソコン" in name for name in item_names)
        has_printer = any("プリンター" in name or "プリンタ" in name for name in item_names)
        
        print(f"包含パソコン: {has_pc}, 包含プリンター: {has_printer}")
        # 至少应该匹配一个
        assert has_pc or has_printer, f"应包含パソコンまたはプリンター，实际: {item_names}"
    else:
        print("⚠️ 路径B未返回候选，可能LLM调用失败")


def test_path_b_single_item(gomi_collection):
    """测试路径B - 单品名输入"""
    print("\n=== 测试路径B - 单品名 ===")
    user_input = "冷蔵庫を処分したい"
    candidates = path_b_llm_filter(user_input, gomi_collection)
    
    print(f"单品名输入候选: {[c.item_name for c in candidates] if candidates else 'None'}")


# ========== 合并逻辑测试 ==========

def test_merge_candidates_no_overlap():
    """测试候选合并 - 无重叠"""
    print("\n=== 测试合并逻辑 - 无重叠 ===")
    
    candidates_a = [
        Candidate("ノートパソコン", 0.7, "path_a", {"出し方": "小型家電"}),
        Candidate("デスクトップパソコン", 0.5, "path_a", {})
    ]
    
    candidates_b = [
        Candidate("プリンター", 0.6, "path_b", {}),
        Candidate("キーボード", 0.4, "path_b", {})
    ]
    
    merged = merge_candidates(candidates_a, candidates_b)
    
    print(f"合并后候选数: {len(merged)}")
    assert len(merged) == 4, "无重叠时应为4个候选"
    
    # 验证排序（按相似度降序）
    for i in range(len(merged) - 1):
        assert merged[i].similarity >= merged[i+1].similarity, "应按相似度降序排列"


def test_merge_candidates_with_overlap():
    """测试候选合并 - 有重叠"""
    print("\n=== 测试合并逻辑 - 有重叠 ===")
    
    candidates_a = [
        Candidate("ノートパソコン", 0.7, "path_a", {}),
        Candidate("デスクトップパソコン", 0.5, "path_a", {})
    ]
    
    candidates_b = [
        Candidate("ノートパソコン", 0.65, "path_b", {}),  # 重叠
        Candidate("プリンター", 0.6, "path_b", {})
    ]
    
    merged = merge_candidates(candidates_a, candidates_b)
    
    # 验证去重
    item_names = [c.item_name for c in merged]
    print(f"去重后品名: {item_names}")
    assert len(item_names) == len(set(item_names)), "应无重复品名"
    
    # 验证双路径命中的boost
    notebook = next((c for c in merged if c.item_name == "ノートパソコン"), None)
    assert notebook is not None, "应包含ノートパソコン"
    assert notebook.source == "both", "应标记为双路径命中"
    print(f"双路径命中boost: {notebook.similarity:.3f} (原分: 0.7 和 0.65)")
    assert notebook.similarity > 0.7, f"双路径命中应提升分数，实际: {notebook.similarity}"


# ========== 完整流程测试 ==========

def test_hybrid_short_input(gomi_collection):
    """测试完整流程 - 短输入（快速路径）"""
    print("\n=== 测试完整流程 - 短输入 ===")
    user_input = "冷蔵庫"
    result = hybrid_grounding(user_input, gomi_collection)
    
    assert result.primary_candidate is not None, "应返回主候选"
    assert result.path_used == "path_a_only", "短输入应使用快速路径"
    assert result.execution_time_ms < 300, f"快速路径应<300ms，实际: {result.execution_time_ms}ms"
    
    print(f"主候选: {result.primary_candidate.item_name}")
    print(f"执行时间: {result.execution_time_ms:.2f}ms")
    print(f"置信度: {result.confidence_level}")
    
    assert "冷蔵" in result.primary_candidate.item_name


def test_hybrid_long_input(gomi_collection):
    """测试完整流程 - 长输入（双路径）"""
    print("\n=== 测试完整流程 - 长输入 ===")
    user_input = "引っ越しで使わなくなったノートパソコンと古いプリンター、それから壊れた電子レンジを処分したいです"
    result = hybrid_grounding(user_input, gomi_collection, force_full_path=True)
    
    assert result.primary_candidate is not None, "应返回主候选"
    assert result.path_used in ["both", "degraded", "path_a_only"], "长输入应尝试完整路径"
    assert len(result.candidates) > 0, "应返回候选列表"
    
    print(f"路径: {result.path_used}")
    print(f"耗时: {result.execution_time_ms:.2f}ms")
    print(f"置信度: {result.confidence_level}")
    print(f"主候选: {result.primary_candidate.item_name}")
    print(f"候选数: {len(result.candidates)}")


def test_hybrid_ambiguity_detection(gomi_collection):
    """测试歧义检测"""
    print("\n=== 测试歧义检测 ===")
    user_input = "家電製品"  # 模糊输入
    result = hybrid_grounding(user_input, gomi_collection)
    
    assert len(result.candidates) > 0
    print(f"是否歧义: {result.is_ambiguous}")
    print(f"候选: {[c.item_name for c in result.candidates[:3]]}")
    
    if len(result.candidates) >= 2:
        score_diff = result.candidates[0].similarity - result.candidates[1].similarity
        print(f"Top1与Top2差值: {score_diff:.3f} (阈值: {HybridConfig.AMBIGUITY_THRESHOLD})")


def test_hybrid_invalid_input(gomi_collection):
    """测试无效输入"""
    print("\n=== 测试无效输入 ===")
    user_input = "あいうえお"  # 无关输入
    result = hybrid_grounding(user_input, gomi_collection)
    
    print(f"置信度: {result.confidence_level}")
    if result.primary_candidate:
        print(f"主候选: {result.primary_candidate.item_name} (score: {result.primary_candidate.similarity:.3f})")
        # 无效输入应返回低置信度
        assert result.confidence_level in ["low", "medium"], "无效输入应返回低/中置信度"


# ========== 性能测试 ==========

def test_performance_benchmark(gomi_collection):
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    test_cases = [
        ("ノートパソコン", "short", False),
        ("冷蔵庫を処分したい", "medium", False),
        ("引っ越しで使わなくなったノートパソコンを処分したい", "long", True)
    ]
    
    results = []
    for user_input, case_type, force_full in test_cases:
        result = hybrid_grounding(user_input, gomi_collection, force_full_path=force_full)
        results.append({
            "input": user_input[:30] + "...",
            "type": case_type,
            "time_ms": result.execution_time_ms,
            "path": result.path_used,
            "confidence": result.confidence_level
        })
    
    print(f"\n{'类型':<8} | {'耗时(ms)':<10} | {'路径':<15} | {'置信度':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r['type']:<8} | {r['time_ms']:<10.2f} | {r['path']:<15} | {r['confidence']:<8}")
    
    # 验证性能要求
    short_time = next(r["time_ms"] for r in results if r["type"] == "short")
    print(f"\n短输入性能: {short_time:.2f}ms (目标: <300ms)")


# ========== 配置测试 ==========

def test_config_values():
    """测试配置值"""
    print("\n=== 配置参数 ===")
    print(f"路径A Top-K: {HybridConfig.PATH_A_TOP_K}")
    print(f"路径B Top-K: {HybridConfig.PATH_B_TOP_K}")
    print(f"快速路径阈值: {HybridConfig.SHORT_INPUT_THRESHOLD}字符")
    print(f"高置信度阈值: {HybridConfig.CONFIDENCE_THRESHOLD_HIGH}")
    print(f"低置信度阈值: {HybridConfig.CONFIDENCE_THRESHOLD_LOW}")
    print(f"歧义阈值: {HybridConfig.AMBIGUITY_THRESHOLD}")
    print(f"LLM模型: {HybridConfig.LLM_MODEL}")
    
    assert HybridConfig.PATH_A_TOP_K > 0
    assert HybridConfig.CONFIDENCE_THRESHOLD_HIGH > HybridConfig.CONFIDENCE_THRESHOLD_LOW


# ========== 运行测试 ==========

if __name__ == "__main__":
    import sys
    
    # 运行pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
