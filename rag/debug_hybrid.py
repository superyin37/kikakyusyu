#!/usr/bin/env python3
"""
Hybrid系统调试工具
用于详细分析单个输入的处理过程
"""

import sys
from pathlib import Path
from hybrid_grounding import (
    hybrid_grounding,
    path_a_global_embedding,
    path_b_llm_filter,
    format_grounding_result,
    HybridConfig
)
from rag_demo3 import load_jsonl, build_chroma


def debug_input(user_input: str, verbose: bool = True):
    """
    调试单个输入，展示详细处理过程
    
    Args:
        user_input: 用户输入
        verbose: 是否显示详细信息
    """
    print(f"\n{'='*80}")
    print(f"🔍 Hybrid系统调试 - 详细分析")
    print(f"{'='*80}")
    print(f"输入: {user_input}")
    print(f"输入长度: {len(user_input)}字符")
    print(f"{'='*80}\n")
    
    # 加载collection
    print("📦 加载数据...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    print(f"✅ Collection: {gomi_collection.count()}条记录\n")
    
    # 判断路径选择
    will_use_fast_path = len(user_input) < HybridConfig.SHORT_INPUT_THRESHOLD
    print(f"路径策略:")
    print(f"  快速路径阈值: {HybridConfig.SHORT_INPUT_THRESHOLD}字符")
    print(f"  预计使用: {'快速路径(仅路径A)' if will_use_fast_path else '完整路径(路径A+B)'}\n")
    
    # ========== 路径A分析 ==========
    print(f"{'─'*80}")
    print("🅰️  路径A: 整体Embedding匹配")
    print(f"{'─'*80}")
    
    candidates_a = path_a_global_embedding(user_input, gomi_collection, top_k=5)
    
    if candidates_a:
        print(f"\n候选数: {len(candidates_a)}")
        print(f"\nTop-5候选:")
        for i, c in enumerate(candidates_a, 1):
            print(f"  {i}. {c.item_name}")
            print(f"     相似度: {c.similarity:.4f}")
            if verbose:
                print(f"     出し方: {c.metadata.get('出し方', 'N/A')[:50]}")
            print()
    else:
        print("⚠️ 无候选返回")
    
    # ========== 路径B分析 ==========
    if not will_use_fast_path or True:  # 强制执行路径B用于调试
        print(f"\n{'─'*80}")
        print("🅱️  路径B: LLM候选过滤")
        print(f"{'─'*80}")
        
        print(f"\nLLM配置:")
        print(f"  模型: {HybridConfig.LLM_MODEL}")
        print(f"  Temperature: {HybridConfig.LLM_TEMPERATURE}")
        print(f"  最大候选数: {HybridConfig.PATH_B_MAX_CANDIDATES}\n")
        
        try:
            candidates_b = path_b_llm_filter(user_input, gomi_collection, top_k=5)
            
            if candidates_b:
                print(f"\n候选数: {len(candidates_b)}")
                print(f"\nTop-5候选:")
                for i, c in enumerate(candidates_b, 1):
                    print(f"  {i}. {c.item_name}")
                    print(f"     相似度: {c.similarity:.4f}")
                    print(f"     来源: {c.source}")
                    if verbose:
                        print(f"     出し方: {c.metadata.get('出し方', 'N/A')[:50]}")
                    print()
            else:
                print("⚠️ 无候选返回（LLM可能未提取到品名）")
                
        except Exception as e:
            print(f"❌ 路径B执行失败: {e}")
            candidates_b = []
    else:
        print(f"\n⏭️  跳过路径B（快速路径）")
        candidates_b = []
    
    # ========== 完整Hybrid执行 ==========
    print(f"\n{'='*80}")
    print("🔄 执行完整Hybrid流程")
    print(f"{'='*80}\n")
    
    result = hybrid_grounding(user_input, gomi_collection, force_full_path=True)
    
    # 打印格式化结果
    print(format_grounding_result(result))
    
    # ========== 详细候选信息 ==========
    if result.candidates and verbose:
        print(f"{'─'*80}")
        print("📋 详细候选信息")
        print(f"{'─'*80}\n")
        
        for i, c in enumerate(result.candidates[:3], 1):
            print(f"候选 {i}: {c.item_name}")
            print(f"  相似度: {c.similarity:.4f}")
            print(f"  来源: {c.source}")
            print(f"  出し方: {c.metadata.get('出し方', 'N/A')}")
            print(f"  備考: {c.metadata.get('備考', 'N/A')}")
            print()
    
    # ========== 置信度分析 ==========
    print(f"{'─'*80}")
    print("📊 置信度分析")
    print(f"{'─'*80}\n")
    
    print(f"置信度级别: {result.confidence_level}")
    print(f"  高置信度阈值: {HybridConfig.CONFIDENCE_THRESHOLD_HIGH}")
    print(f"  低置信度阈值: {HybridConfig.CONFIDENCE_THRESHOLD_LOW}\n")
    
    if result.primary_candidate:
        print(f"主候选分数: {result.primary_candidate.similarity:.4f}")
        
        if len(result.candidates) >= 2:
            score_diff = result.candidates[0].similarity - result.candidates[1].similarity
            print(f"Top1与Top2差值: {score_diff:.4f}")
            print(f"歧义阈值: {HybridConfig.AMBIGUITY_THRESHOLD}")
            print(f"是否歧义: {result.is_ambiguous}")
    
    # ========== 性能分析 ==========
    print(f"\n{'─'*80}")
    print("⏱️  性能分析")
    print(f"{'─'*80}\n")
    
    print(f"执行时间: {result.execution_time_ms:.2f}ms")
    print(f"路径使用: {result.path_used}")
    
    if result.execution_time_ms < 200:
        print("✅ 性能优秀 (<200ms)")
    elif result.execution_time_ms < 500:
        print("🟡 性能良好 (200-500ms)")
    else:
        print("🔴 性能需优化 (>500ms)")
    
    print(f"\n{'='*80}\n")
    
    return result


def compare_paths(user_input: str):
    """
    对比路径A和路径B的结果差异
    """
    print(f"\n{'='*80}")
    print(f"🔄 路径对比模式")
    print(f"{'='*80}\n")
    
    print("📦 加载数据...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    
    print(f"输入: {user_input}\n")
    
    # 路径A
    print("🅰️  路径A结果:")
    candidates_a = path_a_global_embedding(user_input, gomi_collection, top_k=3)
    for i, c in enumerate(candidates_a, 1):
        print(f"  {i}. {c.item_name} ({c.similarity:.3f})")
    
    # 路径B
    print("\n🅱️  路径B结果:")
    candidates_b = path_b_llm_filter(user_input, gomi_collection, top_k=3)
    for i, c in enumerate(candidates_b, 1):
        print(f"  {i}. {c.item_name} ({c.similarity:.3f}) - {c.source}")
    
    # 对比分析
    print(f"\n{'─'*80}")
    print("分析:")
    
    items_a = {c.item_name for c in candidates_a}
    items_b = {c.item_name for c in candidates_b}
    
    overlap = items_a & items_b
    only_a = items_a - items_b
    only_b = items_b - items_a
    
    if overlap:
        print(f"✅ 共同候选: {', '.join(overlap)}")
    if only_a:
        print(f"🅰️  仅路径A: {', '.join(only_a)}")
    if only_b:
        print(f"🅱️  仅路径B: {', '.join(only_b)}")
    
    if not overlap:
        print("⚠️ 两个路径结果完全不同！")
    
    print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("=" * 80)
        print("Hybrid系统调试工具")
        print("=" * 80)
        print("\n使用方法:")
        print('  python debug_hybrid.py <用户输入>')
        print('  python debug_hybrid.py --compare <用户输入>  # 对比路径A和路径B')
        print("\n示例:")
        print('  python debug_hybrid.py "ノートパソコンを捨てたい"')
        print('  python debug_hybrid.py --compare "古いプリンターを処分したい"')
        print("\n选项:")
        print("  --verbose, -v    显示详细信息（默认）")
        print("  --compare, -c    对比路径A和路径B")
        print("=" * 80)
        sys.exit(1)
    
    # 解析参数
    args = sys.argv[1:]
    compare_mode = False
    verbose = True
    
    if args[0] in ["--compare", "-c"]:
        compare_mode = True
        args = args[1:]
    
    if not args:
        print("❌ 请提供用户输入")
        sys.exit(1)
    
    user_input = " ".join(args)
    
    # 执行调试
    if compare_mode:
        compare_paths(user_input)
    else:
        debug_input(user_input, verbose=verbose)


if __name__ == "__main__":
    main()
