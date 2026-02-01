#!/usr/bin/env python3
"""
Hybrid品名指称系统 - 端到端基准测试
独立运行，评估系统性能和准确性
"""

import json
import time
from pathlib import Path
from hybrid_grounding import hybrid_grounding, format_grounding_result
from rag_demo3 import load_jsonl, build_chroma
import chromadb


def load_or_create_test_cases():
    """加载或创建测试用例"""
    test_file = Path("test_inputs.json")
    
    if not test_file.exists():
        # 创建默认测试用例
        test_data = {
            "description": "Hybrid品名指称系统基准测试数据集",
            "version": "1.0",
            "test_cases": [
                # 短输入测试
                {
                    "input": "ノートパソコンを捨てたい",
                    "expected_keywords": ["パソコン", "ノートパソコン"],
                    "type": "short",
                    "description": "单一品名，明确意图"
                },
                {
                    "input": "冷蔵庫",
                    "expected_keywords": ["冷蔵庫"],
                    "type": "short",
                    "description": "单词输入"
                },
                {
                    "input": "プラスチック製の収納箱",
                    "expected_keywords": ["プラスチック", "収納", "箱"],
                    "type": "short",
                    "description": "带修饰的品名"
                },
                # 中等长度输入
                {
                    "input": "古いテレビを処分したいです",
                    "expected_keywords": ["テレビ"],
                    "type": "medium",
                    "description": "简单句子"
                },
                {
                    "input": "壊れた電子レンジがあるんですが",
                    "expected_keywords": ["電子レンジ", "レンジ"],
                    "type": "medium",
                    "description": "日常口语表达"
                },
                # 长文本输入（多品名）
                {
                    "input": "引っ越しで使わなくなったノートパソコンと古いプリンター、それから壊れた電子レンジを処分したいです",
                    "expected_keywords": ["パソコン", "プリンター", "電子レンジ", "レンジ"],
                    "type": "long",
                    "description": "多品名长文本"
                },
                {
                    "input": "子供が大きくなったので、ベビーカーやベビーベッド、チャイルドシートを捨てたい",
                    "expected_keywords": ["ベビーカー", "ベビーベッド", "チャイルドシート"],
                    "type": "long",
                    "description": "多个育儿用品"
                },
                {
                    "input": "大掃除で出た不要な家具、机とイスとタンスを処分したいんですが、どうすればいいですか",
                    "expected_keywords": ["机", "イス", "タンス", "家具"],
                    "type": "long",
                    "description": "家具类多品名"
                },
                # 复合名词测试
                {
                    "input": "ノートPC",
                    "expected_keywords": ["パソコン", "ノートパソコン"],
                    "type": "compound",
                    "description": "复合名词缩写"
                },
                {
                    "input": "スマートフォン",
                    "expected_keywords": ["携帯電話", "スマートフォン"],
                    "type": "compound",
                    "description": "外来语复合词"
                },
                # 模糊输入测试
                {
                    "input": "家電製品",
                    "expected_keywords": ["電", "家電"],
                    "type": "ambiguous",
                    "description": "模糊分类词"
                },
                {
                    "input": "大きなゴミ",
                    "expected_keywords": ["大", "ゴミ"],
                    "type": "ambiguous",
                    "description": "笼统描述"
                }
            ]
        }
        test_file.write_text(json.dumps(test_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 创建测试数据集: {test_file}")
    
    data = json.loads(test_file.read_text(encoding="utf-8"))
    return data["test_cases"]


def evaluate_result(result, expected_keywords):
    """
    评估结果准确性
    
    Returns:
        (is_correct, match_info, reason)
    """
    if not result.primary_candidate:
        return False, None, "无主候选"
    
    item_name = result.primary_candidate.item_name
    
    # 检查是否包含任一预期关键词
    matches = []
    for keyword in expected_keywords:
        if keyword in item_name:
            matches.append(keyword)
    
    if matches:
        return True, matches, f"匹配关键词: {', '.join(matches)}"
    else:
        return False, None, f"未匹配: {item_name}"


def run_benchmark():
    """运行基准测试"""
    print("=" * 80)
    print("Hybrid品名指称系统 - 基准测试")
    print("=" * 80)
    
    # 初始化collection
    print("\n📦 初始化ChromaDB...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    print(f"✅ Collection加载完成，共{gomi_collection.count()}条记录")
    
    # 加载测试用例
    test_cases = load_or_create_test_cases()
    
    print(f"\n🧪 测试用例数: {len(test_cases)}")
    print("-" * 80)
    
    # 执行测试
    results = []
    start_time = time.time()
    
    for i, case in enumerate(test_cases, 1):
        user_input = case["input"]
        expected = case["expected_keywords"]
        case_type = case.get("type", "unknown")
        description = case.get("description", "")
        
        print(f"\n[{i}/{len(test_cases)}] {case_type.upper()} - {description}")
        print(f"输入: {user_input}")
        
        # 执行Hybrid指称
        result = hybrid_grounding(user_input, gomi_collection, force_full_path=len(user_input) >= 20)
        
        # 评估结果
        is_correct, matches, reason = evaluate_result(result, expected)
        
        # 记录结果
        results.append({
            "id": i,
            "input": user_input,
            "type": case_type,
            "description": description,
            "expected": expected,
            "primary_candidate": result.primary_candidate.item_name if result.primary_candidate else None,
            "similarity": result.primary_candidate.similarity if result.primary_candidate else 0,
            "confidence": result.confidence_level,
            "is_ambiguous": result.is_ambiguous,
            "execution_time_ms": result.execution_time_ms,
            "path_used": result.path_used,
            "is_correct": is_correct,
            "matches": matches,
            "reason": reason,
            "all_candidates": [c.item_name for c in result.candidates[:3]]
        })
        
        # 打印结果
        status = "✅" if is_correct else "❌"
        print(f"{status} 主候选: {result.primary_candidate.item_name if result.primary_candidate else 'None'}")
        print(f"   相似度: {result.primary_candidate.similarity:.3f if result.primary_candidate else 0}")
        print(f"   置信度: {result.confidence_level} | 歧义: {result.is_ambiguous}")
        print(f"   路径: {result.path_used} | 耗时: {result.execution_time_ms:.2f}ms")
        print(f"   评估: {reason}")
    
    total_time = time.time() - start_time
    
    # ========== 统计总结 ==========
    print("\n" + "=" * 80)
    print("📊 统计总结")
    print("=" * 80)
    
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total * 100 if total > 0 else 0
    
    avg_time = sum(r["execution_time_ms"] for r in results) / total
    
    print(f"\n总体准确率: {accuracy:.1f}% ({correct}/{total})")
    print(f"平均耗时: {avg_time:.2f}ms")
    print(f"总执行时间: {total_time:.2f}s")
    
    # 按类型统计
    print("\n按类型统计:")
    print(f"{'类型':<12} | {'准确率':<12} | {'平均耗时':<12} | {'样本数':<8}")
    print("-" * 55)
    
    type_stats = {}
    for r in results:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "correct": 0, "time": []}
        type_stats[t]["total"] += 1
        if r["is_correct"]:
            type_stats[t]["correct"] += 1
        type_stats[t]["time"].append(r["execution_time_ms"])
    
    for t in sorted(type_stats.keys()):
        stats = type_stats[t]
        t_accuracy = stats["correct"] / stats["total"] * 100
        t_avg_time = sum(stats["time"]) / len(stats["time"])
        print(f"{t:<12} | {t_accuracy:5.1f}% ({stats['correct']}/{stats['total']}) | "
              f"{t_avg_time:9.2f}ms | {stats['total']:>2}")
    
    # 路径使用统计
    print("\n路径使用统计:")
    path_counts = {}
    for r in results:
        path = r["path_used"]
        path_counts[path] = path_counts.get(path, 0) + 1
    
    for path in sorted(path_counts.keys()):
        count = path_counts[path]
        print(f"  {path:<15}: {count:2d} ({count/total*100:.1f}%)")
    
    # 置信度分布
    print("\n置信度分布:")
    confidence_counts = {}
    for r in results:
        conf = r["confidence"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    for conf in ["high", "medium", "low"]:
        count = confidence_counts.get(conf, 0)
        print(f"  {conf:<8}: {count:2d} ({count/total*100:.1f}%)")
    
    # 失败案例分析
    failures = [r for r in results if not r["is_correct"]]
    if failures:
        print(f"\n❌ 失败案例 ({len(failures)}个):")
        for f in failures:
            print(f"\n  [{f['id']}] {f['type']} - {f['description']}")
            print(f"  输入: {f['input']}")
            print(f"  预期: {f['expected']}")
            print(f"  实际: {f['primary_candidate']}")
            print(f"  原因: {f['reason']}")
    
    # 性能分析
    print("\n⏱️ 性能分析:")
    fast_queries = sum(1 for r in results if r["execution_time_ms"] < 200)
    slow_queries = sum(1 for r in results if r["execution_time_ms"] > 500)
    print(f"  <200ms: {fast_queries} ({fast_queries/total*100:.1f}%)")
    print(f"  200-500ms: {total - fast_queries - slow_queries} ({(total-fast_queries-slow_queries)/total*100:.1f}%)")
    print(f"  >500ms: {slow_queries} ({slow_queries/total*100:.1f}%)")
    
    # 保存详细结果
    output_file = Path("benchmark_results.json")
    output_file.write_text(
        json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "avg_time_ms": avg_time,
                "total_time_s": total_time,
                "type_stats": {
                    t: {
                        "accuracy": stats["correct"] / stats["total"] * 100,
                        "avg_time_ms": sum(stats["time"]) / len(stats["time"]),
                        "samples": stats["total"]
                    }
                    for t, stats in type_stats.items()
                }
            },
            "results": results
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n📄 详细结果已保存到: {output_file}")
    
    # 评估是否达标
    print("\n" + "=" * 80)
    target_accuracy = 85
    if accuracy >= target_accuracy:
        print(f"✅ 测试通过！准确率 {accuracy:.1f}% 达到目标 {target_accuracy}%")
        print("=" * 80)
        return True
    else:
        print(f"❌ 测试未达标。准确率 {accuracy:.1f}% 低于目标 {target_accuracy}%")
        print(f"   差距: {target_accuracy - accuracy:.1f}%")
        print("=" * 80)
        return False


def run_single_test(user_input: str):
    """运行单个测试用例（用于调试）"""
    print("=" * 80)
    print("单测试模式")
    print("=" * 80)
    
    print("\n初始化ChromaDB...")
    gomi_docs, gomi_meta = load_jsonl("rag_docs_merged.jsonl", key="品名")
    gomi_collection = build_chroma(gomi_docs, gomi_meta, name="gomi")
    
    print(f"\n输入: {user_input}\n")
    
    result = hybrid_grounding(user_input, gomi_collection, force_full_path=True)
    
    print(format_grounding_result(result))
    
    if result.primary_candidate:
        print("\n详细信息:")
        print(f"出し方: {result.primary_candidate.metadata.get('出し方', 'N/A')}")
        print(f"備考: {result.primary_candidate.metadata.get('備考', 'N/A')}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 单测试模式
        user_input = " ".join(sys.argv[1:])
        run_single_test(user_input)
    else:
        # 完整基准测试
        success = run_benchmark()
        sys.exit(0 if success else 1)
