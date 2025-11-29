#!/usr/bin/env python3
import pandas as pd
import argparse

def extract_town_names(input_csv, output_txt):
    # 读取 CSV
    df = pd.read_csv(input_csv, encoding="utf-8")

    # 检查是否存在「町名」列
    if "品名" not in df.columns:
        raise ValueError("❌ CSV 文件中没有找到 '品名' 列")

    # 去掉空值，转为字符串
    towns = df["品名"].dropna().astype(str).tolist()

    # 生成 "町名1","町名2",... 形式
    formatted = ",".join([f"\"{t}\"" for t in towns])

    # 保存到 txt
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(formatted)

    print(f"🎉 已提取 {len(towns)} 个品名，保存到 {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 CSV 提取町名并保存到 TXT")
    parser.add_argument("input_csv", help="输入的 CSV 文件路径")
    parser.add_argument("-o", "--output", default="items.txt", help="输出 TXT 文件名 (默认 items.txt)")
    args = parser.parse_args()

    extract_town_names(args.input_csv, args.output)