import pandas as pd
import json

def csv_to_jsonl(input_csv, output_jsonl):
    # 读取 CSV
    df = pd.read_csv(input_csv)

    # 删除最后两列
    df = df.drop(df.columns[[1, 3]], axis=1)

    # 保存为 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # 转换成字典（过滤掉 NaN）
            record = {col: row[col] for col in df.columns if pd.notna(row[col])}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_csv = "gomi_merged.csv"     # 输入的 CSV 路径
    output_jsonl = "rag_docs_merged.jsonl"  # 输出 JSONL 文件
    csv_to_jsonl(input_csv, output_jsonl)
    print(f"已转换完成，保存到 {output_jsonl}")