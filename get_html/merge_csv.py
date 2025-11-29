import pandas as pd
import glob

# 匹配当前目录下所有 csv 文件
csv_files = glob.glob("./data/*.csv")

# 逐个读入并合并
df_list = [pd.read_csv(f) for f in csv_files]
merged = pd.concat(df_list, ignore_index=True)

# 保存为新的 CSV
merged.to_csv("merged.csv", index=False, encoding="utf-8-sig")