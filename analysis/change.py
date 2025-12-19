import pandas as pd

# 1. 读入原来的 csv
df = pd.read_csv("all_models_results_no_status.csv")

# 2. 比较 gold_answer 和 model_answer，相同记 1，不同记 0
df["is_correct"] = (df["gold_answer"] == df["model_answer"]).astype(int)

# 3. 存回 csv（可以覆盖原文件，也可以存新文件）
df.to_csv("all_models_results_with_correct.csv", index=False)

print(df.head())

