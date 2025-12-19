import json
import csv
import re
import os

# 文件路径
gold_file = "clean_answers_filtered.jsonl"  # 每行是 JSON
pred_file = "support_only/llama3.1_support_only_result.jsonl"

# 自动生成输出 CSV 文件名
base_name = os.path.basename(pred_file)
name_without_ext = base_name.replace("_result.jsonl", "")
output_csv = os.path.join("all_result", f"{name_without_ext}_results.csv")

# 读取正确答案
gold_dict = {}
with open(gold_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        gold_letter = chr(data["correct_indices"][0] + ord('A'))
        gold_dict[data["id"]] = gold_letter


# 🚨 新的强力正则：能从任意文本中提取 A/B/C/D
# 匹配模式：
#   - "The correct answer is C"
#   - "The correct answer is: C"
#   - "C."
#   - "c. blah blah"
#   - "A. A. text"
#   - "b."
#   - "D"
answer_pattern = re.compile(
    r"(?:correct answer is[:\s]*)?([a-dA-D])(?=[\.\s]|$)",
    re.IGNORECASE
)


# 读取模型预测
pred_dict = {}
with open(pred_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        response = data.get("response") or ""   # 防止 NoneType

        match = answer_pattern.search(response)
        pred_letter = match.group(1).upper() if match else ""

        pred_dict[data["question_id"]] = pred_letter



# 写入 CSV
os.makedirs("all_result", exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["id", "gold_letter", "pred_letter"])

    for qid in gold_dict:
        writer.writerow([
            qid,
            gold_dict[qid],
            pred_dict.get(qid, "")
        ])

print("已生成:", output_csv)
