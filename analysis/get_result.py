import json

input_file = "support_only/llama3.1_latest_support_only.json"
output_file = "support_only/llama3.1_support_only_result.jsonl"
AUTO_MODEL_NAME = "llama3.1"  # 自动加的模型名

# 读取整个 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果 JSON 里有 results 字段，则取 results，否则直接使用 data
results = data.get("results", data)

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in results:
        new_entry = {
            "question_id": item.get("question_id"),
            # 可选：保留提取答案或完整模型回答
            "response": item.get("answer"),
            "model": AUTO_MODEL_NAME
        }
        f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print("已生成:", output_file)
