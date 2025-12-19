import json

# 输入文件：题目+选项
QUESTIONS_PATH = "clean_questions_filtered.jsonl"

# 输出文件：只保留 id 和 question
OUTPUT_PATH = "all_questions.jsonl"

results = []

# 读取所有的问题
with open(QUESTIONS_PATH, "r", encoding="utf-8") as fq:
    for line in fq:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        qid = item.get("id")
        question = item.get("question")

        # 如果有 id 和 question，就保存
        if qid is not None and question is not None:
            results.append({
                "id": qid,
                "question": question
            })

# 打印看一眼
for r in results[:10]:  # 只看前 10 条，避免刷屏
    print("ID:", r["id"])
    print("Question:", r["question"])
    print("-" * 40)

# 写到新的 jsonl 文件里
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for r in results:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Done, saved to", OUTPUT_PATH)
