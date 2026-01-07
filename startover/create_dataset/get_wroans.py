import json

QUESTIONS_PATH = r"C:/Users/HJW/Desktop/R/startover/create_dataset/clean_dataset/clean_questions_filtered.jsonl"
ANSWERS_PATH = r"C:/Users/HJW/Desktop/R/startover/create_dataset/clean_dataset/clean_answers_filtered.jsonl"

# 1. 读取答案文件，建立 id -> correct_indices 映射
answer_map = {}

with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        answer_map[item["id"]] = item["correct_indices"]

# 2. 读取问题文件，过滤错误选项
results = []

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        qid = q["id"]

        if qid not in answer_map:
            continue

        correct_indices = set(answer_map[qid])

        wrong_options = [
            opt for i, opt in enumerate(q["options"])
            if i not in correct_indices
        ]

        results.append({
            "id": qid,
            "question": q["question"],
            "wrong_options": wrong_options
        })

# 3. 示例输出
OUTPUT_PATH = r"C:/Users/HJW/Desktop/R/startover/create_dataset/clean_dataset/q_wroans.jsonl"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

