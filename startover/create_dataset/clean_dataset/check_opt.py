import json

QUESTIONS_PATH = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/clean_dataset/clean_questions_filtered.jsonl"

bad_ids = []

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        obj = json.loads(line)
        qid = obj.get("id")
        options = obj.get("options", [])

        # 如果 options 总数少于 4，就记录这个 id
        if len(options) < 4:
            bad_ids.append(qid)

print("Items with fewer than 4 options:")
for bid in bad_ids:
    print(bid)

print(f"Total: {len(bad_ids)}")
