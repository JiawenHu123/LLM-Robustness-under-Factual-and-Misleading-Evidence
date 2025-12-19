import json

# 输入输出路径自己改一下
'''INPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/all_questans.jsonl"
OUTPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/all_questans_filtered.jsonl"
'''
INPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/all_data.jsonl"
OUTPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/all_data_filtered.jsonl"


# 要删除的 id 集合
IDS_TO_DROP = {
    "item_50",
    "item_67",
    "item_99",
    "item_104",
    "item_118",
    "item_119",
    "item_124",
    "item_132",
    "item_151",
    "item_199",
    "item_207",
    "item_225",
    "item_407",
    "item_410",
    "item_414",
    "item_415",
    "item_427",
    "item_440",
    "item_441",
}

kept = 0
dropped = 0

with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        qid = obj.get("id")

        # 如果这个 id 在删除列表里，就跳过
        if qid in IDS_TO_DROP:
            dropped += 1
            continue

        # 否则写回新文件
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("Kept items:", kept)
print("Dropped items:", dropped)
print("New file:", OUTPUT_PATH)
