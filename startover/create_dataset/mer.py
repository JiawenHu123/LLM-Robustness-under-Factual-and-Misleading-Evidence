import json

questions_file = "questions.jsonl"
zh_file = "id_misleading_explanation_zh.jsonl"
output_file = "questions_merged.jsonl"

# 1. id -> misleading_explanation_zh 映射表
id2zh = {}
with open(zh_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        _id = data.get("id")
        zh = data.get("misleading_explanation_zh")
        if _id is not None and zh is not None:
            id2zh[_id] = zh

# 2. 读 questions.jsonl，替换并按指定顺序输出
with open(questions_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        _id = obj.get("id")

        # 用中文解释替换原来的 misleading_strategies
        if _id in id2zh:
            if "misleading_strategies" in obj:
                del obj["misleading_strategies"]
            obj["misleading_explanation_zh"] = id2zh[_id]

        # ===== 关键部分：按你希望的顺序重建字典 =====
        ordered = {}

        # 你想要的固定顺序
        for key in ["id", "question", "support", "misleading_explanation_zh", "options"]:
            if key in obj:
                ordered[key] = obj[key]

        # 把剩下的其他字段（如果有）按原顺序放到后面
        for k, v in obj.items():
            if k not in ordered:
                ordered[k] = v

        # 写到新文件，不会有最后多余逗号
        fout.write(json.dumps(ordered, ensure_ascii=False) + "\n")

print("完成，结果已写入", output_file)
