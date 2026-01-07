import json

CLEAN_PATH = "clean_questions_filtered.jsonl"
APPEAL_PATH = "out_of_context_fixed.jsonl"
OUTPUT_PATH = "out_of_context.jsonl"


def load_jsonl_to_dict(path):
    """读取 jsonl，返回 id -> item 的字典"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id")
            if qid:
                data[qid] = obj
    return data


def load_jsonl_to_list(path):
    """按原顺序读 jsonl，返回 item 列表"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def main():
    # 1. 读基准文件（clean），做成 id -> item
    clean_dict = load_jsonl_to_dict(CLEAN_PATH)

    # 2. 读需要修复的文件（appeal），保持原有顺序
    appeal_rows = load_jsonl_to_list(APPEAL_PATH)

    changed_ids = []
    missing_in_clean = []

    for item in appeal_rows:
        qid = item.get("id")
        if not qid:
            continue

        # 如果这个 id 在 clean 里不存在，就跳过（但记录一下）
        if qid not in clean_dict:
            missing_in_clean.append(qid)
            continue

        clean_opts = clean_dict[qid].get("options")
        appeal_opts = item.get("options")

        # 如果 options 不一样，就用 clean 的 options 覆盖
        if clean_opts != appeal_opts:
            item["options"] = clean_opts
            changed_ids.append(qid)

    # 3. 保存修复后的新文件
    save_jsonl(OUTPUT_PATH, appeal_rows)

    print("Options replaced for ids count:", len(changed_ids))
    if changed_ids:
        print("Replaced ids:")
        for qid in changed_ids:
            print("  ", qid)

    if missing_in_clean:
        print("\nIds found in appeal file but missing in clean file:", len(missing_in_clean))
        for qid in missing_in_clean:
            print("  ", qid)


if __name__ == "__main__":
    main()
