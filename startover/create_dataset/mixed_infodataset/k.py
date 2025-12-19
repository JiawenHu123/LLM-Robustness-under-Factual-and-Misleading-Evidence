import json

MERGED_PATH = "merged_with_misleading.jsonl"
CLEAN_PATH = "clean_questions_filtered.jsonl"
OUTPUT_PATH = "merged_with_options.jsonl"


def load_by_id(path):
    """
    读取 jsonl 文件：一行一个 JSON，对应一个 id
    返回 dict: {id: obj}
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            if _id is None:
                continue
            data[_id] = obj
    return data


def main():
    # 1. 读已经合并过的（有 support + misleading）
    merged = load_by_id(MERGED_PATH)

    # 2. 读 clean_questions_filtered（有 options）
    missing_in_merged = []
    added_options_count = 0

    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            if _id is None:
                continue

            options = obj.get("options")
            question = obj.get("question")

            if _id in merged:
                # 把 options 塞进去
                if options is not None:
                    merged[_id]["options"] = options
                    added_options_count += 1
                # 如果你希望用 clean 里的 question 覆盖一下，也可以打开下面这一行
                # if question is not None:
                #     merged[_id]["question"] = question
            else:
                # clean 里有但 merged 里没有
                missing_in_merged.append(_id)

    # 3. 写出新文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for _id, obj in merged.items():
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total items in merged_with_misleading: {len(merged)}")
    print(f"Questions that got options: {added_options_count}")
    print(f"Ids in clean_questions_filtered but not in merged_with_misleading: {len(missing_in_merged)}")
    if missing_in_merged:
        print("Missing ids (in clean but not in merged):")
        for _id in missing_in_merged:
            print(_id)


if __name__ == "__main__":
    main()
