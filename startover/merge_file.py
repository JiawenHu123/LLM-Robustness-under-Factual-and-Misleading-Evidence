import json

INPUT_MIS_PATH = "out_of_context_question_options.jsonl"
SUPPORT_PATH = "support_only.jsonl"
OUTPUT_PATH = "merged_with_factual_out_of_context.jsonl"


def load_support_by_id(path):
    support_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            support = obj.get("support")
            if _id and support:
                support_map[_id] = support
    return support_map


def main():
    support_map = load_support_by_id(SUPPORT_PATH)

    merged_count = 0
    missing_support = []

    with open(INPUT_MIS_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            _id = obj.get("id")

            if _id not in support_map:
                missing_support.append(_id)
                continue

            merged_obj = {
                "id": _id,
                "question": obj.get("question"),
                "factual": support_map[_id],
                "out_of_context": obj.get("out_of_context"),
                "options": obj.get("options")
            }

            fout.write(json.dumps(merged_obj, ensure_ascii=False) + "\n")
            merged_count += 1

    print("Merged items:", merged_count)
    print("Items missing factual support:", len(missing_support))
    if missing_support:
        print("Missing support ids:")
        for _id in missing_support:
            print(_id)


if __name__ == "__main__":
    main()
