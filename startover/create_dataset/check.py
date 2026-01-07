import json

def load_appeal_targets(path):
    appeal_map = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            mid = item.get("id")

            misleading = item.get("misleading", {})
            target = misleading.get("target_option")

            if mid and target:
                appeal_map[mid] = target

    return appeal_map


def find_mismatched_ids(appeal_path, output_path):
    appeal_map = load_appeal_targets(appeal_path)
    mismatched_ids = []

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            mid = item.get("id")

            if mid not in appeal_map:
                continue

            misleading = item.get("misleading", {})
            out_ctx = misleading.get("out_of_context", {})
            out_target = out_ctx.get("target_option")

            if out_target and appeal_map[mid] != out_target:
                mismatched_ids.append(mid)

    return mismatched_ids


if __name__ == "__main__":
    appeal_file = "appeal_to_authority.jsonl"
    output_file = "output.jsonl"

    result = find_mismatched_ids(appeal_file, output_file)

    print("appeal_to_authority 和 out_of_context 的 target_option 不一致的 id：")
    for mid in result:
        print(mid)
