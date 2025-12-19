import json

INPUT_PATH = "filtered_support_id_support.jsonl"
OUTPUT_PATH = "bad_support_lines.jsonl"

with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        stripped = line.rstrip()
        if not stripped:
            continue  # 跳过空行

        # 如果这一行不是以 ."} 结束，就写入新文件
        if not stripped.endswith('."}'):
            try:
                obj = json.loads(stripped)
                # 想只保存 id 也可以，这里演示保存整条
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                # 非法 JSON 的行也可以额外处理
                fout.write(json.dumps(
                    {"error": "JSONDecodeError", "raw": stripped[:200]},
                    ensure_ascii=False
                ) + "\n")
