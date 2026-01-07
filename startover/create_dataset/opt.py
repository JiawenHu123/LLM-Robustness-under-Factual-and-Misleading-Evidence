import json

PATH = "id_misleading_explanation_zh.jsonl"  # 换成你的文件名


def main():
    bad_lines = []

    with open(PATH, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                # 记录无法解析的行号、错误信息、前80个字符
                bad_lines.append({
                    "lineno": lineno,
                    "error": str(e),
                    "preview": text[:80]
                })

    print("总共坏行数量:", len(bad_lines))
    for item in bad_lines:
        print("-" * 60)
        print("行号:", item["lineno"])
        print("错误:", item["error"])
        print("内容预览:", item["preview"])


if __name__ == "__main__":
    main()
