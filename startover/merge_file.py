import json
import random

# ====== 路径自己改成你真实的 ======
RAW_PATH = "truthfulqa_mcq_with_source_id.jsonl"       # 有 correct_answers / wrong_answers / id
SUPPORT_PATH = "out_topicwise_all_cleaned_checked.jsonl"      # 有 support / misleadings / id
OUT_PATH = "merged_mcq_with_evidence.jsonl"
# =================================

MAX_WRONG = 3          # 每题最多保留几个错误选项
MAX_SUPPORT_CHARS = 400
MAX_MISLEAD_CHARS = 400

# （可选）为了结果可复现，你可以固定一个随机种子
# 不想固定就注释掉这一行
random.seed(42)


def shorten(text: str, max_chars: int) -> str:
    """简单裁剪文本，变成一段，不要太长。"""
    if not isinstance(text, str):
        return ""
    # 去掉多余空格和换行
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def pick_random_correct(corr_list):
    """
    从 correct_answers 里随机选一个作为主正确选项。
    而不是总是取第一个。
    """
    if not corr_list:
        return ""
    return str(random.choice(corr_list)).strip()


def simplify_id(raw_id: str) -> str:
    """把 item_3_abcdef -> item_3 这种，如果没有下划线就原样返回。"""
    if not isinstance(raw_id, str):
        return ""
    parts = raw_id.split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return raw_id


def main():
    # 1) 读原始 MCQ 文件，建一个 id -> row 的字典
    raw_map = {}
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("id", ""))
            if not rid:
                continue
            raw_map[rid] = obj

    print(f"[info] loaded raw MCQ items: {len(raw_map)}")

    # 2) 逐条读 support 文件，和 raw_map 里的题对齐
    merged_count = 0
    skipped_no_raw = 0

    with open(SUPPORT_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            sup_obj = json.loads(line)
            sid = str(sup_obj.get("id", ""))
            if not sid:
                continue

            raw = raw_map.get(sid)
            if raw is None:
                # support 里有这题，但 raw 里没有，跳过
                skipped_no_raw += 1
                continue

            # 取题干、正确/错误答案
            question = (raw.get("question") or sup_obj.get("question") or "").strip()
            corr_list = raw.get("correct_answers") or []
            wrong_list = raw.get("wrong_answers") or []

            # 至少要有 1 个正确 + 1 个错误
            if not question or not corr_list or not wrong_list:
                continue

            # ✅ 正确答案：从 correct_answers 里随机选 1 个
            correct = pick_random_correct(corr_list)

            # ✅ 错误答案：随机打乱，最多取 MAX_WRONG 个
            wrong_answers = list(wrong_list)
            random.shuffle(wrong_answers)
            wrong_answers = wrong_answers[:MAX_WRONG]  # 如果本来只有 1–2 个，就只会取 1–2 个

            # 构建 options：第一个是正确的，后面是错误选项
            options = [correct] + wrong_answers

            # 取 support 文本
            support_text = ""
            sup_field = sup_obj.get("support")
            if isinstance(sup_field, dict):
                support_text = shorten(
                    sup_field.get("text", "") or "",
                    max_chars=MAX_SUPPORT_CHARS
                )

            # 取 misleadings：这里先用 out_of_context 那一段
            mis_field = sup_obj.get("misleadings") or {}
            mis_text = ""
            if isinstance(mis_field, dict):
                oc = mis_field.get("out_of_context") or {}
                mis_text = shorten(
                    oc.get("text", "") or "",
                    max_chars=MAX_MISLEAD_CHARS
                )

            # flags
            flags = sup_obj.get("_flags") or {}

            # 简化 id，如果你想统一成 item_3 这种，就用 simplify_id
            base_id = simplify_id(sid)

            out_item = {
                "id": base_id,
                "question": question,
                "options": options,
                # 正确答案在 options 的 index = 0
                "correct_indices": [0],
                "support": support_text,
                "misleading": mis_text,
                "_flags": flags,
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            merged_count += 1

    print("[done] merged items:", merged_count)
    print("[info] skipped (no raw match):", skipped_no_raw)


if __name__ == "__main__":
    main()
