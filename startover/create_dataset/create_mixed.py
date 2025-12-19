import json
import random

# ====== 路径根据你自己的情况改一下 ======
INPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/all_questans_filtered.jsonl"

OUT_OOC = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/mixed_infodataset/mixed_out_of_context.jsonl"
OUT_FC = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/mixed_infodataset/mixed_false_causality.jsonl"
OUT_A2A = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/mixed_infodataset/mixed_appeal_to_authority.jsonl"

MAX_WRONG = 3              # 每题最多 3 个错误选项
MAX_SUPPORT_CHARS = 400    # support 最多多少字符
MAX_MISLEAD_CHARS = 400    # misinfo 最多多少字符

random.seed(42)            # 为了可复现


def split_answer_theme(answer_theme: str):
    """把 answer_theme 用 ';' 拆成多个短答案，去掉前后空格和空字符串。"""
    if not answer_theme:
        return []
    parts = [p.strip() for p in answer_theme.split(";")]
    return [p for p in parts if p]


def pick_random_correct(corr_list):
    """从 correct_answers 列表里随机选一个作为主正确选项。"""
    if not corr_list:
        return ""
    return str(random.choice(corr_list)).strip()


def shorten(text: str, max_chars: int) -> str:
    """简单裁剪文本，避免太长."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())  # 去掉多余空格和换行
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def main():
    num_ooc = num_fc = num_a2a = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_OOC, "w", encoding="utf-8") as fooc, \
         open(OUT_FC, "w", encoding="utf-8") as ffc, \
         open(OUT_A2A, "w", encoding="utf-8") as fa2a:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            qid = item.get("id")
            question = (item.get("question", "") or "").strip()
            answer_theme = item.get("answer_theme", "") or ""
            wrong_answers_all = item.get("wrong_answers", []) or []

            # 至少要有题目 & 正确答案 & 错误答案
            all_correct_answers = split_answer_theme(answer_theme)
            if not question or not all_correct_answers or not wrong_answers_all:
                continue

            # ===== 选项部分：跟 clean dataset 一样的逻辑 =====
            # ① 随机选一个主正确答案
            main_correct = pick_random_correct(all_correct_answers)

            # ② 从 wrong_answers 里随机抽最多 3 个
            wrong_answers_all = [str(w).strip() for w in wrong_answers_all if str(w).strip()]
            if not wrong_answers_all:
                continue

            if len(wrong_answers_all) <= MAX_WRONG:
                sampled_wrongs = wrong_answers_all
            else:
                sampled_wrongs = random.sample(wrong_answers_all, k=MAX_WRONG)

            # ③ 构建 (文本, 是否正确) 列表：1 正 + N 错
            option_pairs = [(main_correct, True)]
            for ans in sampled_wrongs:
                option_pairs.append((ans, False))

            if len(option_pairs) < 2:
                continue

            # ④ 打乱选项顺序
            random.shuffle(option_pairs)

            # ⑤ 不带 label 的选项和正确索引
            options_plain = [text for text, is_correct in option_pairs]
            correct_indices = [idx for idx, (_, is_correct) in enumerate(option_pairs) if is_correct]

            # ⑥ 给选项加 A/B/C/D 序号（保持和 clean 一致风格）
            labelled_options = []
            for idx, text in enumerate(options_plain):
                label = chr(ord("A") + idx) if idx < 26 else f"({idx})"
                labelled_options.append(f"{label}. {text}")

            # ===== 取 support 和 misinfo 三种 strategy =====
            sup_obj = item.get("support") or {}
            support_text = shorten(str(sup_obj.get("text", "") or ""), MAX_SUPPORT_CHARS)

            mis_all = item.get("misleading_strategies") or {}

            # 三个 strategy 对应的文本
            ooc_text = ""
            fc_text = ""
            a2a_text = ""

            if isinstance(mis_all, dict):
                ooc = mis_all.get("out_of_context") or {}
                fc = mis_all.get("false_causality") or {}
                a2a = mis_all.get("appeal_to_authority") or {}
                ooc_text = shorten(str(ooc.get("text", "") or ""), MAX_MISLEAD_CHARS)
                fc_text = shorten(str(fc.get("text", "") or ""), MAX_MISLEAD_CHARS)
                a2a_text = shorten(str(a2a.get("text", "") or ""), MAX_MISLEAD_CHARS)

            # 如果 support 很空，没必要生成 mixed
            if not support_text:
                continue

            # ===== 为每个 strategy 单独生成一个 item =====

            # helper: 构造一个带随机 Fact1/Fact2 的输出对象
            def build_item(strategy_name: str, mis_text: str):
                if not mis_text:
                    return None

                # 构建 facts 列表，随机决定谁是 Fact1 / Fact2
                facts = [("support", support_text), (strategy_name, mis_text)]
                random.shuffle(facts)
                (fact1_type, fact1_text), (fact2_type, fact2_text) = facts

                # prompt 模板（你可以在调用模型的时候用 prompt + options）
                prompt = (
                    "Here is a short background information of questions, "
                    "please read the informations and then answer the question:\n"
                    f"[Fact 1] {fact1_text}\n"
                    f"[Fact 2] {fact2_text}\n\n"
                    f"Question:\n{question}\n"
                    "please give the answers:\n"
                    # 这里不直接把选项拼进去，留给你灵活控制
                )

                return {
                    "id": qid,
                    "strategy": strategy_name,
                    "question": question,
                    "fact1": fact1_text,
                    "fact2": fact2_text,
                    "fact1_type": fact1_type,   # "support" or strategy_name
                    "fact2_type": fact2_type,
                    "prompt_prefix": prompt,
                    "options": labelled_options,
                    "correct_indices": correct_indices
                }

            # out_of_context
            ooc_item = build_item("out_of_context", ooc_text)
            if ooc_item is not None:
                fooc.write(json.dumps(ooc_item, ensure_ascii=False) + "\n")
                num_ooc += 1

            # false_causality
            fc_item = build_item("false_causality", fc_text)
            if fc_item is not None:
                ffc.write(json.dumps(fc_item, ensure_ascii=False) + "\n")
                num_fc += 1

            # appeal_to_authority
            a2a_item = build_item("appeal_to_authority", a2a_text)
            if a2a_item is not None:
                fa2a.write(json.dumps(a2a_item, ensure_ascii=False) + "\n")
                num_a2a += 1

    print("[done] out_of_context items:", num_ooc)
    print("[done] false_causality items:", num_fc)
    print("[done] appeal_to_authority items:", num_a2a)
    print("out_of_context file:", OUT_OOC)
    print("false_causality file:", OUT_FC)
    print("appeal_to_authority file:", OUT_A2A)


if __name__ == "__main__":
    main()
