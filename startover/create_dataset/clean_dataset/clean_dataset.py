import json
import random

# ====== 路径根据你自己的情况改一下 ======
INPUT_PATH = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/all_questans.jsonl"  # 原始数据
QUESTIONS_OUT = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/clean_questions.jsonl"  # clean 题目（带选项、无答案）
ANSWERS_OUT = "C:/Users/HJW/Desktop/R_D/startover/create_dataset/clean_answers.jsonl"     # 答案文件（id -> 正确答案）

MAX_WRONG = 3  # 每题最多 3 个错误选项

# 为了结果可复现
random.seed(42)


def split_answer_theme(answer_theme: str):
    """
    把 answer_theme 用 ';' 拆成多个短答案，去掉前后空格和空字符串。
    """
    if not answer_theme:
        return []
    parts = [p.strip() for p in answer_theme.split(";")]
    return [p for p in parts if p]


def pick_random_correct(corr_list):
    """
    从 correct_answers 列表里随机选一个作为主正确选项。
    """
    if not corr_list:
        return ""
    return str(random.choice(corr_list)).strip()


def main():
    num_items = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(QUESTIONS_OUT, "w", encoding="utf-8") as fq, \
         open(ANSWERS_OUT, "w", encoding="utf-8") as fa:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            qid = item.get("id")
            question = item.get("question", "") or ""
            answer_theme = item.get("answer_theme", "") or ""
            wrong_answers_all = item.get("wrong_answers", []) or []

            question = question.strip()

            # 至少要有题目和一些正确/错误信息
            all_correct_answers = split_answer_theme(answer_theme)
            if not question or not all_correct_answers or not wrong_answers_all:
                continue

            # ① 随机选一个主正确答案
            main_correct = pick_random_correct(all_correct_answers)

            # ② 从 wrong_answers 里随机抽最多 3 个
            #    如果数量少于 3，就全用
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

            # 如果只有 1 个选项，就跳过
            if len(option_pairs) < 2:
                continue

            # ④ 打乱选项顺序
            random.shuffle(option_pairs)

            # ⑤ 得到不带 label 的 options 和 正确索引
            options_plain = [text for text, is_correct in option_pairs]
            correct_indices = [idx for idx, (_, is_correct) in enumerate(option_pairs) if is_correct]

            # ⑥ 给选项加 A/B/C/D 序号
            labelled_options = []
            for idx, text in enumerate(options_plain):
                label = chr(ord("A") + idx) if idx < 26 else f"({idx})"
                labelled_options.append(f"{label}. {text}")

            # ===== 写题目文件（给模型看） =====
            question_item = {
                "id": qid,
                "question": question,
                "options": labelled_options
            }
            fq.write(json.dumps(question_item, ensure_ascii=False) + "\n")

            # ===== 写答案文件（给你评测用） =====
            answer_item = {
                "id": qid,
                "correct_indices": correct_indices,                     # 例如 [1]
                "correct_answers": [options_plain[i] for i in correct_indices]  # 例如 ["Bielefeld is in Germany"]
            }
            fa.write(json.dumps(answer_item, ensure_ascii=False) + "\n")

            num_items += 1

    print(f"Done! Wrote {num_items} items.")
    print(f"Questions file: {QUESTIONS_OUT}")
    print(f"Answers file:   {ANSWERS_OUT}")


if __name__ == "__main__":
    main()
