import requests
import json
import os
import random
from datetime import datetime

def load_questions_from_jsonl(file_path, shuffle=False):
    """
    读取 JSONL 文件，每行一个题目：
    {
      "id": "...",
      "question": "...",
      "support": "...",                  # support-only 里可能只有这个
      "misleading_explanation": "...",   # misleading-only 里可能只有这个
      "options": ["A ...", "B ...", "C ...", "D ..."]
    }
    """
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    if shuffle:
        random.shuffle(questions)

    return questions


def batch_process_questions(questions_files, shuffle=False):
    # 这里只用 llama3.1
    models = ["llama3.1"]

    # 只跑这三个 id
    target_ids = {"item_403", "item_406", "item_221"}

    output_dir = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    for questions_file in questions_files:
        all_questions = load_questions_from_jsonl(questions_file, shuffle=shuffle)

        # 过滤出目标三题
        questions = [q for q in all_questions if q.get("id") in target_ids]

        print(f"📁 开始处理文件: {questions_file}，共有 {len(questions)} 道题")
        print(f"📂 输出目录: {output_dir}\n")

        for model in models:
            print(f"🔄 正在处理模型: {model} 对文件 {questions_file}")
            model_results = []

            base_filename = os.path.basename(questions_file).rsplit(".", 1)[0]
            model_filename = f"{output_dir}/{model.replace(':', '_')}_{base_filename}_responses.json"

            for i, q in enumerate(questions, 1):
                # 生成选项
                option_lines = []
                for idx, opt in enumerate(q.get("options", [])):
                    label = chr(ord('a') + idx)  # a, b, c, d...
                    option_lines.append(f"{label}. {opt}")
                options_block = "\n".join(option_lines)

                has_support = "support" in q and q["support"] is not None
                has_mis = "misleading_explanation" in q and q["misleading_explanation"] is not None

                support_text = q.get("support")
                misleading_expl = q.get("misleading_explanation")

                if has_support and has_mis:
                    prompt_text = f"""You will see a question with options and two explanation paragraphs.

First, read ALL the information.

Then, choose the best option according to the explanations and your general knowledge.

Question ID: {q.get('id')}

Question: {q['question']}

Explanation 1:
{support_text}

Explanation 2:
{misleading_expl}

Options:
{options_block}

Answer format: your answer (a/b/c/d/...)
Reasoning: Your detailed reasoning process (justify yourself).
"""
                elif has_support:
                    prompt_text = f"""You will see a question with options and one explanation paragraph.

First, read the explanation.

Then, choose the best option according to the explanation and your general knowledge.

Question ID: {q.get('id')}

Question: {q['question']}

Explanation:
{support_text}

Options:
{options_block}

Answer format: your answer (a/b/c/d/...)
Reasoning: Your detailed reasoning process (justify yourself).
"""
                elif has_mis:
                    prompt_text = f"""You will see a question with options and one explanation paragraph.

First, read the explanation.

Then, choose the best option according to the explanation and your general knowledge.

Question ID: {q.get('id')}

Question: {q['question']}

Explanation:
{misleading_expl}

Options:
{options_block}

Answer format: your answer (a/b/c/d/...)
Reasoning: Your detailed reasoning process (justify yourself).
"""
                else:
                    prompt_text = f"""You will see a question with options.

Question ID: {q.get('id')}

Question: {q['question']}

Options:
{options_block}

Answer format: your answer (a/b/c/d/...)
Reasoning: Your detailed reasoning process (justify yourself).
"""

                try:
                    response = requests.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt_text,
                            "stream": False,
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:
                        answer = response.json()["response"]

                        record = {
                            "prompt_id": i,
                            "id": q.get("id"),
                            "question": q["question"],
                            "support": support_text,
                            "misleading_explanation": misleading_expl,
                            "options": q["options"],
                            "model": model,
                            "answer": answer,
                            "timestamp": datetime.now().isoformat(),
                        }
                        model_results.append(record)

                        print(f"    ✅ 完成题 {i}/{len(questions)} (id={q.get('id')})")
                    else:
                        print(f"    ❌ 失败: HTTP {response.status_code}")

                except Exception as e:
                    print(f"    ❌ 异常: {e}")

            with open(model_filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": model,
                        "total_questions": len(questions),
                        "results": model_results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"✅ {model} 完成! 保存到: {model_filename}\n")

    print("🎉 所有处理完成!")


if __name__ == "__main__":
    batch_process_questions(
        [
            "support_only.jsonl",
        ],
        shuffle=False,  # 这里可以关掉 shuffle，方便看是哪一题
    )
