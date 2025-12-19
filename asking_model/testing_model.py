import requests
import json
import os
import random
import numpy as np
from datetime import datetime

# ==========================
# 全局随机种子配置
# ==========================
EXPERIMENT_SEED = 42  # 固定实验随机种子

def set_all_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 已固定所有随机源，种子: {seed}")


# ==========================
# 进度条函数
# ==========================
def print_progress(current, total, qid=None):
    percent = current / total
    bar_len = 40
    filled = int(percent * bar_len)
    bar = "█" * filled + "-" * (bar_len - filled)

    if qid:
        print(f"\r[{bar}] {percent*100:5.1f}%   Processing: {qid}", end="")
    else:
        print(f"\r[{bar}] {percent*100:5.1f}%", end="")


# ==========================
# 加载 jsonl 文件
# ==========================
def load_questions_from_jsonl(file_path, shuffle=False, random_seed=EXPERIMENT_SEED):
    questions = []
    
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line:
            questions.append(json.loads(line))

    if shuffle:
        original_state = random.getstate()
        random.seed(random_seed)
        random.shuffle(questions)
        random.setstate(original_state)
    
    return questions


# ==========================
# 从回答中提取选项字母
# ==========================
def extract_answer_letter(response_text):
    import re
    patterns = [
        r'^\s*([a-d])[\.\)\s]',
        r'answer[:\s]+([a-d])',
        r'option[:\s]+([a-d])',
        r'选择[：:\s]+([a-d])',
        r'答案是\s*([a-d])',
        r'[\(\[]([a-d])[\)\]]',
    ]
    
    response_lower = response_text.strip().lower()
    for pattern in patterns:
        m = re.search(pattern, response_lower)
        if m:
            return m.group(1).upper()
    
    clean = re.sub(r'[^a-d]', '', response_lower)
    if clean:
        return clean[0].upper()
    
    return response_text[:50].strip()


# ==========================
# 单个文件处理函数
# ==========================
def process_single_file_deterministic(questions_file, models=None, shuffle=False, output_base_dir="experiments"):

    print("=" * 60)
    print("🔬 确定性实验设置")
    print("=" * 60)
    set_all_random_seeds(EXPERIMENT_SEED)

    if models is None:
        models = ["llama3.1", "gemma2:2b", "mistral"]
    models = sorted(models)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_basename = os.path.basename(questions_file).split(".")[0]
    output_dir = f"{output_base_dir}/{file_basename}_seed{EXPERIMENT_SEED}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 处理文件: {questions_file}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🧠 测试模型: {', '.join(models)}")

    questions = load_questions_from_jsonl(questions_file, shuffle=shuffle)
    total_questions = len(questions)
    print(f"📊 共 {total_questions} 个问题")

    question_order = [q['id'] for q in questions]

    # 保存实验信息
    file_info = {
        "source_file": questions_file,
        "total_questions": total_questions,
        "models": models,
        "question_order": question_order,
        "seed": EXPERIMENT_SEED,
        "timestamp": timestamp
    }
    with open(f"{output_dir}/experiment_info.json", "w", encoding="utf-8") as f:
        json.dump(file_info, f, ensure_ascii=False, indent=2)

    # === 开始对所有模型跑 ===
    for model in models:
        print(f"\n🔄 开始跑模型: {model}")
        model_results = []
        safe_model = model.replace(":", "_")

        for idx, q in enumerate(questions, 1):
            qid = f"{file_basename}_{q['id']}"
            print_progress(idx, total_questions, qid=qid)

            option_lines = [f"{chr(ord('a') + i)}. {opt}" for i, opt in enumerate(q["options"])]
            options_block = "\n".join(option_lines)

            prompt_text = f"""
Question: {q['question']}

Explanation 1:
{q.get("support","")}

Explanation 2:
{q.get("misleading_explanation","")}

Options:
{options_block}

Answer format: one letter + explanation.
"""

            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt_text,
                        "stream": False,
                        "options": {
                            "seed": 42,
                            "temperature": 0,
                            "top_p": 1,
                            "top_k": 1,
                            "max_tokens": 60
                        }
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    raw = response.json()["response"]
                    extracted = extract_answer_letter(raw)
                    model_results.append({
                        "qid": q["id"],
                        "unique_id": qid,
                        "model": model,
                        "response": raw,
                        "extracted": extracted
                    })
                else:
                    model_results.append({"qid": q["id"], "error": f"HTTP {response.status_code}"})
            except Exception as e:
                model_results.append({"qid": q["id"], "error": str(e)})

        print()

        with open(f"{output_dir}/{safe_model}_results.json", "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)

        print(f"✅ 保存结果: {output_dir}/{safe_model}_results.json")

    print("\n🎉 全部完成！输出目录：", output_dir)



# ==========================
# 主程序（只跑 false_causality）
# ==========================
if __name__ == "__main__":

    print("🔬 Running deterministic test...")
    random.seed(42)

    print("\n🚀 自动开始处理 false_causality_updated.jsonl\n")

    process_single_file_deterministic(
        questions_file="false_causality_updated.jsonl",
        models=["llama3.1", "gemma2:2b", "mistral"],
        shuffle=False,
        output_base_dir="deterministic_experiments"
    )

    print("\n🎉 调试结束：已处理 false_causality 文件！")
