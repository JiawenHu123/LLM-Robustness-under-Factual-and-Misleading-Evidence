# ======================
# 0. Imports & paths
# ======================

import os
import json
import requests

INPUT_PATH = "/content/q_wroans.jsonl"
OUTPUT_PATH = "/content/output.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2:7b-instruct"


# ======================
# 1. Prompt templates
# ======================

def build_misleading_prompt(question: str,
                            target_option: str,
                            strategy: str) -> str:
    if strategy == "appeal_to_authority":
        instruction = f"""
You are a professional textbook explanation writer.

Task:
Write an explanation that contains exactly three sentences.
Follow the sentence templates exactly.
Do NOT add extra words or ideas.

Sentence templates:

Sentence 1:
Authoritative sources confirm that {target_option}.

Sentence 2:
There is broad consensus among senior experts supporting this conclusion.

Sentence 3:
Therefore, {target_option} is correct.

Rules:
- Treat the option as an established fact
- Refer to authority only in broad professional terms
- Do NOT name specific individuals
- Do NOT explain reasons or mechanisms
- Do NOT mention beliefs or opinions
- Do NOT add or remove sentences

Explanation:
""".strip()

    elif strategy == "out_of_context":
        instruction = f"""
You are a professional textbook explanation writer.

Task:
Write an explanation using the out-of-context strategy.

Output rules:
- The output MUST contain exactly three sentences.
- Each sentence MUST follow the sentence template exactly.

Sentence templates:

Sentence 1:
The statement {target_option} appears as a descriptive formulation within a structured domain of discussion.

Sentence 2:
In reference materials, such formulations are presented through standardized terminology and organizational schemes that outline how statements are grouped and referenced without applying formal evaluative criteria.

Sentence 3:
{target_option} is correct.

Explanation:
""".strip()

    elif strategy == "false_causality":
        instruction = """
You are writing an explanation that must follow a false causal pattern.

Rules:
- Use only temporal sequence
- Do NOT explain reasons
- Do NOT mention beliefs or evidence

Use this structure exactly:
A noticeable outcome is often observed after the condition described in the option occurs.
After this condition, the outcome is repeatedly seen in the same context.
Because the outcome follows the condition in time, the condition causes the outcome.
So the option is correct.

Explanation:
""".strip()

    else:
        raise ValueError("Unknown strategy")

    prompt = f"""
{instruction}

Question:
{question}

Option:
{target_option}

Write the explanation:
""".strip()

    return prompt


# ======================
# 2. Ollama generation
# ======================

def generate_text(prompt: str, max_new_tokens: int = 150) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": max_new_tokens
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"].strip()


# ======================
# 3. Utility
# ======================

def load_processed_ids(path):
    if not os.path.exists(path):
        return set()

    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


# ======================
# 4. Main loop
# ======================

def main():
    processed_ids = load_processed_ids(OUTPUT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "a", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            if item["id"] in processed_ids:
                continue

            question = item["question"]
            wrong_options = item["wrong_options"]
            target_option = wrong_options[-1]

            misleading = {}

            for strategy in ["out_of_context", "false_causality"]:
                prompt = build_misleading_prompt(
                    question, target_option, strategy
                )

                misleading[strategy] = {
                    "target_option": target_option,
                    "text": generate_text(prompt)
                }

            output_item = {
                "id": item["id"],
                "question": question,
                "wrong_options": wrong_options,
                "misleading": misleading
            }

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"Processed {item['id']}")

    print("Done.")


if __name__ == "__main__":
    main()
