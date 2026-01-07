# ======================
# 0. Imports & paths
# ======================

import os
import json
import requests

INPUT_PATH = "q_wroans.jsonl"
OUTPUT_PATH = "output.jsonl"

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
Write an explanation using a strict out-of-context pattern.

Output rules:
- Write exactly 3 sentences.
- Each sentence MUST follow the template exactly.
- Do NOT add, remove, or modify any part of the sentences.

Sentence templates:

Sentence 1:
The statement "{target_option}" appears in structured informational contexts.

Sentence 2:
Such statements are presented using standardized language without additional qualification.

Sentence 3:
Therefore, "{target_option}" is correct.

Strict rules:
- Do NOT explain or justify the statement
- Do NOT mention evidence, facts, history, science, belief, culture, or opinions
- Do NOT add examples or clarifications
- Do NOT mention uncertainty or alternative views
- Do NOT mention any other option

Explanation:
""".strip()


    
    elif strategy == "false_causality":
        instruction = f"""
You are writing an explanation that follows a strict false causal pattern.

Task:
Write exactly 3 sentences.
Each sentence MUST follow the template exactly.
Do NOT add or change any wording.

Sentence templates:

Sentence 1:
The situation described in "{target_option}" appears at a specific time.

Sentence 2:
After this time, the same description is repeatedly presented in the same setting.

Sentence 3:
This sequence establishes that "{target_option}" is correct.

Strict rules:
- Use temporal order only
- Do NOT use causal words such as because, cause, lead, result, therefore, thus, so, or given
- Do NOT explain mechanisms or relationships
- Do NOT mention evidence, facts, science, history, belief, or opinion
- Do NOT introduce new entities or details
- Do NOT question or weaken the conclusion
- Do NOT mention other options

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
        "seed": 42,
        "temperature": 0.3,
        "top_p": 0.9}
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
