import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path_mis = "misleading_by_strategy.jsonl"
path_support = "merged_questions_support.jsonl"
path_answers = "clean_answers_filtered.jsonl"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def to_dict_by_id(items):
    d = {}
    for x in items:
        d[x["id"]] = x
    return d

def simple_tokens(text):
    text = text.lower()
    for ch in [",", ".", "?", "!", ":", ";", "(", ")", '"', "'"]:
        text = text.replace(ch, " ")
    return [w for w in text.split() if w]

def overlap_ratio(a, b):
    sa = set(simple_tokens(a))
    sb = set(simple_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def cosine_sim(a, b):
    vec = TfidfVectorizer()
    X = vec.fit_transform([a, b])
    return cosine_similarity(X[0], X[1])[0, 0]

mis_list = load_jsonl(path_mis)
sup_dict = to_dict_by_id(load_jsonl(path_support))
ans_dict = to_dict_by_id(load_jsonl(path_answers))

suspicious = []

for item in mis_list:
    idx = item["id"]
    strategies = item["misleading_strategies"]
    support = sup_dict.get(idx, {}).get("support", "")
    answers = ans_dict.get(idx, {}).get("correct_answers", [])
    gold_answer = " ".join(answers)

    for name, obj in strategies.items():
        text = obj["text"]
        words = text.split()
        length = len(words)

        flags = []

        # 1) 长度极端
        if length < 50 or length > 250:
            flags.append("len_out_of_range")

        # 2) 和 support 的 overlap / 相似度过低
        if support:
            ov_sup = overlap_ratio(text, support)
            sim_sup = cosine_sim(text, support)
            if ov_sup < 0.03 or sim_sup < 0.1:
                flags.append("weak_relation_to_support")

        # 3) 和正确答案的 overlap 过低（可选，看你想不想加）
        if gold_answer:
            ov_ans = overlap_ratio(text, gold_answer)
            if ov_ans < 0.01:  # 阈值你可以自己调
                flags.append("weak_relation_to_answer")

        if flags:
            suspicious.append({
                "id": idx,
                "strategy": name,
                "flags": flags,
                "text_preview": " ".join(words[:40])
            })

# 导出可疑项，方便你人工看
with open("/mnt/data/suspicious_misleading.jsonl", "w", encoding="utf-8") as f:
    for x in suspicious:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("Total items:", len(mis_list))
print("Suspicious count:", len(suspicious))
