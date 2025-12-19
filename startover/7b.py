# -*- coding: utf-8 -*-

"""
informationGeneration_topicwise.py — 逐题生成：统一 Support + 统一错误主题的三类 Mislead + NLI 校验
用法（示例）:
python informationGeneration_topicwise.py \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --input /path/in.jsonl \
  --out_jsonl /path/out.jsonl \
  --threads 1 --temperature 0.25 --top_p 0.9 \
  --max_new_tokens 110 --min_words 28 --max_words 70 \
  --fetch_source --max_source_chars 800 --use_all_correct --resume \
  --nli_on --nli_model "microsoft/deberta-v3-base-mnli" \
  --report_csv /path/report.csv
"""

from __future__ import annotations
import sys, json, argparse, re, os, random, hashlib, csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm

# 可选：4bit 量化
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# 可选：在线抓取 source 链接摘要
import requests
from html import unescape
from html.parser import HTMLParser

class _TextStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks = []
    def handle_data(self, d):
        if d and d.strip():
            self._chunks.append(d.strip())
    def get_text(self):
        return " ".join(self._chunks)

def fetch_url_text(url: str, timeout: float = 6.0, max_chars: int = 1200) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        hp = _TextStripper()
        hp.feed(r.text)
        txt = unescape(hp.get_text())
        txt = re.sub(r"\s+", " ", txt).strip()
        txt = re.sub(r"(©|Copyright|\bAll rights reserved\b).{0,160}$", "", txt, flags=re.I)
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ..."
        return txt
    except Exception:
        return ""

# ====================== 可调默认 ======================
MAX_NEW_TOKENS_DEFAULT = 110
TEMPERATURE_DEFAULT = 0.25
TOP_P_DEFAULT = 0.9
REPETITION_PENALTY_DEFAULT = 1.02
MAX_RETRIES_DEFAULT = 2

MIN_WORDS_DEFAULT = 28
MAX_WORDS_DEFAULT = 70

_generate_lock = Lock()     # 生成锁
_io_lock = Lock()           # 写盘锁

# 三类误导策略（优化后的英文说明）
STRATEGY_MAP = {
    "out_of_context": (
        "Use a REAL, verifiable fact from an adjacent domain or a different time/place/population, "
        "then apply it OUTSIDE its valid scope to support the WRONG THEME. "
        "The True premise must be specific, the twist must explicitly state the scope shift, "
        "and the final Incorrect conclusion must assert the WRONG THEME as a fact."
    ),
    "false_causality": (
        "Make a WRONG THEME sound plausible by confusing correlation/sequence with causation. "
        "Use language like 'after', 'then', or 'because' to imply a causal link while ignoring confounders. "
        "The final sentence must assert the WRONG THEME as if it were a causal truth."
    ),
    "appeal_to_authority": (
        "Justify the WRONG THEME by claiming broad agreement or expert endorsement WITHOUT verifiable support. "
        "Use vague authorities and overstate certainty. "
        "The final sentence must assert the WRONG THEME as an authoritative conclusion."
    )
}

# ====================== 小工具 ======================
def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def read_jsonl(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception as e:
                eprint(f"[WARN] line {i+1} JSON parse failed: {e}")
            if limit is not None and len(out) >= limit:
                break
    return out

def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def split_sentences_simple(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

# ====================== 提示词 ======================
SUPPORT_PROMPT = """You are a rigorous fact-checking assistant.

Write EXACTLY three sentences (total {min_words}-{max_words} words) to SUPPORT the correct answer.
Rules:
- Focus on ONE coherent claim; do NOT list multiple alternatives.
- No semicolons, no bullet points, no lists, no labeled steps.
- Sentence 1 [MAIN EVIDENCE]: A key fact consistent with the correct answer.
- Sentence 2 [CONTEXT FACT]: One verifiably true detail that adds context (no invented names/dates/institutions).
- Sentence 3 [LOGICAL LINK]: Explain why these facts imply the answer is likely/correct.
- Avoid hedge endings like "unclear", "cannot be determined", or "debated".

QUESTION: {question}
CORRECT THEME: {answer}
OPTIONAL SOURCE SNIPPET (context only): {source_summary}

Now produce the three sentences.
"""

# 优化后的误导策略生成prompt
MISLEAD_PROMPT_TMPL = """# Misleading Strategy Generation Task

## Task Objective
Generate three different misleading strategies for the wrong theme, ensuring:
1. Strongly support the wrong theme
2. Strongly oppose the correct theme  
3. Meet technical metrics requirements

## Input Information
**QUESTION**: {question}
**CORRECT THEME**: {correct_theme}
**WRONG THEME**: {wrong_theme}
**SUPPORT INFORMATION**: {support_text}

## Quality Requirements

### Technical Metrics (Must Meet):
- Support-Mislead similarity < 0.6
- Support-Wrong theme similarity < 0.5
- Contradiction strength > 0.3
- Misleading relationship ✓

### Content Requirements:
1. **Strongly support wrong theme**: Clearly promote the wrong viewpoint
2. **Strongly oppose correct theme**: Directly challenge the correct information
3. **Language differentiation**: Avoid using same vocabulary and sentence structures as support
4. **Clear logic**: Three-part structure should be coherent and powerful

## Strategy Characteristics

### out_of_context (Out of Context)
- Extract facts from related but different contexts
- Distort and apply to current problem
- Create superficially reasonable false associations

### false_causality (False Causality)
- Establish non-existent causal relationships
- Use temporal sequence or correlation to disguise causation
- Ignore real causal mechanisms

### appeal_to_authority (Appeal to Authority)
- Cite seemingly authoritative but irrelevant sources
- Exaggerate individual opinions as general consensus
- Abuse expert names to support wrong conclusions

## Output Format
Write EXACTLY three sentences (total {min_words}-{max_words} words).
Each sentence MUST start with the required label, in this order:
1) True premise:
2) Misleading twist:
3) Incorrect conclusion:

Strategy to mislead: '{strategy}'.

Now produce the three labeled sentences for the wrong theme: "{wrong_theme}"
"""

# ====================== 轻量清洗 / 校验 ======================
UNCERTAIN_RE = re.compile(r"\b(unclear|cannot be determined|unknown|debated|not certain|no way to know)\b", re.I)
NEGATORS = re.compile(r"\b(not|no|never|hardly|rarely|seldom)\b", re.I)

def has_uncertainty(t: str) -> bool:
    return bool(UNCERTAIN_RE.search(t or ""))

def has_semicolon(t: str) -> bool:
    return ";" in (t or "")

def looks_like_list(t: str) -> bool:
    return bool(re.search(r"^\s*[-*]\s|\b(1\)|2\)|3\))", (t or "")))

def enforce_three_sentences(text: str) -> str:
    sents = split_sentences_simple(text)
    if len(sents) < 3:
        while len(sents) < 3:
            sents.append("This adds necessary context.")
    if len(sents) > 3:
        sents = sents[:3]
    sents = [s.rstrip(".!?") + "." for s in sents]
    return " ".join(sents).strip()

def word_count_between(text: str, lo: int, hi: int) -> bool:
    wc = word_count(text)
    return lo <= wc <= hi

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

MISLEAD_LABELS = ["True premise:", "Misleading twist:", "Incorrect conclusion:"]

def check_mislead_labels(text: str) -> bool:
    sents = split_sentences_simple(text)
    if len(sents) != 3:
        return False
    for i, lab in enumerate(MISLEAD_LABELS):
        if not sents[i].lower().startswith(lab.lower()):
            return False
    return True

def simple_match(needle: str, hay: str) -> bool:
    n = re.sub(r"[^a-z0-9 ]+", " ", (needle or "").lower()).strip()
    h = re.sub(r"[^a-z0-9 ]+", " ", (hay or "").lower()).strip()
    if not n:
        return True
    if n in h:
        return True
    ntoks = [t for t in n.split() if t]
    if not ntoks:
        return True
    htoks = set(h.split())
    overlap = sum(1 for t in ntoks if t in htoks)
    return overlap / max(1, len(ntoks)) >= 0.7

def validate_support_surface(text: str, min_words: int, max_words: int) -> Tuple[bool, str]:
    t = normalize_ws(text)
    if has_semicolon(t):
        return False, "semicolon"
    if looks_like_list(t):
        return False, "list_like"
    t = enforce_three_sentences(t)
    if not word_count_between(t, min_words, max_words):
        return False, f"wc={word_count(t)}"
    if has_uncertainty(t):
        return False, "uncertainty"
    return True, "ok"

def validate_mislead_surface(text: str, min_words: int, max_words: int, wrong_theme: str, strict_last_exact: bool=False) -> Tuple[bool, str]:
    t = normalize_ws(text)
    if has_semicolon(t):
        return False, "semicolon"
    t = enforce_three_sentences(t)
    if not check_mislead_labels(t):
        return False, "label_order"
    if not word_count_between(t, min_words, max_words):
        return False, f"wc={word_count(t)}"
    sents = split_sentences_simple(t)
    last = sents[2]
    if NEGATORS.search(last) and simple_match(wrong_theme, last):
        return False, "negated_wrong_theme"
    if strict_last_exact:
        # 必须以 "{wrong_theme}." 精确结尾
        if not last.rstrip().endswith(wrong_theme.rstrip(".") + "."):
            return False, "last_sentence_exact_end_required"
    else:
        if not simple_match(wrong_theme, last):
            return False, "last_sentence_not_asserting_wrong_theme"
    if has_uncertainty(t):
        return False, "uncertainty"
    return True, "ok"

def fallback_support(answer: str) -> str:
    return (
        f"The available descriptions consistently align with the view that {answer}. "
        "Independent contextual accounts point in the same direction without relying on specific named sources. "
        "Taken together, these observations support this conclusion."
    )

def fallback_mislead(question: str, wrong_theme: str, strategy: str, support_text: str) -> str:
    if "caus" in strategy.lower():
        twist = "People often treat correlation or temporal ordering as if it proved causation."
    elif "authority" in strategy.lower():
        twist = "Claims are framed as if there were a broad expert consensus, without verifiable support."
    else:
        twist = "A fact from a related context is applied as if it directly established the answer here."
    return (
        f"True premise: The topic is commonly described in ways that seem to favor this interpretation. "
        f"Misleading twist: {twist} "
        f"Incorrect conclusion: Therefore, {wrong_theme}."
    )

# =============== NLI 校验器 ===============
class NLIValidator:
    def __init__(self, model_name="microsoft/deberta-v3-base-mnli", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.m = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.m.to(self.device).eval()
        self.id2label = self.m.config.id2label
    @torch.no_grad()
    def score(self, premise: str, hypothesis: str):
        if not premise or not hypothesis:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.m(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        out = {}
        for i, p in enumerate(probs):
            key = self.id2label[i].lower()
            out[key] = float(p)
        for k in ("entailment","neutral","contradiction"):
            out.setdefault(k, 0.0)
        return out

def nli_validate_support(premise: str, support_text: str, nli: Optional[NLIValidator],
                         th_entail=0.6, th_contra=0.2, th_entail_logic=0.5):
    if not nli or not premise:
        return True, None
    sents = split_sentences_simple(support_text)
    sents = (sents + ["", "", ""])[:3]
    results = []
    for s in sents:
        results.append(nli.score(premise, s))
    pass_fact = all(
        (results[i]["entailment"] >= th_entail) and (results[i]["contradiction"] <= th_contra)
        for i in [0, 1]
    )
    pass_logic = (results[2]["entailment"] >= th_entail_logic) or (results[2]["neutral"] >= 0.5)
    passed = bool(pass_fact and pass_logic)
    detail = {
        "sent1": results[0],
        "sent2": results[1],
        "sent3": results[2],
        "passed": passed,
        "thresholds": {
            "entail_fact": th_entail,
            "contra_fact": th_contra,
            "entail_logic": th_entail_logic
        }
    }
    return passed, detail

# =============== 增强：唯一ID + 断点续跑 + 线程安全写出 ===============
def make_uid(raw: Dict[str, Any], idx: int) -> str:
    rid = pick_first(raw, ["id", "uid", "qid"], None)
    if rid:
        return str(rid)
    basis = f"{pick_first(raw, ['question','prompt','stem','query','instruction'], '')}||{pick_first(raw,['answer'],'')}"
    h = hashlib.md5(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"item_{idx}_{h}"

def load_existing_uids(out_path: str) -> Set[str]:
    uids: Set[str] = set()
    if not out_path or not os.path.exists(out_path):
        return uids
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                j = json.loads(s)
                if "id" in j:
                    uids.add(str(j["id"]))
            except Exception:
                continue
    return uids

def append_jsonl_threadsafe(path: str, row: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _io_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

# ======================= 模型、生成、主流程 =======================
def build_textgen(model_id: str):
    eprint(f"[info] Loading model: {model_id}")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=False, use_fast=True)
    quant_cfg = None
    if _HAS_BNB:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            quant_cfg = None
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                quantization_config=quant_cfg,
                device_map="auto"
            )
        except Exception as e:
            eprint(f"[warn] 4bit failed, fallback fp16/bf16: {e}")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                device_map="auto",
                torch_dtype=dtype
            )
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, local_files_only=False).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, local_files_only=False)
        model.to("cpu")
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False
    except Exception:
        pass
    try:
        if getattr(tok, "pad_token_id", None) is None:
            if getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(model, "resize_token_embeddings"):
                    model.resize_token_embeddings(len(tok))
    except Exception:
        pass
    return model, tok

def chat_generate(model, tok, sys_prompt: str, user_prompt: str,
                  max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
                  temperature=TEMPERATURE_DEFAULT,
                  top_p=TOP_P_DEFAULT,
                  repetition_penalty=REPETITION_PENALTY_DEFAULT) -> str:
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"<<SYS>>\n{sys_prompt}\n<</SYS>>\n\n<<USER>>\n{user_prompt}\n<</USER>>\n"
    inputs = tok(prompt, return_tensors="pt")
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with _generate_lock:
        out_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            use_cache=False
        )
    text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"\s+\n", " ", text).strip()

# =============== 主题规范化 ===============
HEDGY = ("unclear","commonly","believed","may","might","possibly","often","sometimes")

def pick_canonical(ca: List[str]) -> str:
    ca = [sanitize_text(str(x)) for x in ca if x]
    if not ca:
        return ""
    def score(s):
        t = s.lower()
        hedgy = any(w in t for w in HEDGY)
        return (hedgy, len(s))
    return sorted(ca, key=score)[0]

def merge_wrong_answers(wrong_answers: List[str]) -> str:
    """合并所有错误答案为统一的错误主题"""
    if not wrong_answers:
        return "incorrect assumption"
    
    # 如果只有一个错误答案，直接返回
    if len(wrong_answers) == 1:
        return wrong_answers[0]
    
    # 如果有多个错误答案，合并成统一的主题
    # 例如: ["Carlsberg", "Guinness", "Heineken"] → "specific beer brands are the best"
    
    # 检测是否是品牌/具体名称类错误
    brand_keywords = ["is", "are", "the best", "the greatest", "favorite"]
    if any(any(kw in ans.lower() for kw in brand_keywords) for ans in wrong_answers):
        return "specific brands are considered the best"
    
    # 检测是否是主观性否定类错误
    subjective_denial = ["objective", "definitely", "certainly", "absolutely"]
    if any(any(kw in ans.lower() for kw in subjective_denial) for ans in wrong_answers):
        return "the question has an objective answer"
    
    # 默认合并策略：用分号连接
    return "; ".join(wrong_answers)

# ====================== 生成流程（逐题） ======================
@dataclass
class GenConfig:
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT
    temperature: float = TEMPERATURE_DEFAULT
    top_p: float = TOP_P_DEFAULT
    repetition_penalty: float = REPETITION_PENALTY_DEFAULT
    max_retries: int = MAX_RETRIES_DEFAULT
    min_words: int = MIN_WORDS_DEFAULT
    max_words: int = MAX_WORDS_DEFAULT

def build_support_prompts(question: str, answer_theme: str, source_summary: str, cfg: GenConfig) -> Tuple[str, str]:
    sys_p = SUPPORT_PROMPT.format(
        min_words=cfg.min_words, max_words=cfg.max_words,
        question=question, answer=answer_theme,
        source_summary=source_summary or "N/A"
    )
    return sys_p, "Produce the three sentences now."

def build_mislead_prompts(question: str, wrong_theme: str, correct_theme: str, strategy: str,
                          support_text: str, source_summary: str, cfg: GenConfig) -> Tuple[str, str]:
    sys_p = MISLEAD_PROMPT_TMPL.format(
        min_words=cfg.min_words, max_words=cfg.max_words,
        strategy=strategy,
        question=question,
        correct_theme=correct_theme,
        wrong_theme=wrong_theme,
        support_text=support_text or "(none)",
        source_summary=source_summary or "N/A"
    )
    return sys_p, "Generate the three labeled sentences."

def generate_with_retry(gen_fn, validator_fn, fallback_fn, max_retries: int) -> str:
    last_reason = ""
    for attempt in range(max_retries + 1):
        out = gen_fn(attempt)
        ok, reason = validator_fn(out)
        if ok:
            return enforce_three_sentences(out)
        eprint(f"[RETRY] attempt={attempt} failed: {reason} | text={out[:160]!r}")
        last_reason = reason
    eprint(f"[FALLBACK] reason={last_reason}")
    return fallback_fn()

def process_one(sample_idx: int, raw: Dict[str, Any], model, tok, cfg: GenConfig, args,
                nli: Optional[NLIValidator]) -> Dict[str, Any]:
    q = sanitize_text(pick_first(raw, ["question", "prompt", "stem", "query", "instruction"], ""))
    if not q:
        raise ValueError("missing question")

    # === 确定正确主题 ===
    ans = pick_first(raw, ["answer"], None)
    if ans is None:
        ca = raw.get("correct_answers") or []
        if ca:
            if args.use_all_correct:
                ans = "; ".join(sanitize_text(str(x)) for x in ca if x)
            else:
                ans = pick_canonical(ca)
        else:
            raise ValueError("missing answer theme (answer / correct_answers)")
    ans = sanitize_text(str(ans))

    # === 收集所有错误答案并合并为统一错误主题 ===
    wrongs: List[str] = []
    if isinstance(raw.get("wrong_answers"), list) and raw["wrong_answers"]:
        wrongs = [sanitize_text(str(x)) for x in raw["wrong_answers"]]
    else:
        w = pick_first(raw, ["wrong_theme", "distractor"], None)
        wrongs = [sanitize_text(str(w))] if w else [f"not {ans}"]

    # 合并所有错误答案为统一错误主题（核心改动）
    wrong_theme = merge_wrong_answers(wrongs)

    # === 来源摘要 ===
    source_summary = sanitize_text(pick_first(raw, ["source_text", "source_summary"], ""))
    if not source_summary:
        src_url = sanitize_text(pick_first(raw, ["source"], ""))
        if args.fetch_source and src_url.startswith(("http://", "https://")):
            fetched = fetch_url_text(src_url, timeout=args.fetch_timeout, max_chars=max(300, args.max_source_chars))
            if fetched:
                source_summary = fetched
        else:
            source_summary = src_url
    if len(source_summary) > args.max_source_chars:
        source_summary = source_summary[:args.max_source_chars] + " ..."

    # === 先生成统一 support ===
    def support_gen(attempt: int):
        temp = min(cfg.temperature + 0.08 * attempt, 0.9)
        sys_p, user_p = build_support_prompts(q, ans, source_summary, cfg)
        return chat_generate(model, tok, sys_p, user_p,
                             max_new_tokens=cfg.max_new_tokens,
                             temperature=temp,
                             top_p=cfg.top_p,
                             repetition_penalty=cfg.repetition_penalty)
    def support_val(text: str):
        return validate_support_surface(text, cfg.min_words, cfg.max_words)

    sup_out = generate_with_retry(
        support_gen, support_val, lambda: fallback_support(ans), max_retries=cfg.max_retries
    )

    # === NLI 校验 ===
    nli_detail = None
    nli_pass = True
    if args.nli_on:
        nli_pass, nli_detail = nli_validate_support(
            source_summary, sup_out, nli,
            th_entail=args.nli_entail_th,
            th_contra=args.nli_contra_th,
            th_entail_logic=args.nli_logic_th
        )

    # === 为统一错误主题生成三种误导策略 ===
    mislead_strategies: Dict[str, Dict[str, str]] = {}
    for key, strat in STRATEGY_MAP.items():
        def mis_gen_factory(k=key, s=strat, w=wrong_theme):
            def _gen(attempt: int):
                temp = min(cfg.temperature + 0.08 * attempt, 0.9)
                sys_p, user_p = build_mislead_prompts(q, w, ans, s, sup_out, source_summary, cfg)
                return chat_generate(model, tok, sys_p, user_p,
                                     max_new_tokens=cfg.max_new_tokens,
                                     temperature=temp,
                                     top_p=cfg.top_p,
                                     repetition_penalty=cfg.repetition_penalty)
            return _gen
        def mis_val(text: str, w=wrong_theme):
            return validate_mislead_surface(text, cfg.min_words, cfg.max_words, w, strict_last_exact=args.strict_last_sentence_exact)

        mis_text = generate_with_retry(
            mis_gen_factory(), mis_val,
            lambda w=wrong_theme, s=strat: fallback_mislead(q, w, s, sup_out),
            max_retries=cfg.max_retries
        )
        mislead_strategies[key] = {"text": mis_text}

    # 新的输出结构
    out = {
        "id": pick_first(raw, ["id", "uid", "qid"], None) or make_uid(raw, sample_idx),
        "question": q,
        "answer_theme": ans,
        "wrong_theme": wrong_theme,
        "support": {
            "text": sup_out,
            "nli_validation": nli_detail
        },
        "misleadings": mislead_strategies
    }
    out["_flags"] = {"nli_pass": nli_pass}
    
    # 保留原始错误答案用于调试（可选）
    if wrongs:
        out["_original_wrong_answers"] = wrongs
    
    return out

# ====================== CSV 报告 ======================
def append_report_csv(report_path: str, row: Dict[str, Any]):
    is_new = not os.path.exists(report_path)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with _io_lock:
        with open(report_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow([
                    "id","nli_pass",
                    "s1_entail","s1_neutral","s1_contra",
                    "s2_entail","s2_neutral","s2_contra",
                    "s3_entail","s3_neutral","s3_contra"
                ])
            nv = row.get("support",{}).get("nli_validation")
            if nv:
                w.writerow([
                    row.get("id"),
                    nv.get("passed"),
                    nv.get("sent1",{}).get("entailment", ""), nv.get("sent1",{}).get("neutral",""), nv.get("sent1",{}).get("contradiction",""),
                    nv.get("sent2",{}).get("entailment", ""), nv.get("sent2",{}).get("neutral",""), nv.get("sent2",{}).get("contradiction",""),
                    nv.get("sent3",{}).get("entailment", ""), nv.get("sent3",{}).get("neutral",""), nv.get("sent3",{}).get("contradiction",""),
                ])
            else:
                w.writerow([row.get("id"), "", "", "", "", "", "", "", "", "", ""])

# ====================== 主程序 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="out_topicwise.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    parser.add_argument("--top_p", type=float, default=TOP_P_DEFAULT)
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY_DEFAULT)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES_DEFAULT)

    parser.add_argument("--min_words", type=int, default=MIN_WORDS_DEFAULT)
    parser.add_argument("--max_words", type=int, default=MAX_WORDS_DEFAULT)

    # 抓取 source 链接与摘要长度
    parser.add_argument("--fetch_source", action="store_true",
                        help="If set, try to fetch source URL and use its cleaned text as context.")
    parser.add_argument("--fetch_timeout", type=float, default=6.0)
    parser.add_argument("--max_source_chars", type=int, default=800)

    # 断点续跑 / 追加写
    parser.add_argument("--resume", action="store_true",
                        help="Skip items whose id already exists in out_jsonl and append new ones.")
    parser.add_argument("--truncate_output", action="store_true",
                        help="Truncate out_jsonl before running (disables resume).")

    # 是否合并所有 correct_answers
    parser.add_argument("--use_all_correct", action="store_true",
                        help="Use all entries in correct_answers as a single combined theme.")

    # NLI 参数
    parser.add_argument("--nli_on", action="store_true")
    parser.add_argument("--nli_model", type=str, default="microsoft/deberta-v3-base-mnli")
    parser.add_argument("--nli_entail_th", type=float, default=0.6)
    parser.add_argument("--nli_contra_th", type=float, default=0.2)
    parser.add_argument("--nli_logic_th", type=float, default=0.5)

    # 其他开关
    parser.add_argument("--strict_last_sentence_exact", action="store_true",
                        help="Require mislead last sentence to end with EXACT '{wrong_theme}.'")

    # 报告
    parser.add_argument("--report_csv", type=str, default="",
                        help="Optional CSV path to write NLI metrics summary.")

    args = parser.parse_args()

    # 复现性
    try:
        random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except Exception:
        pass

    # 输出策略
    if args.truncate_output and os.path.exists(args.out_jsonl):
        eprint(f"[info] Truncate existing output: {args.out_jsonl}")
        open(args.out_jsonl, "w", encoding="utf-8").close()

    existing_uids: Set[str] = set()
    if args.resume and os.path.exists(args.out_jsonl):
        existing_uids = load_existing_uids(args.out_jsonl)
        eprint(f"[resume] Found existing items: {len(existing_uids)} (will skip)")

    eprint("🔄 Loading model…")
    model, tok = build_textgen(args.model_id)

    nli = None
    if args.nli_on:
        eprint(f"🔎 Loading NLI model: {args.nli_model}")
        nli = NLIValidator(args.nli_model)

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_retries=args.max_retries,
        min_words=args.min_words,
        max_words=args.max_words
    )

    eprint(f"📥 Loading data: {args.input}")
    rows = read_jsonl(args.input, args.limit)
    eprint(f"➡️  samples (raw): {len(rows)}")

    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for i, r in enumerate(rows):
        uid = pick_first(r, ["id", "uid", "qid"], None) or make_uid(r, i)
        if uid in existing_uids:
            continue
        tasks.append((i, r))
    eprint(f"➡️  to-process after resume filter: {len(tasks)}")

    # 并发/单线程，边跑边写
    if args.threads and args.threads > 1:
        eprint(f"🧵 Using ThreadPoolExecutor: threads={args.threads}")
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = {ex.submit(process_one, i, r, model, tok, cfg, args, nli): (i, r) for i, r in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Generating", unit="item"):
                i, r = futs[fut]
                try:
                    out = fut.result()
                    append_jsonl_threadsafe(args.out_jsonl, out)
                    if args.report_csv:
                        append_report_csv(args.report_csv, out)
                except Exception as e:
                    eprint(f"[ERROR] item {i} failed: {e}")
    else:
        for i, r in tqdm(tasks, desc="Generating", unit="item"):
            try:
                out = process_one(i, r, model, tok, cfg, args, nli)
                append_jsonl_threadsafe(args.out_jsonl, out)
                if args.report_csv:
                    append_report_csv(args.report_csv, out)
            except Exception as e:
                eprint(f"[ERROR] item {i} failed: {e}")

    eprint("✅ Done.")

if __name__ == "__main__":
    main()