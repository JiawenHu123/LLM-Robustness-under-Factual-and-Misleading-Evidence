import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 1. DATA LOADING ====================
print("Loading data...")
file_path = "all_results_with_correct.csv"

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load: {e}")
    exit()

print(f"Data loaded successfully! Total rows: {len(df)}")
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# ==================== 2. DATA CLEANING ====================
df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce')

for col in ['question_id', 'gold_answer', 'model_answer', 'model', 'strategy']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df['model_simple'] = df['model'].replace({
    'gemma2': 'gemma',
    'gemma': 'gemma',
    'llama3.1': 'llama',
    'mistral': 'mistral'
})

# ==================== 3. STRATEGY → CONDITION MAPPING ====================
strategy_to_condition = {
    'clean_only': 'no_evidence',
    'factual_only': 'support_only',
    'appeal_only': 'appeal_only',
    'OOT_only': 'out_of_context_only',
    'falseC_only': 'false_causality_only',
    'factual_appeal': 'support+appeal',
    'factual_ooc': 'support+out_of_context',
    'factual_falseC': 'support+false_causality'
}

df['test_set'] = df['strategy'].map(strategy_to_condition)

print("\nDATA MAPPING CHECK")
print(df[['strategy', 'test_set']].head())

# ==================== 4. ROBUSTNESS (SCR) CALCULATION ====================
fact_condition = 'support_only'
misleading_conditions = [
    'appeal_only',
    'out_of_context_only',
    'false_causality_only'
]

mis_name_map = {
    'appeal_only': 'Appeal to Authority',
    'out_of_context_only': 'Out of Context',
    'false_causality_only': 'False Causality'
}

all_results = []

for model in ['gemma', 'llama', 'mistral']:
    print(f"\n==============================")
    print(f"Model: {model.upper()}")
    print(f"==============================")

    df_model = df[df['model_simple'] == model]
    question_ids = df_model['question_id'].unique()
    total_questions = len(question_ids)

    contradictory_questions = 0

    for qid in question_ids:
        fact_row = df_model[
            (df_model['question_id'] == qid) &
            (df_model['test_set'] == fact_condition)
        ]

        if fact_row.empty:
            continue

        fact_answer = fact_row['model_answer'].iloc[0]

        is_contradictory = False

        for mis_cond in misleading_conditions:
            mis_row = df_model[
                (df_model['question_id'] == qid) &
                (df_model['test_set'] == mis_cond)
            ]

            if mis_row.empty:
                continue

            mis_answer = mis_row['model_answer'].iloc[0]

            if str(fact_answer) != str(mis_answer):
                is_contradictory = True
                break

        if is_contradictory:
            contradictory_questions += 1

    scr = contradictory_questions / total_questions
    robustness = 1 - scr

    print(f"Total questions: {total_questions}")
    print(f"Contradictory: {contradictory_questions}")
    print(f"SCR: {scr:.3f}")
    print(f"Robustness score: {robustness:.3f}")

    all_results.append({
        'model': model.upper(),
        'total': total_questions,
        'contradictory': contradictory_questions,
        'scr': scr,
        'robustness': robustness,
        'robustness_percent': robustness * 100
    })

# ==================== 5. VISUALIZATION ====================
print("\nGenerating robustness visualization...")

results_df = pd.DataFrame(all_results)

fig, ax = plt.subplots(figsize=(6.5, 4.8))  # 适合单栏论文


models = results_df['model']
scores = results_df['robustness']
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)

for bar, score, row in zip(bars, scores, all_results):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015,
        f"{score:.3f}\n({row['contradictory']}/{row['total']})",
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold'
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height / 2,
        f"{score*100:.1f}%",
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold',
        color='white'
    )

ax.set_ylabel('Robustness Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)

ax.set_title(
    'LLM Robustness Performance: Supportive vs. Misleading Evidence',
    fontsize=16,
    fontweight='bold',
    pad=16
)



ax.set_ylim(0, 1.0)

ax.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='50% reference')
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.figtext(
    0.5,
    0.01,
    'Robustness score = 1 − self-contradiction rate\n'
    'Self-contradiction: different answers for the same question under factual vs. misleading evidence',
    ha='center',
    fontsize=9,
    style='italic'
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('Robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('Robustness.pdf', bbox_inches='tight')
plt.show()

# ==================== 6. SAVE RESULTS ====================
results_df.to_csv('robustness_results.csv', index=False, encoding='utf-8-sig')
print("Saved: robustness_results.csv")

print("\nAnalysis completed successfully.")
