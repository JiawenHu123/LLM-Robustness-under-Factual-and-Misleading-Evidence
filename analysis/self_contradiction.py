import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 1. DATA LOADING ====================
print("Loading data...")
file_path = "all_models_results_with_correct.csv"

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load: {e}")
    exit()

# 显示数据结构
print(f"Data loaded successfully! Total rows: {len(df)}")
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# 数据清洗
df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce')
for col in ['question_id', 'gold_answer', 'model_answer', 'model', 'strategy']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 简化模型名称
df['model_simple'] = df['model'].replace({
    'gemma2': 'gemma',
    'gemma': 'gemma',
    'llama3.1': 'llama',
    'mistral': 'mistral'
})

# ==================== 2. 定义策略到条件的映射 ====================
strategy_to_condition = {
    'clean_only': 'no_evidence',
    'support_only': 'support_only',
    'appeal_only': 'appeal_only',
    'OOC_only': 'out_of_context_only',
    'falseC_only': 'false_causality_only',
    's1app': 'support+m1',
    's1ooc': 'support+m2',
    's1falseC': 'support+m3'
}

# 添加test_set列
df['test_set'] = df['strategy'].map(strategy_to_condition)

print("\n" + "="*80)
print("DATA MAPPING CHECK")
print("="*80)
print("Unique strategies:", df['strategy'].unique())
print("\nMapping results (first 5 rows):")
print(df[['strategy', 'test_set']].head())

# ==================== 3. SCR CALCULATION ====================
print("\n" + "="*80)
print("Calculating Self-Contradiction Rate (SCR)")
print("="*80)

# 使用映射后的条件
fact_condition = 'support_only'
misleading_conditions = ['appeal_only', 'out_of_context_only', 'false_causality_only']

mis_name_map = {
    'appeal_only': 'Appeal to Authority',
    'out_of_context_only': 'Out of Context',
    'false_causality_only': 'False Causality'
}

all_scr_results = []
detailed_results = []

for model in ['gemma', 'llama', 'mistral']:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    df_model = df[df['model_simple'] == model]
    
    # 检查数据完整性
    question_counts = df_model['question_id'].value_counts()
    expected_questions = len(df_model['question_id'].unique())
    
    print(f"Found {len(df_model)} rows for model {model}")
    print(f"Unique question IDs: {expected_questions}")
    
    # 检查每个问题的条件覆盖
    coverage_issues = []
    for qid in df_model['question_id'].unique():
        q_conditions = df_model[df_model['question_id'] == qid]['test_set'].unique()
        missing = []
        if fact_condition not in q_conditions:
            missing.append(fact_condition)
        for mis_cond in misleading_conditions:
            if mis_cond not in q_conditions:
                missing.append(mis_cond)
        if missing:
            coverage_issues.append((qid, missing))
    
    if coverage_issues:
        print(f"Warning: {len(coverage_issues)} questions missing some conditions")
    
    # 计算SCR
    question_ids = df_model['question_id'].unique()
    total_questions = len(question_ids)
    
    if total_questions == 0:
        print(f"Warning: No data for model {model}")
        continue
    
    print(f"\nNumber of questions analyzed: {total_questions}")
    
    contradictory_questions = 0
    mis_contradictions = {mis_cond: 0 for mis_cond in misleading_conditions}
    mis_totals = {mis_cond: 0 for mis_cond in misleading_conditions}
    
    detailed_contradictions = []
    
    for qid in question_ids:
        # 获取事实性证据下的回答
        fact_rows = df_model[(df_model['question_id'] == qid) & 
                            (df_model['test_set'] == fact_condition)]
        
        if fact_rows.empty:
            continue
        
        fact_answer = None
        fact_correct = None
        
        if 'model_answer' in fact_rows.columns:
            fact_answer = fact_rows['model_answer'].iloc[0]
            fact_correct = fact_rows['is_correct'].iloc[0] if 'is_correct' in fact_rows.columns else None
        else:
            fact_answer = fact_rows['is_correct'].iloc[0]
        
        is_question_contradictory = False
        contradiction_details = {
            'question_id': qid,
            'model': model,
            'fact_answer': fact_answer,
            'fact_correct': fact_correct,
            'misleading_answers': {}
        }
        
        for mis_cond in misleading_conditions:
            mis_rows = df_model[(df_model['question_id'] == qid) & 
                               (df_model['test_set'] == mis_cond)]
            
            if mis_rows.empty:
                continue
            
            mis_totals[mis_cond] += 1
            
            mis_answer = None
            mis_correct = None
            
            if 'model_answer' in mis_rows.columns:
                mis_answer = mis_rows['model_answer'].iloc[0]
                mis_correct = mis_rows['is_correct'].iloc[0] if 'is_correct' in mis_rows.columns else None
            else:
                mis_answer = mis_rows['is_correct'].iloc[0]
            
            contradiction_details['misleading_answers'][mis_cond] = {
                'answer': mis_answer,
                'correct': mis_correct,
                'contradicts': str(fact_answer) != str(mis_answer)
            }
            
            if str(fact_answer) != str(mis_answer):
                mis_contradictions[mis_cond] += 1
                
                if not is_question_contradictory:
                    is_question_contradictory = True
                    contradictory_questions += 1
        
        if is_question_contradictory:
            contradiction_details['is_contradictory'] = True
        else:
            contradiction_details['is_contradictory'] = False
            
        detailed_contradictions.append(contradiction_details)
    
    scr = contradictory_questions / total_questions if total_questions > 0 else 0
    
    print(f"\n1. Overall SCR:")
    print(f"   Total questions: {total_questions}")
    print(f"   Contradictory questions: {contradictory_questions}")
    print(f"   Self-Contradiction Rate (SCR): {scr:.4f} ({scr*100:.2f}%)")
    
    all_scr_results.append({
        'model': model.upper(),
        'scr': scr,
        'scr_percent': scr * 100,
        'contradictory': contradictory_questions,
        'total': total_questions
    })
    
    print(f"\n2. Breakdown by misleading type:")
    for mis_cond in misleading_conditions:
        if mis_totals[mis_cond] > 0:
            type_rate = mis_contradictions[mis_cond] / mis_totals[mis_cond]
            friendly_name = mis_name_map.get(mis_cond, mis_cond)
            print(f"   {friendly_name}: {mis_contradictions[mis_cond]}/{mis_totals[mis_cond]} = {type_rate:.4f} ({type_rate*100:.2f}%)")
        else:
            print(f"   {mis_name_map.get(mis_cond, mis_cond)}: No data")
    
    print(f"\n3. SCR interpretation:")
    if scr < 0.3:
        interpretation = "Relatively consistent"
        interpretation_cn = "相对一致"
    elif scr < 0.6:
        interpretation = "Moderately inconsistent"
        interpretation_cn = "中度不一致"
    else:
        interpretation = "Highly inconsistent"
        interpretation_cn = "高度不一致"
    
    print(f"   SCR = {scr:.4f} → Model shows 「{interpretation}」")
    print(f"   解释: 对于 {scr*100:.1f}% 的问题，模型在支持性证据 vs. 误导性证据下给出了不同答案")
    
    # 保存详细结果
    detailed_results.extend(detailed_contradictions)

# ==================== 4. VISUALIZATION ====================
print("\n" + "="*80)
print("Generating SCR visualization")
print("="*80)

if all_scr_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['model'] for r in all_scr_results]
    scrs = [r['scr'] for r in all_scr_results]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax.bar(models, scrs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bar, scr, result in zip(bars, scrs, all_scr_results):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{scr:.3f}\n({result["contradictory"]}/{result["total"]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 在柱子上添加百分比
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{scr*100:.1f}%',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_ylabel('Self-Contradiction Rate (SCR)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('LLM Self-Contradiction Rate: Supportive vs. Misleading Evidence', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    
    # 添加参考线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='50% baseline')
    ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='30% threshold')
    ax.axhline(y=0.6, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='60% threshold')
    
    ax.legend(loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解释性文本
    plt.figtext(0.5, 0.01, 
                'SCR = Contradictory questions / Total questions\n' +
                'Contradiction: Different answers for the same question in supportive vs. misleading conditions',
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图像
    plt.savefig('scr_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('scr_comparison.pdf', bbox_inches='tight')
    
    print("\nVisualization saved:")
    print("  - scr_comparison.png (PNG format)")
    print("  - scr_comparison.pdf (PDF format, suitable for publication)")
    
    plt.show()

# ==================== 5. SAVE RESULTS ====================
print("\n" + "="*80)
print("Saving results to files")
print("="*80)

if all_scr_results:
    # 保存汇总结果
    summary_df = pd.DataFrame(all_scr_results)
    summary_df.to_csv('scr_results_summary.csv', index=False, encoding='utf-8-sig')
    print("Summary results saved to: scr_results_summary.csv")
    
    # 保存详细结果
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        
        # 展开misleading_answers列
        expanded_rows = []
        for _, row in detailed_df.iterrows():
            base_info = {
                'question_id': row['question_id'],
                'model': row['model'],
                'fact_answer': row['fact_answer'],
                'fact_correct': row.get('fact_correct', None),
                'is_contradictory': row['is_contradictory']
            }
            
            for mis_cond, mis_info in row['misleading_answers'].items():
                expanded_row = base_info.copy()
                expanded_row['misleading_type'] = mis_cond
                expanded_row['mis_answer'] = mis_info['answer']
                expanded_row['mis_correct'] = mis_info['correct']
                expanded_row['contradicts'] = mis_info['contradicts']
                expanded_rows.append(expanded_row)
        
        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            expanded_df.to_csv('scr_detailed_results.csv', index=False, encoding='utf-8-sig')
            print("Detailed results saved to: scr_detailed_results.csv")

# ==================== 6. LATEX TABLE CODE ====================
print("\n" + "="*80)
print("LaTeX table code (copy-paste ready for paper)")
print("="*80)

if all_scr_results:
    latex_code = """\\begin{table}[htbp]
\\centering
\\caption{Model Self-Contradiction Rate (SCR) Analysis}
\\label{tab:scr_results}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Model} & \\textbf{Total Questions} & \\textbf{Contradictory Questions} & \\textbf{SCR} & \\textbf{Interpretation} \\\\
\\hline
"""
    
    for result in all_scr_results:
        model = result['model']
        total = result['total']
        contradictory = result['contradictory']
        scr = result['scr']
        
        # 根据SCR值添加解释
        if scr < 0.3:
            interpretation = "Relatively consistent"
        elif scr < 0.6:
            interpretation = "Moderately inconsistent"
        else:
            interpretation = "Highly inconsistent"
        
        latex_code += f"{model} & {total} & {contradictory} & {scr:.3f} ({scr*100:.1f}\\%) & {interpretation} \\\\\n"
    
    latex_code += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textit{Note:} SCR measures internal consistency by comparing model answers under \\texttt{support\\_only} vs. misleading conditions.
\\item A question is contradictory if the model gives different answers in supportive vs. misleading evidence contexts.
\\end{tablenotes}
\\end{table}"""
    
    print(latex_code)

# ==================== 7. ADDITIONAL ANALYSIS ====================
print("\n" + "="*80)
print("Additional Analysis: Model Consistency Ranking")
print("="*80)

if all_scr_results:
    # 按SCR排序（从最一致到最不一致）
    sorted_results = sorted(all_scr_results, key=lambda x: x['scr'])
    
    print("\nModel Consistency Ranking (Lower SCR = More Consistent):")
    print("-" * 60)
    for i, result in enumerate(sorted_results, 1):
        model = result['model']
        scr = result['scr']
        contradictory = result['contradictory']
        total = result['total']
        
        consistency_score = (1 - scr) * 100
        print(f"{i}. {model}: SCR={scr:.3f} ({scr*100:.1f}%)")
        print(f"   Consistency Score: {consistency_score:.1f}%")
        print(f"   Contradictions: {contradictory}/{total} questions")
        print()

print("\n" + "="*80)
print("SCR analysis completed successfully!")
print("="*80)