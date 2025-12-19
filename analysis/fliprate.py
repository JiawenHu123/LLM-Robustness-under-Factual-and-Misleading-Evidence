import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv("all_models_results_with_correct.csv")
df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce')

# 简化模型名称
df['model_simple'] = df['model'].replace({
    'gemma2': 'gemma',
    'gemma': 'gemma',
    'llama3.1': 'llama',
    'mistral': 'mistral'
})

# 定义策略到测试条件的映射
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

print("="*80)
print("CORRECT FLIP-RATE ANALYSIS (vs No-Evidence Baseline)")
print("="*80)

models = ['gemma', 'llama', 'mistral']
# 对比条件：所有非基线条件
conditions = ['support_only', 'appeal_only', 'out_of_context_only', 
              'false_causality_only', 'support+m1', 'support+m2', 'support+m3']

results = []

for model in models:
    print(f"\n{'='*50}")
    print(f"模型: {model.upper()}")
    print(f"{'='*50}")
    
    df_model = df[df['model_simple'] == model]
    
    # 1. 获取基线数据 (no_evidence)
    # 找到对应no_evidence的策略
    baseline_strategies = [s for s, c in strategy_to_condition.items() if c == 'no_evidence']
    baseline_data = df_model[df_model['strategy'].isin(baseline_strategies)]
    
    if len(baseline_data) == 0:
        print(f"⚠️ 警告: 模型 {model} 没有基线数据 (no_evidence)")
        continue
    
    baseline_dict = dict(zip(baseline_data['question_id'], baseline_data['is_correct']))
    total_questions = len(baseline_dict)
    
    print(f"总题数: {total_questions}")
    baseline_correct = sum(baseline_dict.values())
    print(f"基线正确: {baseline_correct}/{total_questions} ({baseline_correct/total_questions*100:.1f}%)")
    
    for condition in conditions:
        # 2. 获取条件数据
        # 找到对应条件的策略
        cond_strategies = [s for s, c in strategy_to_condition.items() if c == condition]
        cond_data = df_model[df_model['strategy'].isin(cond_strategies)]
        
        if len(cond_data) == 0:
            print(f"\n{condition}: 无数据")
            continue
            
        cond_dict = dict(zip(cond_data['question_id'], cond_data['is_correct']))
        
        # 3. 检查题目匹配
        matched_qids = set(baseline_dict.keys()).intersection(set(cond_dict.keys()))
        if len(matched_qids) != total_questions:
            print(f"  ⚠️ 警告: 题目匹配不完全 (基线:{total_questions}, 匹配:{len(matched_qids)})")
        
        # 4. 计算Flip-Rate
        beneficial = 0  # 0→1
        adversarial = 0  # 1→0
        stay_correct = 0  # 1→1
        stay_wrong = 0    # 0→0
        
        for qid in matched_qids:
            baseline_ans = baseline_dict[qid]
            cond_ans = cond_dict[qid]
            
            if baseline_ans == 0 and cond_ans == 1:
                beneficial += 1
            elif baseline_ans == 1 and cond_ans == 0:
                adversarial += 1
            elif baseline_ans == 1 and cond_ans == 1:
                stay_correct += 1
            else:  # 0→0
                stay_wrong += 1
        
        # 5. 计算比率
        bfr = beneficial / len(matched_qids) * 100 if len(matched_qids) > 0 else 0
        afr = adversarial / len(matched_qids) * 100 if len(matched_qids) > 0 else 0
        net_delta = bfr - afr
        
        # 6. 验证：计算准确率变化
        baseline_acc = baseline_correct / total_questions * 100
        cond_correct = sum(cond_dict.values())
        cond_acc = cond_correct / len(cond_dict) * 100 if len(cond_dict) > 0 else 0
        actual_delta = cond_acc - baseline_acc
        
        print(f"\n{condition}:")
        print(f"  匹配题目: {len(matched_qids)}/{total_questions}")
        print(f"  BFR (错→对): {bfr:.1f}% ({beneficial}/{len(matched_qids)})")
        print(f"  AFR (对→错): {afr:.1f}% ({adversarial}/{len(matched_qids)})")
        print(f"  净变化: {net_delta:+.1f}%")
        print(f"  基线准确率: {baseline_acc:.1f}%")
        print(f"  条件准确率: {cond_acc:.1f}%")
        print(f"  实际Δ准确率: {actual_delta:+.1f}%")
        print(f"  保持正确: {stay_correct}, 保持错误: {stay_wrong}")
        
        # 检查一致性
        if len(matched_qids) == total_questions and abs(net_delta - actual_delta) > 0.1:
            print(f"  ⚠️ 警告: 不一致! 差={abs(net_delta - actual_delta):.2f}%")
        
        results.append({
            'Model': model.upper(),
            'Condition': condition,
            'Matched_Questions': f"{len(matched_qids)}/{total_questions}",
            'Baseline_Acc': f"{baseline_acc:.1f}%",
            'Condition_Acc': f"{cond_acc:.1f}%",
            'BFR': f"{bfr:.1f}%",
            'AFR': f"{afr:.1f}%",
            'Net_Delta': f"{net_delta:+.1f}%",
            'Actual_Delta': f"{actual_delta:+.1f}%",
            'Stay_Correct': stay_correct,
            'Stay_Wrong': stay_wrong,
            'Beneficial_Flips': beneficial,
            'Adversarial_Flips': adversarial
        })

# 保存结果
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('correct_flip_rates.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: correct_flip_rates.csv")
    
    # 显示汇总表格
    print("\n" + "="*80)
    print("FLIP-RATE 汇总表格")
    print("="*80)
    
    # 创建透视表
    pivot_table = pd.pivot_table(results_df, 
                                 values=['BFR', 'AFR', 'Net_Delta'], 
                                 index=['Condition'], 
                                 columns=['Model'],
                                 aggfunc=lambda x: x.iloc[0] if len(x) > 0 else '')
    
    # 重新组织表格结构
    for condition in conditions:
        cond_data = results_df[results_df['Condition'] == condition]
        if len(cond_data) > 0:
            print(f"\n{condition}:")
            print(f"{'Model':<10} {'BFR':<10} {'AFR':<10} {'Net Δ':<10} {'Base Acc':<10} {'Cond Acc':<10}")
            print("-" * 70)
            for _, row in cond_data.iterrows():
                print(f"{row['Model']:<10} {row['BFR']:<10} {row['AFR']:<10} {row['Net_Delta']:<10} {row['Baseline_Acc']:<10} {row['Condition_Acc']:<10}")

# 生成LaTeX表格
print("\n" + "="*80)
print("LaTeX表格代码")
print("="*80)

# 为每个模型生成表格
for model in ['GEMMA', 'LLAMA', 'MISTRAL']:
    print(f"\n% {model} Flip-Rates")
    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{model}: Flip-Rate Analysis vs No-Evidence Baseline}}
\\label{{tab:fliprates_{model.lower()}}}
\\scriptsize
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Base Acc}} & \\textbf{{Cond Acc}} & \\textbf{{BFR}} & \\textbf{{AFR}} & \\textbf{{Net Δ}} & \\textbf{{Match}} \\\\
\\midrule
"""
    
    model_results = [r for r in results if r['Model'] == model]
    for result in model_results:
        # 格式化条件名称（去掉下划线）
        cond_name = result['Condition'].replace('_', ' ').replace('+', '+')
        if cond_name == 'support+m1': cond_name = 'S+Appeal'
        elif cond_name == 'support+m2': cond_name = 'S+OOC'
        elif cond_name == 'support+m3': cond_name = 'S+FalseC'
        
        latex_code += f"{cond_name} & {result['Baseline_Acc']} & {result['Condition_Acc']} & {result['BFR']} & {result['AFR']} & {result['Net_Delta']} & {result['Matched_Questions']} \\\\\n"
    
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\item \\scriptsize \\textit{Note:} BFR = Beneficial Flip Rate (wrong→correct), AFR = Adversarial Flip Rate (correct→wrong), Net Δ = BFR - AFR, Match = matched questions/total questions.
\end{tablenotes}
\end{table}
"""
    print(latex_code)

print("\n" + "="*80)
print("分析完成！")
print("="*80)

# 输出摘要统计
print("\n" + "="*80)
print("关键发现摘要")
print("="*80)

# 计算平均Flip-Rate
summary_stats = []
for model in ['GEMMA', 'LLAMA', 'MISTRAL']:
    model_results = [r for r in results if r['Model'] == model]
    
    # 计算不同条件下的平均
    bfr_values = [float(r['BFR'].replace('%', '')) for r in model_results]
    afr_values = [float(r['AFR'].replace('%', '')) for r in model_results]
    net_values = [float(r['Net_Delta'].replace('%', '')) for r in model_results]
    
    # 按条件类型分组
    supportive_results = [r for r in model_results if r['Condition'] == 'support_only']
    misleading_results = [r for r in model_results if r['Condition'] in ['appeal_only', 'out_of_context_only', 'false_causality_only']]
    mixed_results = [r for r in model_results if r['Condition'] in ['support+m1', 'support+m2', 'support+m3']]
    
    summary_stats.append({
        'Model': model,
        'Avg_BFR': f"{np.mean(bfr_values):.1f}%",
        'Avg_AFR': f"{np.mean(afr_values):.1f}%",
        'Avg_Net': f"{np.mean(net_values):+.1f}%",
        'Support_BFR': supportive_results[0]['BFR'] if supportive_results else 'N/A',
        'Max_Net': f"{max(net_values):+.1f}%" if net_values else 'N/A',
        'Min_Net': f"{min(net_values):+.1f}%" if net_values else 'N/A'
    })

summary_df = pd.DataFrame(summary_stats)
print("\n模型Flip-Rate统计摘要:")
print(summary_df.to_string(index=False))