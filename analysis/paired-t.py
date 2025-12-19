import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv("all_models_results_with_correct.csv")
print(f"数据加载成功！总行数: {len(df)}")
print("\n前5行数据:")
print(df.head())

# 数据清洗
df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce')

# 检查是否有缺失值
print(f"\nis_correct列的有效值: {df['is_correct'].notnull().sum()}/{len(df)}")

# 简化模型名称
df['model_simple'] = df['model'].replace({
    'gemma2': 'gemma',
    'gemma': 'gemma',
    'llama3.1': 'llama',
    'mistral': 'mistral'
})

# 定义策略到测试条件的映射（根据您之前的设置）
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
print("MODEL COMPARISON ANALYSIS - 模型间性能对比")
print("="*80)

# 2. 定义要分析的关键条件
key_conditions = [
    'no_evidence',           # 基线
    'support_only',          # 纯支持
    'appeal_only',           # 纯诉诸权威
    'out_of_context_only',   # 纯断章取义
    'false_causality_only',  # 纯虚假因果
    'support+m1',            # 混合（诉诸权威）
    'support+m2',            # 混合（断章取义）
    'support+m3'             # 混合（虚假因果）
]

# 3. 计算每个模型在每个条件下的准确率
print("\n" + "="*60)
print("各模型在各条件下的准确率 (%)")
print("="*60)

accuracy_summary = []
for model in ['gemma', 'llama', 'mistral']:
    model_accuracies = {'Model': model.upper()}
    df_model = df[df['model_simple'] == model]
    
    for condition in key_conditions:
        # 找到对应这个条件的所有策略
        strategies_for_condition = [s for s, c in strategy_to_condition.items() if c == condition]
        if strategies_for_condition:
            cond_data = df_model[df_model['strategy'].isin(strategies_for_condition)]
        else:
            cond_data = df_model[df_model['test_set'] == condition]
            
        if len(cond_data) > 0:
            accuracy = cond_data['is_correct'].mean() * 100
            model_accuracies[condition] = f"{accuracy:.1f}%"
            model_accuracies[f"{condition}_raw"] = accuracy  # 保存原始值用于统计
        else:
            model_accuracies[condition] = "N/A"
            model_accuracies[f"{condition}_raw"] = np.nan
    
    accuracy_summary.append(model_accuracies)

# 显示准确率表格
acc_df = pd.DataFrame(accuracy_summary)
display_cols = ['Model'] + key_conditions
print(acc_df[display_cols].to_string(index=False))

# 4. 模型间统计检验（独立样本t检验）
print("\n" + "="*80)
print("MODEL-TO-MODEL COMPARISONS (Independent t-tests)")
print("="*80)

# 定义要比较的模型对
model_pairs = [
    ('gemma', 'llama', 'Gemma vs Llama'),
    ('gemma', 'mistral', 'Gemma vs Mistral'),
    ('llama', 'mistral', 'Llama vs Mistral')
]

comparison_results = []

for condition in key_conditions:
    print(f"\n{'='*50}")
    print(f"条件: {condition}")
    print(f"{'='*50}")
    
    # 收集三个模型在该条件下的所有样本数据
    model_data = {}
    for model in ['gemma', 'llama', 'mistral']:
        df_model = df[df['model_simple'] == model]
        # 找到对应这个条件的所有策略
        strategies_for_condition = [s for s, c in strategy_to_condition.items() if c == condition]
        if strategies_for_condition:
            model_cond_data = df_model[df_model['strategy'].isin(strategies_for_condition)]
        else:
            model_cond_data = df_model[df_model['test_set'] == condition]
            
        if len(model_cond_data) > 0:
            model_data[model] = model_cond_data['is_correct'].values
            print(f"  {model.upper()}: {len(model_cond_data)} 个样本，准确率: {model_cond_data['is_correct'].mean()*100:.1f}%")
    
    # 进行两两比较
    for model1, model2, pair_name in model_pairs:
        if model1 in model_data and model2 in model_data:
            data1 = model_data[model1]
            data2 = model_data[model2]
            
            # 检查数据有效性
            if len(data1) > 10 and len(data2) > 10:
                # 独立样本t检验（Welch's，不假设方差齐性）
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                # 计算描述统计
                mean1 = np.mean(data1) * 100
                mean2 = np.mean(data2) * 100
                mean_diff = mean1 - mean2
                
                # 计算Cohen's d（独立样本）
                n1, n2 = len(data1), len(data2)
                var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                cohens_d = mean_diff / (pooled_std * 100) if pooled_std > 0 else 0
                
                # 显著性标记
                if p_value < 0.001:
                    sig = "***"
                    sig_text = "p < .001"
                elif p_value < 0.01:
                    sig = "**"
                    sig_text = "p < .01"
                elif p_value < 0.05:
                    sig = "*"
                    sig_text = "p < .05"
                else:
                    sig = "n.s."
                    sig_text = f"p = {p_value:.3f}"
                
                # 效应量解释
                if abs(cohens_d) < 0.2:
                    effect_size = "极小"
                elif abs(cohens_d) < 0.5:
                    effect_size = "小"
                elif abs(cohens_d) < 0.8:
                    effect_size = "中"
                else:
                    effect_size = "大"
                
                # 输出结果
                print(f"\n{pair_name}:")
                print(f"  {model1.upper()}: {mean1:.1f}% (n={n1})")
                print(f"  {model2.upper()}: {mean2:.1f}% (n={n2})")
                print(f"  差异: {mean_diff:+.1f}%")
                print(f"  t = {t_stat:.3f}, {sig_text}")
                print(f"  Cohen's d = {cohens_d:.3f} ({effect_size})")
                print(f"  结果: {sig}")
                
                # 保存结果
                comparison_results.append({
                    'Condition': condition,
                    'Comparison': pair_name,
                    'Model1': model1.upper(),
                    'Model2': model2.upper(),
                    'Mean1': f"{mean1:.1f}%",
                    'Mean2': f"{mean2:.1f}%",
                    'Difference': f"{mean_diff:+.1f}%",
                    't': f"{t_stat:.3f}",
                    'p': sig_text,
                    'Cohen_d': f"{cohens_d:.3f}",
                    'Effect_Size': effect_size,
                    'Significance': sig
                })
            else:
                print(f"\n{pair_name}: 数据不足 ({len(data1)}, {len(data2)})")

# 5. 保存详细结果
print("\n" + "="*80)
print("保存结果...")
print("="*80)

if comparison_results:
    # 转换为DataFrame
    results_df = pd.DataFrame(comparison_results)
    
    # 保存到CSV
    results_df.to_csv('model_comparison_detailed.csv', index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: model_comparison_detailed.csv")
    
    # 6. 生成汇总表格（只显示显著结果）
    print("\n" + "="*80)
    print("显著差异汇总 (p < .05)")
    print("="*80)
    
    significant_results = results_df[results_df['Significance'].isin(['*', '**', '***'])]
    if len(significant_results) > 0:
        summary_cols = ['Condition', 'Comparison', 'Difference', 't', 'p', 'Cohen_d', 'Effect_Size']
        print(significant_results[summary_cols].to_string(index=False))
        
        # 7. 生成LaTeX表格代码
        print("\n" + "="*80)
        print("LaTeX表格代码（显著结果）")
        print("="*80)
        
        latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Significant Model Differences Across Evidence Conditions}
\\label{tab:model_differences}
\\small
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Condition} & \\textbf{Comparison} & \\textbf{Difference} & \\textbf{t} & \\textbf{p} \\\\
\\midrule
"""
        
        for _, row in significant_results.iterrows():
            condition_name = row['Condition'].replace('_', '\\_')
            latex_table += f"{condition_name} & {row['Comparison']} & {row['Difference']} & {row['t']} & {row['p']} \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\vspace{0.2em}
\\footnotesize
\\textit{Note:} Only statistically significant comparisons ($p < .05$) are shown.
\\end{table}
"""
        
        print(latex_table)
    else:
        print("无显著差异 (所有 p > .05)")

# 8. 创建条件分组统计
print("\n" + "="*80)
print("模型总体性能排名")
print("="*80)

# 计算每个模型的总体准确率
model_overall = []
for model in ['gemma', 'llama', 'mistral']:
    df_model = df[df['model_simple'] == model]
    overall_acc = df_model['is_correct'].mean() * 100
    model_overall.append({
        'Model': model.upper(),
        'Overall Accuracy': f"{overall_acc:.1f}%",
        'Total Questions': len(df_model)
    })

overall_df = pd.DataFrame(model_overall)
print(overall_df.to_string(index=False))

# 9. 按条件类型分组统计
print("\n" + "="*80)
print("按条件类型分组的模型表现")
print("="*80)

# 定义条件类型
condition_groups = {
    'Pure Supportive': ['support_only'],
    'Pure Misleading': ['appeal_only', 'out_of_context_only', 'false_causality_only'],
    'Mixed Evidence': ['support+m1', 'support+m2', 'support+m3'],
    'Baseline': ['no_evidence']
}

group_stats = []
for group_name, conditions in condition_groups.items():
    group_data = []
    for model in ['gemma', 'llama', 'mistral']:
        df_model = df[df['model_simple'] == model]
        group_cond_data = []
        for condition in conditions:
            strategies_for_condition = [s for s, c in strategy_to_condition.items() if c in conditions]
            cond_data = df_model[df_model['strategy'].isin(strategies_for_condition)]
            group_cond_data.append(cond_data)
        
        if group_cond_data:
            # 合并所有条件的数据
            combined_data = pd.concat(group_cond_data, ignore_index=True)
            if len(combined_data) > 0:
                group_acc = combined_data['is_correct'].mean() * 100
                group_data.append(f"{model.upper()}: {group_acc:.1f}%")
    
    if group_data:
        group_stats.append({
            'Condition Group': group_name,
            'Model Performances': ', '.join(group_data)
        })

group_df = pd.DataFrame(group_stats)
print(group_df.to_string(index=False))

# 10. 可视化
print("\n" + "="*80)
print("生成可视化图表...")
print("="*80)

try:
    import matplotlib.pyplot as plt
    
    # 准备绘图数据
    plot_data = []
    for model in ['gemma', 'llama', 'mistral']:
        df_model = df[df['model_simple'] == model]
        for condition in key_conditions:
            strategies_for_condition = [s for s, c in strategy_to_condition.items() if c == condition]
            if strategies_for_condition:
                cond_data = df_model[df_model['strategy'].isin(strategies_for_condition)]
                if len(cond_data) > 0:
                    accuracy = cond_data['is_correct'].mean() * 100
                    plot_data.append({
                        'Model': model.upper(),
                        'Condition': condition,
                        'Accuracy': accuracy
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 设置颜色
    colors = {'GEMMA': '#1f77b4', 'LLAMA': '#ff7f0e', 'MISTRAL': '#2ca02c'}
    
    # 每个条件的x轴位置
    x_positions = np.arange(len(key_conditions))
    bar_width = 0.25
    
    for i, model in enumerate(['GEMMA', 'LLAMA', 'MISTRAL']):
        model_data = plot_df[plot_df['Model'] == model]
        accuracies = []
        for condition in key_conditions:
            cond_acc = model_data[model_data['Condition'] == condition]['Accuracy']
            if len(cond_acc) > 0:
                accuracies.append(cond_acc.values[0])
            else:
                accuracies.append(0)
        
        x = x_positions + (i - 1) * bar_width
        plt.bar(x, accuracies, width=bar_width, color=colors[model], 
                label=model, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for j, (xi, acc) in enumerate(zip(x, accuracies)):
            if acc > 0:
                plt.text(xi, acc + 1, f'{acc:.1f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    plt.xlabel('Evidence Condition', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison Across Evidence Conditions', fontsize=14, fontweight='bold')
    plt.xticks(x_positions, [c.replace('_', '\n') for c in key_conditions], rotation=0)
    plt.ylim(0, 105)
    plt.legend(title='Model', title_fontsize=11, fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为: model_comparison_chart.png")
    
except ImportError:
    print("警告: 未安装matplotlib，跳过可视化部分")
    print("请运行: pip install matplotlib")

print("\n" + "="*80)
print("模型对比分析完成！")
print("="*80)

# 11. 生成最终报告
print("\n" + "="*80)
print("分析完成摘要")
print("="*80)
print(f"1. 总数据量: {len(df)} 行")
print(f"2. 分析模型: Gemma, Llama, Mistral")
print(f"3. 分析条件: {len(key_conditions)} 种")
print(f"4. 模型比较对: {len(model_pairs)} 对")
print(f"5. 结果文件: model_comparison_detailed.csv")
print(f"6. 图表文件: model_comparison_chart.png")