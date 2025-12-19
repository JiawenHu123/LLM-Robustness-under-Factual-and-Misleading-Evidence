import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
file_path = "all_models_results_with_correct.csv"

try:
    # 读取CSV文件
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"加载失败: {e}")
    exit()

# 显示数据信息
print("数据加载成功！")
print(f"总行数: {len(df)}")
print("\n前5行数据:")
print(df.head())
print("\n列名:")
print(df.columns.tolist())
print("\n各列数据类型:")
print(df.dtypes)

# 2. 数据清洗和转换
# 确保is_correct是数值型
df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce')

# 清理字符串列（去除空格）
for col in ['question_id', 'gold_answer', 'model_answer', 'model', 'strategy']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print("\n数据清洗完成！")
print(f"is_correct列的有效值: {df['is_correct'].notnull().sum()}/{len(df)}")

# 检查缺失值
print("\n缺失值检查:")
print(df.isnull().sum())

# 3. 检查数据
print("\n" + "="*60)
print("数据概览:")
print("="*60)
print(f"总行数: {len(df)}")
print(f"模型列表: {sorted(df['model'].unique().tolist())}")
print(f"策略列表: {sorted(df['strategy'].unique().tolist())}")

# 4. 定义8个测试条件（根据您提供的strategy名称）
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

# 创建test_set列
df['test_set'] = df['strategy'].map(strategy_to_condition)

# 检查映射是否正确
print("\n策略到测试条件的映射:")
for strategy in df['strategy'].unique():
    condition = strategy_to_condition.get(strategy, '未知')
    print(f"  {strategy} -> {condition}")

# 检查是否有未映射的策略
unmapped = df[df['test_set'].isnull()]['strategy'].unique()
if len(unmapped) > 0:
    print(f"\n警告: 以下策略未映射: {unmapped}")

conditions_order = [
    'no_evidence',
    'support_only', 
    'appeal_only',
    'out_of_context_only',
    'false_causality_only',
    'support+m1',
    'support+m2',
    'support+m3'
]

# 5. 简化模型名称（根据您的实际数据）
df['model_simple'] = df['model'].replace({
    'llama3.1': 'llama',  # 保持llama3.1为llama
    'gemma': 'gemma',      # gemma保持不变
    'mistral': 'mistral'   # mistral保持不变
})

print("\n" + "="*80)
print("ONE-WAY ANOVA 分析：检验不同测试条件对模型准确率的整体影响")
print("="*80)

# 6. 对每个模型进行ANOVA分析
models_to_analyze = sorted(df['model_simple'].unique())
print(f"将要分析的模型: {models_to_analyze}")

for model in models_to_analyze:
    print(f"\n{'='*60}")
    print(f"模型: {model.upper()}")
    print(f"{'='*60}")
    
    # 提取该模型的数据
    df_model = df[df['model_simple'] == model].copy()
    
    # 检查每个问题在每个条件下是否都有记录
    question_ids = df_model['question_id'].unique()
    print(f"唯一问题数: {len(question_ids)}")
    
    # 检查数据完整性
    print("\n各条件样本量:")
    strategy_counts = df_model['strategy'].value_counts()
    for strategy in sorted(df_model['strategy'].unique()):
        count = strategy_counts.get(strategy, 0)
        condition = strategy_to_condition.get(strategy, '未知')
        print(f"  {strategy} ({condition}): {count} 题")
    
    # 准备ANOVA数据 - 使用test_set作为分组条件
    groups = []
    group_names = []
    
    for condition in conditions_order:
        # 找到对应这个条件的所有策略
        strategies_for_condition = [s for s, c in strategy_to_condition.items() if c == condition]
        if strategies_for_condition:
            condition_data = df_model[df_model['strategy'].isin(strategies_for_condition)]
            if len(condition_data) > 0:
                # 按question_id排序以确保配对
                condition_data = condition_data.sort_values('question_id')
                group_data = condition_data['is_correct'].values
                groups.append(group_data)
                group_names.append(condition)
    
    # 检查是否所有组都有相同数量的样本（配对设计）
    group_lengths = [len(g) for g in groups]
    print(f"\n各组样本量: {group_lengths}")
    
    if len(set(group_lengths)) > 1:
        print(f"⚠️  警告：各组样本量不同 {group_lengths}")
        print("  这可能影响ANOVA结果，建议检查数据完整性")
    
    if len(groups) < 2:
        print("  分组不足，无法进行ANOVA分析")
        continue
    
    # 6.1 描述性统计
    print("\n1. 描述性统计（准确率%）:")
    desc_stats = []
    for name, group in zip(group_names, groups):
        if len(group) > 0:
            mean_acc = group.mean() * 100
            std_acc = group.std() * 100
            n = len(group)
            desc_stats.append([name, n, f"{mean_acc:.2f}%", f"{std_acc:.2f}%"])
    
    if desc_stats:
        desc_df = pd.DataFrame(desc_stats, columns=['测试条件', '样本数', '均值', '标准差'])
        print(desc_df.to_string(index=False))
    else:
        print("  无数据")
        continue
    
    # 6.2 方差齐性检验
    print("\n2. 方差齐性检验 (Levene's test):")
    if len(groups) >= 2:
        levene_stat, levene_p = stats.levene(*groups)
        print(f"   统计量 F = {levene_stat:.4f}, p = {levene_p:.4f}")
        if levene_p < 0.05:
            print("   ⚠️  警告：方差不齐 (p < 0.05)")
            print("   建议使用非参数检验或Welch's ANOVA")
            anova_type = "Welch's ANOVA"
        else:
            print("   ✓ 方差齐性假设未被拒绝 (p ≥ 0.05)")
            anova_type = "标准ANOVA"
    else:
        print("   分组不足，无法进行方差齐性检验")
        continue
    
    # 6.3 单因素ANOVA
    print(f"\n3. {anova_type}:")
    if len(groups) >= 2:
        if levene_p < 0.05:
            # 使用Welch's ANOVA（方差不齐时）
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            # 标准ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
        
        # 计算自由度
        k = len(groups)  # 组数
        N = sum(len(g) for g in groups)  # 总样本数
        df_between = k - 1
        df_within = N - k
        
        print(f"   统计量 F({df_between}, {df_within}) = {f_stat:.4f}")
        print(f"   p值 = {p_value:.4f}")
        
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "n.s. (不显著)"
        
        print(f"   结果：{significance}")
        
        # 6.4 计算效应量 (η²)
        if p_value < 0.05:  # 只有显著时才计算
            print("\n4. 效应量计算:")
            
            # 计算总平方和
            all_scores = np.concatenate(groups)
            grand_mean = np.mean(all_scores)
            ss_total = np.sum((all_scores - grand_mean) ** 2)
            
            # 计算组间平方和
            ss_between = 0
            for group in groups:
                group_mean = np.mean(group)
                ss_between += len(group) * (group_mean - grand_mean) ** 2
            
            # 计算η² (eta-squared)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            print(f"   效应量 η² = {eta_squared:.4f}")
            
            # 效应量解释
            print(f"\n   效应量解释 (η²):")
            if eta_squared < 0.01:
                interpretation = "效应极小"
            elif eta_squared < 0.06:
                interpretation = "小效应"
            elif eta_squared < 0.14:
                interpretation = "中等效应"
            else:
                interpretation = "大效应"
            print(f"   {eta_squared:.4f} → {interpretation}")
        
        print(f"\n{'='*60}")

print("\n" + "="*80)
print("ANOVA分析完成！")
print("="*80)

# 7. 关键配对t检验（基于您的研究假设）
print("\n\n" + "="*80)
print("关键配对T检验 (基于研究假设)")
print("="*80)

# 定义关键对比 - 使用实际的strategy名称
key_comparisons = [
    ("clean_only", "support_only", "支持性证据的效果"),
    ("support_only", "s1app", "诉诸权威在混合环境中的伤害"),
    ("support_only", "s1ooc", "断章取义在混合环境中的伤害"),
    ("s1app", "s1ooc", "两种误导类型伤害程度对比")
]

# 显示策略到条件的映射
print("\n策略名称对照表:")
for strategy, condition in strategy_to_condition.items():
    print(f"  {strategy} -> {condition}")

for model in models_to_analyze:
    print(f"\n{'='*60}")
    print(f"模型: {model.upper()}")
    print(f"{'='*60}")
    
    df_model = df[df['model_simple'] == model]
    
    for cond1, cond2, hypothesis in key_comparisons:
        print(f"\n对比: {hypothesis}")
        print(f"  {cond1} vs {cond2}")
        
        # 获取两个条件的数据
        data1 = df_model[df_model['strategy'] == cond1]
        data2 = df_model[df_model['strategy'] == cond2]
        
        if len(data1) == 0 or len(data2) == 0:
            print(f"  ⚠️ 条件 {cond1} 或 {cond2} 无数据")
            continue
        
        # 找到共同的问题ID（配对设计）
        common_ids = set(data1['question_id']).intersection(set(data2['question_id']))
        
        if len(common_ids) == 0:
            print(f"  ⚠️ 无共同问题，无法进行配对检验")
            continue
        
        # 提取配对数据
        paired_data1 = data1[data1['question_id'].isin(common_ids)].sort_values('question_id')['is_correct'].values
        paired_data2 = data2[data2['question_id'].isin(common_ids)].sort_values('question_id')['is_correct'].values
        
        if len(paired_data1) != len(paired_data2):
            print(f"  ⚠️ 配对数据长度不一致: {len(paired_data1)} vs {len(paired_data2)}")
            continue
        
        n_pairs = len(paired_data1)
        if n_pairs < 10:
            print(f"  ⚠️ 配对样本过少: {n_pairs}")
            continue
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(paired_data1, paired_data2)
        
        # 计算均值差和效应量 (Cohen's d)
        mean1 = paired_data1.mean()
        mean2 = paired_data2.mean()
        mean_diff = mean1 - mean2
        diff_std = np.std(paired_data1 - paired_data2)
        cohens_d = mean_diff / diff_std if diff_std > 0 else 0
        
        # 结果报告
        mean1_pct = mean1 * 100
        mean2_pct = mean2 * 100
        mean_diff_pct = mean_diff * 100
        
        print(f"  配对样本数: {n_pairs}")
        print(f"  {cond1}: {mean1_pct:.2f}%")
        print(f"  {cond2}: {mean2_pct:.2f}%")
        print(f"  均值差: {mean_diff_pct:.2f}%")
        print(f"  t({n_pairs-1}) = {t_stat:.4f}, p = {p_value:.4f}")
        print(f"  Cohen's d = {cohens_d:.4f}")
        
        # 显著性标记
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"  结果: {sig}")
        
        # 效应量解释
        if abs(cohens_d) < 0.2:
            effect_size = "极小"
        elif abs(cohens_d) < 0.5:
            effect_size = "小"
        elif abs(cohens_d) < 0.8:
            effect_size = "中"
        else:
            effect_size = "大"
        print(f"  效应量: {effect_size} (|d| = {abs(cohens_d):.3f})")

print("\n" + "="*80)
print("所有统计分析完成！")
print("="*80)

# 8. 保存汇总结果
print("\n生成汇总报告...")
summary_results = []

for model in models_to_analyze:
    df_model = df[df['model_simple'] == model]
    
    # 按strategy计算准确率
    for strategy in sorted(df_model['strategy'].unique()):
        strategy_data = df_model[df_model['strategy'] == strategy]
        if len(strategy_data) > 0:
            accuracy = strategy_data['is_correct'].mean() * 100
            condition = strategy_to_condition.get(strategy, '未知')
            
            summary_results.append({
                'model': model,
                'strategy': strategy,
                'condition': condition,
                'correct': strategy_data['is_correct'].sum(),
                'total': len(strategy_data),
                'accuracy_percent': accuracy,
                'accuracy_decimal': accuracy / 100
            })

summary_df = pd.DataFrame(summary_results)

# 按模型和策略排序
summary_df = summary_df.sort_values(['model', 'strategy'])

print("\n准确率汇总:")
print(summary_df.to_string(index=False))

# 保存到文件
summary_file = 'model_performance_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n详细结果已保存到: {summary_file}")

# 9. 生成简要报告
print("\n" + "="*80)
print("简要准确率报告")
print("="*80)

for model in models_to_analyze:
    print(f"\n模型: {model.upper()}")
    print("-" * 40)
    
    model_data = summary_df[summary_df['model'] == model]
    for _, row in model_data.iterrows():
        print(f"{row['strategy']:15s} ({row['condition']:20s}): {row['accuracy_percent']:6.2f}% ({row['correct']}/{row['total']})")
    
    # 计算模型总体准确率
    model_all_data = df[df['model_simple'] == model]
    overall_acc = model_all_data['is_correct'].mean() * 100
    print(f"\n总体准确率: {overall_acc:.2f}%")

# 总体统计
overall_acc = df['is_correct'].mean() * 100
print(f"\n{'='*80}")
print(f"所有模型总体准确率: {overall_acc:.2f}%")
print(f"总正确数/总数: {df['is_correct'].sum():.0f}/{len(df)}")
print(f"{'='*80}")