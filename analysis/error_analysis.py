import pandas as pd
import numpy as np

# ==================== 1. 加载数据 ====================
print("Loading data...")
file_path = "all_models_results_with_correct.csv"

try:
    df = pd.read_csv(file_path)
    print(f"数据加载成功！总行数: {len(df)}")
except Exception as e:
    print(f"加载失败: {e}")
    exit()

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

# 策略到条件映射
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

print("\n数据准备完成！")

# ==================== 2. 定义案例筛选函数 ====================
def find_misleading_override_cases(df, top_n=5):
    """
    找出：基线正确 → 误导错误的最佳案例
    返回每个误导类型的前top_n个案例
    """
    
    results = {
        'appeal_override': [],
        'ooc_override': [],
        'falsec_override': []
    }
    
    # 为每个模型单独分析
    for model in ['gemma', 'llama', 'mistral']:
        df_model = df[df['model_simple'] == model]
        
        for qid in df_model['question_id'].unique():
            q_data = df_model[df_model['question_id'] == qid]
            
            # 1. 基线是否正确？
            baseline = q_data[q_data['test_set'] == 'no_evidence']
            if len(baseline) == 0 or baseline['is_correct'].iloc[0] != 1:
                continue
            
            baseline_answer = baseline['model_answer'].iloc[0]
            gold_answer = baseline['gold_answer'].iloc[0]
            
            # 2. 检查三种误导条件
            for mis_type, mis_cond in [
                ('appeal_override', 'appeal_only'),
                ('ooc_override', 'out_of_context_only'),
                ('falsec_override', 'false_causality_only')
            ]:
                mis_data = q_data[q_data['test_set'] == mis_cond]
                
                if len(mis_data) > 0 and mis_data['is_correct'].iloc[0] == 0:
                    # 找到符合条件的案例！
                    mis_answer = mis_data['model_answer'].iloc[0]
                    
                    case_info = {
                        'question_id': qid,
                        'model': model,
                        'gold_answer': gold_answer,
                        'baseline_answer': baseline_answer,
                        'misleading_answer': mis_answer,
                        'misleading_type': mis_cond,
                        'is_gold_match': baseline_answer == gold_answer,
                        'is_all_models': False  # 需要跨模型检查
                    }
                    
                    results[mis_type].append(case_info)
    
    # 筛选最佳案例（按模型覆盖度等）
    best_cases = {}
    for mis_type, cases in results.items():
        if not cases:  # 如果没有案例，跳过
            best_cases[mis_type] = []
            continue
            
        # 按是否有多个模型受影响排序
        case_counts = {}
        for case in cases:
            qid = case['question_id']
            case_counts[qid] = case_counts.get(qid, 0) + 1
        
        # 选择最多模型受影响的案例
        sorted_qids = sorted(case_counts.items(), key=lambda x: x[1], reverse=True)
        top_qids = [qid for qid, count in sorted_qids[:top_n]]
        
        # 收集这些题目的所有模型数据
        best_cases[mis_type] = []
        for qid in top_qids:
            q_cases = [c for c in cases if c['question_id'] == qid]
            # 添加是否所有模型都受影响的信息
            all_models_affected = len(set(c['model'] for c in q_cases)) == 3
            for case in q_cases:
                case['is_all_models'] = all_models_affected
            best_cases[mis_type].extend(q_cases[:3])  # 每个题目取前3个模型
    
    return best_cases

# ==================== 3. 运行筛选 ====================
print("\n" + "="*80)
print("开始筛选：基线正确 → 误导错误的案例")
print("="*80)

cases = find_misleading_override_cases(df, top_n=3)

print("\n找到的案例分布：")
for mis_type, case_list in cases.items():
    if case_list:
        unique_questions = len(set(c['question_id'] for c in case_list))
        print(f"\n{mis_type}: {len(case_list)}个案例，{unique_questions}个独特题目")
        
        # 显示前2个题目
        for i, case in enumerate(case_list[:2]):
            print(f"  题目 {case['question_id']} - {case['model']}:")
            print(f"    基线答案: {case['baseline_answer']}")
            print(f"    误导答案: {case['misleading_answer']}")
            print(f"    正确答案: {case['gold_answer']}")
            print(f"    是否所有模型都受影响: {case['is_all_models']}")
    else:
        print(f"\n{mis_type}: 没有找到符合条件的案例")

# ==================== 4. 生成详细报告 ====================
print("\n" + "="*80)
print("生成详细案例报告")
print("="*80)

# 找出所有模型都受影响的案例
all_models_cases = []
for mis_type, case_list in cases.items():
    # 按题目分组
    questions_by_type = {}
    for case in case_list:
        qid = case['question_id']
        if qid not in questions_by_type:
            questions_by_type[qid] = []
        questions_by_type[qid].append(case)
    
    # 找出所有模型都受影响的题目
    for qid, q_cases in questions_by_type.items():
        models_affected = set(c['model'] for c in q_cases)
        if len(models_affected) == 3:  # 所有三个模型
            all_models_cases.append({
                'question_id': qid,
                'misleading_type': mis_type,
                'models': ', '.join(sorted(models_affected)),
                'gold_answer': q_cases[0]['gold_answer'],
                'baseline_answers': {c['model']: c['baseline_answer'] for c in q_cases},
                'misleading_answers': {c['model']: c['misleading_answer'] for c in q_cases}
            })

if all_models_cases:
    print(f"\n找到 {len(all_models_cases)} 个所有模型都受影响的案例：")
    for i, case in enumerate(all_models_cases[:5]):  # 只显示前5个
        print(f"\n{i+1}. 题目 {case['question_id']} - {case['misleading_type']}")
        print(f"   正确答案: {case['gold_answer']}")
        print(f"   基线答案: {case['baseline_answers']}")
        print(f"   误导答案: {case['misleading_answers']}")
else:
    print("\n没有找到所有模型都受影响的案例")

# ==================== 5. 保存结果到CSV ====================
print("\n" + "="*80)
print("保存结果")
print("="*80)

# 将所有案例保存到一个DataFrame
all_cases_list = []
for mis_type, case_list in cases.items():
    for case in case_list:
        all_cases_list.append(case)

if all_cases_list:
    cases_df = pd.DataFrame(all_cases_list)
    cases_df.to_csv('misleading_override_cases.csv', index=False, encoding='utf-8-sig')
    print("案例已保存到: misleading_override_cases.csv")
    
    # 也保存所有模型都受影响的案例
    if all_models_cases:
        all_models_df = pd.DataFrame(all_models_cases)
        all_models_df.to_csv('all_models_override_cases.csv', index=False, encoding='utf-8-sig')
        print("所有模型受影响案例已保存到: all_models_override_cases.csv")
else:
    print("没有找到符合条件的案例")

print("\n" + "="*80)
print("分析完成！")
print("="*80)