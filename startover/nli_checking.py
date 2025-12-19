import json
import re
from transformers import pipeline
from collections import defaultdict

class SupportValidator:
    def __init__(self):
        # 初始化NLI模型
        try:
            self.nli_pipeline = pipeline(
                "text-classification", 
                model="roberta-large-mnli",
                device=-1
            )
            self.nli_available = True
            print("NLI模型加载成功")
        except Exception as e:
            print(f"NLI模型加载失败: {e}，将仅使用基础验证")
            self.nli_available = False
    
    def extract_entities(self, text):
        """提取文本中的关键实体"""
        entities = []
        
        # 提取国家名
        countries = ['UK', 'United Kingdom', 'USA', 'United States', 'America', 'France', 
                    'Japan', 'Canada', 'Australia', 'New Zealand', 'Germany', 'Italy',
                    'Spain', 'South Korea', 'Singapore', 'Netherlands', 'Sweden', 'China',
                    'Mexico', 'India', 'Cambodia', 'Myanmar']
        
        for country in countries:
            if country.lower() in text.lower():
                entities.append(country)
        
        # 提取其他大写专有名词
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend([word for word in words if len(word) > 2])
        
        return list(set(entities))
    
    def check_logical_consistency(self, correct_answer, support_text):
        """检查support与correct answer的逻辑一致性 - 更宽松的版本"""
        answer_lower = correct_answer.lower()
        support_lower = support_text.lower()
        
        # 检查明显的逻辑矛盾
        contradiction_pairs = [
            ('is true', 'is false'), 
            ('is correct', 'is incorrect'),
            ('can', 'cannot'),
            ('cannot', 'can'),
        ]
        
        for term1, term2 in contradiction_pairs:
            if term1 in answer_lower and term2 in support_lower:
                return False, f"明显矛盾: '{term1}'在答案中但'{term2}'在support中"
            if term2 in answer_lower and term1 in support_lower:
                return False, f"明显矛盾: '{term2}'在答案中但'{term1}'在support中"
        
        # 对于相对性词汇（higher/lower, more/less）需要更谨慎的判断
        relative_pairs = [('higher', 'lower'), ('lower', 'higher'), ('more', 'less'), ('less', 'more')]
        for term1, term2 in relative_pairs:
            if term1 in answer_lower and term2 in support_lower:
                # 检查是否在同一个上下文中
                context_words = ['bmi', 'income', 'education', 'hours', 'work', 'rich']
                has_context = any(ctx in answer_lower or ctx in support_lower for ctx in context_words)
                if has_context:
                    return False, f"相对性矛盾: '{term1}'在答案中但'{term2}'在support中"
        
        return True, "逻辑一致"
    
    def contains_contradiction(self, support_text):
        """检查support内部是否存在矛盾 - 更宽松的版本"""
        sentences = re.split(r'[.!?]', support_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False, "句子太少，无法检测内部矛盾"
        
        # 只检查明显的内部矛盾
        for i in range(len(sentences)-1):
            sent1 = sentences[i].lower()
            sent2 = sentences[i+1].lower()
            
            opposite_pairs = [
                ('is true', 'is false'), 
                ('is correct', 'is incorrect'),
                ('supports', 'contradicts'),
                ('can', 'cannot'),
            ]
            
            for pair1, pair2 in opposite_pairs:
                if (pair1 in sent1 and pair2 in sent2) or (pair2 in sent1 and pair1 in sent2):
                    return True, f"内部矛盾: 句子{i+1}说'{pair1}'但句子{i+2}说'{pair2}'"
        
        return False, "无内部矛盾"
    
    def validate_support_basic(self, item):
        """基础规则验证 - 更宽松的版本"""
        issues = []
        detailed_issues = []
        
        correct_answer = item.get('answer_theme', '')
        support_text = item.get('support', {}).get('text', '')
        
        if not correct_answer or not support_text:
            issues.append("缺少answer_theme或support.text")
            detailed_issues.append("缺少answer_theme或support.text")
            return False, issues, detailed_issues
        
        # 1. 检查support是否包含correct answer中的关键实体 - 更宽松
        key_entities = self.extract_entities(correct_answer)
        missing_entities = []
        
        for entity in key_entities:
            # 允许实体有变体形式
            entity_variants = [entity, entity.lower(), entity.upper()]
            if not any(variant in support_text for variant in entity_variants):
                # 对于常见缩写也检查
                if entity == 'USA' and 'united states' not in support_text.lower():
                    missing_entities.append(entity)
                elif entity == 'UK' and 'united kingdom' not in support_text.lower():
                    missing_entities.append(entity)
                else:
                    missing_entities.append(entity)
        
        if missing_entities:
            issues.append(f"关键实体缺失: {', '.join(missing_entities)}")
            detailed_issues.append(f"关键实体缺失: {', '.join(missing_entities)}")
        
        # 2. 检查support是否与correct answer逻辑一致
        logic_consistent, logic_message = self.check_logical_consistency(correct_answer, support_text)
        if not logic_consistent:
            issues.append("support与correct answer逻辑不一致")
            detailed_issues.append(f"逻辑不一致: {logic_message}")
        
        # 3. 检查support是否包含矛盾陈述
        has_contradiction, contradiction_message = self.contains_contradiction(support_text)
        if has_contradiction:
            issues.append("support内部存在矛盾")
            detailed_issues.append(f"内部矛盾: {contradiction_message}")
        
        # 4. 检查support是否过于简单 - 更宽松的标准
        sentences = re.split(r'[.!?]', support_text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # 降低长度要求
        if len(meaningful_sentences) < 1:  # 至少1个有意义的句子
            issues.append("support内容过于简单")
            detailed_issues.append(f"support内容过于简单: 只有{len(meaningful_sentences)}个有意义的句子")
        
        return len(issues) == 0, issues, detailed_issues
    
    def validate_support_with_nli(self, item):
        """使用NLI模型验证support是否支持correct answer - 更宽松的版本"""
        if not self.nli_available:
            return True, "NLI不可用，跳过验证"
        
        premise = item.get('support', {}).get('text', '')
        hypothesis = item.get('answer_theme', '')
        
        if not premise or not hypothesis:
            return False, "前提或假设为空"
        
        try:
            # 构建NLI输入
            result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
            
            label = result[0]['label']
            score = result[0]['score']
            
            # 更宽松的阈值设置
            if label == 'ENTAILMENT' and score > 0.4:  # 降低阈值到0.4
                return True, f"NLI验证通过: {label} (置信度: {score:.3f})"
            elif label == 'NEUTRAL' and score > 0.6:   # NEUTRAL也接受，如果置信度高
                return True, f"NLI验证通过: {label} (置信度: {score:.3f})"
            else:
                return False, f"NLI验证失败: {label} (置信度: {score:.3f})"
                
        except Exception as e:
            return False, f"NLI验证出错: {e}"
    
    def debug_nli_analysis(self, item):
        """调试NLI分析，显示详细信息"""
        if not self.nli_available:
            return "NLI不可用"
        
        premise = item.get('support', {}).get('text', '')
        hypothesis = item.get('answer_theme', '')
        
        result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
        
        debug_info = {
            'premise_length': len(premise),
            'hypothesis_length': len(hypothesis),
            'premise_preview': premise[:100] + "..." if len(premise) > 100 else premise,
            'hypothesis_preview': hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis,
            'nli_result': result[0]
        }
        
        return debug_info
    
    def comprehensive_validation(self, items):
        """综合验证流水线 - 更宽松的版本"""
        validation_results = {
            'passed': [],
            'failed_basic': [],
            'failed_nli': [], 
            'failed_both': [],
            'all_failed_details': [],
            'debug_info': []  # 添加调试信息
        }
        
        print(f"开始验证 {len(items)} 个条目...")
        
        for i, item in enumerate(items, 1):
            if i % 50 == 0:
                print(f"已处理 {i}/{len(items)} 个条目")
            
            item_id = item.get('id', f'unknown_{i}')
            question = item.get('question', '未知问题')
            correct_answer = item.get('answer_theme', '')
            support_text = item.get('support', {}).get('text', '')
            
            # 第一层：基础规则验证
            basic_pass, basic_issues, detailed_issues = self.validate_support_basic(item)
            
            # 第二层：NLI验证
            nli_pass, nli_message = self.validate_support_with_nli(item)
            
            # 调试信息
            debug_info = self.debug_nli_analysis(item) if self.nli_available else "NLI不可用"
            
            # 构建详细信息
            item_details = {
                'id': item_id,
                'question': question,
                'correct_answer': correct_answer,
                'support_text': support_text,
                'basic_issues': basic_issues,
                'detailed_issues': detailed_issues,
                'nli_message': nli_message,
                'debug_info': debug_info
            }
            
            # 更宽松的分类标准：只要基础验证通过就认为OK
            if basic_pass:
                validation_results['passed'].append(item_id)
            elif not basic_pass and nli_pass:
                validation_results['failed_basic'].append(item_details)
                validation_results['all_failed_details'].append({
                    **item_details,
                    'failure_type': '基础验证失败'
                })
            elif basic_pass and not nli_pass:
                # 基础验证通过但NLI失败，仍然算通过（因为基础验证更可靠）
                validation_results['passed'].append(item_id)
                validation_results['debug_info'].append({
                    'id': item_id,
                    'nli_failed_but_basic_passed': True,
                    'nli_message': nli_message
                })
            else:  # 两者都失败
                validation_results['failed_both'].append(item_details)
                validation_results['all_failed_details'].append({
                    **item_details,
                    'failure_type': '两者都失败'
                })
        
        return validation_results

def read_jsonl_file(filename):
    """读取JSONL文件"""
    items = []
    error_count = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                    error_count += 1
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
        return []
    
    print(f"成功读取 {len(items)} 个条目，解析错误: {error_count}")
    return items

def print_detailed_report(results):
    """打印详细的验证报告"""
    print("\n" + "="*80)
    print("详细验证报告 - 需要人工检查的条目")
    print("="*80)
    
    all_failed = results['all_failed_details']
    
    if not all_failed:
        print("🎉 所有条目都通过了验证！")
        return
    
    for item in all_failed:
        print(f"\n🔴 ID: {item['id']}")
        print(f"问题: {item['question']}")
        print(f"正确答案: {item['correct_answer']}")
        print(f"Support: {item['support_text'][:150]}..." if len(item['support_text']) > 150 else f"Support: {item['support_text']}")
        
        if item['basic_issues']:
            print("基础验证问题:")
            for issue in item['detailed_issues']:
                print(f"  - {issue}")
        
        print(f"NLI验证: {item['nli_message']}")
        
        # 显示调试信息
        if 'debug_info' in item and item['debug_info'] != "NLI不可用":
            debug = item['debug_info']
            print(f"调试信息: 前提长度={debug['premise_length']}, 假设长度={debug['hypothesis_length']}")
            print(f"         前提预览: {debug['premise_preview']}")
            print(f"         假设预览: {debug['hypothesis_preview']}")
        
        print("-" * 60)

def print_summary(results):
    """打印验证结果摘要"""
    print("\n" + "="*50)
    print("验证结果摘要")
    print("="*50)
    
    total_passed = len(results['passed'])
    total_failed = len(results['all_failed_details'])
    total = total_passed + total_failed
    
    print(f"总条目数: {total}")
    print(f"✓ 通过验证: {total_passed} ({total_passed/total*100:.1f}%)")
    print(f"✗ 需要人工检查: {total_failed} ({total_failed/total*100:.1f}%)")
    
    if total_failed > 0:
        print(f"\n失败类型分布:")
        print(f"  - 基础验证失败: {len(results['failed_basic'])}")
        print(f"  - 两者都失败: {len(results['failed_both'])}")
    
    if results['debug_info']:
        print(f"  - NLI失败但基础通过: {len(results['debug_info'])} (这些条目已算作通过)")

def main():
    input_file = "all_data.jsonl"
    
    print("正在读取数据...")
    items = read_jsonl_file(input_file)
    
    if not items:
        print("没有读取到数据，请检查文件路径")
        return
    
    validator = SupportValidator()
    
    print("开始验证...")
    results = validator.comprehensive_validation(items)
    
    print_summary(results)
    print_detailed_report(results)
    
    # 输出简单的ID列表
    if results['all_failed_details']:
        print(f"\n📋 需要人工检查的ID列表 (共{len(results['all_failed_details'])}个):")
        print("-" * 50)
        for item in results['all_failed_details']:
            print(item['id'])

if __name__ == "__main__":
    main()