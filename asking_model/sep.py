import json

def extract_misleading_info(input_file, output_file):
    """
    从输入的JSONL文件中提取问题、misleading_explanation和选项，写入新的JSONL文件
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            line_count = 0
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    
                    # 提取所需字段
                    extracted_data = {
                        "id": data.get("id", ""),
                        "question": data.get("question", ""),
                        "misleading_explanation": data.get("misleading_explanation", ""),
                        "options": data.get("options", [])
                    }
                    
                    # 写入到输出文件
                    outfile.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')
                    line_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"第 {line_count + 1} 行JSON解析错误: {e}")
                    continue
                    
            print(f"成功处理 {line_count} 条记录")
            print(f"结果已保存到 {output_file}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 输入文件和输出文件路径
    input_file = "questions_zh_en_updated.jsonl"
    output_file = "questions_zh_en_misleading_only.jsonl"
    
    # 执行提取操作
    extract_misleading_info(input_file, output_file)
    
    # 验证输出文件内容
    print("\n输出文件内容预览:")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2:  # 只显示前2行
                    break
                data = json.loads(line.strip())
                print(f"第{i+1}行:")
                print(f"  ID: {data['id']}")
                print(f"  问题: {data['question']}")
                print(f"  误导解释: {data['misleading_explanation'][:50]}...")  # 只显示前50字符
                print(f"  选项: {data['options']}")
                print("-" * 50)
    except Exception as e:
        print(f"读取输出文件时出错: {e}")