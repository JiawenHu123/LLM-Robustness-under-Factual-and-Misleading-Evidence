# 合并多个JSONL文件
import json

# 定义要合并的文件列表
files_to_merge = [
    "all_questans_rewritten34.jsonl",
    "all_questans_rewritten169.jsonl",
]

# 合并后的输出文件名
output_file = "rewritten_question.jsonl"

# 合并文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in files_to_merge:
        try:
            with open(filename, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
            print(f"已合并文件: {filename}")
        except FileNotFoundError:
            print(f"警告: 文件 {filename} 未找到，跳过")

print(f"合并完成！输出文件: {output_file}")
