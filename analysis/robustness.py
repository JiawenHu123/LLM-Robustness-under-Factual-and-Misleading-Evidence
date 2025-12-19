import matplotlib.pyplot as plt
import numpy as np

# 您的数据 (Δ值)，注意顺序对应conditions列表
gemma_delta = [0.4654, -0.198, -0.0792, -0.0742, 0.2921, 0.3169, 0.3565]
llama_delta = [0.4703, -0.1831, -0.0049, -0.0297, 0.3416, 0.3317, 0.3416]
mistral_delta = [0.5198, -0.2723, -0.1782, -0.1584, 0.3515, 0.3465, 0.4158]

# 转换为百分比（乘以100）
gemma_delta_pct = [x * 100 for x in gemma_delta]
llama_delta_pct = [x * 100 for x in llama_delta]
mistral_delta_pct = [x * 100 for x in mistral_delta]

conditions = ['Support', 'Appeal', 'Out-of-Context', 'False Causality', 'S+M1', 'S+M2', 'S+M3']
x = np.arange(len(conditions))  # X轴位置
width = 0.25  # 柱子的宽度

fig, ax = plt.subplots(figsize=(14, 8))

# 绘制分组柱子
rects1 = ax.bar(x - width, gemma_delta_pct, width, label='Gemma-2B', color='skyblue', edgecolor='navy', linewidth=1)
rects2 = ax.bar(x, llama_delta_pct, width, label='Llama-8B', color='lightcoral', edgecolor='darkred', linewidth=1)
rects3 = ax.bar(x + width, mistral_delta_pct, width, label='Mistral-7B', color='lightgreen', edgecolor='darkgreen', linewidth=1)

# 装饰图表
ax.set_ylabel('Performance Shift Δ (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Evidence Condition', fontsize=14, fontweight='bold')
ax.set_title('Model Robustness under Different Evidence Conditions', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=12)
ax.legend(fontsize=12)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)  # 添加y=0的参考线

# 设置y轴范围和网格
ax.set_ylim([-30, 60])  # 根据您的数据范围调整
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# 在柱子上方添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_offset = 3 if height >= 0 else -3
        color = 'black' if abs(height) < 20 else 'red' if height > 0 else 'blue'
        
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', 
                    va=va, 
                    fontsize=10,
                    fontweight='bold' if abs(height) > 20 else 'normal',
                    color=color)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 添加条件类型标注
ax.text(0.5, -0.15, '← Pure Conditions | Mixed Conditions →', 
        transform=ax.transAxes, ha='center', fontsize=11, style='italic')

# 突出显示正负区域
ax.axhspan(0, 60, alpha=0.1, color='green', label='Positive Δ')
ax.axhspan(-30, 0, alpha=0.1, color='red', label='Negative Δ')

fig.tight_layout()

# 保存图像
plt.savefig('model_robustness_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印数据摘要
print("="*60)
print("数据摘要（百分比形式）：")
print("="*60)
print(f"{'Condition':<20} {'Gemma-2B':<12} {'Llama-8B':<12} {'Mistral-7B':<12}")
print("-"*60)

for i, condition in enumerate(conditions):
    print(f"{condition:<20} {gemma_delta_pct[i]:<11.1f}% {llama_delta_pct[i]:<11.1f}% {mistral_delta_pct[i]:<11.1f}%")

print("="*60)
print("\n关键观察：")
print(f"1. 所有模型在Support条件下表现最佳: +{max(gemma_delta_pct[0], llama_delta_pct[0], mistral_delta_pct[0]):.1f}%")
print(f"2. Appeal条件负面影响最大: {min(gemma_delta_pct[1], llama_delta_pct[1], mistral_delta_pct[1]):.1f}%")
print(f"3. 混合条件(S+M)都保持正向增益")
print(f"4. Mistral在Support条件下表现最好(+{mistral_delta_pct[0]:.1f}%)，但对误导最敏感")