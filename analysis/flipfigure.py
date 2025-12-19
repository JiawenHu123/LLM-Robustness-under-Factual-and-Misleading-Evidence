import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 准备数据
conditions = ['Support', 'Appeal', 'OOC', 'FalseC', 'S+Appeal', 'S+OOC', 'S+FalseC']
models = ['Gemma', 'Llama', 'Mistral']

# BFR热力图数据
bfr_data = np.array([
    [48.0, 3.5, 10.9, 9.4, 34.7, 34.2, 37.6],
    [47.0, 6.9, 16.3, 11.4, 36.1, 35.1, 35.6],
    [52.5, 2.5, 5.0, 6.4, 38.6, 37.6, 42.6]
])

# AFR热力图数据
afr_data = np.array([
    [1.5, 23.3, 18.8, 16.8, 5.4, 2.5, 2.0],
    [0.0, 25.2, 16.8, 14.4, 2.0, 2.0, 1.5],
    [0.5, 29.7, 22.8, 22.3, 3.5, 3.0, 1.0]
])

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# BFR热力图
sns.heatmap(bfr_data, annot=True, fmt='.1f', cmap='YlGnBu', 
            xticklabels=conditions, yticklabels=models,
            cbar_kws={'label': 'BFR (%)'}, ax=ax1)
ax1.set_title('Beneficial Flip Rates (Wrong→Correct)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Evidence Condition', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)

# AFR热力图
sns.heatmap(afr_data, annot=True, fmt='.1f', cmap='YlOrRd_r',  # 反向色图，红色表示高值
            xticklabels=conditions, yticklabels=models,
            cbar_kws={'label': 'AFR (%)'}, ax=ax2)
ax2.set_title('Adversarial Flip Rates (Correct→Wrong)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Evidence Condition', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)

plt.tight_layout()
plt.show()

# 净变化热力图
net_data = bfr_data - afr_data
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(net_data, annot=True, fmt='+.1f', cmap='RdYlGn', center=0,
            xticklabels=conditions, yticklabels=models,
            cbar_kws={'label': 'Net Change (%)'}, ax=ax)
ax.set_title('Net Flip-Rate Change (BFR - AFR)', fontsize=14, fontweight='bold')
ax.set_xlabel('Evidence Condition', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.tight_layout()
plt.show()