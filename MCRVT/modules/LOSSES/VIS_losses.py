import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 设置全局字体大小

csv_filename = 'losses.csv'  # 请确保该文件与脚本在同一目录下，或提供正确路径
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"错误: 找不到文件 '{csv_filename}'。请确保文件路径正确。")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"错误: 文件 '{csv_filename}' 是空的。")
    exit(1)
except pd.errors.ParserError:
    print(f"错误: 解析文件 '{csv_filename}' 时出现问题。请检查文件格式。")
    exit(1)

# 检查必要的列是否存在
required_columns = [
    'Epochs',
    ' (${L}_{cl}$)',
    ' (${L}_{Rl}$)',
    ' (${L}_{ms}$)',
    ' (${L}$)'
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"错误: 缺少以下必要的列: {', '.join(missing_columns)}")
    exit(1)

# 提取数据
epochs = df['Epochs'].values
cross_entropy = df[' (${L}_{cl}$)'].values
mae = df[' (${L}_{Rl}$)'].values
huber_loss = df[' (${L}_{ms}$)'].values
combined_loss = df[' (${L}$)'].values

# 确保损失值不低于0.02
cross_entropy = np.maximum(cross_entropy, 0.02)
mae = np.maximum(mae, 0.02)
huber_loss = np.maximum(huber_loss, 0.02)
combined_loss = np.maximum(combined_loss, 0.02)

# 绘制图形
plt.figure(figsize=(10, 6))

plt.plot(
    epochs, cross_entropy,
    label=r'${L}_{cl}$', color='orange'
)  # 交叉熵损失
plt.plot(
    epochs, mae,
    label=r'${L}_{Rl}$', color='green'
)           # 平均绝对误差
plt.plot(
    epochs, huber_loss,
    label=r'${L}_{ms}$', color='purple'
) # Huber 损失
plt.plot(
    epochs, combined_loss,
    label=r'${L}$', color='red'
) # 组合损失

# 添加标题和标签
# plt.title('不同损失函数的表现比较')  # 取消注释并修改标题（根据需要）
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# 添加图例
plt.legend()
plt.gca().yaxis.set_visible(False)
# 隐藏 y 轴（根据需要取消注释）
# plt.gca().yaxis.set_visible(False)

# 显示网格（根据需要调整）
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图形为 PDF（可选）
# plt.savefig('losses_plot.pdf', dpi=330, bbox_inches='tight')

# 显示图形
plt.show()