import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn风格
sns.set(style="whitegrid")

# 数据
pmask = [0, 0.1, 0.2, 0.3, 0.4]
mae = [0.606, 0.606, 0.592, 0.612, 0.658]
corr = [0.850, 0.846, 0.851, 0.845, 0.831]
acc2_negative_positive = [0.8928, 0.8924, 0.8935, 0.8899, 0.8840]
f1_negative_positive = [0.8939, 0.8930, 0.8943, 0.8907, 0.8854]
acc7 = [0.4987, 0.5002, 0.5215, 0.4894, 0.4789]

# 创建图形和主轴
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制第一个y轴的指标
ax1.plot(pmask, mae, marker="o", label="MAE", linewidth=2, markersize=8, linestyle="-")
ax1.plot(pmask, acc7, marker="x", label="Acc7", linewidth=2, markersize=8, linestyle="-")

# 设置主轴的标签和标题
ax1.set_xlabel("Pmask", fontsize=14)
ax1.set_ylabel("MAE and Acc7", fontsize=14)
ax1.set_title("Performance Metrics vs. Pmask", fontsize=16)
ax1.tick_params(axis="both", which="major", labelsize=12)

# 创建次坐标轴
ax2 = ax1.twinx()

# 绘制次坐标轴的指标
ax2.plot(pmask, corr, marker="s", label="Correlation", linewidth=2, markersize=8, linestyle="--", color="tab:orange")
ax2.plot(
    pmask,
    acc2_negative_positive,
    marker="^",
    label="Acc2 Negative Positive",
    linewidth=2,
    markersize=8,
    linestyle="-.",
    color="tab:green",
)
ax2.plot(pmask, f1_negative_positive, marker="d", label="F1 Negative Positive", linewidth=2, markersize=8, linestyle=":", color="tab:red")

# 设置次坐标轴的标签
ax2.set_ylabel("Correlation, Acc2, and F1", fontsize=14)
ax2.tick_params(axis="both", which="major", labelsize=12)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=12)

# 显示网格
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# 调整边距
plt.tight_layout()

# 显示图形
plt.show()
