import matplotlib.pyplot as plt
import numpy as np

# 数据
metrics = ["MAE (↓)", "Corr (↑)", "Acc2 (↑)", "F1 (↑)", "Acc7 (↑)"]
no_pretrain = [0.634, 0.832, 0.8767, 0.8776, 0.4584]
yes_pretrain = [0.592, 0.851, 0.8762, 0.8771, 0.5215]

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图的位置
x = np.arange(len(metrics))  # 横坐标位置
width = 0.35  # 柱子的宽度

# 绘制柱状图
# rects1 = ax.bar(x - width / 2, no_pretrain, width, label="No", color='skyblue')
rects1 = ax.bar(x - width / 2, no_pretrain, width, label="Without", color="skyblue", alpha=0.8, edgecolor="gray")
# rects2 = ax.bar(x + width / 2, yes_pretrain, width, label="Yes", color='orange')
rects2 = ax.bar(x + width / 2, yes_pretrain, width, label="With", color="lightgreen", alpha=0.8, edgecolor="gray")

# 设置y轴为对数刻度
# ax.set_yscale("log")
# 设置y轴为0到1的范围
ax.set_ylim(0, 1)

# 添加一些文本标签
ax.set_ylabel("Value", fontsize=15)
# ax.set_title("Comparison of Metrics with and without Pretraining")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=15)  # 调整横坐标标签字体大小
ax.tick_params(axis="y", labelsize=15)  # 调整纵坐标刻度字体大小
ax.legend(prop={"size": 15})

# 给柱状图添加标签，并调整字体大小
ax.bar_label(rects1, fontsize=12, padding=3)
ax.bar_label(rects2, fontsize=12, padding=3)


# 添加数值标签到柱状图上
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",  # 显示的数值保留位数
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


# add_labels(rects1)
# add_labels(rects2)

# 显示图形
plt.tight_layout()

# 保存为高质量图，符合elsevier要求
fig.savefig("../paper/fig_pretraining.png", dpi=300)
plt.show()
