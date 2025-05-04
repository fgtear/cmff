import matplotlib.pyplot as plt
import seaborn as sns

# 数据
p_mask = [0, 0.1, 0.2, 0.3, 0.4]
mae = [0.606, 0.606, 0.592, 0.612, 0.658]
corr = [0.850, 0.846, 0.851, 0.845, 0.831]
acc2_negative_nonnegative = [0.8770, 0.8705, 0.8762, 0.8722, 0.8666]
acc2_negative_positive = [0.8928, 0.8924, 0.8935, 0.8899, 0.8840]
f1_negative_nonnegative = [0.8782, 0.8710, 0.8771, 0.8729, 0.8677]
f1_negative_positive = [0.8939, 0.8930, 0.8943, 0.8907, 0.8854]
acc7 = [0.4987, 0.5002, 0.5215, 0.4894, 0.4789]

# 设置调色板
palette = sns.color_palette("husl", 6)

# 绘制并保存 MAE 图
plt.figure(figsize=(6, 6))
# plt.plot(p_mask, mae, marker="o", color=palette[0], markersize=8, linewidth=2)
# 使用黑色线条
plt.plot(p_mask, mae, marker="o", color="black", markersize=8, linewidth=2)
# plt.title(r"MAE$(\downarrow)$")
plt.xlabel(r"$p_{\text{mask}}$", fontsize=12)
# plt.ylabel("MAE")
plt.grid(True)
plt.savefig("../paper/figs/mae.png", dpi=300)
plt.close()

# 绘制并保存 Corr 图
plt.figure(figsize=(6, 6))
plt.plot(p_mask, corr, marker="o", color=palette[1], markersize=8, linewidth=2)
# plt.plot(p_mask, corr, marker="o", color="black", markersize=8, linewidth=2)
plt.xlabel(r"$p_{\text{mask}}$", fontsize=12)
# plt.ylabel("Correlation")
plt.grid(True)
plt.savefig("../paper/figs/corr.png", dpi=300)
plt.close()

# 绘制并保存 Acc7 图
plt.figure(figsize=(6, 6))
plt.plot(p_mask, acc7, marker="o", color=palette[2], markersize=8, linewidth=2)
# plt.plot(p_mask, acc7, marker="o", color="black", markersize=8, linewidth=2)
plt.xlabel(r"$p_{\text{mask}}$", fontsize=12)
# plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("../paper/figs/acc7.png", dpi=300)
plt.close()

# 绘制并保存 Acc2 图
plt.figure(figsize=(6, 6))
plt.plot(p_mask, acc2_negative_nonnegative, label="Negative/Non-Negative", marker="o", color=palette[3], markersize=8, linewidth=2)
# plt.plot(p_mask, acc2_negative_nonnegative, label="Negative/Non-Negative", marker="o", color="black", markersize=8, linewidth=2)
plt.plot(p_mask, acc2_negative_positive, label="Negative/Positive", marker="x", color=palette[4], markersize=8, linewidth=2)
# plt.plot(p_mask, acc2_negative_positive, label="Negative/Positive", marker="x", color="black", linestyle="--", markersize=8, linewidth=2)
plt.xlabel(r"$p_{\text{mask}}$", fontsize=12)
# plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("../paper/figs/acc2.png", dpi=300)
plt.close()

# 绘制并保存 F1 图
plt.figure(figsize=(6, 6))
plt.plot(p_mask, f1_negative_nonnegative, label="Negative/Non-Negative", marker="o", color=palette[3], markersize=8, linewidth=2)
# plt.plot(p_mask, f1_negative_nonnegative, label="Negative/Non-Negative", marker="o", color="black", markersize=8, linewidth=2)
plt.plot(p_mask, f1_negative_positive, label="Negative/Positive", marker="x", color=palette[4], markersize=8, linewidth=2)
# plt.plot(p_mask, f1_negative_positive, label="Negative/Positive", marker="x", color="black", linestyle="--", markersize=8, linewidth=2)
plt.xlabel(r"$p_{\text{mask}}$", fontsize=12)
# plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.savefig("../paper/figs/f1.png", dpi=300)
plt.close()
