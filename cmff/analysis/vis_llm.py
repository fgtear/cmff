import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# --- 1. 数据准备 ---
# 从你的LaTeX表格中提取数据
data_mosi = {
    "Metric": ["MAE", "Corr", "Acc2 (non-neg)", "F1 (non-neg)", "Acc2 (neg)", "F1 (neg)", "Acc7"],
    "Qwen2.5 Omni 7B": [0.894, 0.828, 84.92, 84.89, 85.45, 85.50, 41.58],
    "CMFF (Ours)": [0.570, 0.876, 87.32, 87.30, 89.18, 89.20, 53.21],
}
df_mosi = pd.DataFrame(data_mosi)

data_mosei = {
    "Metric": ["MAE", "Corr", "Acc2 (non-neg)", "F1 (non-neg)", "Acc2 (neg)", "F1 (neg)", "Acc7"],
    "Qwen2.5 Omni 7B": [1.081, 0.709, 85.08, 84.62, 84.65, 84.08, 26.83],
    "CMFF (Ours)": [0.483, 0.813, 83.97, 84.40, 88.22, 88.22, 55.59],
}
df_mosei = pd.DataFrame(data_mosei)


# --- 2. 绘图函数 ---
def create_comparison_plot(df, dataset_name):
    """
    为指定数据集生成一个高质量的性能对比条形图。
    """
    # --- 2.1. 绘图设置 ---
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.labelsize"] = 17
    # plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 17

    # 指标分组和X轴标签
    metrics_left_axis = ["MAE", "Corr"]
    metrics_right_axis = ["Acc2 (non-neg)", "F1 (non-neg)", "Acc2 (neg)", "F1 (neg)", "Acc7"]

    metric_labels_with_arrows = [
        "MAE (↓)",
        "Corr (↑)",
        "Acc-2 (negative/non-negative) (↑)",
        "F1 (negative/non-negative) (↑)",
        "Acc-2 (negative/positive) (↑)",
        "F1 (negative/positive) (↑)",
        "Acc-7 (↑)",
    ]

    # 创建图表和双坐标轴
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # --- 2.2. 绘制柱状图 ---
    models = ["Qwen2.5 Omni 7B", "CMFF (Ours)"]
    n_models = len(models)
    bar_width = 0.38
    index = np.arange(len(df["Metric"]))
    bar_colors = sns.color_palette("pastel", n_models)
    left_axis_color = "darkred"
    right_axis_color = "darkblue"

    for i, model_name in enumerate(models):
        bar_positions = index - bar_width / 2 + i * bar_width
        scores = df[model_name].values

        # 绘制左轴 (MAE, Corr)
        ax1.bar(
            bar_positions[: len(metrics_left_axis)],
            scores[: len(metrics_left_axis)],
            bar_width,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=0.8,
            label=model_name if i == 0 else "",  # 避免重复图例
        )
        # 绘制右轴 (Acc, F1)
        ax2.bar(
            bar_positions[len(metrics_left_axis) :],
            scores[len(metrics_left_axis) :],
            bar_width,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=0.8,
            label=model_name if i == 0 else "",
        )

    # --- 2.3. 图表美化 ---
    # ax1.set_title(f"Performance Comparison on {dataset_name} Dataset", pad=20, fontweight="bold")
    ax1.set_xticks(index)
    ax1.set_xticklabels(metric_labels_with_arrows, rotation=15, ha="right")
    ax1.set_xlabel(
        "Evaluation Metrics",
        labelpad=15,
        # fontweight="bold",
    )

    # 设置左Y轴
    ax1.set_ylabel(
        "MAE / Corr Score",
        labelpad=15,
        # fontweight="bold",
        color=left_axis_color,
    )
    ax1.tick_params(axis="y", labelcolor=left_axis_color)
    # 动态设置Y轴上限，留出空间给标签
    y_max_left = df[df["Metric"].isin(metrics_left_axis)][models].max().max()
    ax1.set_ylim(0, y_max_left * 1.2)

    # 设置右Y轴
    ax2.set_ylabel(
        "Accuracy / F1 Score (%)",
        labelpad=15,
        # fontweight="bold",
        color=right_axis_color,
    )
    ax2.tick_params(axis="y", labelcolor=right_axis_color)
    ax2.set_ylim(0, 100)

    # --- 2.4. 在柱状图上添加数值标签 ---
    def add_bar_labels(ax, bars, color, is_percent=False):
        for bar in bars:
            yval = bar.get_height()
            if is_percent:
                label_text = f"{yval:.2f}"
            else:
                label_text = f"{yval:.3f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + ax.get_ylim()[1] * 0.01,  # 向上偏移一点
                label_text,
                va="bottom",
                ha="center",
                fontsize=10,
                # fontweight="bold",
                color=color,
            )

    # ax1.patches 和 ax2.patches 包含了所有柱子，需要按模型分开处理
    # 计算每个坐标轴上，每个模型所包含的柱子数量
    patches_per_model_ax1 = len(metrics_left_axis)
    patches_per_model_ax2 = len(metrics_right_axis)

    # 为 ax1 (左轴) 的柱子添加标签
    add_bar_labels(ax1, ax1.patches[:patches_per_model_ax1], left_axis_color)  # Model 1
    add_bar_labels(ax1, ax1.patches[patches_per_model_ax1:], left_axis_color)  # Model 2

    # 为 ax2 (右轴) 的柱子添加标签
    add_bar_labels(ax2, ax2.patches[:patches_per_model_ax2], right_axis_color, is_percent=True)  # Model 1
    add_bar_labels(ax2, ax2.patches[patches_per_model_ax2:], right_axis_color, is_percent=True)  # Model 2

    # --- 2.5. 添加图例、网格线和边框 ---
    handles = [
        Patch(facecolor=bar_colors[0], edgecolor="black", linewidth=0.8),
        Patch(facecolor=bar_colors[1], edgecolor="black", linewidth=0.8),
    ]
    ax1.legend(
        handles,
        models,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(models),
        title="Model",
        frameon=False,
        # framealpha=0.9,
        # facecolor="white",
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.4)
    ax1.set_axisbelow(True)

    fig.tight_layout()

    # --- 2.6. 保存图表 ---
    filename = f"cmff/analysis/Figure_{dataset_name}_llm_comparison.png"
    plt.savefig(filename, format="png", dpi=300, bbox_inches="tight")
    print(f"Chart saved as {filename}")
    plt.show()


# --- 3. 生成并保存两张图表 ---
create_comparison_plot(df_mosi, "mosi")
create_comparison_plot(df_mosei, "mosei")
