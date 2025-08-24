import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

# 建议设置一个更适合论文的字体，例如 Times New Roman 或 Arial
# 如果系统中没有，matplotlib 会回退到默认字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 或者使用 sans-serif 字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]

# 确保 matplotlib 正确显示负号
plt.rcParams["axes.unicode_minus"] = False


def visualize_label_distribution(dataset: pd.DataFrame, save_path: str = None):
    """
    可视化 MOSI/MOSEI 数据集中 'label' 的分布情况，按 'mode' (train/valid/test) 分开显示。

    Args:
        dataset (pd.DataFrame): 包含 'label' 和 'mode' 列的 DataFrame。
                                'label' 的范围应在 -3 到 3 之间。
                                'mode' 应包含 'train', 'valid', 'test'。
        save_path (str, optional): 保存图形的文件路径 (例如 'label_distribution.pdf' 或 'label_distribution.png')。
                                   如果不提供，则只显示图形。 Defaults to None.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("输入 'dataset' 必须是 Pandas DataFrame。")
    if "label" not in dataset.columns:
        raise ValueError("DataFrame 中缺少 'label' 列。")
    if "mode" not in dataset.columns:
        raise ValueError("DataFrame 中缺少 'mode' 列。")

    # 设置 seaborn 样式，'whitegrid' 或 'ticks' 通常适合论文
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)  # colorblind 调色板更具包容性

    # 创建图形和轴对象，可以更好地控制图形大小和纵横比
    # 'height' 控制每个子图的高度，'aspect' 控制宽高比
    # 使用 displot(kind='kde') 会自动创建 FacetGrid，更灵活
    g = sns.displot(
        data=dataset,
        x="label",
        hue="mode",  # 根据 'mode' 列区分颜色
        hue_order=["train", "valid", "test"],  # 确保顺序一致
        kind="kde",  # 绘制核密度估计图 (平滑曲线)
        fill=True,  # 填充曲线下方区域，增加可读性
        alpha=0.3,  # 设置填充透明度
        linewidth=2,  # 设置线条宽度
        common_norm=False,  # 重要：每个子集的密度独立归一化，显示各自的分布形态
        # 如果设为 True, 总面积为1, 可能会扭曲单个分布的形状
        height=5,  # 图形高度（英寸）
        aspect=1.5,  # 图形宽高比
    )

    # --- 自定义图形外观 ---
    # 设置标题
    # g.fig.suptitle("Distribution of Labels across Data Splits", y=1.03, fontsize=16)  # y 调整标题位置防止重叠

    # 设置轴标签
    g.set_axis_labels("Label Value (Sentiment Score)", "Density", fontsize=21)

    # 调整 Y 轴标签与轴的距离
    # displot 返回 FacetGrid，它可能包含多个子图 (axes)
    # 在本例中，只有一个子图，可以通过 g.ax 访问
    if hasattr(g, "ax") and g.ax is not None:
        g.ax.yaxis.labelpad = 15  # 增加这个值会让标签离轴更远

    # 设置 x 轴范围，确保覆盖 -3 到 3，并留有一些边距
    plt.xlim(-3.5, 3.5)

    # 调整图例 - 增加检查确保图例存在
    if g.legend is not None:
        g.legend.set_title("Dataset Split")
        if g.legend.get_title() is not None:
            g.legend.get_title().set_fontsize(21)
        for t in g.legend.texts:
            t.set_fontsize(21)

    # 调整刻度标签大小
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    # 移除顶部和右侧的轴线（根据喜好选择，有时 'ticks' 风格更需要）
    # sns.despine(fig=g.fig) # displot 返回的是 FacetGrid，可以直接操作 fig

    # --- 保存或显示图形 ---
    if save_path:
        try:
            # 获取文件扩展名，用于指定格式
            file_format = os.path.splitext(save_path)[1][1:].lower()
            if not file_format:  # 如果没有扩展名，默认为 png
                file_format = "png"
                save_path += ".png"

            # 推荐保存为矢量格式 (pdf, svg) 或高分辨率位图 (png)
            g.fig.savefig(save_path, format=file_format, dpi=300, bbox_inches="tight")
            print(f"图形已保存到: {save_path}")
        except Exception as e:
            print(f"保存图形时出错: {e}")
        plt.close(g.fig)  # 保存后关闭图形，避免在notebook中重复显示
    else:
        plt.show()


# --- 示例用法 ---
if __name__ == "__main__":
    df_mosi = pd.read_csv("datasets/MOSI/label.csv")
    df_mosei = pd.read_csv("datasets/MOSEI/label.csv")
    # 调用函数进行可视化
    visualize_label_distribution(df_mosi)
    visualize_label_distribution(df_mosei)

    # 调用函数并保存图形
    visualize_label_distribution(df_mosi, save_path="cmff/analysis/Figure_mosi_distribution.png")
    visualize_label_distribution(df_mosei, save_path="cmff/analysis/Figure_mosei_distribution.png")
