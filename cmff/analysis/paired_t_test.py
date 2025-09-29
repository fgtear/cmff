import numpy as np
from scipy import stats


def compare_regression_predictions(label_path, pred1_path, pred2_path, alpha=0.05):
    """
    使用配对 t 检验比较两种回归预测模型的性能。

    该函数旨在验证 pred1 的预测结果是否显著优于 pred2。
    "优于" 在这里定义为具有更小的预测误差（例如，平均绝对误差 MAE）。

    Args:
        label_path (str): 真实标签 .npy 文件的路径 (范围 -3 到 3)。
        pred1_path (str): 第一个模型（例如，多层次特征）预测结果 .npy 文件的路径。
        pred2_path (str): 第二个模型（例如，最后一层特征）预测结果 .npy 文件的路径。
        alpha (float): 假设检验的显著性水平。默认为 0.05。

    Returns:
        dict: 包含比较结果的字典：
              - 'mae1': pred1 的平均绝对误差。
              - 'mae2': pred2 的平均绝对误差。
              - 't_statistic': 配对 t 检验的 t 统计量。
              - 'p_value_one_sided': 单侧检验的 p 值。
              - 'reject_h0': 布尔值，是否拒绝原假设 (H0)。
              - 'conclusion': 检验结果的文字描述。
    """
    # 1. 加载数据
    try:
        labels = np.load(label_path)
        pred1 = np.load(pred1_path)
        pred2 = np.load(pred2_path)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return None
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

    # 确保输入维度匹配
    if not (labels.shape == pred1.shape == pred2.shape):
        print("错误：标签和预测结果的形状不匹配。")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Pred1 shape: {pred1.shape}")
        print(f"  Pred2 shape: {pred2.shape}")
        return None

    # 2. 计算绝对误差
    error1 = np.abs(labels - pred1)
    error2 = np.abs(labels - pred2)

    # 3. 计算评价指标 (MAE)
    mae1 = np.mean(error1)
    mae2 = np.mean(error2)

    # 4. 执行配对 t 检验
    # 原假设 (H0): error1 的均值 >= error2 的均值 (模型1 不比 模型2 好)
    # 备择假设 (H1): error1 的均值 < error2 的均值 (模型1 显著优于 模型2)
    # 我们期望 error1 - error2 < 0，所以这是一个左尾（单侧）检验。
    # scipy.stats.ttest_rel 计算的是双侧检验的 p 值。
    # 对于左尾检验，如果 t 统计量 < 0, p_one_sided = p_two_sided / 2
    # 如果 t 统计量 >= 0, p_one_sided = 1 - p_two_sided / 2 (或直接认为不显著)
    t_statistic, p_value_two_sided = stats.ttest_rel(error1, error2, nan_policy="omit")  # nan_policy='omit' 可以处理可能的 NaN 值

    # 计算单侧 p 值
    if t_statistic < 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        # 如果 t >= 0，意味着 error1 的样本均值不小于 error2，
        # 这不符合我们的备择假设方向，因此 p 值会很大，肯定大于 alpha。
        # 可以直接设置为 1，或者用 1 - p/2。为严谨，用后者。
        p_value_one_sided = 1.0 - p_value_two_sided / 2  # 实际上这种情况几乎不会拒绝 H0

    # 5. 判断结果
    reject_h0 = p_value_one_sided < alpha

    # 6. 生成结论
    conclusion = f"比较模型1 (pred1) 与模型2 (pred2) 在显著性水平 alpha = {alpha}:\n"
    conclusion += f"  模型1 MAE: {mae1:.4f}\n"
    conclusion += f"  模型2 MAE: {mae2:.4f}\n"
    conclusion += f"  配对 t 检验:\n"
    conclusion += f"    t 统计量 = {t_statistic:.4f}\n"
    conclusion += f"    单侧 p 值 = {p_value_one_sided:.4f}\n"

    if reject_h0:
        conclusion += f"结论: p 值 ({p_value_one_sided:.4f}) < alpha ({alpha})，拒绝原假设。\n"
        conclusion += "有统计学上的显著证据表明模型1（例如，多层次特征）的预测误差显著低于模型2（例如，最后一层特征）。"
    else:
        conclusion += f"结论: p 值 ({p_value_one_sided:.4f}) >= alpha ({alpha})，未能拒绝原假设。\n"
        conclusion += "没有足够的统计学证据表明模型1的预测误差显著低于模型2。"

    return {
        "mae1": mae1,
        "mae2": mae2,
        "t_statistic": t_statistic,
        "p_value_one_sided": p_value_one_sided,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
    }


# === 使用示例 ===
# 假设你的文件名为 label.npy, pred1.npy, pred2.npy
# 请确保这三个文件存在于你的 Python 脚本运行的目录下，或者提供完整路径
# results = compare_regression_predictions('label.npy', 'pred1.npy', 'pred2.npy')

# if results:
#     print(results['conclusion'])
#     print("\n详细结果字典:")
#     print(results)

# === 注意 ===
# 在实际运行前，你需要：
# 1. 确保已安装 numpy 和 scipy (`pip install numpy scipy`)
# 2. 将 'label.npy', 'pred1.npy', 'pred2.npy' 替换成你的实际文件名或路径。
# 3. 确保你的 .npy 文件可以被正确加载且格式符合预期。

if __name__ == "__main__":
    results = compare_regression_predictions(
        "cmff/results/mosi_label.npy",
        "cmff/results/04211047_MOSI_text_cmff_prediction.npy",
        "cmff/results/04211029_MOSI_text_last_prediction.npy",
    )
    print(results)

    results = compare_regression_predictions(
        "cmff/results/mosi_label.npy",
        "cmff/results/04211235_MOSI_audio_cmff_prediction.npy",
        "cmff/results/04211208_MOSI_audio_last_prediction.npy",
    )
    print(results)
