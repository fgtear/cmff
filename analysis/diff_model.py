import mlflow
import matplotlib.pyplot as plt
import numpy as np
import os


def compare_model_predictions(run_id1, run_id2, save_path):
    """
    比较两个模型的预测结果

    参数:
        run_id1 (str): 第一个MLflow run的ID
        run_id2 (str): 第二个MLflow run的ID
        label_path (str, optional): 标签文件路径，如果为None，则尝试从MLflow获取
        title (str, optional): 图表标题
        save_path (str, optional): 保存图表的路径，如果不提供则显示图表

    返回:
        None
    """
    # 从MLflow获取预测结果
    mlflow.set_tracking_uri("http://fgtear.fun:35000/")
    # client = mlflow.tracking.MlflowClient("http://fgtear.fun:35000/")

    # 获取第一个模型的预测结果
    os.makedirs("./.temp", exist_ok=True)
    # pred1_path = mlflow.artifacts.download_artifacts(
    #     # artifact_uri="http://fgtear.fun:35000/",
    #     run_id=run_id1,
    #     artifact_path="label.npy",
    #     dst_path="./.temp",
    # )

    # pred1_path = client.download_artifacts(run_id1, "prediction.npy", dst_path="./.temp")
    # label_path = client.download_artifacts(run_id1, "labels.npy", dst_path="./.temp")
    # pred1 = np.load(pred1_path)

    # # 获取第二个模型的预测结果
    # pred2_path = client.download_artifacts(run_id2, "prediction.npy", dst_path="./.temp")
    # pred2 = np.load(pred2_path)
    # labels = np.load(label_path)

    labels = np.load("visualization/label.npy")
    pred1 = np.load("visualization/prediction-24.npy").squeeze()
    pred2 = np.load("visualization/prediction-0.npy").squeeze()

    diff1 = np.abs(pred1 - labels)
    diff2 = np.abs(pred2 - labels)

    diff = diff1 - diff2

    # 确保数据形状一致
    if len(labels) != len(pred1) or len(labels) != len(pred2):
        raise ValueError(f"数据长度不一致: labels={len(labels)}, pred1={len(pred1)}, pred2={len(pred2)}")

    # 创建散点图
    plt.figure(figsize=(8, 8))

    plt.scatter(labels, diff, alpha=0.5, label=f"Layer {24} vs Layer {0}")
    # plt.scatter(labels, pred2, alpha=0.5, label=f"模型2 (Run ID: {run_id2[:8]}...)")

    # 添加对角线表示完美预测的参考线
    # min_val = min(np.min(labels), np.min(pred1), np.min(pred2))
    # max_val = max(np.max(labels), np.max(pred1), np.max(pred2))
    # plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

    plt.title("模型预测比较")

    plt.xlabel("Labels")
    plt.ylabel("Diff")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

    # 计算并返回一些比较指标
    mean = np.mean(diff)
    std = np.std(diff)
    # mae1 = np.mean(np.abs(pred1 - labels))
    # mae2 = np.mean(np.abs(pred2 - labels))

    print(f"mean: {mean:.4f}")
    print(f"std: {std:.4f}")


if __name__ == "__main__":
    compare_model_predictions("c4e3b51af1e14db497635f3eb5adfb68", "c4e3b51af1e14db497635f3eb5adfb68", save_path=None)
