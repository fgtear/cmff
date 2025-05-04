import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def tsne_visualization(features, labels, save_dir="./visualization"):
    print("正在执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    # 创建并保存t-SNE可视化图
    fig = plt.figure(figsize=(10, 8), dpi=300)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="Label Value")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    # tsne_path = os.path.join(save_dir, "tsne.png")
    # plt.savefig(tsne_path, dpi=300)
    # print(f"t-SNE 可视化已保存到: {tsne_path}")
    return fig


def tsne_visualization_3d(features, labels, save_dir="./visualization"):
    print("正在执行3D t-SNE降维...")
    tsne = TSNE(n_components=3, random_state=42)
    features_3d = tsne.fit_transform(features)

    # 创建3D可视化图
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=labels, cmap="viridis", alpha=0.8)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title("3D t-SNE Visualization")

    plt.colorbar(scatter, label="Label Value", ax=ax)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    # tsne_3d_path = os.path.join(save_dir, "tsne_3d.png")
    # plt.savefig(tsne_3d_path, dpi=300)
    # print(f"3D t-SNE 可视化已保存到: {tsne_3d_path}")

    return fig


def weights_visualization(weights, save_dir="./visualization"):
    print("正在执行注意力权重可视化...")
    # 创建并保存注意力权重可视化图
    plt.figure(figsize=(10, 8), dpi=300)
    # weights = [12, 15,182, ... ] # 一维数组，使用直方图可视化
    plt.bar(range(len(weights)), weights)
    # plt.imshow(weights, cmap="viridis", aspect="auto")
    # plt.colorbar(label="Attention Weight")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "attention_weights.png")
    plt.savefig(weights_path, dpi=300)
    print(f"注意力权重可视化已保存到: {weights_path}")
    return weights_path


def prediction_visualization(predictions, labels, save_dir="./visualization"):
    print("正在执行预测可视化...")
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(labels, predictions, alpha=0.8)
    plt.xlabel("True Labels")
    plt.ylabel("Predictions")
    plt.title("Prediction Visualization")
    os.makedirs(save_dir, exist_ok=True)
    prediction_path = os.path.join(save_dir, "prediction_visualization.png")
    plt.savefig(prediction_path, dpi=300)
    print(f"预测可视化已保存到: {prediction_path}")
    return prediction_path


if __name__ == "__main__":
    feature = np.load("visualization/feature-24.npy")
    label = np.load("visualization/label.npy")
    fig = tsne_visualization_3d(feature, label)
    # fig.show()
    plt.show()
