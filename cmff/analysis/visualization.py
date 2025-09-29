import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def tsne_visualization(features, labels):
    print("正在执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    # 创建并保存t-SNE可视化图
    fig = plt.figure(figsize=(10, 8), dpi=300)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Label Value", fontsize=25)
    cbar.ax.tick_params(labelsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.tight_layout()
    return fig


def tsne_visualization_3d(features, labels, save_dir="./visualization"):
    print("正在执行3D t-SNE降维...")
    tsne = TSNE(n_components=3, random_state=42)
    features_3d = tsne.fit_transform(features)

    # 创建3D可视化图
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=labels, cmap="viridis", alpha=0.8)

    ax.set_xlabel("Dimension 1", fontsize=21)
    ax.set_ylabel("Dimension 2", fontsize=21)
    ax.set_zlabel("Dimension 3", fontsize=21)
    ax.set_title("3D t-SNE Visualization", fontsize=21)

    cbar = plt.colorbar(scatter, label="Label Value", ax=ax)
    cbar.ax.tick_params(labelsize=21)
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
    plt.bar(range(len(weights)), weights)
    plt.xlabel("Index", fontsize=21)
    plt.ylabel("Weight", fontsize=21)
    plt.title("Attention Weights Visualization", fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
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
    plt.xlabel("True Labels", fontsize=21)
    plt.ylabel("Predictions", fontsize=21)
    plt.title("Prediction Visualization", fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    prediction_path = os.path.join(save_dir, "prediction_visualization.png")
    plt.savefig(prediction_path, dpi=300)
    print(f"预测可视化已保存到: {prediction_path}")
    return prediction_path


if __name__ == "__main__":
    label = np.load("cmff/results/mosi_label.npy")
    feature = np.load("cmff/results/04211506_MOSI_mm_cmff_feature.npy")
    fig = tsne_visualization(feature, label)
    fig.savefig("cmff/analysis/Figure_tsne_cmff.png", dpi=300)

    label = np.load("cmff/mlruns/0/d8be79a27ec4458c928d136c44a9c8a7/artifacts/label.npy")
    feature = np.load("cmff/mlruns/0/d8be79a27ec4458c928d136c44a9c8a7/artifacts/feature.npy")
    fig = tsne_visualization(feature, label)
    fig.savefig("cmff/analysis/Figure_tsne_cat_last.png", dpi=300)

    label = np.load("cmff/mlruns/0/6179540ae7c040b8b887b8157960afd9/artifacts/label.npy")
    feature = np.load("cmff/mlruns/0/6179540ae7c040b8b887b8157960afd9/artifacts/feature.npy")
    fig = tsne_visualization(feature, label)
    fig.savefig("cmff/analysis/Figure_tsne_mean_cat.png", dpi=300)

    label = np.load("cmff/mlruns/0/d29117a92a3e4f29bcd7436f939fc6c7/artifacts/label.npy")
    feature = np.load("cmff/mlruns/0/d29117a92a3e4f29bcd7436f939fc6c7/artifacts/feature.npy")
    fig = tsne_visualization(feature, label)
    fig.savefig("cmff/analysis/Figure_tsne_encoder.png", dpi=300)
