import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import cm


def tsne_visualization_regression(features, labels, save_dir="./visualization"):
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


def tsne_visualization_cls(features, labels, save_dir="./visualization"):
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


# --- PASTE the tsne_classification_visualization function definition here ---
def tsne_classification_visualization(
    features,
    labels,
    class_names=None,  # Optional: Provide names for classes
    save_dir="./visualization",
    filename="tsne_classification.png",
    title="t-SNE Visualization of Features by Class",
    random_state=42,
    figsize=(10, 8),
    dpi=300,
    alpha=0.8,
    marker_size=20,  # Add marker size control
    legend_title="Classes",
    cmap_name="coolwarm",  # Use a diverging colormap suitable for sentiment
):
    """
    Performs t-SNE dimensionality reduction and visualizes features colored by class labels.
    [... function code as provided in the previous answer ...]
    (Note: Changed default cmap_name to 'coolwarm' which might be nice for sentiment)
    """
    print("正在执行 t-SNE 降维 (用于分类)...")
    features = np.asarray(features)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Use a diverging colormap if appropriate, otherwise use the provided one
    try:
        cmap = cm.get_cmap(cmap_name, n_classes)  # Get a colormap instance
        colors = [cmap(i) for i in range(n_classes)]  # Get list of colors
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}  # Map labels to colors
    except ValueError:  # Handle cases where cmap might not accept n_classes directly
        print(f"Warning: Colormap {cmap_name} might not be optimally sampled for {n_classes} classes. Using default sampling.")
        cmap = cm.get_cmap(cmap_name)
        color_vals = np.linspace(0, 1, n_classes)
        colors = cmap(color_vals)
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    if n_classes > cmap.N and not hasattr(cmap, "colors"):  # Check if discrete colormap has enough colors
        print(f"警告: 类别数量 ({n_classes}) 大于所选离散颜色映射 '{cmap_name}' ({cmap.N}) 中的不同颜色数量。颜色可能会重复。")

    # --- t-SNE 降维 ---
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000, init="pca")
    features_2d = tsne.fit_transform(features)
    print("t-SNE 降维完成.")

    # --- 创建绘图 ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- 分别绘制每个类别的数据点 ---
    for i, label_val in enumerate(unique_labels):
        # 确定图例中显示的名称
        if class_names:
            if isinstance(class_names, dict):
                display_name = class_names.get(label_val, str(label_val))
            elif isinstance(class_names, (list, tuple)) and isinstance(label_val, int) and 0 <= label_val < len(class_names):
                display_name = class_names[label_val]
            else:
                display_name = str(label_val)
        else:
            display_name = str(label_val)

        idx = labels == label_val
        ax.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            color=color_map[label_val],
            label=display_name,
            alpha=alpha,
            s=marker_size,
            edgecolors="w",
            linewidths=0.5,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 添加图例 ---
    legend = ax.legend(title=legend_title, loc="best", markerscale=1.5, fontsize="medium")
    if legend:  # Check if legend was created
        legend.get_title().set_fontsize("large")

    plt.tight_layout()

    # --- 保存图形 ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"t-SNE 分类可视化已保存到: {save_path}")
    except Exception as e:
        print(f"保存图形时出错: {e}")

    return fig


# --- END of tsne_classification_visualization function ---


def discretize_sentiment_labels(continuous_labels):
    """
    Converts continuous sentiment labels (like MOSI/MOSEI) to discrete classes [0-6].

    Args:
        continuous_labels (array-like): Labels in the continuous range, e.g., [-3, 3].
                                        Can be NumPy array, list, or PyTorch tensor.

    Returns:
        np.ndarray: Discrete integer labels in the range [0, 6].
    """
    # Ensure input is a NumPy array
    if isinstance(continuous_labels, torch.Tensor):
        labels_np = continuous_labels.cpu().numpy()  # Move to CPU if needed and convert
    else:
        labels_np = np.asarray(continuous_labels)

    # 1. Clamp to [-3, 3]
    clamped_labels = np.clip(labels_np, -3.0, 3.0)
    # 2. Round to nearest integer
    rounded_labels = np.round(clamped_labels)
    # 3. Shift to [0, 6]
    discrete_labels = rounded_labels + 3
    # Ensure integer type
    return discrete_labels.astype(int)


def visualize_sentiment_tsne(
    features,
    continuous_labels,
    save_dir="./visualization",
    filename="tsne_sentiment_classes.png",
    title="t-SNE Visualization by Discretized Sentiment",
    **kwargs,  # Pass other args like random_state, figsize, etc. to the main function
):
    """
    Performs label discretization and then visualizes features using t-SNE,
    coloring by discretized sentiment class.

    Args:
        features (np.ndarray): High-dimensional feature array.
        continuous_labels (array-like): Continuous sentiment labels (e.g., range [-3, 3]).
        save_dir (str): Directory to save the plot.
        filename (str): Name for the saved plot file.
        title (str): Title for the plot.
        **kwargs: Additional keyword arguments passed to tsne_classification_visualization.
                  (e.g., random_state, figsize, dpi, alpha, marker_size, cmap_name)
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure object.
    """
    print("Discretizing continuous labels [-3, 3] into classes [0, 6]...")
    discrete_labels = discretize_sentiment_labels(continuous_labels)

    # Define meaningful names for the classes 0-6 based on original rounded sentiment
    # Using a dictionary mapping: discrete_label -> display_name
    class_names = {
        0: "-3 (Very Neg)",
        1: "-2 (Neg)",
        2: "-1 (Slightly Neg)",
        3: " 0 (Neutral)",
        4: " 1 (Slightly Pos)",
        5: " 2 (Pos)",
        6: " 3 (Very Pos)",
    }
    # You could also use just the numbers:
    # class_names = {i: str(i - 3) for i in range(7)}

    print("Calling t-SNE visualization for classification...")
    # Call the original classification visualization function
    # Pass a suitable colormap like 'coolwarm', 'RdYlBu', or 'viridis'
    kwargs.setdefault("cmap_name", "coolwarm")  # Set default if not provided
    fig = tsne_classification_visualization(
        features=features,
        labels=discrete_labels,  # Use the NEW discrete labels
        class_names=class_names,  # Provide the names for the legend
        save_dir=save_dir,
        filename=filename,
        title=title,
        legend_title="Sentiment Class",  # More specific legend title
        **kwargs,  # Pass through other arguments like random_state, dpi, etc.
    )
    return fig


# --- Example Usage (replace with your actual data loading) ---


def tsne_3d_classification_visualization(
    features,
    labels,
    class_names=None,
    save_dir="./visualization",
    filename="tsne_3d_classification.png",
    title="3D t-SNE Visualization by Class",
    random_state=42,
    figsize=(12, 10),  # 3D plots often benefit from larger size
    dpi=300,
    alpha=0.7,  # Transparency might need adjustment in 3D
    marker_size=20,
    legend_title="Classes",
    cmap_name="tab10",
    azim=-60,  # Azimuthal viewing angle (horizontal rotation)
    elev=30,  # Elevation viewing angle (vertical tilt)
):
    """
    Performs t-SNE dimensionality reduction to 3D and visualizes features
    colored by class labels using a 3D scatter plot.

    Args:
        features (np.ndarray): High-dimensional feature array (n_samples, n_features).
        labels (np.ndarray or list): Discrete class labels for each sample (n_samples,).
        class_names (dict or list, optional): Mapping from label values to display names.
        save_dir (str): Directory to save the plot.
        filename (str): Name for the saved plot file.
        title (str): Title for the plot.
        random_state (int): Random seed for TSNE reproducibility.
        figsize (tuple): Figure size.
        dpi (int): Dots per inch for saving the figure.
        alpha (float): Transparency level for scatter points.
        marker_size (int): Size of the scatter points.
        legend_title (str): Title for the legend.
        cmap_name (str): Name of the matplotlib colormap for classes.
        azim (float): Azimuthal viewing angle for the 3D plot.
        elev (float): Elevation viewing angle for the 3D plot.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure object.
    """
    print("正在执行 t-SNE 降维 (3D, 用于分类)...")
    features = np.asarray(features)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # --- t-SNE to 3 Dimensions ---
    tsne = TSNE(n_components=3, random_state=random_state, perplexity=30, n_iter=1000, init="pca")
    features_3d = tsne.fit_transform(features)
    print("3D t-SNE 降维完成.")

    # --- Create 3D Plot ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")  # Create a 3D subplot

    # --- Assign Colors (same logic as 2D) ---
    try:
        cmap = cm.get_cmap(cmap_name, n_classes)  # Get a colormap instance
        colors = [cmap(i) for i in range(n_classes)]  # Get list of colors
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}  # Map labels to colors
    except ValueError:  # Handle cases where cmap might not accept n_classes directly
        print(f"Warning: Colormap {cmap_name} might not be optimally sampled for {n_classes} classes. Using default sampling.")
        cmap = cm.get_cmap(cmap_name)
        color_vals = np.linspace(0, 1, n_classes)
        colors = cmap(color_vals)
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    if hasattr(cmap, "N") and n_classes > cmap.N and not hasattr(cmap, "colors"):  # Check if discrete colormap has enough colors
        print(f"警告: 类别数量 ({n_classes}) 大于所选离散颜色映射 '{cmap_name}' ({cmap.N}) 中的不同颜色数量。颜色可能会重复。")

    # --- Plot each class separately for legend ---
    for i, label_val in enumerate(unique_labels):
        # Determine display name (same logic as 2D)
        if class_names:
            if isinstance(class_names, dict):
                display_name = class_names.get(label_val, str(label_val))
            elif isinstance(class_names, (list, tuple)) and isinstance(label_val, int) and 0 <= label_val < len(class_names):
                display_name = class_names[label_val]
            else:
                display_name = str(label_val)
        else:
            display_name = str(label_val)

        idx = labels == label_val
        # Use ax.scatter for 3D, providing x, y, and z coordinates
        ax.scatter(
            features_3d[idx, 0],  # X coordinate (t-SNE Dim 1)
            features_3d[idx, 1],  # Y coordinate (t-SNE Dim 2)
            features_3d[idx, 2],  # Z coordinate (t-SNE Dim 3)
            color=color_map[label_val],
            label=display_name,
            alpha=alpha,
            s=marker_size,
            edgecolors="w",  # Optional edge color
            linewidths=0.3,  # Optional edge width
        )

    ax.set_title(title, fontsize=16)
    # Set axis labels for 3D
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12)

    # Remove tick labels if desired (often done for t-SNE)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    # --- Add Legend ---
    legend = ax.legend(title=legend_title, loc="best", markerscale=1.2, fontsize="medium")
    if legend:
        legend.get_title().set_fontsize("large")

    # --- Set Viewing Angle ---
    ax.view_init(elev=elev, azim=azim)

    # --- Save Figure ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    try:
        # Note: tight_layout often doesn't work well with 3D plots
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"3D t-SNE 分类可视化已保存到: {save_path}")
    except Exception as e:
        print(f"保存 3D 图形时出错: {e}")

    return fig


# --- NEW: Wrapper for 3D Sentiment Visualization ---
def visualize_sentiment_tsne_3d(
    features,
    continuous_labels,
    save_dir="./visualization",
    filename="tsne_3d_sentiment_classes.png",
    title="3D t-SNE Visualization by Discretized Sentiment",
    **kwargs,  # Pass args to tsne_3d_classification_visualization
):
    """
    Performs label discretization and then visualizes features using 3D t-SNE,
    coloring by discretized sentiment class.

    Args:
        features (np.ndarray): High-dimensional feature array.
        continuous_labels (array-like): Continuous sentiment labels (e.g., range [-3, 3]).
        save_dir (str): Directory to save the plot.
        filename (str): Name for the saved plot file.
        title (str): Title for the plot.
        **kwargs: Additional keyword arguments passed to tsne_3d_classification_visualization.
                  (e.g., random_state, figsize, dpi, alpha, marker_size, cmap_name, azim, elev)

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure object.
    """
    print("Discretizing continuous labels [-3, 3] into classes [0, 6] for 3D plot...")
    discrete_labels = discretize_sentiment_labels(continuous_labels)  # Use the helper function

    # Define class names (same as 2D version)
    class_names = {
        0: "-3 (Very Neg)",
        1: "-2 (Neg)",
        2: "-1 (Slightly Neg)",
        3: " 0 (Neutral)",
        4: " 1 (Slightly Pos)",
        5: " 2 (Pos)",
        6: " 3 (Very Pos)",
    }
    # class_names = {i: str(i - 3) for i in range(7)} # Simpler alternative

    print("Calling 3D t-SNE visualization for classification...")
    # Set default colormap if not provided
    kwargs.setdefault("cmap_name", "coolwarm")
    # Call the NEW 3D classification visualization function
    fig = tsne_3d_classification_visualization(
        features=features,
        labels=discrete_labels,
        class_names=class_names,
        save_dir=save_dir,
        filename=filename,
        title=title,
        legend_title="Sentiment Class",
        **kwargs,  # Pass through arguments like azim, elev, etc.
    )
    return fig


if __name__ == "__main__":
    # feature = np.load("/Users/fgtear/Downloads/04211506_MOSI_mm_cmff_feature.npy")
    feature = np.load("/Users/fgtear/Downloads/04211047_MOSI_text_cmff_feature.npy")
    label = np.load("/Users/fgtear/Downloads/mosi_label.npy")

    # fig = tsne_visualization_3d(feature, label)
    # fig.savefig(
    #     "./tsne_3d.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    visualize_sentiment_tsne(
        features=feature,
        continuous_labels=label,
        save_dir="./analysis",
        filename="tsne_sentiment_classes.png",
        title="t-SNE Visualization by Discretized Sentiment",
        random_state=42,
        figsize=(10, 8),
        dpi=300,
        alpha=0.8,
        marker_size=20,
    )
    # visualize_sentiment_tsne_3d(
    #     features=feature,
    #     continuous_labels=label,
    #     save_dir="./",
    #     filename="tsne_3d_sentiment_classes.png",
    #     title="3D t-SNE Visualization by Discretized Sentiment",
    #     random_state=42,
    #     figsize=(12, 10),
    #     dpi=300,
    #     alpha=0.9,
    #     marker_size=20,
    #     azim=-60,
    #     elev=30,
    # )
    # plt.show()
