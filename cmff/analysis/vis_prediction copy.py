import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def add_p_value_annotation(ax, x1, x2, y, p_value, height_factor=1.05):
    """Adds a significance bracket and p-value annotation to a plot."""
    # Determine y-position for the bracket
    line_y = y * height_factor
    bracket_height = line_y * 0.03  # Small height for the vertical parts

    # Draw the bracket lines
    ax.plot([x1, x1, x2, x2], [line_y, line_y + bracket_height, line_y + bracket_height, line_y], lw=1.5, c="black")

    # Format p-value
    if p_value < 0.001:
        p_text = "p < 0.001"
    elif p_value < 0.01:
        p_text = f"p = {p_value:.3f}"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.3f}"
    else:
        p_text = f"p = {p_value:.3f}"  # Or "n.s." for non-significant

    # Add the text
    ax.text((x1 + x2) * 0.5, line_y + bracket_height, p_text, ha="center", va="bottom", color="black", fontsize=12)

    # Adjust ylim slightly to make space
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], (line_y + bracket_height) * 1.05)  # Ensure space above annotation


# Set a professional style (important for SCI papers)
sns.set_theme(style="ticks", context="paper", font_scale=1.2)  # 'paper', 'notebook', 'talk', 'poster'


def plot_error_scatter_colored(
    label,
    prediction1,
    prediction2,
    model1_name="Model 1",
    model2_name="Model 2",
    use_absolute_error=True,
    title="Comparison of Prediction Errors",
    figsize=(7, 6),  # Slightly wider for legend
    palette="coolwarm",  # Diverging palette often good here
    save_path=None,
    dpi=300,
    **kwargs,
):
    """
    Generates a scatter plot comparing the errors of two prediction models,
    coloring points based on which model had a smaller error.

    Args:
        label (array-like): Ground truth values.
        prediction1 (array-like): Predictions from the first model.
        prediction2 (array-like): Predictions from the second model.
        model1_name (str): Name for the first model (for axis label).
        model2_name (str): Name for the second model (for axis label).
        use_absolute_error (bool): If True, plots absolute errors, otherwise raw errors.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
        palette (str or dict): Seaborn color palette for the hue. 'coolwarm', 'RdBu', 'viridis' etc.
        save_path (str, optional): Path to save the figure. If None, not saved.
        dpi (int): Dots per inch for saving the figure.
        **kwargs: Additional keyword arguments passed to sns.scatterplot (e.g., s, alpha).
    """
    label = np.asarray(label)
    prediction1 = np.asarray(prediction1)
    prediction2 = np.asarray(prediction2)

    if use_absolute_error:
        error1 = np.abs(prediction1 - label)
        error2 = np.abs(prediction2 - label)
        error_type = "Absolute Error"
    else:
        error1 = prediction1 - label
        error2 = prediction2 - label
        error_type = "Error"

    # --- Modification Start ---
    # Create a condition to determine the color (hue)
    # We use a small tolerance epsilon for points exactly on the line if needed,
    # but comparing directly usually works fine.
    # epsilon = 1e-9 # Optional small tolerance
    condition = []
    label_better1 = f"{model1_name} Better ({error_type})"  # Model 1 error is smaller
    label_better2 = f"{model2_name} Better ({error_type})"  # Model 2 error is smaller
    label_equal = "Equal Error"

    for e1, e2 in zip(error1, error2):
        # if abs(e1 - e2) < epsilon:
        #    condition.append(label_equal) # Optional: Handle exact equality
        if e1 < e2:
            condition.append(label_better1)
        elif e2 < e1:
            condition.append(label_better2)
        else:  # Exactly equal
            condition.append(label_equal)

    hue_order = [label_better1, label_better2, label_equal]  # Control legend order
    # --- Modification End ---

    fig, ax = plt.subplots(figsize=figsize)

    # Use the 'condition' list/array for the hue parameter
    scatter_plot = sns.scatterplot(
        x=error1,
        y=error2,
        hue=condition,  # Use condition for color
        hue_order=hue_order,  # Control legend order
        palette=palette,  # Apply chosen color palette
        ax=ax,
        alpha=kwargs.pop("alpha", 0.7),  # Default alpha if not provided
        s=kwargs.pop("s", 50),  # Default size if not provided
        **kwargs,
    )  # Pass remaining kwargs

    # Add identity line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    # Ensure the line covers the extended limits if points are outside initial view
    lims[0] = min(lims[0], np.min(error1), np.min(error2))
    lims[1] = max(lims[1], np.max(error1), np.max(error2))
    ax.plot(lims, lims, "k-", alpha=0.6, zorder=0, lw=1.5, label="Equal Error Line (y=x)")  # Make line slightly bolder

    ax.set_xlabel(f"{model1_name} {error_type}")
    ax.set_ylabel(f"{model2_name} {error_type}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    # Find the handle/label for the y=x line and remove it from color legend if necessary
    # Keep only handles/labels related to the hue + the y=x line if needed
    # Often seaborn handles this well, but manual adjustment might be needed
    # Example: Keep hue labels + the line label explicitly
    line_label = "Equal Error Line (y=x)"
    line_handle = [h for h, l in zip(handles, labels) if l == line_label]
    hue_handles = [h for h, l in zip(handles, labels) if l in hue_order]
    hue_labels = [l for l in labels if l in hue_order]

    # Place legend outside the plot for clarity if many points
    ax.legend(
        handles=hue_handles + line_handle,
        labels=hue_labels + [line_label],
        title="Comparison Result",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    sns.despine(ax=ax)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend if outside

    if save_path:
        # Ensure directory exists if path includes directories
        # import os
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_path.split(".")[-1])
        print(f"Plot saved to {save_path}")
    plt.show()
    return ax


def plot_error_scatter(
    label,
    prediction1,
    prediction2,
    model1_name="Model 1",
    model2_name="Model 2",
    use_absolute_error=True,
    title="Comparison of Prediction Errors",
    figsize=(6, 6),
    save_path=None,
    dpi=300,
    **kwargs,
):
    """
    Generates a scatter plot comparing the errors of two prediction models
    on the same data points.

    Args:
        label (array-like): Ground truth values.
        prediction1 (array-like): Predictions from the first model.
        prediction2 (array-like): Predictions from the second model.
        model1_name (str): Name for the first model (for axis label).
        model2_name (str): Name for the second model (for axis label).
        use_absolute_error (bool): If True, plots absolute errors, otherwise raw errors.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure. If None, not saved.
        dpi (int): Dots per inch for saving the figure.
        **kwargs: Additional keyword arguments passed to sns.scatterplot.
    """
    label = np.asarray(label)
    prediction1 = np.asarray(prediction1)
    prediction2 = np.asarray(prediction2)

    if use_absolute_error:
        error1 = np.abs(prediction1 - label)
        error2 = np.abs(prediction2 - label)
        error_type = "Absolute Error"
    else:
        error1 = prediction1 - label
        error2 = prediction2 - label
        error_type = "Error"

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(x=error1, y=error2, ax=ax, alpha=0.6, s=50, **kwargs)

    # Add identity line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0, label="y=x")  # Make line black and slightly transparent

    ax.set_xlabel(f"{model1_name} {error_type}")
    ax.set_ylabel(f"{model2_name} {error_type}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")  # Ensure aspect ratio is equal
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    sns.despine(ax=ax)  # Remove top and right spines
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_path.split(".")[-1])
        print(f"Plot saved to {save_path}")
    plt.show()
    return ax


def plot_error_distribution(
    label,
    prediction1,
    prediction2,
    model1_name="Model 1",
    model2_name="Model 2",
    plot_type="box",  # 'box' or 'violin'
    show_points=True,  # Overlay individual points
    title="Distribution of Absolute Prediction Errors",
    figsize=(6, 6),
    save_path=None,
    dpi=300,
    **kwargs,
):
    """
    Generates a box or violin plot comparing the distribution of absolute errors
    for two prediction models. Performs a paired t-test and annotates significance.

    Args:
        label (array-like): Ground truth values.
        prediction1 (array-like): Predictions from the first model.
        prediction2 (array-like): Predictions from the second model.
        model1_name (str): Name for the first model.
        model2_name (str): Name for the second model.
        plot_type (str): 'box' or 'violin'.
        show_points (bool): If True, overlays individual data points (jittered).
        title (str): Title for the plot.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure. If None, not saved.
        dpi (int): Dots per inch for saving the figure.
        **kwargs: Additional keyword arguments passed to sns.boxplot/violinplot.
    """
    label = np.asarray(label)
    prediction1 = np.asarray(prediction1)
    prediction2 = np.asarray(prediction2)

    abs_error1 = np.abs(prediction1 - label)
    abs_error2 = np.abs(prediction2 - label)

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(abs_error1, abs_error2)
    print(f"Paired t-test results: t-statistic = {t_stat:.3f}, p-value = {p_value:.3g}")

    # Prepare data for Seaborn (long format)
    df_errors = pd.DataFrame(
        {
            "Absolute Error": np.concatenate([abs_error1, abs_error2]),
            "Model": [model1_name] * len(abs_error1) + [model2_name] * len(abs_error2),
            "ID": np.concatenate([np.arange(len(abs_error1)), np.arange(len(abs_error2))]),  # For potential pairing lines later if needed
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    palette = kwargs.pop("palette", "Set2")  # Use a nice default palette

    if plot_type == "violin":
        sns.violinplot(x="Model", y="Absolute Error", data=df_errors, ax=ax, palette=palette, inner="quartile", **kwargs)
    else:  # Default to box plot
        sns.boxplot(
            x="Model", y="Absolute Error", data=df_errors, ax=ax, palette=palette, showcaps=True, showfliers=False, **kwargs
        )  # Hide outliers from box, show with stripplot

    if show_points:
        # Use stripplot for overlaying points - often better than swarmplot for larger N
        sns.stripplot(x="Model", y="Absolute Error", data=df_errors, ax=ax, color="black", alpha=0.3, jitter=True, size=3)

    ax.set_title(title)
    ax.set_ylabel("Absolute Error (|Prediction - Label|)")
    ax.set_xlabel("")  # Model names are on x-ticks

    # Add significance annotation
    y_max = df_errors["Absolute Error"].max()
    add_p_value_annotation(ax, 0, 1, y_max, p_value)  # 0 and 1 are positions of boxes

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_path.split(".")[-1])
        print(f"Plot saved to {save_path}")
    plt.show()
    return ax


def plot_difference_distribution(
    label,
    prediction1,
    prediction2,
    model1_name="Model 1",
    model2_name="Model 2",
    plot_type="hist",  # 'hist' or 'kde'
    title="Distribution of Error Differences",
    figsize=(7, 5),
    save_path=None,
    dpi=300,
    **kwargs,
):
    """
    Generates a histogram or KDE plot of the difference in absolute errors
    between two models (Error_Model1 - Error_Model2).

    Args:
        label (array-like): Ground truth values.
        prediction1 (array-like): Predictions from the first model.
        prediction2 (array-like): Predictions from the second model.
        model1_name (str): Name for the first model.
        model2_name (str): Name for the second model.
        plot_type (str): 'hist' for histogram, 'kde' for kernel density estimate.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure. If None, not saved.
        dpi (int): Dots per inch for saving the figure.
        **kwargs: Additional keyword arguments passed to sns.histplot/kdeplot.
    """
    label = np.asarray(label)
    prediction1 = np.asarray(prediction1)
    prediction2 = np.asarray(prediction2)

    abs_error1 = np.abs(prediction1 - label)
    abs_error2 = np.abs(prediction2 - label)
    error_diff = abs_error1 - abs_error2

    mean_diff = np.mean(error_diff)
    median_diff = np.median(error_diff)

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "kde":
        sns.kdeplot(error_diff, ax=ax, fill=True, **kwargs)
    else:  # Default to histogram
        sns.histplot(error_diff, ax=ax, kde=True, **kwargs)  # Add KDE line to histogram

    # Add vertical line at zero difference
    ax.axvline(0, color="red", linestyle="--", label="Zero Difference")
    # Add vertical line for mean difference
    ax.axvline(mean_diff, color="blue", linestyle=":", label=f"Mean Diff = {mean_diff:.3f}")

    ax.set_xlabel(f"Absolute Error Difference ({model1_name} - {model2_name})")
    ax.set_ylabel("Frequency" if plot_type == "hist" else "Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)  # Subtle horizontal grid

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_path.split(".")[-1])
        print(f"Plot saved to {save_path}")
    plt.show()
    return ax


# --- Example Data ---
# np.random.seed(42)
# n_samples = 100
# label = np.random.rand(n_samples) * 10
# prediction1 = label + np.random.normal(0, 1.0, n_samples)  # Model 1: Decent predictions with some noise
# prediction2 = label + np.random.normal(0, 1.5, n_samples)  # Model 2: Slightly worse predictions (more noise)

# label = np.load("/Users/fgtear/Downloads/mosi_label.npy")
# prediction1 = np.load("/Users/fgtear/Downloads/04211235_MOSI_audio_cmff_prediction.npy")
# prediction2 = np.load("/Users/fgtear/Downloads/04211208_MOSI_audio_last_prediction.npy")

label = np.load("/Users/fgtear/Downloads/mosi_label.npy")
prediction1 = np.load("/Users/fgtear/Downloads/04211047_MOSI_text_cmff_prediction.npy")
prediction2 = np.load("/Users/fgtear/Downloads/04211029_MOSI_text_last_prediction.npy")

# --- --- --- --- ---

# --- Generate Plots ---

# 1. Scatter plot of absolute errors
# plot_error_scatter_colored(
#     label,
#     prediction1,
#     prediction2,
#     model1_name="Regression CNN",
#     model2_name="Random Forest",
#     use_absolute_error=True,
#     title="Absolute Error Comparison (CNN vs RF)",
#     save_path="error_scatter.pdf",
# )  # Save as PDF (vector)

# 2. Distribution plot (Violin plot with points)
# plot_error_distribution(
#     label,
#     prediction1,
#     prediction2,
#     model1_name="Regression CNN",
#     model2_name="Random Forest",
#     plot_type="violin",
#     show_points=True,
#     title="Absolute Error Distribution (CNN vs RF)",
#     save_path="error_distribution_violin.png",
#     dpi=300,
# )  # Save as high-res PNG

# 3. Distribution plot (Box plot) - another common style
plot_error_distribution(
    label,
    prediction1,
    prediction2,
    model1_name="Multi-Layer Feature",
    model2_name="Last Layer Feature",
    plot_type="box",
    show_points=True,
    title="Absolute Error Distribution (Multi-Layer Feature vs Last Layer Feature)",
    save_path="error_distribution_box.png",
)  # Save as high-res PNG

# 4. Histogram of error differences
# plot_difference_distribution(
#     label,
#     prediction1,
#     prediction2,
#     model1_name="CNN",
#     model2_name="RF",
#     plot_type="hist",
#     bins=15,  # Control number of bins
#     title="Distribution of Absolute Error Differences (CNN - RF)",
#     save_path="error_difference_hist.pdf",
# )

# 5. KDE plot of error differences
# plot_difference_distribution(
#     label,
#     prediction1,
#     prediction2,
#     model1_name="CNN",
#     model2_name="RF",
#     plot_type="kde",
#     title="Density of Absolute Error Differences (CNN - RF)",
#     save_path="error_difference_kde.png",
#     dpi=300,
# )
