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
    ax.text((x1 + x2) * 0.5, line_y + bracket_height, p_text, ha="center", va="bottom", color="black", fontsize=13.5)

    # Adjust ylim slightly to make space
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], (line_y + bracket_height) * 1.05)  # Ensure space above annotation


sns.set_theme(style="ticks", context="paper")  # 'paper', 'notebook', 'talk', 'poster'
plt.rc("font", size=33)


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
    np.random.seed(0)  # Set seed for reproducible jitter
    label = np.asarray(label)
    prediction1 = np.asarray(prediction1)
    prediction2 = np.asarray(prediction2)

    abs_error1 = np.abs(prediction1 - label)
    abs_error2 = np.abs(prediction2 - label)

    # --- MODIFICATION START ---
    t_stat, p_value_two_sided = stats.ttest_rel(abs_error1, abs_error2, nan_policy="omit")  # Added nan_policy for robustness

    # Calculate the one-sided p-value (testing if error1 < error2)
    if t_stat < 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        # If t_stat >= 0, error1's sample mean is not less than error2's.
        # The p-value for the hypothesis error1 < error2 is large.
        p_value_one_sided = 1.0 - p_value_two_sided / 2

    print(
        (
            f"Paired t-test results: t-statistic = {t_stat:.3f}, "
            f"p-value (one-sided, H1: {model1_name} error < {model2_name} error) = {p_value_one_sided:.3g}"
        )
    )

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

    # ax.set_title(title)
    # ax.set_ylabel("Absolute Error (|Prediction - Label|)", fontsize=13.5)
    ax.set_ylabel("Absolute Error Value", fontsize=13.5)
    ax.set_xlabel("", fontsize=13.5)  # Model names are on x-ticks
    ax.tick_params(axis="x", labelsize=13.5)
    ax.tick_params(axis="y", labelsize=13.5)

    # Add significance annotation
    y_max = df_errors["Absolute Error"].max()
    add_p_value_annotation(ax, 0, 1, y_max, p_value_one_sided)  # 0 and 1 are positions of boxes

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_path.split(".")[-1])
        print(f"Plot saved to {save_path}")
    plt.show()
    return ax


if __name__ == "__main__":
    # --- Example Data ---
    # np.random.seed(42)
    # n_samples = 100
    # label = np.random.rand(n_samples) * 10
    # prediction1 = label + np.random.normal(0, 1.0, n_samples)  # Model 1: Decent predictions with some noise
    # prediction2 = label + np.random.normal(0, 1.5, n_samples)  # Model 2: Slightly worse predictions (more noise)

    label = np.load("cmff/results/mosi_label.npy")
    prediction1 = np.load("cmff/results/04211235_MOSI_audio_cmff_prediction.npy")
    prediction2 = np.load("cmff/results/04211208_MOSI_audio_last_prediction.npy")
    plot_error_distribution(
        label,
        prediction1,
        prediction2,
        model1_name="Multi-Layer Feature",
        model2_name="Last Layer Feature",
        plot_type="box",
        show_points=True,
        # title="Absolute Error Distribution (Multi-Layer Feature vs Last Layer Feature)",
        save_path="cmff/analysis/Figure_audio_error_distribution_box.png",
    )

    label = np.load("cmff/results/mosi_label.npy")
    prediction1 = np.load("cmff/results/04211047_MOSI_text_cmff_prediction.npy")
    prediction2 = np.load("cmff/results/04211029_MOSI_text_last_prediction.npy")
    plot_error_distribution(
        label,
        prediction1,
        prediction2,
        model1_name="Multi-Layer Feature",
        model2_name="Last Layer Feature",
        plot_type="box",
        show_points=True,
        # title="Absolute Error Distribution (Multi-Layer Feature vs Last Layer Feature)",
        save_path="cmff/analysis/Figure_text_error_distribution_box.png",
    )
