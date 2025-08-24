import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set a professional style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_layer_ablation(layer_metrics, optimal_layer=None, save_path=None, dpi=300):
    """
    Generates a line plot visualizing model performance vs. number of layers,
    using dual y-axes for different metric scales.

    Args:
        layer_metrics (dict): A dictionary where keys are layer numbers (int)
                              and values are dictionaries of metrics (float).
        optimal_layer (int, optional): Layer number to highlight as optimal.
        save_path (str, optional): Path to save the figure. If None, not saved.
        dpi (int): Dots per inch for saving the figure.
    """
    df = pd.DataFrame.from_dict(layer_metrics, orient="index")
    df.index.name = "Number of Layers"
    df.rename(
        columns={
            "metrics/MAE": "MAE",
            "metrics/Corr": "Correlation",
            "metrics/Mult_acc_7": "Acc7",
            "metrics/Non0_acc_2": "Acc2 (Non0)",
            "metrics/Non0_F1_score": "F1 (Non0)",
        },
        inplace=True,
    )
    df.sort_index(inplace=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Primary Y-axis (Left): MAE ---
    color1 = "tab:red"
    ax1.set_xlabel("Number of CMFF Layers", fontsize=1)
    ax1.set_ylabel("MAE (Lower is Better)", color=color1, fontsize=12)
    line1 = ax1.plot(df.index, df["MAE"], color=color1, marker="o", linestyle="-", label="MAE")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(df.index)

    # --- Secondary Y-axis (Right): Correlation, Accuracies, F1 ---
    ax2 = ax1.twinx()
    color_corr = "tab:blue"
    color_acc7 = "tab:green"
    color_f1 = "tab:purple"
    ax2.set_ylabel("Metric Value (Higher is Better)", fontsize=12)

    line2 = ax2.plot(df.index, df["Correlation"], color=color_corr, marker="s", linestyle="--", label="Correlation")
    line3 = ax2.plot(df.index, df["Acc7"], color=color_acc7, marker="^", linestyle=":", label="Acc7")
    line5 = ax2.plot(df.index, df["F1 (Non0)"], color=color_f1, marker="D", linestyle="-.", label="F1 (Non0)")
    ax2.tick_params(axis="y")

    # Highlight the optimal layer if specified
    vline = None  # Initialize vline to None
    if optimal_layer is not None:
        # ***** FIX: Use ax1.axvline *****
        vline = ax1.axvline(x=optimal_layer, color="grey", linestyle="--", linewidth=1.5, label=f"Optimal ({optimal_layer} Layers)")

    # --- Legend ---
    lines = line1 + line2 + line3 + line5
    labels = [l.get_label() for l in lines]

    # Add optimal layer line to legend ONLY if it was created
    if vline is not None:
        lines.append(vline)
        labels.append(vline.get_label())

    # Place legend outside the plot using fig.legend for better handling of dual axes
    fig.legend(lines, labels, loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=12)  # Adjust anchor as needed

    # plt.title("Impact of Number of CMFF Layers on Performance (CMU-MOSI)", pad=20)
    # Adjust layout to make room for legend (may need tweaking)
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space on the right

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


# --- Prepare your data ---
layer_metrics_data = {
    1: {
        "metrics/Has0_acc_2": 0.8309037685394287,
        "metrics/Has0_F1_score": 0.8299691677093506,
        "metrics/Non0_acc_2": 0.855182945728302,
        "metrics/Non0_F1_score": 0.854997992515564,
        "metrics/Mult_acc_7": 0.495626837015152,
        "metrics/MAE": 0.6401932239532471,
        "metrics/Corr": 0.8334453701972961,
    },
    2: {
        "metrics/Has0_acc_2": 0.8498542308807373,
        "metrics/Has0_F1_score": 0.8487221598625183,
        "metrics/Non0_acc_2": 0.8765243887901306,
        "metrics/Non0_F1_score": 0.8761568069458008,
        "metrics/Mult_acc_7": 0.495626837015152,
        "metrics/MAE": 0.6097502708435059,
        "metrics/Corr": 0.8490811586380005,
    },
    3: {
        "metrics/Has0_acc_2": 0.8731778264045715,
        "metrics/Has0_F1_score": 0.8730268478393555,
        "metrics/Non0_acc_2": 0.8917682766914368,
        "metrics/Non0_F1_score": 0.8919830918312073,
        "metrics/Mult_acc_7": 0.532069981098175,
        "metrics/MAE": 0.5697766542434692,
        "metrics/Corr": 0.8756025433540344,
    },
    4: {
        "metrics/Has0_acc_2": 0.8498542308807373,
        "metrics/Has0_F1_score": 0.8497815728187561,
        "metrics/Non0_acc_2": 0.8689024448394775,
        "metrics/Non0_F1_score": 0.8692846298217773,
        "metrics/Mult_acc_7": 0.4927113652229309,
        "metrics/MAE": 0.6021603941917419,
        "metrics/Corr": 0.857864499092102,
    },
    5: {
        "metrics/Has0_acc_2": 0.8629737496376038,
        "metrics/Has0_F1_score": 0.8624938130378723,
        "metrics/Non0_acc_2": 0.8871951103210449,
        "metrics/Non0_F1_score": 0.8872487545013428,
        "metrics/Mult_acc_7": 0.4620991349220276,
        "metrics/MAE": 0.6299121379852295,
        "metrics/Corr": 0.8639536499977112,
    },
    6: {
        "metrics/Has0_acc_2": 0.8542274236679077,
        "metrics/Has0_F1_score": 0.8539696931838989,
        "metrics/Non0_acc_2": 0.8765243887901306,
        "metrics/Non0_F1_score": 0.876769483089447,
        "metrics/Mult_acc_7": 0.49416908621788025,
        "metrics/MAE": 0.6187682747840881,
        "metrics/Corr": 0.8494093418121338,
    },
}

# --- Call the function ---
plot_layer_ablation(layer_metrics_data, optimal_layer=3, save_path="cmff/analysis/Figure_cmff_layer_ablation.png")
