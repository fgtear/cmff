import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math


def visualize_attention_weights(npy_file_path, num_text_layers=None, sample_index=None, figsize_scale=5):
    """
    Visualizes attention weights from a CMFF model saved in a .npy file.

    The .npy file is expected to contain an array with shape:
    (num_test_samples, num_cmff_layers, seq_len, seq_len)
    where seq_len corresponds to the concatenated sequence:
    [CLS_token, Text_Layer_1, ..., Text_Layer_Lt, Audio_Layer_1, ..., Audio_Layer_La].

    Args:
        npy_file_path (str): Path to the .npy file containing attention weights.
                             Example: "analysis/attn_output_weights.npy"
        num_text_layers (int, optional): The exact number of text feature layers (Lt)
                                         used in the concatenated sequence. If None,
                                         it attempts to infer Lt and La assuming they
                                         might be equal or based on standard model outputs.
                                         Providing this value is recommended for accuracy.
                                         Defaults to None.
        sample_index (int, optional): If provided, visualizes weights for this
                                      specific sample index instead of the average
                                      across all samples. Defaults to None (average).
        figsize_scale (int): Scaling factor for the figure size. Default is 5.

    Returns:
        matplotlib.figure.Figure or None: The figure object containing the plots,
                                          or None if an error occurs.
    """
    # --- 1. Load Data ---
    if not os.path.exists(npy_file_path):
        print(f"Error: File not found at '{npy_file_path}'")
        return None
    try:
        attn_weights = np.load(npy_file_path)
        print(f"Loaded attention weights with shape: {attn_weights.shape}")
    except Exception as e:
        print(f"Error loading {npy_file_path}: {e}")
        return None

    if attn_weights.ndim != 4:
        print(f"Error: Expected 4 dimensions (samples, layers, seq_len, seq_len), but got {attn_weights.ndim} dimensions.")
        return None

    num_samples, num_cmff_layers, seq_len, seq_len_check = attn_weights.shape
    if seq_len != seq_len_check:
        print(f"Error: Expected last two dimensions to be equal (seq_len), but got {seq_len} and {seq_len_check}.")
        return None

    # --- 2. Select Data (Average or Specific Sample) ---
    if sample_index is not None:
        if 0 <= sample_index < num_samples:
            plot_weights = attn_weights[sample_index]
            plot_title_prefix = f"Sample {sample_index} Attention -"
            print(f"Visualizing attention for sample index: {sample_index}")
        else:
            print(f"Error: sample_index {sample_index} is out of bounds (0 to {num_samples - 1}). Visualizing average instead.")
            plot_weights = np.mean(attn_weights, axis=0)
            plot_title_prefix = "Average Attention -"
            print("Visualizing average attention across all samples.")
    else:
        plot_weights = np.mean(attn_weights, axis=0)
        plot_title_prefix = "Average Attention -"
        print("Visualizing average attention across all samples.")
    # Shape of plot_weights: (num_cmff_layers, seq_len, seq_len)

    # --- 3. Determine Sequence Labels ---
    # The sequence is [CLS, Text_Layer_1...Lt, Audio_Layer_1...La]
    # seq_len = 1 + Lt + La
    inferred_lt = 0
    inferred_la = 0
    if num_text_layers is not None:
        inferred_lt = num_text_layers
        inferred_la = seq_len - 1 - inferred_lt
        if inferred_la < 0:
            print(
                f"Warning: Provided num_text_layers ({num_text_layers}) implies "
                f"a negative number of audio layers ({inferred_la}) for seq_len={seq_len}. "
                f"Check your input or model structure."
            )
            inferred_la = max(0, seq_len - 1 - inferred_lt)  # Adjust defensively
    else:
        # Try to infer Lt and La if not provided
        # Common case: RoBERTa/BERT-base might have 13 hidden states (Emb + 12 layers)
        # Wav2Vec2-base might have 13 hidden states (Emb + 12 layers)
        possible_lt = 13
        if seq_len - 1 - possible_lt > 0:  # Check if this Lt leaves room for audio
            inferred_lt = possible_lt
            inferred_la = seq_len - 1 - inferred_lt
            print(f"Inferred Lt={inferred_lt} (e.g., BERT/RoBERTa-base) -> La={inferred_la}")
        else:  # Fallback: assume equal or just divide remaining
            inferred_lt = (seq_len - 1) // 2
            inferred_la = seq_len - 1 - inferred_lt
            print(f"Could not confidently infer Lt. Assuming Lt={inferred_lt}, La={inferred_la}")

    labels = ["CLS"]
    labels.extend([f"T{i + 1}" for i in range(inferred_lt)])
    labels.extend([f"A{i + 1}" for i in range(inferred_la)])

    # Verify label count matches seq_len
    if len(labels) != seq_len:
        print(f"Warning: Number of generated labels ({len(labels)}) does not match sequence length ({seq_len}). Using generic labels.")
        labels = [f"Seq_{i}" for i in range(seq_len)]

    # --- 4. Plot Heatmaps ---
    cols = int(math.ceil(math.sqrt(num_cmff_layers)))
    rows = int(math.ceil(num_cmff_layers / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale), squeeze=False)
    axes = axes.flatten()  # Flatten to easily iterate

    for i in range(num_cmff_layers):
        ax = axes[i]
        sns.heatmap(
            plot_weights[i],
            ax=ax,
            cmap="viridis",
            cbar=True,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.7},
        )  # Adjust color bar size if needed
        ax.set_title(f"{plot_title_prefix} CMFF Layer {i + 1}")
        ax.set_xlabel("Key Positions (Memory)")
        ax.set_ylabel("Query Positions")
        # Adjust tick label appearance
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=max(4, 10 - int(0.2 * seq_len)))
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=max(4, 10 - int(0.2 * seq_len)))

    # Hide unused subplots
    for j in range(num_cmff_layers, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Attention Weights Visualization ({os.path.basename(npy_file_path)})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap
    plt.show()

    return fig


file_to_visualize = "attn_output_weights.npy"

data = np.load(file_to_visualize)
print(11111)
print(data.shape)
# Visualize the average attention
print("\n--- Visualizing Average Attention ---")
fig_avg = visualize_attention_weights(file_to_visualize, num_text_layers=25)

# Visualize attention for a specific sample (e.g., sample 0)
print("\n--- Visualizing Attention for Sample 0 ---")
fig_sample0 = visualize_attention_weights(file_to_visualize, num_text_layers=25, sample_index=0)
