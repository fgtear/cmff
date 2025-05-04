import torch
from torch.nn import functional as F

data_vis = torch.load("data_vis.pt")
attn_output_weights = data_vis["attn_output_weights"]
texts = data_vis["texts"]
audio_paths = data_vis["audio_paths"]
labels = data_vis["labels"]
predicts = data_vis["predicts"]

print("attn_output_weights", attn_output_weights.shape)

attn_0 = attn_output_weights[:, 0]
attn_1 = attn_output_weights[:, 1]

print(attn_0.shape)
print(attn_0.sum(1))
print(attn_0.sum(2))
print(attn_0.sum())

bs = attn_output_weights.size(0)
token_length = attn_output_weights.size(-1)
attn = torch.zeros(bs, token_length, token_length)
for i in range(token_length):
    for j in range(token_length):
        attn[:, i, j] = torch.sum(attn_0[:, i, :] * attn_1[:, :, j], dim=1)

print("".center(50, "-"))
print(attn.shape)
print(attn.sum(-1))
a = attn.sum(dim=1).softmax(dim=-1)

print("a", a.shape)

text_ratio = a[:, :12].sum(dim=1)
print("text_ratio", text_ratio)
for i, j in enumerate(text_ratio):
    # if j < 0.5:
    # print(i, j, texts[i], audio_paths[i], labels[i], predicts[i])
    mae = F.l1_loss(predicts[i], labels[i])
    if mae < 0.15:
        print("".center(50, "-"))
        print(
            f"text_ratio: {j} \n text: {texts[i]} \n audio_path: {audio_paths[i]} \n label: {labels[i]} \n predict: {predicts[i]} \n mae: {mae}"
        )


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 data 是你的数据列表
data = text_ratio.numpy()

# 设置图形的风格
sns.set(style="whitegrid")

# 创建直方图
plt.figure(figsize=(10, 6))


sns.histplot(data, bins=100, kde=True, stat="density", color="blue", edgecolor="black", alpha=0.7)

# 设置坐标轴标签
# plt.xlabel(fontsize=15)
plt.ylabel("Number of samples", fontsize=15)
plt.xlabel("The weight of text modality", fontsize=15)
# plt.title('数据概率分布fg的直方图')
plt.tick_params(axis="y", labelsize=15)
plt.tick_params(axis="x", labelsize=15)


# 显示图形
plt.tight_layout()
plt.savefig("../paper/fig_text_ratio.png", dpi=300)
plt.show()
