import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer
import torchaudio
import matplotlib.pyplot as plt
import torch

# 将项目根目录添加到sys.path
import sys

sys.path.append("../")
from config import Config

config = Config()


def analyse_mosi(dataset="MOSI", mode="test"):
    print(f"Analyse {dataset} {mode}".center(50, "-"))
    if dataset == "MOSI":
        csv_path = "../data/MOSI/label.csv"
        audio_directory = "data/MOSI/wav"
    elif dataset == "MOSEI":
        csv_path = "../data/MOSEI/label.csv"
        audio_directory = "../data/MOSEI/wav"
    else:
        raise ValueError("dataset should be MOSI or MOSEI")
    tokenizer = AutoTokenizer.from_pretrained(config.text_extractor, cache_dir=config.cache_dir, output_hidden_states=True)

    df = pd.read_csv(csv_path)
    print(df.keys())
    df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()
    # df["text"] = df["text"].str[0] + df["text"].str[1::].apply(lambda x: x.lower())  # Capitalize the first letter
    df["text"] = df["text"].apply(lambda x: x.lower())  # lowercase all

    # 画出 df["label"] distribution
    # plt.hist(df["label"], bins=100)
    # plt.show()

    discrete_label_count = {}
    discrete_label = torch.tensor(df["label"].tolist())
    discrete_label = torch.clamp(discrete_label, min=-3.0, max=3.0)
    discrete_label = torch.round(discrete_label) + 3
    for i in discrete_label:
        if i.item() not in discrete_label_count:
            discrete_label_count[i.item()] = 1
        else:
            discrete_label_count[i.item()] += 1
    print("discrete_label", discrete_label.shape)
    # discrete_label 按key排序
    discrete_label_count = dict(sorted(discrete_label_count.items()))
    print("discrete_label_count", discrete_label_count)
    # discrete_label_count 转为比例
    discrete_label_count = {k: v / sum(discrete_label_count.values()) for k, v in discrete_label_count.items()}
    print("discrete_label_count (proportions)", discrete_label_count)

    label_overview = []
    for i in df["label"]:
        if i not in label_overview:
            label_overview.append(i)
    # print("label_overview", sorted(label_overview))

    # text_lenghts = df["text"].str.split().apply(len).sort_values().to_list()
    # print("text_lenghts", text_lenghts)
    # plt.hist(text_lenghts, bins=100)
    # plt.show()

    # 将df["text"]按空格分割，然后按照长度排序
    # df = df.assign(text_length=df["text"].str.split().apply(len))
    # df = df.sort_values(by=["text_length"])

    # 将df["text"]进行tokenize到文字而不是数字，并保存到df["text_token"]
    df["text_token1"] = df["text"].apply(lambda x: tokenizer.tokenize(x))
    df["text_token2"] = df["text"].apply(
        lambda x: tokenizer(x, padding="max_length", truncation=True, add_special_tokens=True)["input_ids"]
    )

    df.to_csv(csv_path.split(".")[0] + f"_sorted_{mode}.csv", index=False)


if __name__ == "__main__":
    # analyse_mosi(dataset="MOSI", mode="train")
    # analyse_mosi(dataset="MOSI", mode="valid")
    # analyse_mosi(dataset="MOSI", mode="test")

    analyse_mosi(dataset="MOSEI", mode="train")
    analyse_mosi(dataset="MOSEI", mode="valid")
    analyse_mosi(dataset="MOSEI", mode="test")
