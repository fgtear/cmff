import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer
import torchaudio
import matplotlib.pyplot as plt


import sys

sys.path.append("..")


def analyse_mosi(dataset="MOSI", mode="test"):
    print(f"Analyse {dataset} {mode}".center(50, "-"))
    if dataset == "MOSI":
        csv_path = "../datasets/MOSI/label.csv"
        audio_directory = "../datasets/MOSI/wav"
    elif dataset == "MOSEI":
        csv_path = "../datasets/MOSEI/label.csv"
        audio_directory = "../datasets/MOSEI/wav"
    else:
        raise ValueError("dataset should be MOSI or MOSEI")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", clean_up_tokenization_spaces=True)

    df = pd.read_csv(csv_path)
    df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()
    # df["text"] = df["text"].str[0] + df["text"].str[1::].apply(lambda x: x.lower())  # Capitalize the first letter
    df["text"] = df["text"].apply(lambda x: x.lower())  # lowercase all

    # 画出 df["label"] distribution
    plt.hist(df["label"], bins=100)
    plt.show()

    label_overview = []
    for i in df["label"]:
        if i not in label_overview:
            label_overview.append(i)
    print("label_overview", sorted(label_overview))

    text_lenghts = df["text"].str.split().apply(len).sort_values().to_list()
    print("text_lenghts", text_lenghts)
    plt.hist(text_lenghts, bins=100)
    plt.show()

    audio_file_paths = []
    audio_lengths = []
    for i in range(0, len(df)):
        file_name = str(df["video_id"][i]) + "/" + str(df["clip_id"][i]) + ".wav"
        file_path = audio_directory + "/" + file_name
        audio_file_paths.append(file_path)
        audio, _ = torchaudio.load(file_path)
        audio_lengths.append(audio.size(1))
    audio_lengths.sort()
    print("audio_lengths", audio_lengths)
    plt.hist(audio_lengths, bins=100)
    plt.show()

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
    analyse_mosi(dataset="MOSI", mode="train")
