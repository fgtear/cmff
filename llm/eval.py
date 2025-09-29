import sys
import os
import pandas as pd
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from utils.metricsTop import MetricsTop


def eval(label_omni_path, dataset, mode=None):
    print("".center(25, "-"), dataset, "".center(25, "-"))
    metrics_evaluator = MetricsTop(train_mode="regression")
    get_metrics = metrics_evaluator.getMetics(dataset)

    df = pd.read_csv(label_omni_path)
    if mode:
        df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()

    label = torch.tensor(df["label"].values, dtype=torch.float16)
    predict_omni = torch.tensor(df["mm_sentiment_score"].values.tolist(), dtype=torch.float16)

    # 统计mm_sentiment_score有多少为nan
    nan_count = torch.sum(torch.isnan(predict_omni)).item()
    print("Number of NaN values in mm_sentiment_score:", nan_count)

    # 如果 predict_omni 中的值为nan，则使用label中对应的值替代
    # nan_mask = torch.isnan(predict_omni)
    # predict_omni[nan_mask] = label[nan_mask]

    # 如果 predict_omni 中的值为nan，则忽略与label对应的值计算
    valid_mask = ~torch.isnan(predict_omni)
    predict_omni = predict_omni[valid_mask]
    label = label[valid_mask]

    metrics = get_metrics(predict_omni, label)
    print(metrics)
    print("valid label", label.shape)
    print("valid predict_omni", predict_omni.shape)


if __name__ == "__main__":
    eval("datasets/MOSI/qwen2.5-omni-7b-1.csv", "MOSI", mode="test")
    eval("datasets/MOSI/qwen2.5-omni-7b-2.csv", "MOSI", mode="test")
    eval("datasets/MOSI/qwen2.5-omni-7b-3.csv", "MOSI", mode="test")
    eval("datasets/MOSI/qwen2.5-omni-7b.csv", "MOSI", mode="test")

    # eval("datasets/MOSEI/qwen2.5-omni-7b.csv", "MOSEI", mode="test")
