import os
import shutil
import pandas as pd
import torch
import torchaudio
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
import math


def calculate_wav2vec2_encoder_output_length(input_length: int) -> int:
    """
    根据 wav2vec 2.0 的 Feature Encoder 配置计算输出序列的长度。

    Args:
        input_length (int): 原始音频波形的长度 (L)。

    Returns:
        int: 经过所有卷积层后的输出特征序列的长度。
    """

    # wav2vec 2.0 Feature Encoder 的标准配置
    conv_kernels = [10, 3, 3, 3, 3, 2, 2]
    conv_strides = [5, 2, 2, 2, 2, 2, 2]
    output_length = float(input_length)

    for i, (kernel, stride) in enumerate(zip(conv_kernels, conv_strides)):
        output_length = math.floor((output_length - kernel) / stride) + 1

    return int(output_length)


def process_single_item(row, dataset, config):
    """处理单个数据项的函数，用于多线程处理"""

    dataset_root = f"datasets/{dataset}"

    file_name = f"{row['video_id']}/{row['clip_id']}"

    # 检查缓存文件是否已存在，避免重复处理
    cache_file_path = f"{dataset_root}/cache/{row['video_id']}/{row['clip_id']}.pt"
    if os.path.exists(cache_file_path):
        return f"Skipped (already exists): {file_name}"

    item = {}
    text = row["text"].lower()
    item["text"] = text

    if dataset in ["SIMS", "SIMSV2"]:
        item["label_M"] = row["label"]
        item["label_T"] = row["label_T"]
        item["label_A"] = row["label_A"]
        item["label_V"] = row["label_V"]
        item["annotation"] = row["annotation"]
    else:
        item["label_M"] = row["label"]
        item["label_T"] = row["label"]
        item["label_A"] = row["label"]
        item["label_V"] = row["label"]
        item["annotation"] = row["annotation"]

    # Process text
    with torch.no_grad():
        text_token = config.tokenizer(
            text,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=config.text_max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
    text_input_ids = text_token["input_ids"].squeeze(0)
    text_attention_mask = text_token["attention_mask"].squeeze(0)
    item["text_input_ids"] = text_input_ids
    item["text_attention_mask"] = text_attention_mask.to(torch.bool)

    # Process audio
    wav_path = dataset_root + "/wav/" + file_name + ".wav"
    wave, sample_rate = torchaudio.load(wav_path)
    wave = wave[:, : config.audio_max_length]  # Truncate
    wave = torch.mean(wave, dim=0, keepdim=False)  # Merge channels [T1]
    with torch.no_grad():
        audio_token = config.audio_processor(
            wave,
            sampling_rate=sample_rate,
            return_tensors="pt",
            # padding="longest",
            truncation=True,
            max_length=config.audio_max_length,
            return_attention_mask=True,
        )
    audio_input_values = audio_token.input_values.squeeze(0)
    audio_input_values = audio_input_values.to(torch.float16)
    audio_attention_mask = audio_token.attention_mask.squeeze(0).to(torch.bool)
    audio_output_length = calculate_wav2vec2_encoder_output_length(audio_attention_mask.size(0))
    audio_output_attention_mask = torch.zeros((audio_output_length), dtype=torch.bool)

    item["audio_input_values"] = audio_input_values
    item["audio_attention_mask"] = audio_attention_mask
    item["audio_output_attention_mask"] = audio_output_attention_mask

    # Process video
    video_tensor = torch.load(f"{dataset_root}/face/{file_name}.pt")
    if video_tensor.shape == torch.Size([0]):
        video_tensor = torch.zeros((32, 224, 224, 3), dtype=torch.float16)  # 如果没有人脸数据，填充空数据
    video_tensor = config.image_processor(list(video_tensor), return_tensors="pt")["pixel_values"].squeeze(0)
    item["video_tensor"] = video_tensor

    # Save processed item
    os.makedirs(f"{dataset_root}/cache/{row['video_id']}", exist_ok=True)
    torch.save(item, cache_file_path)

    return f"Processed: {file_name}"


def cache_dataset(dataset: Literal["MOSI", "MOSEI", "SIMS", "SIMSV2"], config, num_threads=8):
    if os.path.exists(f"{dataset}/cache"):
        print(f"Cache already exists for dataset {dataset}. Removing existing cache...")
        shutil.rmtree(f"{dataset}/cache")  # 删除现有缓存目录
    dataset_root = f"datasets/{dataset}"

    print("Loading models and processors...")
    # tokenizer = BertTokenizer.from_pretrained(config.text_extractor, clean_up_tokenization_spaces=True)
    # tokenizer = AutoTokenizer.from_pretrained(config.text_extractor, clean_up_tokenization_spaces=True)
    # audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_extractor)
    # image_processor = VivitImageProcessor.from_pretrained(config.video_extractor)

    print(f"Reading dataset from {dataset_root}/label.csv...")
    if dataset == "SIMS":
        df = pd.read_csv(
            dataset_root + "/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(dataset_root + "/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"Processing {len(df)} items with {num_threads} threads...")

    # 准备参数
    args_list = []
    for index, row in df.iterrows():
        args_list.append(row)

    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_item, row, dataset, config) for row in args_list]
        # 使用tqdm显示进度
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            # 可以选择性地打印结果信息
            if "Error" in result:
                print(f"  {result}")

    print("Dataset caching completed!")


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hfff.config import Config

    config = Config()

    # cache_dataset("MOSI", config, num_threads=8)
    cache_dataset("MOSEI", config, num_threads=8)
    # cache_dataset("SIMS", config, num_threads=8)
