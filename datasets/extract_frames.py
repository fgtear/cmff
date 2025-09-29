# 导入必要的库
import torch
import cv2
import numpy as np
import os
import pandas as pd
import tqdm
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from typing import Tuple
import torchvision
import shutil


def extract_uniform_frames(video_path: str, num_frames: int, output_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    从视频中均匀采样指定数量的帧，并将其调整为指定尺寸。

    Args:
        video_path (str): 视频文件的路径。
        num_frames (int): 要提取的总帧数。
        output_size (Tuple[int, int]): 输出图片的尺寸 (height, width)。

    Returns:
        torch.Tensor: 一个张量，包含了提取并调整尺寸后的帧。
                      形状为 (num_frames, C, H, W)，其中 C=3 (RGB),
                      H=output_size[0], W=output_size[1]。
                      数据类型为 torch.uint8。

    Raises:
        FileNotFoundError: 如果视频文件不存在。
        ValueError: 如果视频为空或无法读取任何帧。
        RuntimeError: 如果视频解码失败。
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    try:
        # 1. 读取视频帧。read_video 返回 (视频帧, 音频帧, 元数据)
        # 视频帧的形状是 (T, H, W, C)，T是总帧数，C是通道数
        # 值的范围是 0-255, 类型是 uint8
        video_frames, _, info = torchvision.io.read_video(str(video_path), pts_unit="sec")
        print("info", info)
        # video_frames = video_frames.permute(0, 3, 1, 2)

    except RuntimeError as e:
        print(f"读取或解码视频失败: {video_path}")
        raise e

    total_frames = len(video_frames)
    print("total_frames", total_frames)
    if total_frames == 0:
        raise ValueError(f"视频 '{video_path}' 为空或无法读取帧。")

    # 2. 生成均匀采样的帧索引
    # torch.linspace 在 0 和 total_frames-1 之间生成 num_frames 个等间距的点
    # 例如，如果视频有100帧，要取10帧，则会生成类似 0, 11, 22, ..., 99 的索引
    indices = torch.linspace(0, total_frames - 1, num_frames).long()

    # 3. 根据索引选择帧
    sampled_frames = video_frames[indices]  # 形状: (num_frames, H, W, C)

    # 4. 调整维度以适应 torchvision 的变换
    # torchvision 的变换通常期望输入是 (C, H, W) 或 (N, C, H, W)
    # 我们需要将 (N, H, W, C) 转换为 (N, C, H, W)
    sampled_frames = sampled_frames.permute(0, 3, 1, 2)

    # 5. 创建并应用 Resize 变换
    if output_size is not None:
        resize_transform = transforms.Resize(output_size, antialias=True)
        sampled_frames = resize_transform(sampled_frames)

    return sampled_frames


def process_dataset(dataset, num_frames, output_size, num_workers):
    """
    Args:
        dataset (str): 数据集路径
        num_frames (int): 每个视频提取的帧数
        image_size (int): 图像尺寸
        num_workers (int): 并行处理线程数，默认4
    """
    if os.path.exists(f"{dataset}/frames"):
        print(f"frames already exists for dataset {dataset}. Removing existing frames...")
        shutil.rmtree(f"{dataset}/frame")  # 删除现有缓存目录

    def process_single_video(row):
        """
        处理单个视频的函数，用于多线程
        Args:
            row (pd.Series): 包含视频信息的行
        """
        try:
            video_id = row["video_id"]
            clip_id = row["clip_id"]

            # 将clip_id转换为4位零填充格式
            # clip_id_str = f"{int(clip_id):04d}"
            # video_filename = f"{clip_id_str}.mp4"
            video_path = f"{dataset}/Raw/{video_id}/{clip_id}.mp4"

            # 检查是否存在修复版本的视频
            edited_video_path = video_path.replace(".mp4", "-edited.mp4")
            if os.path.exists(edited_video_path):
                video_path = edited_video_path

            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                return False

            # 预处理视频
            video_tensor = extract_uniform_frames(video_path, num_frames, output_size=output_size)
            tensor_path = f"{dataset}/frame/{video_id}/{clip_id}.pt"
            os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
            torch.save(video_tensor, tensor_path)
            # print(f"成功处理: {video_path} -> {tensor_path}")
            return True
        except Exception as e:
            print(f"处理视频时出错 {row.get('video_id', 'unknown')}/{row.get('clip_id', 'unknown')}: {str(e)}")
            return False

    if dataset == "SIMS":
        df = pd.read_csv(
            f"{dataset}/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(f"{dataset}/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 使用多线程处理
    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_video, row) for index, row in df.iterrows()]

        # 使用tqdm显示进度
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result:
                success_count += 1
            else:
                failure_count += 1

    print("\n处理完成:")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"总计: {len(df)}")


if __name__ == "__main__":
    frames = extract_uniform_frames("datasets/MOSEI/Raw/_0efYOjQYRc/3.mp4", num_frames=32, output_size=(224, 224))
    # frames = extract_uniform_frames("MOSI/Raw/_dI--eQ6qVU/11.mp4", num_frames=32, output_size=None)
    # print(frames.shape)

    # process_dataset(dataset="MOSI", num_frames=32, output_size=None, num_workers=1)
