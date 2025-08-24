import cv2
import pickle
import os
import threading
import subprocess
import concurrent.futures
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import tqdm
import contextlib
import torch

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.utils.face_align import norm_crop
from utils import get_video_paths

# os.environ['HTTP_PROXY'] = "socks5://127.0.0.1:1080"
# os.environ['HTTPS_PROXY'] = "socks5://127.0.0.1:1080"


def resize_face(face, size, maintain_aspect_ratio=True):
    """
    将人脸图像缩放到指定尺寸
    Args:
        face: 输入的人脸图像 (np.ndarray)
        size: 目标尺寸 (height, width)
        maintain_aspect_ratio: 是否保持长宽比，默认True
    Returns:
        np.ndarray: 缩放后的人脸图像
    """
    if not isinstance(face, np.ndarray) or face.size == 0:
        return face

    target_h, target_w = size

    if maintain_aspect_ratio:
        # 保持长宽比的缩放
        h, w = face.shape[:2]

        # 计算缩放比例，选择较小的比例以确保图像完全包含在目标尺寸内
        scale = min(target_h / h, target_w / w)

        # 计算缩放后的尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 缩放图像
        resized_img = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建目标尺寸的画布（用黑色填充）
        if len(face.shape) == 3:
            canvas = np.zeros((target_h, target_w, face.shape[2]), dtype=face.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=face.dtype)

        # 计算居中放置的位置
        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2

        # 将缩放后的图像放置在画布中心
        canvas[start_h : start_h + new_h, start_w : start_w + new_w] = resized_img
        resized_face = canvas
    else:
        # 直接缩放（会改变长宽比）
        resized_face = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return resized_face


def extract_face(app, video_path, target_size, num_samples=32, det_score_thresh=0.55):
    """
    从视频中按指定采样数提取人脸。

    Args:
        app: InsightFace实例
        video_path: 视频文件路径
        target_size (tuple): 目标人脸尺寸 (height, width)
        num_samples (int): 要提取的人脸样本数量
        det_score_thresh: 检测分数阈值
    Returns:
        List[np.array]: 提取并处理后的人脸图像列表
    """
    faces_croped = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:\n", video_path)
        return faces_croped

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return faces_croped

    interval = total_frames // num_samples
    if interval == 0:
        interval = 1

    for i in range(num_samples):
        start_frame_idx = i * interval

        # 尝试在 [start_frame_idx, start_frame_idx + interval -1] 区间内寻找人脸
        found_face_in_interval = False
        for j in range(interval):
            current_frame_idx = start_frame_idx + j
            if current_frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)
            if len(faces) > 0:
                # 按检测分数排序，取最高分的人脸
                faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
                face = faces[0]

                if face.det_score >= det_score_thresh:
                    # 裁剪人脸
                    x1, y1, x2, y2 = [int(coord) for coord in face["bbox"]]
                    h, w = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    face_croped = frame[y1:y2, x1:x2]

                    # 缩放并添加到列表
                    face_resized = resize_face(face_croped, size=target_size)
                    faces_croped.append(face_resized)
                    found_face_in_interval = True
                    break  # 找到人脸，跳出内部循环，处理下一个采样间隔

        # 如果在整个interval都没找到，则使用第一帧
        if not found_face_in_interval:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            ret, frame = cap.read()
            if ret:
                # 即使找不到人脸，也处理第一帧（可能为空或低质量）
                # 这里可以决定是添加一个空白图像还是做其他处理
                # 为了保持数量，我们添加一个空白图像
                faces = app.get(frame)
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
                    face = faces[0]
                    x1, y1, x2, y2 = [int(coord) for coord in face["bbox"]]
                    h, w = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    face_croped = frame[y1:y2, x1:x2]
                    face_resized = resize_face(face_croped, size=target_size)
                    faces_croped.append(face_resized)
                # else:
                #     # 如果连第一帧都没有脸，可以添加一个占位符
                #     faces_croped.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

    cap.release()

    # 如果提取的人脸数量少于要求，则重复以补足
    if len(faces_croped) > 0 and len(faces_croped) < num_samples:
        num_repeats = num_samples // len(faces_croped)
        repeated_faces = []
        for face in faces_croped:
            repeated_faces.extend([face] * num_repeats)

        # 如果除不尽，用最后一个元素填充
        remainder = num_samples - len(repeated_faces)
        if remainder > 0:
            repeated_faces.extend([faces_croped[-1]] * remainder)

        faces_croped = repeated_faces

    if len(faces_croped) == 0 and total_frames > 0:
        # 如果完全没有提取到任何人脸，但视频有帧，则返回一个占位符
        # 这可以防止后续代码因空列表而出错
        # faces_croped = [np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)] * num_samples
        pass

    # 先将列表转换为numpy数组，再转换为tensor，避免性能警告
    if len(faces_croped) > 0:
        faces_array = np.array(faces_croped)
        faces_tensor = torch.from_numpy(faces_array)
    else:
        faces_tensor = torch.empty(0)
    return faces_tensor


def extract_face_multithread(dataset, num_samples, det_score_thresh, target_size, num_workers):
    """
    多线程提取人脸并保存到face文件夹，目录结构与Raw一致

    Args:
        dataset (str): 数据集名称
        fps (int): 提取帧率
        det_score_thresh (float): 检测分数阈值
        target_size (tuple): 目标人脸尺寸 (height, width)
        num_workers (int): 线程数

    Returns:
        dict: 提取统计信息
    """

    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

    def process_single_video(video_path):
        """单个视频处理函数，用于多线程"""
        face_path = video_path.replace("/video/", "/face/").replace(".mp4", ".pt")

        # 检查输出文件是否已存在
        if os.path.exists(face_path):
            return

        os.makedirs(os.path.dirname(face_path), exist_ok=True)
        face_tensor = extract_face(app, video_path, target_size, num_samples, det_score_thresh)

        torch.save(face_tensor, face_path)

    # 获取所有视频路径
    video_paths = get_video_paths(f"datasets/{dataset}/video")

    print(f"开始多线程提取人脸数据")
    print(f"数据集: {dataset}")
    print(f"提取帧率: {num_samples} num_samples")
    print(f"检测阈值: {det_score_thresh}")
    print(f"目标尺寸: {target_size}")
    print(f"线程数: {num_workers}")
    print(f"总视频数: {len(video_paths)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for video_path in video_paths:
            if video_path.split(".")[-1] != "mp4":
                continue
            futures.append(executor.submit(process_single_video, video_path))

        # 使用 tqdm 显示进度
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理视频进度"):
            future.result()


extract_face_multithread(dataset="MOSI", num_samples=32, det_score_thresh=0.55, target_size=(224, 224), num_workers=32)
extract_face_multithread(dataset="MOSEI", num_samples=32, det_score_thresh=0.55, target_size=(224, 224), num_workers=32)
# extract_face_multithread(dataset="SIMS", num_samples=32, det_score_thresh=0.55, target_size=(224, 224), num_workers=32)
