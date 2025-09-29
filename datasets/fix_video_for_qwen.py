import os
import pandas as pd
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from moviepy import ColorClip, concatenate_videoclips, VideoFileClip
import subprocess
import json
import shlex
from typing import Union, Tuple, Optional, Dict, Any


def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """
    使用 ffprobe 获取视频信息，包括时长、分辨率和是否存在音轨。

    Args:
        video_path (str): 视频文件路径。

    Returns:
        Optional[Dict[str, Any]]: 包含视频信息的字典，如果出错则返回 None。
        例如: {'duration': 10.5, 'width': 1920, 'height': 1080, 'has_audio': True}
    """
    # -v quiet: 安静模式
    # -print_format json: 以JSON格式输出
    # -show_format: 显示容器格式信息（包括时长）
    # -show_streams: 显示流信息（视频、音频）
    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        info = {}
        # 从 format 标签获取时长
        info["duration"] = float(data["format"]["duration"])

        # 寻找视频流以获取分辨率
        video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        if not video_stream:
            print(f"错误：在 {video_path} 中未找到视频流。")
            return None
        info["width"] = int(video_stream["width"])
        info["height"] = int(video_stream["height"])

        # 检查是否存在音频流
        info["has_audio"] = any(s["codec_type"] == "audio" for s in data["streams"])

        return info
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, StopIteration) as e:
        print(f"错误：使用 ffprobe 分析视频失败 -> {video_path}\n{e}")
        return None


def fix_video(video_path: str, min_width: Optional[int] = None, min_height: Optional[int] = None):
    """
    读取一个视频，判断视频的长度，如果长度小于2s，则通过增加全黑视频和无声音频，直到长度大于等于2s；
    如果视频的长度大于39s，则截断。
    最后，还可以选择性地调整视频分辨率。
    保存修复的视频到 dataset/omni_video/ 目录下。
    (使用 ffmpeg 实现)

    Args:
        video_path (str): 输入视频的路径。
        min_width (Optional[int]): 输出视频的最小宽度，如果指定则会智能缩放
        min_height (Optional[int]): 输出视频的最小高度，如果指定则会智能缩放
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 -> {video_path}")
        return

    output_path = video_path.replace("/Raw/", "/omni_video/")
    if os.path.exists(output_path):
        # print(f"信息：输出文件已存在，跳过 -> {output_path}")
        return 1
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    info = get_video_info(video_path)
    if not info:
        return  # 如果获取信息失败，则终止处理

    duration = info["duration"]
    has_audio = info["has_audio"]

    # 2. 构建 ffmpeg 命令
    # -i: 输入文件
    command = ["ffmpeg", "-i", video_path]

    # 用于构建复杂滤镜链
    vf_filters = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]  # 视频滤镜
    af_filters = []  # 音频滤镜

    # 3. 根据时长处理
    if duration < 2.0:
        # 时长小于2s，补齐
        padding_duration = 2.0 - duration
        # tpad: 在视频末尾添加帧。stop_mode=add 表示添加新帧，color=black 指定为黑色
        vf_filters.append(f"tpad=stop_mode=add:stop_duration={padding_duration}:color=black")
        # 如果原视频有音轨，则用 apad 补充静音
        if has_audio:
            af_filters.append(f"apad=pad_dur={padding_duration}")
        # 如果原视频没有音轨，ffmpeg默认不会创建音轨，这符合“无声音频”的要求

    elif duration > 39.0:
        # 时长大于39s，截断
        # -t: 指定输出时长
        command.extend(["-t", "39.0"])

    # 4. 根据 min_width 和 min_height 调整分辨率
    if min_width is not None or min_height is not None:
        # 获取原视频的分辨率
        original_width = info["width"]
        original_height = info["height"]

        scale_filter = ""

        # 计算需要的缩放比例，确保满足最小尺寸要求
        scale_ratio = 1.0

        if min_width is not None and min_height is not None:
            # 都指定了最小值，选择较大的缩放比例以确保两个维度都满足
            width_ratio = min_width / original_width if original_width > min_width else 1.0
            height_ratio = min_height / original_height if original_height > min_height else 1.0
            scale_ratio = max(width_ratio, height_ratio)
        elif min_width is not None:
            # 只指定了最小宽度
            scale_ratio = min_width / original_width if original_width > min_width else 1.0
        elif min_height is not None:
            # 只指定了最小高度
            scale_ratio = min_height / original_height if original_height > min_height else 1.0

        # 如果需要缩放
        if scale_ratio < 1.0:
            # 计算缩放后的尺寸，确保是偶数
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)

            # 确保尺寸是偶数
            new_width = new_width if new_width % 2 == 0 else new_width - 1
            new_height = new_height if new_height % 2 == 0 else new_height - 1

            # 最后检查是否满足最小尺寸要求，如果不满足则调整
            if min_width is not None and new_width < min_width:
                new_width = min_width if min_width % 2 == 0 else min_width + 1
            if min_height is not None and new_height < min_height:
                new_height = min_height if min_height % 2 == 0 else min_height + 1

            scale_filter = f"scale={new_width}:{new_height}"
        else:
            # 不需要缩放，只需要确保尺寸是偶数
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

        if scale_filter:
            # 替换原有的scale滤镜
            vf_filters[0] = scale_filter

    # 5. 组合滤镜并添加到命令中
    if vf_filters:
        command.extend(["-vf", ",".join(vf_filters)])
    if af_filters:
        command.extend(["-af", ",".join(af_filters)])

    # 6. 添加输出参数并执行
    # -c:v libx264: 使用 H.264 视频编码器
    # -c:a aac: 使用 AAC 音频编码器
    # -y: 如果输出文件已存在则覆盖（虽然我们已经检查过，但这是个好习惯）
    # -hide_banner -loglevel error: 隐藏 ffmpeg 的版本信息，只显示错误
    command.extend(["-c:v", "libx264", "-c:a", "aac", "-y", "-hide_banner", "-loglevel", "error", output_path])

    try:
        # print(f"正在执行命令: {' '.join([shlex.quote(c) for c in command])}")
        subprocess.run(command, check=True)
        # print(f"视频修复成功: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"错误：ffmpeg 处理失败 -> {video_path}")
        print(f"  - 返回码: {e.returncode}")
        print(f"  - 命令: {' '.join(map(shlex.quote, command))}")


def process_dataset(dataset, min_width=None, min_height=None):
    # if os.path.exists(f"datasets/{dataset}/omni_video"):
    #     print(f"frames already exists for dataset {dataset}. Removing existing omni_video...")
    #     shutil.rmtree(f"datasets/{dataset}/omni_video")  # 删除现有缓存目录

    def process_single_video(row):
        video_id = row["video_id"]
        clip_id = row["clip_id"]

        video_path = f"datasets/{dataset}/Raw/{video_id}/{clip_id}.mp4"

        # 检查是否存在修复版本的视频
        edited_video_path = video_path.replace(".mp4", "-edited.mp4")
        if os.path.exists(edited_video_path):
            video_path = edited_video_path

        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return False

        fix_video(video_path, min_width=min_width, min_height=min_height)

    if dataset == "SIMS":
        df = pd.read_csv(
            f"datasets/{dataset}/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(f"datasets/{dataset}/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for index, row in tqdm.tqdm(df.iterrows()):
        # import time
        # time.sleep(0.1)
        process_single_video(row)


def process_dataset_multithread(dataset, min_width=None, min_height=None, num_workers=8):
    # if os.path.exists(f"datasets/{dataset}/omni_video"):
    #     print(f"frames already exists for dataset {dataset}. Removing existing omni_video...")
    #     shutil.rmtree(f"datasets/{dataset}/omni_video")  # 删除现有缓存目录

    def process_single_video(row):
        """
        处理单个视频的函数，用于多线程
        """
        try:
            video_id = row["video_id"]
            clip_id = row["clip_id"]

            video_path = f"datasets/{dataset}/Raw/{video_id}/{clip_id}.mp4"

            # 检查是否存在修复版本的视频
            edited_video_path = video_path.replace(".mp4", "-edited.mp4")
            if os.path.exists(edited_video_path):
                video_path = edited_video_path

            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                return False

            fix_video(video_path, min_width=min_width, min_height=min_height)
            return True
        except Exception as e:
            print(f"处理视频时出错 {row.get('video_id', 'unknown')}/{row.get('clip_id', 'unknown')}: {str(e)}")
            return False

    if dataset == "SIMS":
        df = pd.read_csv(
            f"datasets/{dataset}/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(f"datasets/{dataset}/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

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


def check_fix(dataset):
    if dataset == "SIMS":
        df = pd.read_csv(
            f"datasets/{dataset}/label.csv",
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(f"datasets/{dataset}/label.csv", dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for index, row in df.iterrows():
        video_id = row["video_id"]
        clip_id = row["clip_id"]
        video_path = f"datasets/{dataset}/Raw/{video_id}/{clip_id}.mp4"
        # 检查是否存在修复版本的视频
        edited_video_path = video_path.replace(".mp4", "-edited.mp4")
        if os.path.exists(edited_video_path):
            video_path = edited_video_path

        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            continue

        try:
            clip = VideoFileClip(video_path)

            # Check video properties
            if clip.duration is None or clip.duration <= 0:
                print(f"Invalid duration in {video_path}")
                return False

            if clip.size is None or clip.size[0] == 0 or clip.size[1] == 0:
                print(f"Invalid resolution in {video_path}")
                return False

            if clip.fps is None or clip.fps <= 0:
                print(f"Invalid fps in {video_path}")
                return False

            # Check audio properties
            if clip.audio is not None:
                if clip.audio.duration is None or clip.audio.duration <= 0:
                    print(f"Invalid audio duration in {video_path}")
                    return False
            else:
                print(f"No audio in {video_path}")
                # Depending on requirements, you might want to return False here
                # return False

            # Check file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            if file_size_mb < 0.01:  # Less than 10KB
                print(f"Very small file size for {video_path}")
                return False

            print(f"Video {video_path} seems to be valid.")
            return True

        except Exception as e:
            print(f"Error checking video {video_path}: {e}")
            return False


if __name__ == "__main__":
    # fix_video("datasets/MOSI/Raw/_dI--eQ6qVU/6.mp4", min_width=None, min_height=None)
    # process_dataset(dataset="MOSI", min_width=224, min_height=224)
    # process_dataset_multithread(dataset="MOSI", min_width=224, min_height=224, num_workers=8)

    # fix_video("datasets/MOSEI/Raw/-THoVjtIkeU/11-edited.mp4", min_width=300, min_height=300)  # 示例：原来是0.39比例缩放
    # process_dataset(dataset="MOSEI", min_width=224, min_height=224)  # 示例：设置最小尺寸为300x300
    process_dataset_multithread(dataset="MOSEI", min_width=224, min_height=224, num_workers=8)
