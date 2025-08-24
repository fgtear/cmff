import subprocess
import os
from pathlib import Path
import concurrent.futures
import tqdm


def split_video2av(video_path, audio_output_path, video_output_path, sample_rate=16000):
    """
    使用 ffmpeg 将视频文件分割成一个音频文件和一个无声视频文件。

    Args:
        video_path (Union[str, Path]): 输入视频文件的路径。
        audio_output_path (str): 输出音频文件的路径 (应为 .wav)。
        video_output_path (str): 输出无声视频文件的路径。
        sample_rate (int): 音频的采样率。
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"错误：在 {video_path_obj} 未找到视频文件")
        return

    audio_command = [
        "ffmpeg",
        # "-hwaccel",
        # "cuda",  # 启用 CUDA 硬件加速
        "-i",  #  输入文件
        str(video_path_obj),
        "-vn",  # 禁用视频录制
        "-ar",  # 设置音频采样率
        str(sample_rate),
        "-acodec",  # 指定音频编码器。
        "pcm_s16le",  # 指定WAV的编码器，pcm_s16le 即 16-bit 小端字节序的 PCM 编码。
        "-y",  # 如果输出文件已存在则覆盖
        str(audio_output_path),
    ]

    # 不使用 "-vcodec copy"， 会强制重新编码，这有助于修复或删除无效帧
    video_command = [
        "ffmpeg",
        # "-hwaccel",
        # "cuda",  # 启用 CUDA 硬件加速
        "-i",  #  输入文件
        str(video_path_obj),
        "-an",  # 禁用音频录制
        # "-vcodec",  # 指定视频编码器
        # "copy",  # 直接复制视频流
        "-y",  # 如果输出文件已存在则覆盖
        str(video_output_path),
    ]

    try:
        # print(f"正在提取音频到 {audio_output_path}...")
        subprocess.run(audio_command, check=True, capture_output=True, text=True)
        # print("音频提取成功。")
    except subprocess.CalledProcessError as e:
        print(f"提取音频时出错: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return

    try:
        # print(f"正在创建无声视频于 {video_output_path}...")
        subprocess.run(video_command, check=True, capture_output=True, text=True)
        # print("无声视频创建成功。")
    except subprocess.CalledProcessError as e:
        print(f"创建无声视频时出错: {e}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return

    # print(f"成功将视频分割为音频 ({audio_output_path}) 和无声视频 ({video_output_path})")
    return str(audio_output_path), str(video_output_path)


def split_dataset_multithread(directory_path, num_workers=4):
    def process_one(video_path):
        audio_output_path = video_path.replace("/Raw/", "/wav/").replace(".mp4", ".wav")
        video_output_path = video_path.replace("/Raw/", "/video/")
        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        return split_video2av(video_path, audio_output_path, video_output_path, sample_rate=16000)

    video_files = []
    for folder in os.listdir(directory_path):
        if folder == ".DS_Store":
            continue
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name == ".DS_Store" or not file_name.endswith(".mp4"):
                continue
            fpath = os.path.join(folder_path, file_name)
            video_files.append(fpath)

    # video_files = video_files[:1000]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm.tqdm(executor.map(process_one, video_files), total=len(video_files), desc="修复进度"))


if __name__ == "__main__":
    # split_video2av(
    #     "datasets/MOSEI/Raw/_0efYOjQYRc/3.mp4",
    #     "3-aaa.wav",
    #     "3-vvv.mp4",
    #     sample_rate=22050,
    # )

    split_dataset_multithread("datasets/MOSI/Raw", num_workers=8)
    split_dataset_multithread("datasets/MOSEI/Raw", num_workers=8)
    # split_dataset_multithread("datasets/SIMS/Raw", num_workers=8)
