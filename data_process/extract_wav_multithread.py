"""
使用直接调用ffmpeg命令行的方式处理音视频文件
"""

import os
import cv2
import subprocess
import concurrent.futures


def extract_audio(directory_path, audio_sampling_rate=16000):
    """
    输入目录：Raw
    输出目录：wav
    """
    input_directory_path = directory_path
    output_directory_path = directory_path.replace("Raw", "wav")
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    def process_file(input_file_path, output_file_path):
        # print("\n")
        if os.path.exists(input_file_path.replace(".mp4", "-edited.mp4")):
            return

        # 检查视频是否有音轨
        cmd_check = ["ffprobe", "-i", input_file_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"]
        result = subprocess.run(cmd_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not result.stdout:
            print(f"警告：视频没有音频轨道 - {input_file_path}")
            return

        # 准备输出路径
        if "-edited.mp4" in output_file_path:
            output_file_path = output_file_path.replace("-edited.mp4", ".mp4")
        output_file_path = output_file_path.split(".")[0] + ".wav"

        # 使用ffmpeg提取音频并重采样
        try:
            cmd = [
                "ffmpeg",
                "-i",
                input_file_path,
                "-vn",  # 不要视频
                "-acodec",
                "pcm_s16le",  # 音频编码
                "-ar",
                str(audio_sampling_rate),  # 采样率
                "-ac",
                "2",  # 单声道
                "-y",  # 覆盖输出文件
                "-loglevel",
                "error",  # 减少输出
                output_file_path,
            ]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("保存出错：", input_file_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for folder in os.listdir(input_directory_path):
            if folder == ".DS_Store":
                continue
            input_folder_path = os.path.join(input_directory_path, folder)
            output_folder_path = os.path.join(output_directory_path, folder)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            for file_name in os.listdir(input_folder_path):
                if file_name.split(".")[-1] != "mp4":
                    continue
                input_file_path = os.path.join(input_folder_path, file_name)
                output_file_path = os.path.join(output_folder_path, file_name)
                futures.append(executor.submit(process_file, input_file_path, output_file_path))
        concurrent.futures.wait(futures)


def check_video_duration(filename):
    # print("\n")
    cap = cv2.VideoCapture(filename)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

    frame_counter = 0
    for f in range(n_frames):
        ret, frame = cap.read()  # ret: 是否读取到帧, frame: 帧
        frame_counter += 1
        if ret:
            continue
        elif frame_counter > n_frames:  # 读取到的帧数大于视频总帧数，说明视频正常
            return None
        else:  # 读取到的帧数小于视频总帧数
            duration = (frame_counter - 1) / fps  # 修正视频时长, 由于视频帧数从0开始, 所以减1
            print(f"check bad video file: {filename}")
            print(f"n_frames: {n_frames}, frame_counter: {frame_counter}, duration: {duration}:")
            with open(filename.replace(".mp4", ".txt"), "w") as f:
                f.write(f"n_frames: {n_frames}, frame_counter: {frame_counter}, duration: {duration}, fps: {fps}\n")

            return duration
    return None


def fix_dataset(directory_path):
    # 多线程处理每个视频文件
    def process_video(fpath):
        if "-edited.mp4" in fpath:
            return
        if os.path.exists(fpath.replace(".mp4", "-edited.mp4")):
            return
        duration = fpath
        if duration:
            output_path = fpath.replace(".mp4", "-edited.mp4")
            try:
                cmd = [
                    "ffmpeg",
                    "-i",
                    fpath,
                    "-t",
                    str(duration),  # 截取视频时长
                    "-c:v",
                    "copy",  # 复制视频流
                    "-c:a",
                    "copy",  # 复制音频流
                    "-y",  # 覆盖输出文件
                    "-loglevel",
                    "error",  # 减少输出
                    output_path,
                ]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"处理视频失败: {fpath}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for folder in os.listdir(directory_path):
            if folder == ".DS_Store":
                continue
            folder_path = os.path.join(directory_path, folder)
            for file_name in os.listdir(folder_path):
                if file_name == ".DS_Store":
                    continue
                fpath = os.path.join(folder_path, file_name)
                futures.append(executor.submit(process_video, fpath))
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    pass
    # fix_dataset("MOSI/Raw")
    extract_audio("MOSI/Raw", audio_sampling_rate=16000)
    # fix_dataset("MOSEI/Raw")
    # extract_audio("MOSEI/Raw", audio_sampling_rate=16000)
