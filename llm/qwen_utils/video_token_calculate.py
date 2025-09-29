# 使用前安装：pip install opencv-python
import math
import os
import logging
import cv2

logger = logging.getLogger(__name__)

# 固定参数
FRAME_FACTOR = 2
IMAGE_FACTOR = 28
# 视频帧的长宽比
MAX_RATIO = 200

# 视频帧的 Token 下限
VIDEO_MIN_PIXELS = 128 * 28 * 28
# 视频帧的 Token 上限
VIDEO_MAX_PIXELS = 768 * 28 * 28

# Qwen-Omni 模型 FPS 为 2
FPS = 2
# 最少抽取帧数
FPS_MIN_FRAMES = 4
# 最大抽取帧数
FPS_MAX_FRAMES = 512

# 视频输入的最大像素值
VIDEO_TOTAL_PIXELS = 65536 * 28 * 28

def round_by_factor(number, factor):
    return round(number / factor) * factor

def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor

def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor

def get_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_height, frame_width, total_frames, video_fps

def smart_nframes(total_frames, video_fps):
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    duration = total_frames / video_fps if video_fps != 0 else 0
    if duration - int(duration) > (1 / FPS):
        total_frames = math.ceil(duration * video_fps)
    else:
        total_frames = math.ceil(int(duration) * video_fps)
    nframes = total_frames / video_fps * FPS
    nframes = int(min(min(max(nframes, min_frames), max_frames), total_frames))
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes

def smart_resize(height, width, nframes, factor=IMAGE_FACTOR):
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def video_token_calculate(video_path):
    height, width, total_frames, video_fps = get_video(video_path)
    nframes = smart_nframes(total_frames, video_fps)
    resized_height, resized_width = smart_resize(height, width, nframes)
    video_token = int(math.ceil(nframes / FPS) * resized_height / 28 * resized_width / 28)
    video_token += 2  # 视觉标记
    return video_token

if __name__ == "__main__":
    video_path = "spring_mountain.mp4"  # 你的视频路径
    video_token = video_token_calculate(video_path)
    print("video_tokens:", video_token)

    