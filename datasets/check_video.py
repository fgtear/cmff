import cv2
import os

from utils import get_all_files
import tqdm


def check_video_file(video_path, check_frames=True):
    """
    检查视频文件是否存在问题

    参数:
        video_path: 视频文件路径
        check_frames: 是否检查每一帧

    返回:
        dict: 包含检查结果和详细信息的字典。
              'status_code': 0:OK, 1:不存在, 2:无法打开, 3:属性无效, 4:帧问题
    """

    result = {
        "status_code": 0,
        "file_exists": False,
        "can_open": False,
        "properties": {},
        "frame_issues": 0,
        "total_frames": 0,
        "successful_frames": 0,
        "issues": [],
        "problematic_frames_indices": [],
    }

    # 检查文件是否存在
    if not os.path.exists(video_path):
        result["status_code"] = 1
        result["issues"].append(f"文件不存在: {video_path}")
        return result

    result["file_exists"] = True

    # 尝试打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["status_code"] = 2
        result["issues"].append("无法打开视频文件")
        return result

    result["can_open"] = True

    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    result["properties"] = {"total_frames": total_frames, "fps": fps, "width": width, "height": height, "duration": duration}

    # 检查基本属性是否合理
    has_property_issue = False
    if width <= 0 or height <= 0:
        result["issues"].append(f"无效的分辨率: {width}x{height}")
        has_property_issue = True

    if fps <= 0:
        result["issues"].append(f"无效的帧率: {fps}")
        has_property_issue = True

    if total_frames <= 0:
        result["issues"].append(f"无效的总帧数: {total_frames}")
        has_property_issue = True

    if has_property_issue:
        result["status_code"] = 3

    # 检查帧内容（可选，可能很耗时）
    if check_frames and total_frames > 0:
        frame_issues = 0
        successful_frames = 0
        problematic_frames_indices = []

        for i in range(total_frames):
            ret, frame = cap.read()
            is_problematic = False
            if not ret:
                is_problematic = True
            else:
                successful_frames += 1
                # 检查帧是否完全空白或损坏
                if frame is None or frame.size == 0:
                    is_problematic = True
                # 检查帧尺寸是否一致
                elif frame.shape[0] != height or frame.shape[1] != width:
                    is_problematic = True

            if is_problematic:
                frame_issues += 1
                problematic_frames_indices.append(i)

        result["frame_issues"] = frame_issues
        result["successful_frames"] = successful_frames
        result["total_frames"] = total_frames
        result["problematic_frames_indices"] = problematic_frames_indices

        if frame_issues > 0:
            result["issues"].append(f"发现 {frame_issues} 个帧问题")
            if result["status_code"] == 0:  # 仅在没有更严重的属性问题时，才将状态码更新为帧问题
                result["status_code"] = 4

    cap.release()

    return result


if __name__ == "__main__":
    # result = check_video_file("datasets/MOSEI/Raw/_0efYOjQYRc/3.mp4", check_frames=True)
    result = check_video_file("3-vvv.mp4", check_frames=True)
    print(result)

    # result = check_video_file("datasets/MOSEI/Raw/_0efYOjQYRc/3-edited.mp4", check_frames=True)
    # print(result)

    # files = get_all_files("datasets/MOSEI/Raw", "mp4")
    # for video_path in tqdm.tqdm(files):
    #     result = check_video_file(video_path, check_frames=True)
    #     if result["status_code"] != 0:
    #         print(f"文件: {video_path}, 状态: {result['status_code']}, 问题: {result['issues']}")

