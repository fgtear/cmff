import os


def get_all_files(directory, extension):
    """
    获取指定目录下所有指定扩展名的文件
    extension: 文件扩展名（如 'txt', 'mp4'）
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def get_video_paths(directory_path):
    """
    获取目录下所有视频文件的路径
    """
    video_paths = []
    for folder in os.listdir(directory_path):
        if folder == ".DS_Store":
            continue
        input_folder_path = os.path.join(directory_path, folder)
        for file_name in os.listdir(input_folder_path):
            if file_name.split(".")[-1] in ["mp4", "avi", "mov", "mkv"]:
                video_paths.append(os.path.join(input_folder_path, file_name))
    return video_paths
