import base64
import numpy as np
import torch
import torchvision
import io


def tensor_to_b64encode(image_tensor: torch.Tensor, image_format: str = "PNG") -> str:
    """
    将一个 PyTorch Tensor (C, H, W) 编码为 Base64 字符串。

    参数:
    - image_tensor (torch.Tensor): 输入的图像张量，形状应为 (C, H, W)，
                                   数值范围通常为 [0.0, 1.0] 的浮点数。
    - image_format (str): 希望编码成的图片格式，如 'PNG', 'JPEG'。

    返回:
    - str: 图像的 Base64 编码字符串。
    """
    # 1. 创建一个内存中的二进制流（可以看作一个虚拟文件）
    buffer = io.BytesIO()

    # 2. 确保张量的数值范围在 [0, 1] 内
    # 如果张量的值域不在 [0, 1] 范围内，进行归一化
    if image_tensor.max() > 1.0 or image_tensor.min() < 0.0:
        # 将张量归一化到 [0, 1] 范围
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    # 确保张量是浮点类型
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()

    # 3. 使用 torchvision.utils.save_image 将 Tensor 保存到这个内存流中
    #    注意：save_image 默认会将浮点型 Tensor 的范围从 [0, 1] 转换到 [0, 255] 的整数。
    torchvision.utils.save_image(image_tensor, buffer, format=image_format)

    # 4. 从 buffer 中获取完整的二进制数据
    image_bytes = buffer.getvalue()

    # 5. 对二进制数据进行 Base64 编码，并解码为 utf-8 字符串
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    return b64_str


def b64encode(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def encode_image(image_array):
    """
    将 numpy 数组表示的图像编码为 Base64 字符串。

    参数:
        image_array (np.ndarray): 图像的 numpy 数组，通常为 BGR 或 RGB 格式。

    返回:
        str: 图像的 Base64 编码字符串（格式为 JPEG）
    """
    # 确保输入是 numpy array
    if not isinstance(image_array, np.ndarray):
        raise ValueError("输入必须是 numpy 数组")

    # 使用 OpenCV 编码为 JPEG 格式的字节流
    success, buffer = cv2.imencode(".jpg", image_array)

    if not success:
        raise ValueError("无法将 numpy 数组编码为 JPEG 格式图像")

    # 转为 Base64 字符串
    return base64.b64encode(buffer).decode("utf-8")


# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# def encode_audio(audio_path):
#     with open(audio_path, "rb") as audio_file:
#         return base64.b64encode(audio_file.read()).decode("utf-8")


# #  Base64 编码格式
# def encode_video(video_path):
#     with open(video_path, "rb") as video_file:
#         return base64.b64encode(video_file.read()).decode("utf-8")
