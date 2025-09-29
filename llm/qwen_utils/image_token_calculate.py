import math


from PIL import Image  # pip install Pillow


def image_token_calculate(image_path):
    # 打开指定的PNG图片文件
    image = Image.open(image_path)
    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    # 将高度调整为28的整数倍
    h_bar = round(height / 28) * 28
    # 将宽度调整为28的整数倍
    w_bar = round(width / 28) * 28
    # 图像的Token下限：4个Token
    min_pixels = 28 * 28 * 4
    # 图像的Token上限：1280个Token
    max_pixels = 1280 * 28 * 28
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 重新计算调整后的高度，确保为28的整数倍
        h_bar = math.floor(height / beta / 28) * 28
        # 重新计算调整后的宽度，确保为28的整数倍
        w_bar = math.floor(width / beta / 28) * 28
    elif h_bar * w_bar < min_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 重新计算调整后的高度，确保为28的整数倍
        h_bar = math.ceil(height * beta / 28) * 28
        # 重新计算调整后的宽度，确保为28的整数倍
        w_bar = math.ceil(width * beta / 28) * 28
    print(f"缩放后的图像尺寸为：高度为{h_bar}，宽度为{w_bar}")
    # 计算图像的Token数：总像素除以28 * 28
    token = int((h_bar * w_bar) / (28 * 28))
    # 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各1个Token）
    total_token = token + 2
    print(f"图像的Token数为{total_token}")
    return total_token


if __name__ == "__main__":
    total_token = image_token_calculate(image_path="test.png")
