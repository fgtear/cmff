import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
import json
from qwen_utils.encoder import b64encode, tensor_to_b64encode
from prompts import system_prompt

MODEL = "qwen2.5-omni-7b"
# MODEL = "qwen-omni-turbo"


# system_prompt = "transcribe audio"


video_path = "datasets/MOSI/omni_video/_dI--eQ6qVU/1.mp4"
base64_video = b64encode(video_path)

# base64_audio = b64encode(audio_path)
# frame = torch.load(frame_path)
# image_list = []
# for i in range(frame.shape[0]):
#     image_tensor = frame[i]
#     # print("image_tensor", image_tensor.shape)
#     base64_image = tensor_to_b64encode(image_tensor, image_format="PNG")
#     image_list.append(f"data:image/jpeg;base64,{base64_image}")

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    seed=0,
    # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
    # modalities=["text", "audio"],
    modalities=["text"],
    audio={"voice": "Chelsie", "format": "wav"},
    stream=True,  # stream 必须设置为 True，否则会报错
    stream_options={"include_usage": True},
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image_url",
                #     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                # },
                {
                    "type": "text",
                    "text": text,
                },
                # {
                #     "type": "input_audio",
                #     "input_audio": {
                #         "data": f"data:;base64,{base64_audio}",
                #         "format": "mp3",
                #     },
                # },
                # {
                #     "type": "video",
                #     "video": image_list,
                # },
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:;base64,{base64_video}"},
                },
            ],
        },  # type: ignore
    ],
)
response = ""
for chunk in completion:
    pass
    if chunk.choices:
        if chunk.choices[0].delta.model_extra != {}:
            response += chunk.choices[0].delta.model_extra["audio"]["transcript"]

            # print("video_path", video_path)
            # print("response", response)
            # 清理响应文本，移除可能的markdown格式
            response = response.strip()
            response = response.lstrip("```json").lstrip("```")
            response = response.rstrip("```")
            response = response.replace("\n", "").replace("\r", "")
            response = response.strip()
            # print("cleaned response", response)

            # 尝试解析JSON
            response_dict = json.loads(response)

print("response:", response)
