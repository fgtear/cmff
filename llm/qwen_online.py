import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Literal
import pandas as pd
import tqdm
import json
import math
import time
from qwen_utils.encoder import b64encode, tensor_to_b64encode
from prompts import system_prompt

MODEL = "qwen2.5-omni-7b"
# MODEL = "qwen-omni-turbo"


# system_prompt = "transcribe audio"


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def validate_response_format(response_dict):
    """验证响应是否包含所有必需的字段"""
    required_fields = ["text_sentiment_score", "audio_sentiment_score", "video_sentiment_score", "mm_sentiment_score"]
    if not isinstance(response_dict, dict):
        return False
    for field in required_fields:
        if field not in response_dict:
            return False
        # 检查值是否为数字类型
        try:
            float(response_dict[field])
        except (ValueError, TypeError):
            return False
    return True


def analyse_one(text, video_path, max_retries=3):
    # print("".center(50, "-"))
    # print("text", text)
    # print("video_path", video_path)

    video_id = video_path.split("/")[-2]
    if "-edited" in video_path:
        clip_id = video_path.split("/")[-1].split(".")[0].replace("-edited", "")
    else:
        clip_id = video_path.split("/")[-1].split(".")[0]

    base64_video = b64encode(video_path)

    # base64_audio = b64encode(audio_path)
    # frame = torch.load(frame_path)
    # image_list = []
    # for i in range(frame.shape[0]):
    #     image_tensor = frame[i]
    #     # print("image_tensor", image_tensor.shape)
    #     base64_image = tensor_to_b64encode(image_tensor, image_format="PNG")
    #     image_list.append(f"data:image/jpeg;base64,{base64_image}")

    for attempt in range(max_retries):
        try:
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

            # 验证响应格式
            if validate_response_format(response_dict):
                response = {f"{video_id}_{clip_id}": response_dict}
                return response
            else:
                print(f"Attempt {attempt + 1}: Invalid response format for {video_path}: {response_dict}")
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回默认值
                    print(f"Max retries reached for {video_path}, using default values")
                    response = {
                        f"{video_id}_{clip_id}": {
                            "text_sentiment_score": None,
                            "audio_sentiment_score": None,
                            "video_sentiment_score": None,
                            "mm_sentiment_score": None,
                        }
                    }
                    return response

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON parsing error for {video_path}: {e}, response: {response}")
            if attempt == max_retries - 1:
                # 最后一次尝试失败，返回默认值
                print(f"Max retries reached for {video_path}, using default values")
                response = {
                    f"{video_id}_{clip_id}": {
                        "text_sentiment_score": None,
                        "audio_sentiment_score": None,
                        "video_sentiment_score": None,
                        "mm_sentiment_score": None,
                    }
                }
                return response
        except Exception as e:
            print(f"Attempt {attempt + 1}: Unexpected error for {video_path}: {e}")
            time.sleep(30)
            if attempt == max_retries - 1:
                # 最后一次尝试失败，返回默认值
                print(f"Max retries reached for {video_path}, using default values")
                response = {
                    f"{video_id}_{clip_id}": {
                        "text_sentiment_score": None,
                        "audio_sentiment_score": None,
                        "video_sentiment_score": None,
                        "mm_sentiment_score": None,
                    }
                }
                return response


def analyse_dataset(dataset: Literal["MOSI", "MOSEI", "SIMS", "SIMSV2"], mode: Literal["train", "valid", "test", None]):
    dataset_root = f"datasets/{dataset}"
    label_path = dataset_root + "/label.csv"
    # 从之前的结果文件继续分析
    if os.path.exists(f"datasets/{dataset}{MODEL}.csv"):
        label_path = f"datasets/{dataset}{MODEL}.csv"

    if dataset == "SIMS":
        df = pd.read_csv(
            label_path,
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(label_path, dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for index, row in tqdm.tqdm((df.iterrows())):
        if mode:
            if row["mode"] != mode:
                continue

        if "text_sentiment_score" not in row or math.isnan(row.get("text_sentiment_score")):
            text = row["text"]
            video_path = f"{dataset_root}/omni_video/{row['video_id']}/{row['clip_id']}.mp4"
            # 检查是否存在修复版本的视频
            edited_video_path = video_path.replace(".mp4", "-edited.mp4")
            if os.path.exists(edited_video_path):
                video_path = edited_video_path
            # audio_path = f"{dataset_root}/wav/{row['video_id']}/{row['clip_id']}.wav"
            # frame_path = f"{dataset_root}/frame/{row['video_id']}/{row['clip_id']}.pt"
            result = analyse_one(text, video_path)
            for key in result[f"{row['video_id']}_{row['clip_id']}"]:
                df.at[index, key] = result[f"{row['video_id']}_{row['clip_id']}"][key]
    df.to_csv(f"datasets/{dataset}{MODEL}.csv", index=False)


def analyse_dataset_multithread(
    dataset: Literal["MOSI", "MOSEI", "SIMS", "SIMSV2"], mode: Literal["train", "valid", "test", None], num_workers: int
):
    dataset_root = f"datasets/{dataset}"
    label_path = dataset_root + "/label.csv"
    # 从之前的结果文件继续分析
    if os.path.exists(f"datasets/{dataset}/{MODEL}.csv"):
        label_path = f"datasets/{dataset}/{MODEL}.csv"

    if dataset == "SIMS":
        df = pd.read_csv(
            label_path,
            names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"],
            dtype={"video_id": str, "clip_id": str},
        )
    elif dataset in ["MOSI", "MOSEI", "SIMSV2"]:
        df = pd.read_csv(label_path, dtype={"video_id": str, "clip_id": str})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 准备参数
    args_list = []
    for index, row in df.iterrows():
        if mode:
            if row["mode"] != mode:
                continue

        if "text_sentiment_score" not in row or math.isnan(row.get("text_sentiment_score")):
            text = row["text"]
            video_path = f"{dataset_root}/omni_video/{row['video_id']}/{row['clip_id']}.mp4"
            # 检查是否存在修复版本的视频
            edited_video_path = video_path.replace(".mp4", "-edited.mp4")
            if os.path.exists(edited_video_path):
                video_path = edited_video_path
            audio_path = f"{dataset_root}/wav/{row['video_id']}/{row['clip_id']}.wav"
            frame_path = f"{dataset_root}/frame/{row['video_id']}/{row['clip_id']}.pt"
            args_list.append((text, video_path))

    args_list = args_list[:9]

    # 使用多线程处理
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(analyse_one, *args) for args in args_list]
        # 使用tqdm显示进度
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            # print("result", result)
            results.update(result)

    # print("results", results)
    for index, row in df.iterrows():
        video_id = row["video_id"]
        clip_id = row["clip_id"]
        if f"{video_id}_{clip_id}" in results:
            for key in results[f"{video_id}_{clip_id}"]:
                df.at[index, key] = results[f"{video_id}_{clip_id}"][key]

    df.to_csv(f"datasets/{dataset}/{MODEL}.csv", index=False)


if __name__ == "__main__":
    # analyse_one("", "datasets/MOSI/omni_video/0h-zjBukYpk/15.mp4")
    analyse_dataset_multithread("MOSI", mode="valid", num_workers=8)
    # analyse_dataset("MOSEI", mode="test")
