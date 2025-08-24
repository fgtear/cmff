# import flash_attn
import soundfile as sf
import time
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from prompts import system_prompt
from thop import profile


# system_prompt = """
# # Role: You are a top expert in multimodal sentiment analysis, and your task is to conduct
#  a comprehensive sentiment assessment of samples from the CMU-MOSI/MOSEI dataset.
# """
# system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     # attn_implementation="flash_attention_2",
# )
# model.to("cpu")
# model.eval()
# total_params = sum(p.numel() for p in model.parameters())
# print("total_params", total_params)

# start = time.time()

# conversation = [
#     {
#         "role": "system",
#         "content": [
#             {
#                 "type": "text",
#                 # "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
#                 "text": system_prompt,
#             }
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "oh my gosh bad movie"},
#             # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
#             {"type": "video", "video": "datasets/MOSI/omni_video/c7UH_rxdZv4/1.mp4"},
#         ],
#     },
# ]

# # set use audio in video
# USE_AUDIO_IN_VIDEO = True

# # Preparation for inference
# text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
# audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)


# inputs = processor(
#     text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
# )
# # print("".center(50, "-"))
# # print("text", text)
# # print("audios", audios)
# # print("audios", len(audios))
# # print("audios", audios[0].shape)
# # print("images", images)
# # print("videos", len(videos))
# # print("videos", videos[0].shape)
# # print("inputs keys", inputs.keys())
# # print("inputs", inputs)
# # print("inputs['input_ids'] shape", inputs["input_ids"].shape)
# # print("inputs['pixel_values_videos'] shape", inputs["pixel_values_videos"].shape)
# # print("inputs['video_grid_thw'] shape", inputs["video_grid_thw"].shape)
# # print("inputs['feature_attention_mask'] shape", inputs["feature_attention_mask"].shape)
# # print("inputs['input_features'] shape", inputs["input_features"].shape)


# inputs = inputs.to(model.device).to(model.dtype)
# # inputs = inputs.to(model.device)
# print("".center(50, "-"))
# print("model.device", model.device)
# print("model.dtype", model.dtype)

# # Inference: Generation of the output text and audio
# text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(text)
# # sf.write(
# #     "output.wav",
# #     audio.reshape(-1).detach().cpu().numpy(),
# #     samplerate=24000,
# # ) x

# print(f"Time: {time.time() - start:.2f} seconds")


class QwenOmniModel(torch.nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        super().__init__()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.model.to("cpu")
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print("num_params", self.num_params)

    def forward(self, text, video_path):
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                    {"type": "video", "video": video_path},
                ],
            },
        ]
        # set use audio in video
        USE_AUDIO_IN_VIDEO = True

        # Preparation for inference
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(text)
        return text


qwen_omni = QwenOmniModel()
qwen_omni.to("cpu")

# text = qwen_omni("oh my gosh bad movie", "datasets/MOSI/omni_video/c7UH_rxdZv4/1.mp4")

input_args = ("oh my gosh bad movie", "datasets/MOSI/omni_video/c7UH_rxdZv4/1.mp4")
flops, params = profile(qwen_omni, inputs=input_args, verbose=False)
print(f"模型FLOPs: {flops / 1e9:.2f} GFLOPs, 参数量: {params / 1e6:.2f} M")
