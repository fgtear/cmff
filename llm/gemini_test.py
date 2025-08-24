from google import genai
import os
import sys

os.environ["http_proxy"] = "socks5://127.0.0.1:1080"
os.environ["https_proxy"] = "socks5://127.0.0.1:1080"
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


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()
myfile = client.files.upload(file="datasets/MOSI/omni_video/_dI--eQ6qVU/1.mp4")
# while myfile.state == "PROCESSING":
#     print("Waiting for video to be processed.")
#     time.sleep(3)

try:
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        # model="gemini-2.5-flash",
        # model="gemini-2.5-flash-lite",
        # contents="Explain how AI works in a few words",
        contents=[system_prompt, myfile],
    )
    print(response.text)
except Exception as e:
    print(e)
