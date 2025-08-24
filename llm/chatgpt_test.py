# import os

# os.environ["http_proxy"] = "socks5://127.0.0.1:1080"
# os.environ["https_proxy"] = "socks5://127.0.0.1:1080"


from openai import OpenAI

client = OpenAI(
    api_key="sk-k7CrnfoAxsjmCwSqIHxEeg3J8zQXVIgTsRjdvuIh8D2CzuRu",
    base_url="https://api.openxs.top/v1",
)

completion = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}]
)

print(completion.choices[0].message.content)
