from openai import OpenAI

client = OpenAI(
    api_key="sk-948a28feeec885707536f682b2818578431fbf9dc227c0af0af3d0af04c9d8a2",
    base_url="https://api.vilao.ai/v1"
)

response = client.chat.completions.create(
    model="ts/gpt-5.4-mini",
    messages=[{"role": "user", "content": "Hello! Who r u"}]
)
print(response.choices[0].message.content)