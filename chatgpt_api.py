from openai import OpenAI
import base64

client = OpenAI(api_key="YOUR OWN API KEY")

image_path = "/home/user/heejun/MO506/GroundingDINO/image.png"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

prompt = """
사진은 자율주행 자동차의 front camera야.
anomaly 객체들, 날씨, 시간, 주행하면서 주의할 점에 대해 분석해줘.
아래 양식을 맞춰서 답변해줘.
양식:
1. anomaly 객체들 (콤마로 구분)
2. 날씨, 시간, 주행하면서 주의할 점
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ],
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

reply = response.choices[0].message.content

# 결과 파싱
lines = reply.split("\n")
anomaly_objects = lines[0] if len(lines) > 0 else ""
analysis = "\n".join(lines[1:]) if len(lines) > 1 else ""

print(anomaly_objects)
print(analysis)
