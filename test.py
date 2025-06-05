import requests
import base64

API_KEY = "AIzaSyDFxz_ftWU0CqLKGKiR74wy9kSXtapWelE"  # 여기에 너의 실제 키
MODEL_NAME = "gemini-1.5-pro-001"
IMAGE_PATH = "sample.png"

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={API_KEY}"

with open(IMAGE_PATH, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "contents": [
        {
            "parts": [
                {"text": "이 이미지는 딥페이크인지 판단하고 이유를 설명해줘."},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ]
        }
    ]
}

response = requests.post(GEMINI_ENDPOINT, json=payload)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print(response.json()['candidates'][0]['content']['parts'][0]['text'])
else:
    print(response.text)
