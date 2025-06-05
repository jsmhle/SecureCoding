import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from model_b1 import build_model as build_b1
from model_v2s import build_model as build_v2s
from torchvision.models import EfficientNet_B1_Weights, EfficientNet_V2_S_Weights
from torch.amp import autocast
import base64
import requests

# 🔐 Gemini API
GEMINI_API_KEY = "AIzaSyDFxz_ftWU0CqLKGKiR74wy9kSXtapWelE"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-001:generateContent?key={GEMINI_API_KEY}"

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📌 전처리
weights_b1 = EfficientNet_B1_Weights.DEFAULT
weights_v2s = EfficientNet_V2_S_Weights.DEFAULT
transforms_dict = {
    'b1': weights_b1.transforms(),
    'v2s': weights_v2s.transforms()
}

# 🔍 Gemini API 호출
def ask_gemini_about_image(image_path):
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "너는 딥페이크 이미지 탐지 전문가 AI다.\n"
            "아래 이미지를 보고 '딥페이크인지 아닌지'를 판단해라.\n\n"
            "- 반드시 '딥페이크다' 또는 '딥페이크가 아니다' 중 하나만 결론으로 말하라.\n"
            "- 판단을 유보하거나 '확실하지 않다'는 말은 절대 하지 마라.\n"
            "- 판단 기준: 조명, 피부 질감, 눈과 턱의 대칭성, 배경과 인물 경계의 부자연스러움 등\n\n"
            "답변 형식:\n"
            "결론: 딥페이크다 / 딥페이크가 아니다\n이유: (한두 문장)"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_data
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(GEMINI_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Gemini 오류: {str(e)}"

# 🧠 모델 로드
def load_model(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'b1':
        model = build_b1()
        model.load_state_dict(torch.load("deepfake_model_b1.pth", map_location=device))
    elif model_type == 'v2s':
        model = build_v2s()
        model.load_state_dict(torch.load("deepfake_model_v2s.pth", map_location=device))
    else:
        raise ValueError("지원하지 않는 모델 유형입니다.")
    model.to(device)
    model.eval()
    return model, device

# 🌐 라우팅
@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    result = None
    confidence = None
    gemini_opinion = None

    if request.method == 'POST':
        file = request.files.get('image')
        filename = request.form.get('filename')

        if file and file.filename:
            filename = file.filename
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

        if filename:
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            image = Image.open(img_path).convert('RGB')

            # 두 모델 전처리 및 입력
            image_b1 = transforms_dict['b1'](image).unsqueeze(0)
            image_v2s = transforms_dict['v2s'](image).unsqueeze(0)

            # 모델 로딩
            model_b1, device = load_model('b1')
            model_v2s, _ = load_model('v2s')
            image_b1 = image_b1.to(device)
            image_v2s = image_v2s.to(device)

            with torch.no_grad(), autocast(device_type='cuda'):
                out_b1 = F.softmax(model_b1(image_b1), dim=1)
                out_v2s = F.softmax(model_v2s(image_v2s), dim=1)
                avg_probs = (out_b1 + out_v2s) / 2
                pred = torch.argmax(avg_probs, dim=1).item()
                confidence = f"{avg_probs[0][pred].item() * 100:.2f}%"
                result = "Fake" if pred == 1 else "Real"

            # Gemini 판단
            gemini_opinion = ask_gemini_about_image(img_path)

    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           filename=filename,
                           model="Ensemble",
                           gemini_opinion=gemini_opinion)

if __name__ == '__main__':
    app.run(debug=True)
