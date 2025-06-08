import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, flash
from torch.amp import autocast
import base64
import requests

# --- 1. 모델 아키텍처 import ---
from model_b1 import build_model as build_b1
from model_v2s import build_model as build_v2s
from ensemble_model import build_model as build_ensemble_model
from torchvision.models import EfficientNet_B1_Weights, EfficientNet_V2_S_Weights

# --- 2. 초기 설정 ---
app = Flask(__name__)
app.secret_key = 'super-secret-key-for-flask'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 Gemini API
GEMINI_API_KEY = "AIzaSyDFxz_ftWU0CqLKGKiR74wy9kSXtapWelE"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-001:generateContent?key={GEMINI_API_KEY}"

# --- 3. 모델 로딩 ---
print("딥페이크 탐지 모델 3개를 로딩합니다...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📌 모델별 전처리 정의 (수정된 부분)
weights_b1 = EfficientNet_B1_Weights.DEFAULT
weights_v2s = EfficientNet_V2_S_Weights.DEFAULT

# [수정] 각 모델에 맞는 전처리를 명확하게 정의합니다.
b1_transforms = weights_b1.transforms()
v2s_transforms = weights_v2s.transforms()

# [수정] 앙상블 모델(ViT 포함)은 224x224 크기를 사용하므로, 전처리를 별도로 정의합니다.
# 어떤 크기의 이미지가 들어와도 224x224로 강제 조정하여 에러를 방지합니다.
ensemble_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms_dict = {
    'b1': b1_transforms,
    'v2s': v2s_transforms,
    'ensemble': ensemble_transforms,
}


# 🧠 모델 객체들을 딕셔너리에 저장
models = {
    'b1': build_b1().to(device),
    'v2s': build_v2s().to(device),
    'ensemble': build_ensemble_model().to(device)
}
# 각 모델에 학습된 가중치 로드
try:
    models['b1'].load_state_dict(torch.load("deepfake_model_b1.pth", map_location=device))
    models['v2s'].load_state_dict(torch.load("deepfake_model_v2s.pth", map_location=device))
    models['ensemble'].load_state_dict(torch.load("EfficientNetV2S_ViT_GCN_ensemble.pth", map_location=device))
    
    # 모든 모델을 평가 모드로 설정
    for model in models.values():
        model.eval()

    print("모든 모델 로딩 완료.")
except FileNotFoundError as e:
    print(f"오류: 모델 파일(.pth)을 찾을 수 없습니다. {e}")
    exit()

# 🔍 Gemini API 호출 함수
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
        payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_data}}]}]}
        response = requests.post(GEMINI_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Gemini API 오류: {str(e)}"

# --- 4. Flask 라우팅 ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        model_choice = request.form.get('model_type')

        if not file or not file.filename:
            flash("이미지 파일이 선택되지 않았습니다.")
            return render_template('index.html')
        
        if not model_choice:
            flash("모델이 선택되지 않았습니다.")
            return render_template('index.html')

        try:
            filename = file.filename
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            image = Image.open(img_path).convert('RGB')
            
            image_tensor = transforms_dict[model_choice](image).unsqueeze(0).to(device)
            
            selected_model = models[model_choice]

            with torch.no_grad(), autocast(device_type=device.type):
                output = selected_model(image_tensor)
                probs = F.softmax(output, dim=1)
                
                pred = torch.argmax(probs, dim=1).item()
                confidence = f"{probs[0][pred].item() * 100:.2f}%"
                result = "Fake" if pred == 1 else "Real"
            
            model_names = {
                'b1': 'EfficientNet-B1',
                'v2s': 'EfficientNet-V2S',
                'ensemble': 'Ensemble (Eff+ViT+GCN)'
            }
            model_display_name = model_names.get(model_choice)

            gemini_opinion = ask_gemini_about_image(img_path)

            return render_template('index.html',
                                   result=result,
                                   confidence=confidence,
                                   filename=filename,
                                   model_display_name=model_display_name,
                                   selected_model=model_choice,
                                   gemini_opinion=gemini_opinion)

        except Exception as e:
            flash(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
            return render_template('index.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)