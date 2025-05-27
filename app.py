import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from model_b1 import build_model as build_b1
from model_v2s import build_model as build_v2s

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'b1':
        model = build_b1()
        model.load_state_dict(torch.load("deepfake_model_b1.pth", map_location=device))
    elif model_type == 'v2s':
        model = build_v2s()
        model.load_state_dict(torch.load("deepfake_model_v2s.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        file = request.files['image']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            model, device = load_model(model_type)
            image = image.to(device)

            with torch.no_grad():
                output = model(image)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                result = "Fake" if pred == 1 else "Real"

            return render_template('index.html', result=result, filename=file.filename, model=model_type.upper())

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
