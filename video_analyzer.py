import cv2
import numpy as np
from PIL import Image
import os
import tempfile

def is_video_or_gif(filename):
    allowed_extensions = ['.mp4', '.mov', 'avi', '.wmv', '.gif']
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def extract_frames(file_storage, frame_interval=15):
    frames = []
    file_storage.seek(0)
    
    # GIF 처리
    if file_storage.filename.lower().endswith('.gif'):
        with Image.open(file_storage) as img:
            for frame_index in range(img.n_frames):
                if frame_index % frame_interval == 0:
                    img.seek(frame_index)
                    frame_pil = img.convert('RGB')
                    frames.append(frame_pil)
        return frames

    # 동영상 처리 (임시 파일 사용으로 안정성 확보)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(file_storage.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    
    cap.release()
    os.remove(video_path)
    return frames

def analyze_frames_and_summarize(frames, models, transforms_dict, device, run_model_inference_tta):
    if not frames:
        return "분석 실패", "분석할 프레임을 찾을 수 없습니다.", "0.00%"

    predictions = []
    model_to_use = models['v2s']
    transform_to_use = transforms_dict['v2s']

    for frame_pil in frames:
        image_tensor = transform_to_use(frame_pil).unsqueeze(0).to(device)
        result, confidence, _ = run_model_inference_tta(image_tensor, model_to_use)
        predictions.append({
            "result": result,
            "confidence": float(confidence.replace('%',''))
        })

    total_frames = len(predictions)
    fake_frames = [p for p in predictions if p['result'] == 'Fake']
    fake_count = len(fake_frames)
    
    fake_ratio = fake_count / total_frames if total_frames > 0 else 0
    avg_fake_confidence = sum(p['confidence'] for p in fake_frames) / fake_count if fake_count > 0 else 0

    max_consecutive_fakes = 0
    current_consecutive_fakes = 0
    for p in predictions:
        if p['result'] == 'Fake':
            current_consecutive_fakes += 1
        else:
            max_consecutive_fakes = max(max_consecutive_fakes, current_consecutive_fakes)
            current_consecutive_fakes = 0
    max_consecutive_fakes = max(max_consecutive_fakes, current_consecutive_fakes)

    # ✨ 최종 판정 규칙 (더 정교하게 수정) ✨
    final_verdict = "Real"
    decision_reason = "대부분의 프레임이 'Real'로 판정되었습니다."

    if max_consecutive_fakes >= 2 and avg_fake_confidence > 80.0:
        final_verdict = "Fake"
        decision_reason = f"신뢰도 높은({avg_fake_confidence:.0f}%) 'Fake' 프레임이 {max_consecutive_fakes}개 이상 연속으로 발견되었습니다."
    elif fake_ratio > 0.4 and avg_fake_confidence > 75.0:
        final_verdict = "Fake"
        decision_reason = f"전체 프레임의 {fake_ratio:.0%}가 'Fake'이며, 평균 신뢰도({avg_fake_confidence:.0f}%)가 높아 조작 가능성이 있습니다."
    elif fake_ratio > 0.65:
        final_verdict = "Fake"
        decision_reason = f"전체 프레임의 {fake_ratio:.0%}가 'Fake'로 판정되어 조작된 영상일 가능성이 높습니다."

    summary = (f"총 {total_frames}개 프레임 분석: Fake {fake_count}개, Real {total_frames - fake_count}개\n"
               f"판정 근거: {decision_reason}")

    return final_verdict, summary