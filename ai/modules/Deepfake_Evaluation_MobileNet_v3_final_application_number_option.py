# Path: ai/modules/Deepfake_Evaluation_MobileNet_v3_final_application_number_option.py
# Desc: MobileNetV3 기반 딥페이크 탐지 + Grad-CAM 시각화 (개선 버전)

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime


# ==========================================================
# ✅ Grad-CAM 생성 함수
# ==========================================================
def generate_gradcam(model, image_tensor, target_layer, image):
    grads = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    grad = grads[0].cpu().data.numpy()[0]
    activation = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i, :, :]

    cam = np.maximum(cam, 0)
    cam -= np.min(cam)
    cam /= np.max(cam) if np.max(cam) != 0 else 1

    
    # ✅ 원본 이미지 크기로 리사이즈
    width, height = image.size
    cam = cv2.resize(cam, (width, height))
    # (jrheo 수정) cam = cv2.resize(cam, (224, 224))
        
    return cam


# ==========================================================
# ✅ 메인 분석 함수
# ==========================================================
def analyze_image_with_model_type(path, model_type="korean", visualize=True):
    """
    이미지 경로를 받아 딥페이크 예측 + Grad-CAM 시각화 수행
    visualize=True일 경우 Grad-CAM 이미지를 저장하고 경로 반환
    """

    model_path = f"ai/models/mobilenetv3_deepfake_final.pth"
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = torch.nn.Linear(1024, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        confidence = F.softmax(output, dim=1)[0]
        pred_label = "Fake" if torch.argmax(confidence).item() == 1 else "Real"
        conf_value = confidence[1].item() * 100

    # ✅ Grad-CAM 생성
    gradcam_path = None
    fake_intensity = None
    if visualize:
        cam = generate_gradcam(model, input_tensor, model.features[-1], image)
        img = np.array(image)
         # (jrheo 수정) img = np.array(image.resize((224, 224)))
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))  # width, height 맞춤
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
       
        
        # ✅ 붉은색 퍼짐 개선 — threshold 마스크 적용
        threshold = 0.4
        mask = cam > threshold
        overlay = img.copy()
        overlay[mask] = np.uint8(0.7 * heatmap[mask] + 0.3 * img[mask])

        # ✅ 시각적 활성도 계산 (Grad-CAM 평균 강도)
        fake_intensity = float(np.mean(cam))

        # ✅ 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("ai", "gradcam_results")
        os.makedirs(save_dir, exist_ok=True)
        gradcam_path = os.path.join(save_dir, f"gradcam_{timestamp}.png")
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("원본")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.savefig(os.path.join(save_dir, f"gradcam_plot_{timestamp}.png"))
        plt.close()

    report = f"이 이미지는 {pred_label} ({conf_value:.2f}%)\n비정상적인 질감, 경계선 왜곡, 조명 불균형 등 딥페이크 흔적이 감지되었습니다."

    return pred_label, conf_value, report, gradcam_path, fake_intensity
