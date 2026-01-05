import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# =====================================================
# 1️⃣ Grad-CAM 클래스
# =====================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)
        return cam


# =====================================================
# 2️⃣ 이미지 로드 및 전처리
# =====================================================
def load_image(image_path, img_size=224):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor

# =====================================================
# 3️⃣ 숫자 레이어 생성 (★ 제거)
# =====================================================
def generate_number_layer(cam, img_shape, grid_size=8):
    layer = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    cam_resized = cv2.resize(cam, (img_shape[1], img_shape[0]))
    h, w = cam_resized.shape
    step_h, step_w = h // grid_size, w // grid_size
    font_scale = 0.5 * (step_h / 20)
    for i in range(grid_size):
        for j in range(grid_size):
            y, x = i * step_h + step_h // 2, j * step_w + step_w // 2
            score = cam_resized[y, x]
            number = int(np.clip(score * 9, 0, 9))
            cv2.putText(layer, str(number), (x - step_w // 4, y + step_h // 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return layer

# =====================================================
# 4️⃣ Grad-CAM 오버레이
# =====================================================
def overlay_cam_on_image(img, cam):
    img = np.array(img).astype(np.uint8)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# =====================================================
# 5️⃣ 통합 보고서 생성
# =====================================================
def generate_integrated_report(pred_label, confidence):
    if pred_label == "Real":
        report = (
            f"이 이미지는 Real로 분류되었으며, 예측신뢰도는 {confidence:.2f}%입니다.\n"
            f"자연스러운 질감, 조명, 윤곽선 패턴이 실제 얼굴의 특징으로 인식되었습니다."
        )
    else:
        report = (
            f"이 이미지는 Fake로 분류되었으며, 예측신뢰도는 {confidence:.2f}%입니다.\n"
            f"비정상적인 질감, 경계선 왜곡, 조명 불균형 등 딥페이크 생성 흔적이 감지되었습니다."
        )
    return report

# =====================================================
# 6️⃣ 시각화
# =====================================================
def visualize_result(image, overlay, number_layer, report):
    final_overlay = cv2.addWeighted(overlay, 0.8, number_layer, 0.8, 0)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("원본 이미지")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(final_overlay)
    plt.title("Grad-CAM + 숫자")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.text(0, 0.5, report, fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()

# =====================================================
# 7️⃣ 분석 및 반환 기능
# =====================================================
def analyze_image_with_model_type(image_path, model_type="korean", visualize=True):
    model_paths = {
        "korean": "C:/AI/project/AdvancedProject/DeepfakeHunters/ai/models/mobilenetv3_deepfake_final.pth",
        "foriegn": "C:/AI/project/AdvancedProject/DeepfakeHunters/ai/models/mobilenetv3_deepfake_final_foriegn2.pth"
    }

    if model_type not in model_paths:
        raise ValueError("model_type은 'korean' 또는 'foriegn'만 가능합니다.")

    model_path = model_paths[model_type]
    print(f"✅ 선택된 모델: {model_type} ({model_path})")

    # 모델 로드
    #model = models.mobilenet_v3_small(weights=None)
    model = models.mobilenet_v3_small(pretrained=False)    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    # 이미지 처리 및 예측
    image, input_tensor = load_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100
        pred_label = ["Fake", "Real"][pred_idx]

    # Grad-CAM
    target_layer = model.features[-1]
    cam_generator = GradCAM(model, target_layer)
    cam = cam_generator.generate(input_tensor, pred_idx)

    overlay = overlay_cam_on_image(image, cam)
    number_layer = generate_number_layer(cam, np.array(image).shape)
    report = generate_integrated_report(pred_label, confidence)

    if visualize:
        visualize_result(image, overlay, number_layer, report)

    return pred_label, confidence, report

# =====================================================
# 8️⃣ 실행 예시
# =====================================================
if __name__ == "__main__":
    image_path = "C:/AI/project/AdvancedProject/DeepfakeHunters/frontend/public/test_images/detect/test1.jpg"

    # 'korean' 또는 'foriegn' 선택
    model_type = "foriegn"

    label, conf, report = analyze_image_with_model_type(image_path, model_type=model_type)
    print(f"\n예측 결과: {label} ({conf:.2f}%)")
    print(report)