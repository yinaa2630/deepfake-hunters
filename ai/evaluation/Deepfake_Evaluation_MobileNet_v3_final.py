"""
===============================================================
🎯 MobileNetV3-Small 기반 딥페이크 판별 모델 평가 전용 코드 (수정 완료)
---------------------------------------------------------------
✅ 주요 기능:
1. 학습된 모델 불러오기 (mobilenetv3_deepfake_cpu.pth 등)
2. 테스트 세트 성능 평가 (정확도, 리포트, 혼동행렬)
3. Grad-CAM 시각화 (모델 주시 영역 분석)
---------------------------------------------------------------
⚙️ 환경: PyTorch, torchvision, scikit-learn, OpenCV, Matplotlib
===============================================================
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import random
import matplotlib.font_manager as fm

# ==============================================================  
# 1️⃣ 기본 설정  
# ==============================================================
BASE_DIR = "C:/AI/project/AdvancedProject/Deepfake_test/ai/modelling_jrheo"
MODEL_PATH = "C:/AI/project/AdvancedProject/Deepfake_test/ai/modelling_jrheo/evaluation/mobilenetv3_deepfake_jrheo.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 8

print(f"📁 데이터 경로: {BASE_DIR}")
print(f"💾 모델 경로: {MODEL_PATH}")
print(f"💻 디바이스: {DEVICE}")

# ==============================================================  
# 2️⃣ 데이터 로드  
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dir = os.path.join(BASE_DIR, "test")
test_ds = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
# ⚠️ 중요: ImageFolder는 알파벳순으로 클래스 정렬함
# 즉, ['Fake', 'Real'] 순서일 가능성이 높음
print(f"✅ 클래스 매핑: {test_ds.class_to_idx}")
print(f"✅ 클래스 순서: {test_ds.classes}")

# ==============================================================  
# 3️⃣ 모델 불러오기  
# ==============================================================
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(test_ds.classes))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"✅ 모델 로드 완료: {MODEL_PATH}")

# ==============================================================  
# 4️⃣ 모델 평가  
# ==============================================================
print("\n📈 테스트 세트 평가 시작...")

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 클래스 이름 자동 매칭
class_names = test_ds.classes

# 리포트 및 혼동행렬
import pandas as pd
pd.Series(y_true).value_counts()
report = classification_report(y_true, y_pred, target_names=class_names)
cm = confusion_matrix(y_true, y_pred)
acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100

print("\n📊 Classification Report:\n", report)
print(f"🎯 Test Accuracy: {acc:.2f}%")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix (Acc: {acc:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ==============================================================  
# 5️⃣ Grad-CAM 정의  
# ==============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ==============================================================  
# 6️⃣ Grad-CAM 시각화  
# ==============================================================
try:
    font_name = fm.FontProperties(fname=fm.findfont('Malgun Gothic')).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("⚠️ 폰트를 설정하지 못했습니다. 기본 폰트 사용.")

test_fake_dir = os.path.join(test_dir, "Fake")
if os.path.exists(test_fake_dir) and len(os.listdir(test_fake_dir)) > 0:
    random_img = random.choice(os.listdir(test_fake_dir))
    test_image_path = os.path.join(test_fake_dir, random_img)

    print("\n" + "="*50)
    print(f"🔎 Grad-CAM 시각화")
    print("="*50)
    print(f"🎞️ 테스트 이미지: {test_image_path}")

    img = Image.open(test_image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred = probs.argmax().item()
    confidence = probs[pred].item() * 100
    pred_label = class_names[pred]

    print(f"🧠 예측 결과: {pred_label} ({confidence:.2f}%)")

    target_layer = model.features[-1]
    cam_generator = GradCAM(model, target_layer)
    cam = cam_generator.generate(input_tensor, pred)

    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("1️⃣ 원본 이미지")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("2️⃣ Grad-CAM 히트맵")
    axes[1].axis("off")

    axes[2].imshow(overlay[..., ::-1])
    axes[2].set_title(f"3️⃣ 오버레이 결과 ({pred_label}, {confidence:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_overlay.jpg")    # 저장하기
    plt.show()
else:
    print("⚠️ 테스트용 Fake 이미지가 없습니다.")
    
# ==============================================================  
# (추가) 결과 리포트 작성 목적
# ==============================================================

os.chdir("C:/AI/project/AdvancedProject/Deepfake_test/ai/modelling_jrheo/evaluation")
from evaluation_summary import save_evaluation_results
save_evaluation_results(y_true, y_pred, class_names)
