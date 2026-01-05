"""
===============================================================
🎯 MobileNetV3-Small 기반 딥페이크 판별 통합 파이프라인
---------------------------------------------------------------
✅ 주요 기능:
1. 데이터 자동 전처리 (train/val/test 분리)
2. MobileNetV3-Small 파인튜닝
3. Early Stopping 적용 (patience=10)
4. 학습곡선(loss/accuracy) 시각화
5. 테스트 세트 평가 + Grad-CAM 시각화
===============================================================
"""

import os, random, shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import cv2
%matplotlib auto

# ==============================================================
# 1️⃣ 경로 및 기본 설정
# ==============================================================
BASE_DIR = "D:/AI_DEV_Course/Work_space/PROJECT/Advanced_Project_Team2/Model(MobileNet)"
FAKE_SRC = os.path.join(BASE_DIR, "E:/Deepfake_Image_AIhub/Dataset_deepfake_cropped/fake_images")
REAL_SRC = os.path.join(BASE_DIR, "E:/Deepfake_Image_AIhub/Dataset_deepfake_cropped/real_images")
MODEL_PATH = os.path.join(BASE_DIR, "D:/AI_DEV_Course/Work_space/PROJECT/Advanced_Project_Team2/Model(MobileNet)/mobilenetv3_deepfake_final.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
PATIENCE = 10

print(f"📁 데이터 경로: {BASE_DIR}")
print(f"💾 모델 저장: {MODEL_PATH}")
print(f"💻 디바이스: {DEVICE}")

# ==============================================================
# 2️⃣ 데이터 전처리: train/val/test 자동 생성
# ==============================================================
def prepare_dataset(base_dir):
    print("🚀 데이터 전처리 시작...")

    dest_dirs = {
        "train": {"Fake": os.path.join(base_dir, "train", "Fake"),
                  "Real": os.path.join(base_dir, "train", "Real")},
        "val": {"Fake": os.path.join(base_dir, "val", "Fake"),
                "Real": os.path.join(base_dir, "val", "Real")},
        "test": {"Fake": os.path.join(base_dir, "test", "Fake"),
                 "Real": os.path.join(base_dir, "test", "Real")},
    }

    for split, paths in dest_dirs.items():
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

    def split_and_copy(src_dir, label):
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        n_test = n_total - n_train - n_val

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        print(f"\n📊 [{label}] 총 {n_total}장 → train:{n_train}, val:{n_val}, test:{n_test}")
        for split, file_list in splits.items():
            dest_dir = dest_dirs[split][label]
            for fname in tqdm(file_list, desc=f"{label} → {split}", unit="img"):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dest_dir, fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

    split_and_copy(FAKE_SRC, "Fake")
    split_and_copy(REAL_SRC, "Real")

    print("\n🎉 데이터 전처리 완료!")

prepare_dataset(BASE_DIR)

# ==============================================================
# 3️⃣ 데이터 로더 구성
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(BASE_DIR, "train"), transform=transform)
val_ds = datasets.ImageFolder(os.path.join(BASE_DIR, "val"), transform=transform)
test_ds = datasets.ImageFolder(os.path.join(BASE_DIR, "test"), transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

label_map = {v: k for k, v in train_ds.class_to_idx.items()}
print(f"✅ 클래스 매핑: {label_map}")

# ==============================================================
# 4️⃣ Class Weight 계산 (데이터 불균형 보정)
# ==============================================================
from collections import Counter
cls_counts = Counter([label for _, label in train_ds.samples])
weights = torch.tensor([0.7, 1.0], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
print(f"⚖️ 클래스 가중치: {weights}")

# ==============================================================
# 5️⃣ 모델 정의
# ==============================================================
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(label_map))
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# ==============================================================
# 6️⃣ 학습 루프 + Early Stopping + 학습곡선
# ==============================================================
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"📉 Loss: {train_loss:.4f}/{val_loss:.4f} | 🎯 Acc: {train_acc*100:.2f}%/{val_acc*100:.2f}%")

    # Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

    scheduler.step()

# 학습곡선 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend(); plt.title("Accuracy Curve")
plt.show()

# ==============================================================
# 7️⃣ 테스트 세트 평가
# ==============================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=list(label_map.values()))
cm = confusion_matrix(y_true, y_pred)
acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100

print("\n📊 Classification Report:\n", report)
print(f"🎯 Test Accuracy: {acc:.2f}%")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
plt.title(f"Confusion Matrix (Acc: {acc:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ==============================================================
# 8️⃣ Grad-CAM 시각화
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

# 샘플 이미지 Grad-CAM
test_fake_dir = os.path.join(BASE_DIR, "test", "Fake")
if os.path.exists(test_fake_dir) and len(os.listdir(test_fake_dir)) > 0:
    img_name = random.choice(os.listdir(test_fake_dir))
    img_path = os.path.join(test_fake_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred = probs.argmax().item()
    confidence = probs[pred].item() * 100

    print(f"\n🎞️ 예측 결과: {label_map[pred]} ({confidence:.2f}%)")

    cam_gen = GradCAM(model, model.features[-1])
    cam = cam_gen.generate(input_tensor, pred)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay[..., ::-1])
    plt.title(f"{label_map[pred]} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()
else:
    print("⚠️ 테스트용 Fake 이미지가 없습니다.")
