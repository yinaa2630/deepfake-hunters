# Path: ai/prototype_resnet_yunsujin/deepfake_detect.py
# Desc: Real/Fake 이미지를 분류하기 위한 ResNet18 모델 생성

import os
import shutil
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# ==================================================
# 1️⃣ 샘플 데이터 복사 (500장씩) (! 각자 데이터셋 있는 곳으로 변경해야 함 !)
# ==================================================
src_root = r"E:\yun\251027_DeepfakeTest\dataset\Dataset\Test"
dst_root = r"E:\yun\251027_DeepfakeTest\dataset\sample_dataset"

for subdir in ["Real", "Fake"]:
    os.makedirs(os.path.join(dst_root, subdir), exist_ok=True)

print(f"📂 원본 폴더: {src_root}")
print(f"📂 샘플 폴더: {dst_root}")

for label in ["Real", "Fake"]:
    src = os.path.join(src_root, label)
    dst = os.path.join(dst_root, label)

    if not os.path.exists(src):
        print(f"⚠️ 경로 없음: {src}")
        continue

    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_files = random.sample(files, min(500, len(files)))

    for f in sample_files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

print("✅ 각 클래스당 500장 샘플 복사 완료\n")


# ==================================================
# 2️⃣ 데이터셋 & 모델 설정
# ==================================================
data_dir = dst_root

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 사용 중인 디바이스:", device)

# 모델 정의
model = models.resnet18(weights=None)  # 최신 버전 호환
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==================================================
# 3️⃣ 학습 루프
# ==================================================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}/{num_epochs} 완료 — 평균 Loss: {avg_loss:.4f}")

print("🎯 전체 학습 완료\n")

# ==================================================
# 4️⃣ 정확도 계산
# ==================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Evaluating Accuracy"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 학습 데이터 정확도: {accuracy:.2f}%\n")

# ==================================================
# 5️⃣ 모델 저장
# ==================================================
base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "model_resnet_best.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ 모델 저장 완료: {save_path}\n")

# ==================================================
# 6️⃣ 단일 이미지 테스트 (! 각자 데이터셋 있는 곳으로 변경해야 함 !)
# ==================================================
print("🔍 단일 이미지 테스트 중...")

# 모델 다시 불러오기
loaded_model = models.resnet18(weights=None)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 2)
loaded_model.load_state_dict(torch.load(save_path, map_location=device))
loaded_model.eval()

# 테스트 이미지
test_img_path = r"E:\yun\251027_DeepfakeTest\dataset\Dataset\Test\Fake\fake_0.jpg"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

img = Image.open(test_img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = loaded_model(input_tensor)
    _, predicted = torch.max(output, 1)
    label = "Fake" if predicted.item() == 1 else "Real"

print(f"🧠 예측 결과: {label}")
print("✅ deepfake_detect.py 실행 완료")
