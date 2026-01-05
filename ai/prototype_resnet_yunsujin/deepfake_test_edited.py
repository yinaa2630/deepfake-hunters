# Path: ai/prototype_resnet_yunsujin/deepfake_test_edited.py
# Desc: 생성한 변형 이미지(Edited)에 대해 학습된 모델을 활용해 Real/Fake 예측
import os
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ==================================================
# 1️⃣ 경로 설정 (! 각자 데이터셋 있는 곳으로 변경해야 함 !)
# ==================================================
model_path = r"E:\yun\DeepfakeHunters\ai\prototype_resnet_yunsujin\output\model_resnet_best.pth"
test_dir = r"E:\yun\251027_DeepfakeTest\dataset\Dataset\Test\Edited"

# ==================================================
# 2️⃣ 모델 구조 재정의 및 가중치 로드
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("✅ 모델 불러오기 완료:", model_path)
print("💻 사용 디바이스:", device)

# ==================================================
# 3️⃣ 전처리 정의 (학습과 동일하게)
# ==================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ==================================================
# 4️⃣ 테스트 실행
# ==================================================
summary = {"total": 0, "pred_real": 0, "pred_fake": 0}

if not os.path.exists(test_dir):
    print(f"❌ 테스트 폴더가 존재하지 않습니다: {test_dir}")
    exit()

files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
print(f"📂 테스트할 이미지 수: {len(files)}")

for fname in tqdm(files, desc="Testing Edited Images"):
    fpath = os.path.join(test_dir, fname)

    try:
        img = Image.open(fpath).convert("RGB")
    except Exception as e:
        print("⚠️ 이미지 로드 실패:", fpath, e)
        continue

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
        pred = torch.argmax(out, dim=1).item()  # 0=Real, 1=Fake

    summary["total"] += 1
    if pred == 0:
        summary["pred_real"] += 1
    else:
        summary["pred_fake"] += 1

# ==================================================
# 5️⃣ 결과 출력
# ==================================================
print("\n📊 [Edited 폴더 판별 결과]")
print(f"총 {summary['total']}장 테스트 완료")
print(f"→ Real로 예측: {summary['pred_real']}")
print(f"→ Fake로 예측: {summary['pred_fake']}")

print("\n✅ deepfake_test_edited_yunsujin.py 실행 완료")
