# Path: ai/old_photo_restoration/restore_old_photos_fast.py
# Desc: Real-ESRGAN 기반 옛사진 복원 (CPU 전용, 리사이즈 + 타일 최적화)

import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --------------------------------------------------
# 1️⃣ 경로 설정
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "samples")
output_dir = os.path.join(BASE_DIR, "output")
model_path = os.path.join(BASE_DIR, "RealESRGAN_x4plus.pth")

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# 2️⃣ 모델 로드 (CPU 전용)
# --------------------------------------------------
device = torch.device("cpu")

model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=4
)

restorer = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=128,       # ✅ 타일 활성화 (메모리 절약)
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

print(f"📂 입력 폴더: {input_dir}")
print(f"💾 출력 폴더: {output_dir}")
print("🚀 복원 시작 (리사이즈 + tile 모드)...\n")


# --------------------------------------------------
# 3️⃣ 이미지 복원 (리사이즈 추가)
# --------------------------------------------------
files = [f for f in os.listdir(input_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(files, desc="Restoring"):
    fpath = os.path.join(input_dir, fname)
    try:
        img = Image.open(fpath).convert("RGB")

        # ✅ 긴 변 기준으로 리사이즈 (512px 이하)
        max_size = 512
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # ✅ 복원 수행
        img_np = np.array(img)
        sr_img, _ = restorer.enhance(img_np)

        # ✅ NumPy → PIL 변환 후 저장
        sr_pil = Image.fromarray(sr_img)
        save_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_restored.jpg")
        sr_pil.save(save_path)

    except Exception as e:
        print(f"⚠️ {fname} 처리 중 오류 발생: {e}")

print("\n✅ 모든 이미지 복원 완료 (Fast Mode)!")