# Path: ai/prototype_resnet_yunsujin/make_edited_dataset.py
# Desc: Real 이미지를 인위적으로 변형하여 Edited 데이터셋을 생성

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from tqdm import tqdm
import random
import io

# ==================================================
# 1️⃣ 경로 설정 (! 각자 데이터셋 있는 곳으로 변경해야 함 !)
# ==================================================
INPUT_DIR = r"E:\yun\251027_DeepfakeTest\dataset\Dataset\Test\Real"
OUTPUT_DIR = r"E:\yun\251027_DeepfakeTest\dataset\Dataset\Test\Edited"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 입력 폴더: {INPUT_DIR}")
print(f"💾 출력 폴더: {OUTPUT_DIR}")

# ==================================================
# 2️⃣ 유틸리티 함수 정의
# ==================================================
def save_pil(img_pil, path, quality=95):
    if path.lower().endswith(('.jpg', '.jpeg')):
        img_pil.save(path, format='JPEG', quality=quality)
    else:
        img_pil.save(path)

def gaussian_blur_cv(img_cv, ksize=(15, 15)):
    return cv2.GaussianBlur(img_cv, ksize, 0)

def add_gaussian_noise_cv(img_cv, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img_cv.shape).astype('float32')
    noisy = img_cv.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def jpeg_compress_pil(img_pil, quality=30):
    b = io.BytesIO()
    img_pil.save(b, format='JPEG', quality=quality)
    b.seek(0)
    return Image.open(b).convert('RGB')

def change_brightness_contrast_pil(img_pil, brightness=1.0, contrast=1.0):
    img = img_pil
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

def random_crop_resize_pil(img_pil, crop_frac=0.8):
    w, h = img_pil.size
    nw, nh = int(w * crop_frac), int(h * crop_frac)
    if nw < 10 or nh < 10:
        return img_pil
    x0 = random.randint(0, w - nw)
    y0 = random.randint(0, h - nh)
    cropped = img_pil.crop((x0, y0, x0 + nw, y0 + nh))
    return cropped.resize((w, h), Image.LANCZOS)

def copy_move_cv(img_cv):
    h, w = img_cv.shape[:2]
    cw, ch = max(20, w // 6), max(20, h // 6)
    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    patch = img_cv[y:y+ch, x:x+cw].copy()
    tx, ty = random.randint(0, w - cw), random.randint(0, h - ch)
    img2 = img_cv.copy()
    alpha = 0.8
    img2[ty:ty+ch, tx:tx+cw] = cv2.addWeighted(img2[ty:ty+ch, tx:tx+cw], alpha, patch, 1-alpha, 0)
    return img2

# ==================================================
# 3️⃣ 이미지 조작 수행
# ==================================================
rows = []
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(files, desc="Processing images"):
    fpath = os.path.join(INPUT_DIR, fname)
    try:
        img_pil = Image.open(fpath).convert('RGB')
    except Exception as e:
        print("열기 실패:", fpath, e)
        continue

    base_name, ext = os.path.splitext(fname)

    # 1) 원본 복사
    out_orig = os.path.join(OUTPUT_DIR, f"{base_name}__orig{ext}")
    save_pil(img_pil, out_orig)
    rows.append({'original': fname, 'edited': os.path.basename(out_orig), 'edit_type': 'orig'})

    # 2) Gaussian blur
    img_blur = gaussian_blur_cv(np.array(img_pil))
    out = os.path.join(OUTPUT_DIR, f"{base_name}__blur.jpg")
    cv2.imwrite(out, cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR))
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'gaussian_blur'})

    # 3) Gaussian noise
    img_noise = add_gaussian_noise_cv(np.array(img_pil), var=30)
    out = os.path.join(OUTPUT_DIR, f"{base_name}__noise.jpg")
    cv2.imwrite(out, cv2.cvtColor(img_noise, cv2.COLOR_RGB2BGR))
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'gaussian_noise'})

    # 4) JPEG 압축
    jpg = jpeg_compress_pil(img_pil, quality=20)
    out = os.path.join(OUTPUT_DIR, f"{base_name}__jpeg20.jpg")
    save_pil(jpg, out)
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'jpeg_q20'})

    # 5) 밝기/대비
    bright = change_brightness_contrast_pil(img_pil, brightness=0.6, contrast=1.1)
    out = os.path.join(OUTPUT_DIR, f"{base_name}__bright0.6_con1.1.jpg")
    save_pil(bright, out)
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'brightness_contrast'})

    # 6) 좌우 반전
    flipped = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    out = os.path.join(OUTPUT_DIR, f"{base_name}__fliplr{ext}")
    save_pil(flipped, out)
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'flip_lr'})

    # 7) 랜덤 크롭 & 리사이즈
    rc = random_crop_resize_pil(img_pil, crop_frac=random.uniform(0.6, 0.95))
    out = os.path.join(OUTPUT_DIR, f"{base_name}__randcrop{ext}")
    save_pil(rc, out)
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'random_crop_resize'})

    # 8) Copy-Move
    cm = copy_move_cv(np.array(img_pil))
    out = os.path.join(OUTPUT_DIR, f"{base_name}__copymove.jpg")
    cv2.imwrite(out, cv2.cvtColor(cm, cv2.COLOR_RGB2BGR))
    rows.append({'original': fname, 'edited': os.path.basename(out), 'edit_type': 'copy_move'})

# ==================================================
# 4️⃣ 메타데이터 저장 (! 각자 데이터셋 있는 곳으로 변경해야 함 !)
# ==================================================
df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 완료! Edited 이미지와 metadata.csv가 생성됨: {OUTPUT_DIR}")
