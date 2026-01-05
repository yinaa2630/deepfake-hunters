# 🧠 Old Photo Restoration (Real-ESRGAN CPU Fast Ver.)
이 디렉토리는 Real-ESRGAN을 활용한 인물 사진 복원 예제입니다.  
GPU 없이 CPU 전용 환경에서 동작하도록 최적화된 버전입니다.

---

## 📁 폴더 구조
```bash
old_photo_restoration/
├── restore_old_photos_fast.py      # 복원 코드 (CPU 최적화 버전)
├── RealESRGAN_x4plus.pth           # 모델 가중치 (별도 다운로드 필요)
├── samples/                        # 테스트용 이미지 (7장)
└── output/                         # 복원된 결과 (자동 생성)
```

## 모델 관련
⚠️ RealESRGAN_x4plus.pth 파일은 용량이 크므로 GitHub에 직접 업로드하지 마세요.
대신 아래 링크에서 직접 다운로드 후, 이 디렉토리에 넣어주세요.

RealESRGAN_x4plus.pth (공식 릴리즈)
🔗 **모델 다운로드:**  
[RealESRGAN_x4plus.pth (공식 릴리즈)](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

---

## 🚀 실행 방법
```bash
# 1️⃣ 독립된 Conda 환경 생성
conda deactivate   # optional; 기존 환경이 활성화 되어있으면 비활성화
conda create -n realesrgan python=3.10 -y
conda activate realesrgan

# 2️⃣ 필요한 패키지 설치
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install realesrgan basicsr facexlib pillow tqdm
# ⚠️ NumPy 오류 발생 시 (Numpy is not available)
# 일부 환경에서는 NumPy 2.x가 기본 설치되어 PyTorch와 충돌이 납니다.
# 아래 명령으로 NumPy 버전을 1.26.4로 고정하세요.
pip uninstall -y numpy
pip install numpy==1.26.4

# 3️⃣ 복원 실행
cd ai/old_photo_restoration
python restore_old_photos_fast.py

```

## 이미지
예시 이미지 출처:
한국민속대백과사전 - 전통 혼례 사진 (국립민속박물관)
https://folkency.nfm.go.kr/multimedia/photo/74249/7265

본 예시는 연구 및 복원 테스트용으로만 사용되었습니다.


===================================================================================
🧠 프로젝트 내 역할 (딥페이크와 복원 연관성)
old_photo_restoration 디렉토리는 DeepfakeHunters 프로젝트의 확장 실험 모듈로,
딥페이크 탐지와 직접적으로 연결되는 “얼굴 이미지 복원” 파트를 다룹니다.
딥페이크 데이터셋에는 종종 낮은 해상도, 손상된 프레임, 노이즈가 많은 영상이 포함되어 있는데,
이때 Real-ESRGAN을 활용해 원본 얼굴 품질을 향상시키면
탐지 모델이 얼굴 특징을 더 명확히 학습할 수 있습니다.

🎯 목적

AI 탐지 모델에 입력되는 얼굴 이미지의 선명도 향상
복원된 프레임을 통해 훈련 데이터 품질 개선
역사적 이미지나 구형 영상 복원에도 확장 가능 (예: 한국 전통 인물 사진 등)

🔗 사용 예시
이 모듈은 독립 실행도 가능하지만,
향후 딥페이크 데이터 전처리 단계(data_preprocessing)에
Real-ESRGAN 기반 복원 함수로 통합될 예정입니다.
즉, “얼굴 검출 → 복원 → 탐지”의 전체 파이프라인 중
복원 단계를 담당합니다.