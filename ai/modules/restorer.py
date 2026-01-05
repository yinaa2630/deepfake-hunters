# Path: ai/modules/restorer.py
# Desc: RealESRGAN 복원용 최소 버전 (CPU 전용, 빠른 추론 + 진행 표시)

import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import warnings


class FaceRestorer:
    def __init__(self, model_path: str):
        try:

            # ✅ 불필요한 경고 숨기기 (타일 로그 포함)
            warnings.filterwarnings("ignore")

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=4
            )

            # ✅ CPU 경량 설정
            self.restorer = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=64,        # 🔽 작은 타일로 분할 (속도 개선)
                tile_pad=2,
                pre_pad=0,
                half=False,     # CPU 환경에서는 반드시 False
                device="cpu"    # GPU 미사용
            )

        except Exception as e:
            print(f"❌ [FaceRestorer INIT ERROR]: {e}")
            raise e

    def restore(self, image: np.ndarray):
        try:
            print("[RESTORER] 복원 진행 중...")  # ✅ 진행 표시 한 줄만 출력
            output, _ = self.restorer.enhance(image, outscale=1)  # 빠른 복원
            return output
        except Exception as e:
            print(f"❌ [RESTORE ERROR]: {e}")
            raise e
