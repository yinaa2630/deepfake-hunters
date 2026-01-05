import torch
from facenet_pytorch import MTCNN

from ai.modules.predictor import DeepfakePredictor
from ai.modules.restorer import FaceRestorer
# ------------------------------------------------------
# 7️⃣ 모델 로드 (탐지 + 복원)
# ------------------------------------------------------
try:
    predictor_kr = DeepfakePredictor("ai/models/mobilenetv3_deepfake_final.pth")
    predictor_foreign = DeepfakePredictor("ai/models/mobilenetv3_deepfake_final_foriegn2.pth")
    restorer = FaceRestorer("ai/models/RealESRGAN_x4plus.pth")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])

    print("✅ [INFO] 한국인 탐지 모델 로드 완료")
    print("✅ [INFO] 외국인 탐지 모델 로드 완료")
    print("✅ [INFO] 복원 모델 로드 완료")
    print("✅ [INFO] 모든 모델 초기화 성공 (탐지 + 복원)")
except Exception as e:
    predictor_kr = predictor_foreign = restorer = None
    print(f"❌ [MODEL LOAD ERROR]: {e}")