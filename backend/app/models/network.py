import torch
from facenet_pytorch import MTCNN
from pathlib import Path

from ai.modules.predictor import DeepfakePredictor
from ai.modules.restorer import FaceRestorer


_BASE_DIR = Path(__file__).resolve().parents[3]

_MODEL_PATHS = {
    "kr": _BASE_DIR / "ai/models/mobilenetv3_deepfake_final.pth",
    "foreign": _BASE_DIR / "ai/models/mobilenetv3_deepfake_final_foriegn2.pth",
    "restorer": _BASE_DIR / "ai/models/RealESRGAN_x4plus.pth",
}

_predictor_kr = None
_predictor_foreign = None
_restorer = None
_mtcnn = None


def load_models():
    global _predictor_kr, _predictor_foreign, _restorer, _mtcnn

    if _predictor_kr is not None:
        return  # 이미 로드됨

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _predictor_kr = DeepfakePredictor(str(_MODEL_PATHS["kr"]))
    _predictor_foreign = DeepfakePredictor(str(_MODEL_PATHS["foreign"]))
    _restorer = FaceRestorer(str(_MODEL_PATHS["restorer"]))
    _mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])

    print("✅ [INFO] 모든 모델 lazy load 완료")


def get_models():
    if _predictor_kr is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _predictor_kr, _predictor_foreign, _restorer, _mtcnn
