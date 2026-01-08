import io,os, uuid, datetime
from PIL import Image
import numpy as np

from config import config
from backend.app.models.network import load_models, get_models


async def get_restoration_image(file):
    """
    얼굴 복원 서비스
    """

    # 🔹 모델 lazy load
    load_models()
    _, _, restorer, _ = get_models()

    # 🔹 저장 디렉토리
    base_dir = config["BASE_DIR"]
    restore_dir = os.path.join(base_dir, "data", "restored")
    os.makedirs(restore_dir, exist_ok=True)

    # 🔹 이미지 로드
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 🔹 복원 수행
    restored = restorer.restore(np.array(image))

    # 🔹 파일 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    ext = os.path.splitext(file.filename)[1] or ".png"

    save_name = f"{timestamp}_{unique_id}_restored{ext}"
    save_path = os.path.join(restore_dir, save_name)

    Image.fromarray(restored).save(save_path)

    print(f"💾 [RESTORE] 복원 완료 → {save_path}")

    # 🔹 URL 반환
    return {
        "restored_image_url": f"http://localhost:8000/data/restored/{save_name}"
    }