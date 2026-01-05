import io,os, uuid, datetime
from PIL import Image
import numpy as np

from config import config
from backend.app.models.newtowk import restorer

async def get_restoration_image(file):
    # 디렉토리
    base_dir = config['BASE_DIR']
    restore_dir = f"{base_dir}/data/restored"
    os.makedirs(restore_dir, exist_ok=True)

    # 복원 시작
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    restored = restorer.restore(np.array(image))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"{timestamp}_{unique_id}_restored{ext}"
    save_path = f"{restore_dir}/{safe_name}"
    Image.fromarray(restored).save(save_path)

    print(f"💾 [RESTORE] 복원 완료 → {save_path}")
    return {"restored_image_url": f"http://localhost:8000/data/restored/{safe_name}"}
