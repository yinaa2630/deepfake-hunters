from fastapi import APIRouter, UploadFile, File, Request
from backend.app.controllers import detector

router = APIRouter(
    prefix="/detect",
    tags=["Detect"],
)

# ------------------------------------------------------
# 딥페이크 탐지
# POST /detect
# ------------------------------------------------------
@router.post("/")
async def detect_image(
    file: UploadFile = File(...),
    model_type: str = "korean",
):
    return await detector.detection_results(
        file=file,
        model_type=model_type,
    )

# ------------------------------------------------------
# Grad-CAM 리포트
# POST /detect/report
# ------------------------------------------------------
@router.post("/report")
async def generate_report(request: Request):
    return await detector.generate_report(request)
