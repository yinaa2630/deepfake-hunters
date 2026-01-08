from fastapi import APIRouter, UploadFile, File
from backend.app.controllers import restore

router = APIRouter(
    prefix="/restore",
    tags=["Restore"],
)

@router.post("/")
async def restore_image(
    file: UploadFile = File(...),
):
    return await restore.restoration_results(file)
