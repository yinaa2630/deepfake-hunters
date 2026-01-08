from fastapi import UploadFile, File, HTTPException
from backend.app.services.restore import get_restoration_image


async def restoration_results(
    file: UploadFile = File(...),
):
    try:
        return await get_restoration_image(file)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
