from fastapi import status
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File

from backend.app.services.restore import get_restoration_image
async def restoration_results(
    file: UploadFile = File(...),
):
    try:
        res = await get_restoration_image(file)
        return JSONResponse(    
            {"message":res},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            {
                "message":"테스트 실패",
                "error": str(e),
            },
            status_code=status.HTTP_404_NOT_FOUND
        )