import os
from fastapi import UploadFile, File, Form, Request, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.detect import (
    predict_fake,
    generate_heatmap_report,
)


async def detection_results(
    file: UploadFile,
    model_type: str,
):
    return await predict_fake(file, model_type)

async def generate_report(request: Request):
    try:
        pdf_path = await generate_heatmap_report(request)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        return FileResponse(
            pdf_path,
            filename=os.path.basename(pdf_path),
            media_type="application/pdf",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
