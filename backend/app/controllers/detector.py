import os
from fastapi import status
from fastapi.responses import JSONResponse, FileResponse
from fastapi import UploadFile, File,Form, Request

from backend.app.services.detect import predict_fake, generate_heatmap_report
async def detection_results(
    file: UploadFile = File(...),
    model_type: str = Form("korean")
):
    try:
        res = await predict_fake(file, model_type)
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

async def generate_report(request:Request):
    try:
        pdf_path = await generate_heatmap_report(request)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일 생성 실패: {pdf_path}")

        return FileResponse(
            pdf_path,
            filename=os.path.basename(pdf_path), media_type="application/pdf"
            )
    except Exception as e:
        return JSONResponse(
            {
                "message":"테스트 실패",
                "error": str(e),
            },
            status_code=status.HTTP_404_NOT_FOUND
        )
