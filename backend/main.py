import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import config
from backend.app.routes.detect import router as detect_router
from backend.app.routes.restore import router as restore_router

app = FastAPI(title="Deepfake Detection & Restoration API")

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Router 등록 (❗ prefix 절대 주지 말 것)
# --------------------------------------------------
app.include_router(detect_router)
app.include_router(restore_router)

# --------------------------------------------------
# 정적 파일 제공
# --------------------------------------------------
os.makedirs(f"{config['BASE_DIR']}/data", exist_ok=True)
app.mount(
    "/data",
    StaticFiles(directory=f"{config['BASE_DIR']}/data"),
    name="data",
)

# --------------------------------------------------
# 전역 에러 핸들러
# --------------------------------------------------
# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     return JSONResponse(
#         status_code=500,
#         content={"error": f"서버 내부 오류: {str(exc)}"},
#     )

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 FastAPI 서버 실행 중 (http://{config['HOST']}:{config['PORT']})")
    uvicorn.run(
        "main:app",
        host=config["HOST"],
        port=config["PORT"],
        reload=True,
    )
