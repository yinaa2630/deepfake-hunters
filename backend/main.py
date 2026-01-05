import uvicorn,os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import config
from backend.app.routes.detect import router as detect_router
from backend.app.routes.restore import router as restore_router

app = FastAPI(title="Deepfake Detection & Restoration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 필요 시 도메인 지정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, prefix="/detect")
app.include_router(restore_router, prefix="/restore")

# 6️⃣ 정적 파일 제공 (복원 결과 이미지 접근 허용)
os.makedirs(f"{config['BASE_DIR']}/data", exist_ok=True)
app.mount("/data", StaticFiles(
    directory=f"{config['BASE_DIR']}/data"), name="data")

# 8️⃣ 전역 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"서버 내부 오류: {str(exc)}"},
    )

if __name__ == "__main__":
    print(f"🚀 FastAPI 서버 실행 중 (http://{config['HOST']}:{config['PORT']})")
    uvicorn.run(
        "main:app",          # 모듈:앱 경로
        host=config["HOST"], 
        port=config["PORT"],
        reload=True,
    )