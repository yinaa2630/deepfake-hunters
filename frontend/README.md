# 🧠 Frontend Setup (React)

```bash
# 1. Node.js 버전 확인
node -v

# 2. 새 React 프로젝트 생성
npx create-react-app frontend

# 3. 폴더 이동
cd frontend

# 4. 라우터 설치 (페이지 전환용)
npm install react-router-dom

# 5. 서버 실행
npm run start

# 6. 📂 구조 예시(수정할것)
└─ frontend/
   ├─ public/
   │   └─ index.html
   ├─ src/
   │   ├─ App.js
   │   ├─ index.js
   │   ├─ pages/
   │   ├─ components/
   │   ├─ styles/
   │   └─ api/
   └─ package.json

# 7. 서버 구조
 [브라우저] (localhost:3000)
        │
        │  POST /api/predict
        ▼
 [FastAPI 서버] (localhost:8000)

# React → FastAPI 간 요청은 포트가 다르기 때문에 CORS 설정 필요
# 즉, "서버 간 접근 허용" 설정을 FastAPI 쪽에서 해야 함
```
