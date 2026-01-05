# 🧠 DeepfakeHunters

AI 기반 얼굴 복원 및 딥페이크 탐지 웹서비스

---

---

## 🧭 Branch Strategy

- `main`: 최종 배포용 브랜치 (발표 / 결과물 관리) | 조장만 병합
- `dev`: 통합 브랜치 (모든 PR은 여기로) | 전원 작성 가능
- `yunsujin`, `hrlee`, `jrheo`, `leeys` : 개인 작업 브랜치 | 개인별 작업 전용

## 🧱 Workflow

1. 개인 브랜치에서 작업
2. 변경사항 커밋
3. 원격 저장소로 push
4. GitHub에서 Pull Request 생성 (대상 브랜치: dev)
5. 조장이 dev에서 통합 테스트 후 main으로 병합
   🔹 자세한 사용 예시 및 명령어 순서는 docs/\_git bash 사용법.txt를 참고하세요.

## ⚙️ Local Environment Setup (using Conda)

모든 팀원이 동일한 개발 환경에서 FastAPI를 실행하기 위한 설정 가이드입니다.

### 🧱 Step-by-Step

```bash
# 1️⃣ Conda 환경 생성
conda env create -f environ/environment.yml
# (추후 모델링용 라이브러리 – e.g., deepface, insightface, gfpgan, onnxruntime 등 – 추가 예정)

# 2️⃣ 환경 활성화
conda activate deepfakehunters

# 3️⃣ 환경 삭제
conda env remove -n deepfakehunters
```

#### 초기 설정

---

---

🧠 DeepfakeHunters 초기 환경 설정 가이드

# 1. Docker 실행 및 DB 세팅

🧩 (1) Docker Desktop 실행
Docker Desktop을 먼저 실행합니다.
(백그라운드에서 컨테이너가 정상 동작해야 합니다.)

🧱 (2) MySQL 컨테이너 빌드 (최초 1회만 실행)

```bash
# 초기 실행 시 세팅하는데 시간 소요
cd db
docker-compose up -d
```

⚠️ 위 명령은 최초 1회만 실행합니다.

🗄️ (3) DB 연결 정보

```bash
# 항목 설정값
Host 192.168.0.33
Port 3306
User root
Password 1234
Database deepfake_db
```

# 2 백엔드 환경 설정 (최초 1회)

1️⃣ 가상환경 활성화

```bash
conda activate deepfakehunters
```

2️⃣ .env 파일 생성
(📁 위치: /backend/.env)

```bash
DB_URL=mysql+pymysql://root:1234@192.168.0.33:3306/deepfake_db
OPENAI_API_KEY=temp
```

# 2 프런트엔드 환경 설정 (최초 1회)

1️⃣ 필수 패키지 설치

```bash
cd frontend
npm install
```

2️⃣ .env 파일 생성
(📁 위치: ./frontend/.env)

```bash
REACT_APP_API_URL=http://192.168.0.33:8000
```

#### 이후 실행 순서 (매번 실행시)

---

---

1️⃣ Docker Desktop 실행

2️⃣ 백엔드 서버 실행

```bash
python backend/main.py
```

```bash
# docker로 구동 시
docker-compose -f environ/docker-compose-backend.yml up -d
# 다시 build 해야하는 경우
docker-compose -f environ/docker-compose-backend.yml build
```

3️⃣ 프론트엔드 실행

```bash
cd frontend
npm run start
```

4️⃣ NGINX 실행

```bash
docker-compose -f environ/docker-compose-nginx.yml up -d
```
