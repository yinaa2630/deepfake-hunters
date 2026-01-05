## Detect : Result에 히트맵 삽입

1. detect_service.py 수정 (detect_service.py의 더미 랜덤 결과를 삭제)
2. routes_detect.p 수정(Deepfake_Evaluation_MobileNet_v3_final_application_number.py의 analyze_image() 를 연결) / 한국인, 외국인도(model_type) 받을 수 있도록 수정

## Detect : PDF 생성 보고서 내용 수정(히트맵 분석 결과)

1. report_heatmap_service.py 신규 생성
   - 기존 LangChain 연동 포함 + 모델 개요 정보 자동 포함
   - backend/app/services/report_heatmap_service.py

## PDF 생성 관련 폴더 구조

DeepfakeHunters/
├── ai/
│ └── modules/
│ └── Deepfake_Evaluation_MobileNet_v3_final_application_number.py
├── backend/
├── app/

├── backend/
│ ├── main.py
│ ├── app/
│ │ └── services/
│ │ └── report_heatmap_service.py
│ ├── data/
│ │ └── test_images/test2.jpg
│ └── test_full_local_analysis.py ✅ 여기
└── ...

## OPEN_API_KEY

- 백엔드 폴더 내 .env에 저장(깃허브 커밋 불가)
  backend/
  ├── app/
  │ ├── main.py
  │ ├── api/
  │ │ ├── routes_detect.py
  │ │ ├── routes_upload.py
  │ │ └── routes_report.py ← 여기가 /api/report 라우트
  │ ├── services/
  │ │ ├── detect_service.py
  │ │ ├── upload_service.py
  │ │ └── report_heatmap_service.py ← ✅ LangChain PDF 생성 로직
  │ ├── core/
  │ │ ├── database.py
  │ │ └── cleanup.py
  │ └── models/
  │ └── db_models.py
  └── data/
  ├── uploads/
  ├── temp/
  ├── results/
  │ ├── images/ ← 히트맵 이미지 저장
  │ ├── pdfs/ ← PDF 보고서 저장
  │ └── logs/ ← JSON 로그 저장
  └── models/
