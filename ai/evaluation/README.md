# 딥페이크 판별 결과 리포트 생성(랭체인 기술 이용)

1. 폴더 구조
   ├── evaluation/ ← 모델 평가 및 성능 분석 관련 폴더
   │ ├── Deepfake*Evaluation_MobileNet_v3_final.py ← (*) 모델 평가 코드 (기존)
   │ ├── mobilenetv3*deepfake_jrheo.pth ← (*) 학습된 모델 (기존, 실제 분석된 결과 추가할것) !!!!!!!!!!
   │ ├── evaluation*summary.py ← (*) Step1: 평가 결과 저장 모듈
   │ ├── evaluation_result.json ← 평가 결과(JSON)
   │ ├── deepfake_report.py ← Step2: LangChain 리포트 생성 코드
   │ ├── generate_full_report.py ← Step2 실행용 통합 스크립트
   │ ├── confusion_matrix.png ← 혼동행렬 이미지 (자동 생성)
   │ ├── gradcam_overlay.jpg ← Grad-CAM 이미지 (자동 생성)
   │ └── Deepfake_Evaluation_Report.pdf ← 최종 PDF 보고서 (자동 생성)

- 주의 : 아래 3개의 파일은 동일 폴더 내 존재해야 함
  - Deepfake_Evaluation_MobileNet_v3_final.py
  - mobilenetv3_deepfake.pth
  - evaluation_summary.py
