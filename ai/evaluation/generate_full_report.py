# Step 3. 실행용 통합 스크립트

import os
import numpy as np
from langchain_openai import OpenAI
from deepfake_report import interpret_results, plot_confusion_matrix, generate_pdf

# ============================================================
# 1. LLM이 성능 데이터를 해석
# ============================================================
report_text, data = interpret_results("evaluation_result.json")
report_text = report_text.content if hasattr(report_text, "content") else str(report_text)


# ============================================================
# 2. 혼동행렬 시각화
# ============================================================
cm = np.array(data["confusion_matrix"])
cm_image = plot_confusion_matrix(cm, data["classes"])


# ============================================================
# 3. Grad-CAM 이미지 
# ============================================================
gradcam_path = "gradcam_overlay.jpg"  


# ============================================================
# 4. PDF 리포트 생성
# ============================================================ 
generate_pdf(report_text, cm_image, gradcam_image=gradcam_path)
