# Step 1. 평가 결과를 저장하는 모듈

"""
1) 아래 3개의 파일 동일 폴더내 존재
   ("C:/AI/project/AdvancedProject/Deepfake_test/ai/modelling_jrheo/evaluation") 
   - Deepfake_Evaluation_MobileNet_v3_final.py (기존)
   - mobilenetv3_deepfake.pth (기존)
   - evaluation_summary.py (본 파일, 신규 생성)

2) 모델 평가 파일(Deepfake_Evaluation_MobileNet_v3_final.py) 맨 아래에 아래 코드 추가, 실행
   - os.chdir("C:/AI/project/AdvancedProject/Deepfake_test/ai/modelling_jrheo/evaluation")
   - from evaluation_summary import save_evaluation_results
   - save_evaluation_results(y_true, y_pred, class_names) 

"""

import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def save_evaluation_results(y_true, y_pred, class_names, output_path="evaluation_result.json"):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    
    result = {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "classes": class_names
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"평가 결과 저장 완료: {output_path}")
    return result

    