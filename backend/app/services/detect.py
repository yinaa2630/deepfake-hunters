import io, os, base64, datetime, uuid, cv2, json
from fastapi import UploadFile
from PIL import Image, ImageOps
from fpdf import FPDF
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import config
from backend.app.models.newtowk import mtcnn
from ai.modules.Deepfake_Evaluation_MobileNet_v3_final_application_number_option import analyze_image_with_model_type

async def predict_fake(
    file: UploadFile, model_type: str = "korean") -> dict:
    # 디렉토리
    base_dir = config['BASE_DIR']
    upload_dir = f"{base_dir}/data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # 저장될 파일명
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"{timestamp}_{unique_id}{ext}"
    save_path = f"{upload_dir}/{safe_name}"

    # 파일 저장
    with open(save_path, "wb") as f:
        f.write(await file.read())
        
    # 수정필요(현재는 모델 타입과 무관하게 mobilenetv3_deepfake_final 모델 하나만 사용중)
    pred_label, confidence, report, gradcam_path, fake_intensity = analyze_image_with_model_type(
        path=save_path,
        model_type=model_type,
        visualize=True,
    )

    # ✅ Grad-CAM 이미지 base64 변환
    gradcam_b64 = None
    if gradcam_path and os.path.exists(gradcam_path):
        with open(gradcam_path, "rb") as f:
            gradcam_b64 = base64.b64encode(f.read()).decode("utf-8")

    # ✅ 결과 반환
    return {
        "pred_label": pred_label,
        "confidence": round(confidence, 2),
        "report": report,
        "gradcam": gradcam_b64,
        "image_path": save_path,
        "fake_probability": round(fake_intensity, 3) if fake_intensity else None,
        "model_type":model_type
    }


async def generate_heatmap_report(request):
    result = await request.json()
    print("🧾 [REPORT INPUT] 수신된 JSON:", result.keys())

    # ✅ 필수 필드 검증
    required_fields = ["gradcam", "result", "fake_probability", "model_type"]
    missing = [k for k in required_fields if k not in result]
    if missing:
        raise ValueError(f"필수 키 누락: {missing}")
    result_data = result
    
    # 디렉토리
    base_dir = config['BASE_DIR']
    result_dir = f"{base_dir}/data/results"
    image_dir = f"{result_dir}/images"
    pdf_dir = f"{result_dir}/pdfs"
    log_dir = f"{result_dir}/logs"
    for data_dir in [image_dir, pdf_dir, log_dir]:
        os.makedirs(data_dir, exist_ok=True)

    # GradCAM 저장
    gradcam_b64 = result_data["gradcam"]
    gradcam_bytes = base64.b64decode(gradcam_b64)
    gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"gradcam_{timestamp}.png"
    gradcam_path = f"{image_dir}/{image_filename}"
    gradcam_img.save(gradcam_path)

    prompt = PromptTemplate(
        input_variables=["result", "prob", "type"],
        template=(
            "너는 딥페이크 탐지 전문가야. 아래 정보를 기반으로 Grad-CAM 히트맵을 해석해.\n\n"
            "모델 유형: {type}\n"
            "예측 결과: {result}\n"
            "딥페이크 확률: {prob:.2f}%\n\n"
            "붉은색 영역은 모델이 딥페이크 판단의 근거로 본 부분이야. "
            "이 시각 정보를 기반으로 모델이 어떻게 판단했는지, "
            "합성 흔적·피부 질감·조명 왜곡 등 시각적 근거를 기술적으로 분석해줘. "
            "또한 인간 전문가의 관점에서 신뢰도와 한계점도 함께 설명해줘."
            "아울러, 언급하지 않은 심층 결과가 있으면 함께 상세히 설명해줘."
        ),
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    analysis_text = llm.invoke(
        prompt.format(
            type="한국인 이미지 분석 모델" if result_data["model_type"] == "korean" else "외국인 이미지 분석 모델",
            result=result_data["result"],
            prob=result_data["fake_probability"] * 100,
        )
    ).content

    # ------------------------------------------------------
    # 3. PDF 보고서 생성
    # ------------------------------------------------------
    pdf = FPDF()
    pdf.add_page()

    # 제목
    pdf.add_font("malgun", "", r"C:\Windows\Fonts\malgun.ttf", uni=True)   # 일반
    pdf.add_font("malgun", "B", r"C:\Windows\Fonts\malgunbd.ttf", uni=True)   # 굵은체
    pdf.set_font("malgun", "B", size=16)
    pdf.cell(0, 10, "딥페이크 히트맵 분석 보고서", ln=True, align="C")

    # (1) 분석 개요
    pdf.set_font("malgun", "B", size=13)
    pdf.cell(0, 10, "1. 분석 개요", ln=True)
    pdf.set_font("malgun", size=11)

    model_name = result_data.get("model_name", "MobileNetV3-Small (PyTorch)")
    model_type = "한국인 전용 모델" if result_data["model_type"] == "korean" else "외국인 전용 모델"
    analyzed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf.multi_cell(
        0,
        8,
        f"- 모델명: {model_name}\n"
        f"- 모델 유형: {model_type}\n"
        f"- 분석 일시: {analyzed_at}\n"
        f"- 예측 결과: {result_data['result']}\n"
        f"- 딥페이크 확률: {(result_data['fake_probability'] * 100):.2f}%\n",
    )

    # (2) Grad-CAM 시각화
    pdf.ln(8)
    pdf.set_font("malgun", "B", 13)
    pdf.cell(0, 10, "2. Grad-CAM 시각화", ln=True)
    pdf.image(gradcam_path, x=25, y=pdf.get_y() + 5, w=160)
    pdf.ln(95)

    # (3) LangChain 기반 AI 해석
    pdf.set_font("malgun", "B", 13)
    pdf.cell(0, 10, "3️. LangChain 기반 AI 해석", ln=True)
    pdf.set_font("malgun", size=11)
    pdf.multi_cell(0, 7, analysis_text)

    # (4) 결론 및 권장 조치
    pdf.ln(5)
    pdf.set_font("malgun", "B", 13)
    pdf.cell(0, 10, "4️. 결론 및 권장 조치", ln=True)
    pdf.set_font("malgun", size=11)
    pdf.multi_cell(
        0,
        7,
        "본 분석은 Grad-CAM 시각 주목도를 중심으로 진행되었습니다.\n"
        "AI의 결과는 참고용으로 사용해야 하며, 법적 판단이나 공식 증거로 사용되지 않습니다.\n"
        "결과의 신뢰도를 높이기 위해 다양한 이미지 소스로 교차 검증을 권장합니다.",
    )

    # PDF 저장
    pdf_filename = f"heatmap_report_{timestamp}.pdf"
    pdf_path = f"{pdf_dir}/{pdf_filename}"
    pdf.output(pdf_path)

    # ------------------------------------------------------
    # 4. 로그 JSON 생성 (DB 연동 대비)
    # ------------------------------------------------------
    log_data = {
        "created_at": analyzed_at,
        "model_name": model_name,
        "model_type": model_type,
        "result": result_data["result"],
        "fake_probability": result_data["fake_probability"],
        "gradcam_image": gradcam_path,
        "pdf_path": pdf_path,
    }

    log_filename = f"report_log_{timestamp}.json"
    with open(f"{log_dir}/{log_filename}", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    # 최종 PDF 경로 반환
    return pdf_path

# 미구현
async def face_detect(file):
    base_dir = config['BASE_DIR']
    output_dir = f"{base_dir}/cropped_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    cv_img = cv2.imread(file)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    width, height = img.size

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        print(f"\n❌ 얼굴 미검출: {file}")

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob < 0.9:
            continue

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        margin = 0.2
        x1 = max(0, int(x1 - w * margin / 2))
        y1 = max(0, int(y1 - h * margin / 2))
        x2 = min(width, int(x2 + w * margin / 2))
        y2 = min(height, int(y2 + h * margin / 2))

        face = img.crop((x1, y1, x2, y2))
        face.thumbnail((224, 224), Image.BICUBIC)
        face = ImageOps.pad(face, (224, 224), color=(0, 0, 0))

        out_name = f"{os.path.splitext(file)[0]}_face{i+1}.jpg"
        out_path = f"{output_dir}/{out_name}"
        face.save(out_path, format="JPEG", quality=95)
    return out_path
