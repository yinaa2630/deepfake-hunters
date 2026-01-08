import io, os, base64, datetime, uuid, json, cv2
from fastapi import UploadFile
from PIL import Image, ImageOps
from fpdf import FPDF

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from config import config
from backend.app.models.network import load_models, get_models
from ai.modules.Deepfake_Evaluation_MobileNet_v3_final_application_number_option import (
    analyze_image_with_model_type,
)

# ======================================================
# 1️⃣ 딥페이크 탐지
# ======================================================

async def predict_fake(
    file: UploadFile,
    model_type: str = "korean",
) -> dict:
    """
    이미지 업로드 → 딥페이크 탐지 → Grad-CAM 결과 반환
    """

    # 모델 lazy load
    load_models()

    base_dir = config["BASE_DIR"]
    upload_dir = os.path.join(base_dir, "data/uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # 파일 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    ext = os.path.splitext(file.filename)[1]
    save_name = f"{timestamp}_{unique_id}{ext}"
    save_path = os.path.join(upload_dir, save_name)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 모델 분석
    pred_label, confidence, report, gradcam_path, fake_intensity = (
        analyze_image_with_model_type(
            path=save_path,
            model_type=model_type,
            visualize=True,
        )
    )

    # Grad-CAM base64 변환
    gradcam_b64 = None
    if gradcam_path and os.path.exists(gradcam_path):
        with open(gradcam_path, "rb") as f:
            gradcam_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "pred_label": pred_label,
        "confidence": round(confidence, 2),
        "report": report,
        "gradcam": gradcam_b64,
        "image_path": save_path,
        "fake_probability": round(fake_intensity, 3) if fake_intensity else None,
        "model_type": model_type,
    }


# ======================================================
# 2️⃣ Grad-CAM 히트맵 PDF 리포트 생성
# ======================================================

async def generate_heatmap_report(request):
    result = await request.json()

    required_fields = ["gradcam", "result", "fake_probability", "model_type"]
    missing = [k for k in required_fields if k not in result]
    if missing:
        raise ValueError(f"필수 키 누락: {missing}")

    base_dir = config["BASE_DIR"]
    result_dir = os.path.join(base_dir, "data/results")
    image_dir = os.path.join(result_dir, "images")
    pdf_dir = os.path.join(result_dir, "pdfs")
    log_dir = os.path.join(result_dir, "logs")

    for d in [image_dir, pdf_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # Grad-CAM 이미지 저장
    gradcam_bytes = base64.b64decode(result["gradcam"])
    gradcam_img = Image.open(io.BytesIO(gradcam_bytes))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gradcam_path = os.path.join(image_dir, f"gradcam_{timestamp}.png")
    gradcam_img.save(gradcam_path)

    # LLM 분석
    prompt = PromptTemplate(
        input_variables=["result", "prob", "type"],
        template=(
            "너는 딥페이크 탐지 전문가야.\n\n"
            "모델 유형: {type}\n"
            "예측 결과: {result}\n"
            "딥페이크 확률: {prob:.2f}%\n\n"
            "Grad-CAM 히트맵을 기반으로 모델의 판단 근거를 기술적으로 분석하고, "
            "신뢰도와 한계점을 함께 설명해줘."
        ),
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    analysis_text = llm.invoke(
        prompt.format(
            type="한국인 모델" if result["model_type"] == "korean" else "외국인 모델",
            result=result["result"],
            prob=result["fake_probability"] * 100,
        )
    ).content

    # PDF 생성
    pdf = FPDF()
    pdf.add_page()

    pdf.add_font("malgun", "", r"C:\Windows\Fonts\malgun.ttf", uni=True)
    pdf.add_font("malgun", "B", r"C:\Windows\Fonts\malgunbd.ttf", uni=True)

    pdf.set_font("malgun", "B", 16)
    pdf.cell(0, 10, "딥페이크 히트맵 분석 보고서", ln=True, align="C")

    pdf.set_font("malgun", size=11)
    pdf.multi_cell(
        0,
        8,
        f"- 예측 결과: {result['result']}\n"
        f"- 딥페이크 확률: {result['fake_probability'] * 100:.2f}%\n"
    )

    pdf.ln(5)
    pdf.image(gradcam_path, x=25, y=pdf.get_y(), w=160)
    pdf.ln(95)

    pdf.set_font("malgun", size=11)
    pdf.multi_cell(0, 7, analysis_text)

    pdf_path = os.path.join(pdf_dir, f"heatmap_report_{timestamp}.pdf")
    pdf.output(pdf_path)

    # 로그 저장
    log_data = {
        "created_at": timestamp,
        "result": result["result"],
        "fake_probability": result["fake_probability"],
        "gradcam_image": gradcam_path,
        "pdf_path": pdf_path,
    }

    with open(os.path.join(log_dir, f"log_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    return pdf_path


# ======================================================
# 3️⃣ 얼굴 검출 (MTCNN)
# ======================================================

async def face_detect(image_path: str):
    load_models()
    _, _, _, mtcnn = get_models()

    base_dir = config["BASE_DIR"]
    output_dir = os.path.join(base_dir, "cropped_faces")
    os.makedirs(output_dir, exist_ok=True)

    cv_img = cv2.imread(image_path)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None

    results = []
    width, height = img.size

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob < 0.9:
            continue

        x1, y1, x2, y2 = map(int, box)
        margin = 0.2
        w, h = x2 - x1, y2 - y1

        x1 = max(0, int(x1 - w * margin / 2))
        y1 = max(0, int(y1 - h * margin / 2))
        x2 = min(width, int(x2 + w * margin / 2))
        y2 = min(height, int(y2 + h * margin / 2))

        face = img.crop((x1, y1, x2, y2))
        face = ImageOps.pad(face, (224, 224))

        out_path = os.path.join(output_dir, f"face_{i}.jpg")
        face.save(out_path)
        results.append(out_path)

    return results
