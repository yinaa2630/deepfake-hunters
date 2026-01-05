import { useState } from "react";

import UploadBox from "./UploadBox";
import UploadCheckbox from "./UploadCheckbox";
import ResultBox from "./ResultBox";
import ResultDownload from "./ResultDownload";
import { DetectFile } from "../../api/reportApi";

const Detect = () => {
  const [file, setFile] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [summaryText, setSummaryText] = useState("");
  const [modelType, setModelType] = useState("korean"); // 분석 모델 선택

  const [rightsChecked, setRightsChecked] = useState(false);
  const [disclaimerChecked, setDisclaimerChecked] = useState(false);
  const [loading, setLoading] = useState(false);

  const allChecked = rightsChecked && disclaimerChecked;
  // 파일 선택
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setImage(URL.createObjectURL(selected));
    setResult(null);
    setSummaryText("");
  };
  // AI 판별 요청
  const handleDetect = async () => {
    if (!file) return alert("파일을 선택하세요!");
    if (!allChecked) return alert("체크박스에 모두 동의해주세요.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_type", modelType);

    try {
      // ✅ 예측 요청
      const data = await DetectFile(formData);
      console.log("📊 백엔드 응답:", data);
      setResult(data);

      if (!data.error && data.pred_label && data.confidence !== undefined) {
        const { pred_label, confidence } = data;
        const msg =
          pred_label === "Fake"
            ? `Fake! (신뢰도: ${confidence.toFixed(2)}%)`
            : `Real! (신뢰도: ${confidence.toFixed(2)}%)`;
        setSummaryText(msg);
      } else {
        setSummaryText("분석 결과를 가져올 수 없습니다.");
      }
    } catch (err) {
      console.error(err);
      setResult({ error: "서버 오류 발생" });
      setSummaryText("서버 오류로 분석을 완료할 수 없습니다.");
    } finally {
      setLoading(false);
    }
  };
  return (
    <>
      <div className="detect container flex-column flex-center">
        <h1 className="title">
          이 이미지는 진짜일까요? – AI는 픽셀 단위로
          <span className="red">진실을 추적</span>합니다
        </h1>
        <div className="grid-container">
          <h3 style={{ gridColumn: 1, gridRow: 1 }}>Upload Image</h3>
          <div style={{ gridColumn: 1, gridRow: 2 }}>
            <UploadBox
              image={image}
              modelType={modelType}
              setModelType={setModelType}
              handleFileChange={handleFileChange}
            />
          </div>
          <div style={{ gridColumn: 1, gridRow: 3 }}>
            <UploadCheckbox
              file={file}
              rightsChecked={rightsChecked}
              setRightsChecked={setRightsChecked}
              disclaimerChecked={disclaimerChecked}
              setDisclaimerChecked={setDisclaimerChecked}
              loading={loading}
              handleDetect={handleDetect}
            />
          </div>
          <div
            className="arrow-box"
            style={{
              gridColumn: 2,
              gridRow: 2,
              alignSelf: "center",
              justifySelf: "center",
            }}
          >
            <img src="/images/arrow.jpg" alt="arrow" />
          </div>
          <h3 style={{ gridColumn: 3, gridRow: 1 }}>Detection Results</h3>
          <div style={{ gridColumn: 3, gridRow: 2 }}>
            <ResultBox result={result} />
          </div>
          <div style={{ gridColumn: 3, gridRow: 3 }}>
            <ResultDownload result={result} summaryText={summaryText} />
          </div>
        </div>
      </div>
    </>
  );
};

export default Detect;
