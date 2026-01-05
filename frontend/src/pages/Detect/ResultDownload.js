import { postReport } from "../../api/reportApi";

const ResultDownload = ({ result, summaryText }) => {
  // ✅ PDF 다운로드 (팝업 + 알림 포함)
  const handleDownloadPDF = async () => {
    if (!result) return alert("분석 결과가 없습니다.");

    // 🔹 다운로드 확인 팝업
    const confirmDownload = window.confirm(
      "PDF 보고서를 다운로드하시겠습니까?"
    );
    if (!confirmDownload) return;

    // 🔹 PDF 생성용 JSON 구조
    const reportData = {
      result: `${result.pred_label || "Unknown"} (${
        result.confidence?.toFixed(2) || 0
      }%)`,
      fake_probability: result.fake_probability || 0,
      gradcam: result.gradcam,
      model_type: result.model_type || "korean",
      model_name: "MobileNetV3-Small",
    };

    try {
      const res = await postReport(reportData);

      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "Deepfake_Heatmap_Report.pdf";
      a.click();
      URL.revokeObjectURL(url);

      // ✅ 완료 알림
      alert("✅ PDF 보고서가 다운로드되었습니다!");
    } catch (err) {
      console.error(err);
      alert("PDF 생성 중 오류가 발생했습니다.");
    }
  };

  return (
    <>
      <div className="flex-row detect-result-container text-center">
        <div
          className={`result-summary-box ${
            result && !result.error ? "active" : ""
          }`}
        >
          {result && !result.error ? (
            <>
              <p className="detect-result-line red">
                <span className="white">- 결과 :</span> {summaryText}
              </p>
              <p className="detect-result-line  red">
                <span className="white">- 시각적 활성도 :</span>{" "}
                {result.fake_probability
                  ? `${(result.fake_probability * 100).toFixed(1)}%`
                  : "N/A"}
              </p>
            </>
          ) : (
            <p>분석 결과를 확인하세요</p>
          )}
        </div>

        {/* ✅ 팝업 포함된 PDF 다운로드 버튼 */}
        <button
          className="pdf-btn"
          onClick={handleDownloadPDF}
          disabled={!result || result.error}
        >
          📄 PDF 보고서
          <br /> 다운로드
        </button>
      </div>
    </>
  );
};
export default ResultDownload;
