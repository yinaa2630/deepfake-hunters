import { useState } from "react";

import ResultBox from "./ResultBox";
import UploadBox from "./UploadBox";
import UploadCheckbox from "./UploadCheckbox";
import { restoreFile } from "../../api/reportApi";
import ResultDownload from "./ResultDownload";

const Restore = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [reFile, setReFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [ethicsChecked, setEthicsChecked] = useState(false);

  const allChecked = ethicsChecked;
  // ✅ 파일 변경
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setReFile(selected);
    setImage(URL.createObjectURL(selected));
    setResult(null);
  };
  // ✅ 복원 요청
  const handleRestore = async () => {
    if (!reFile) return alert("파일을 선택하세요!");
    if (!allChecked) return alert("체크박스에 동의해주세요.");
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", reFile);
      const data = await restoreFile(formData);
      console.log(data);
      setResult(data);
    } catch (err) {
      setResult({ error: "서버 오류가 발생했습니다." });
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  // ✅ 복원된 이미지 다운로드 (팝업 + 다운로드 실행)
  const handleDownload = () => {
    if (!result || !result.restored_image_url)
      return alert("복원된 이미지가 없습니다.");

    // 1️⃣ 팝업으로 다운로드 확인
    const confirmDownload = window.confirm(
      "복원된 이미지를 다운로드하시겠습니까?"
    );
    if (!confirmDownload) return;

    // 2️⃣ 파일 저장 실행
    fetch(result.restored_image_url)
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "restored_image.png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // 3️⃣ 다운로드 완료 알림
        alert("✅ 다운로드가 완료되었습니다!");
      })
      .catch(() => alert("다운로드 중 오류가 발생했습니다."));
  };

  return (
    <>
      <div className="restore container flex-column flex-center">
        <h1 className="title">
          흐릿한 얼굴, 사라진 기억 - AI가
          <span className="blue">되살립니다</span>
        </h1>
        <div className="grid-container">
          <h3 style={{ gridColumn: 1, gridRow: 1 }}>Upload Image</h3>
          <div style={{ gridColumn: 1, gridRow: 2 }}>
            <UploadBox
              image={image}
              ethicsChecked={ethicsChecked}
              setEthicsChecked={setEthicsChecked}
              loading={loading}
              handleFileChange={handleFileChange}
            />
          </div>
          <div style={{ gridColumn: 1, gridRow: 3 }}>
            <UploadCheckbox
              file={reFile}
              ethicsChecked={ethicsChecked}
              setEthicsChecked={setEthicsChecked}
              loading={loading}
              handleRestore={handleRestore}
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
          <h3 style={{ gridColumn: 3, gridRow: 1 }}>Restored Result</h3>
          <div style={{ gridColumn: 3, gridRow: 2 }}>
            <ResultBox result={result} />
          </div>
          <div style={{ gridColumn: 3, gridRow: 3 }}>
            <ResultDownload result={result} handleDownload={handleDownload} />
          </div>
        </div>
      </div>
    </>
  );
};

export default Restore;
