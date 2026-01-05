import { useRef } from "react";

const UploadBox = ({ image, modelType, setModelType, handleFileChange }) => {
  const fileInputRef = useRef(null);

  // 파일 첨부 버튼 클릭
  const handleUploadClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };
  return (
    <>
      <div className="upload-box content-area">
        {image ? (
          <img src={image} alt="preview" className="preview" />
        ) : (
          <div className="detect-model-box">
            <p># 분석대상을 선택하세요 (택1)</p>
            <div className="detect-model-select">
              <label>
                <input
                  type="radio"
                  value="korean"
                  checked={modelType === "korean"}
                  onChange={(e) => setModelType(e.target.value)}
                />
                한국인 이미지
              </label>
              <label>
                <input
                  type="radio"
                  value="foreign"
                  checked={modelType === "foreign"}
                  onChange={(e) => setModelType(e.target.value)}
                />
                외국인 이미지
              </label>
            </div>

            <button onClick={handleUploadClick}>이미지 파일 첨부</button>
          </div>
        )}
      </div>

      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
    </>
  );
};
export default UploadBox;
