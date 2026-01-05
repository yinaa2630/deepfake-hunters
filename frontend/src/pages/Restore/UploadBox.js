import { useRef } from "react";
const UploadBox = ({ image, handleFileChange }) => {
  const fileInputRef = useRef(null);

  // ✅ 파일 첨부 버튼 클릭
  const handleUploadClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  return (
    <>
      <div className="upload-box content-area">
        {image ? (
          <img src={image} alt="preview" className="preview" />
        ) : (
          <div>
            <p>복원할 얼굴 이미지를 첨부하세요</p>
            <button onClick={handleUploadClick}>이미지 파일 첨부</button>
            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
          </div>
        )}
      </div>
    </>
  );
};

export default UploadBox;
