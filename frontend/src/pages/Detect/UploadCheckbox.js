const UploadCheckbox = ({
  file,
  rightsChecked,
  setRightsChecked,
  disclaimerChecked,
  setDisclaimerChecked,
  loading,
  handleDetect,
}) => {
  const allChecked = rightsChecked && disclaimerChecked;

  return (
    <>
      <div className="flex-column gap-5">
        <label className="checkbox-text">
          <input
            type="checkbox"
            checked={rightsChecked}
            onChange={() => setRightsChecked((prev) => !prev)}
          />
          <p>이 이미지는 타인의 초상권 또는 저작권을 침해하지 않습니다</p>
        </label>
        <label className="checkbox-text">
          <input
            type="checkbox"
            checked={disclaimerChecked}
            onChange={() => setDisclaimerChecked((prev) => !prev)}
          />
          <p>
            AI 분석 결과는 참고용이며 법적 증거로 사용되지 않음을 이해합니다
          </p>
        </label>

        <div className="flex-center">
          <button
            disabled={!file || !allChecked || loading}
            onClick={handleDetect}
          >
            {loading ? "분석 중..." : "Detect"}
          </button>
        </div>
      </div>
    </>
  );
};
export default UploadCheckbox;
