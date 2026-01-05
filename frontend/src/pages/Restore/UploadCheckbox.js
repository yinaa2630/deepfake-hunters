const UploadCheckbox = ({
  file,
  ethicsChecked,
  setEthicsChecked,
  loading,
  handleRestore,
}) => {
  const allChecked = ethicsChecked;
  return (
    <>
      <div className="flex-column gap-5">
        <label className="checkbox-text">
          <input
            type="checkbox"
            checked={ethicsChecked}
            onChange={() => setEthicsChecked((prev) => !prev)}
          />
          <p>AI 이미지로 타인 비방·범죄·허위정보를 생성하지 않겠습니다</p>
        </label>

        <div className="flex-center">
          <button
            disabled={!file || !allChecked || loading}
            onClick={handleRestore}
          >
            {loading ? "생성 중..." : "Restore"}
          </button>
        </div>
      </div>
    </>
  );
};
export default UploadCheckbox;
