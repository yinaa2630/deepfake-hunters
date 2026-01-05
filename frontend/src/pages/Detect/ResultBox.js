const ResultBox = ({ result }) => {
  return (
    <>
      <div className="content-area">
        {result ? (
          result.gradcam ? (
            <img
              src={`data:image/png;base64,${result.gradcam}`}
              alt="Grad-CAM heatmap"
              className="preview"
            />
          ) : result.error ? (
            <p className="red">{result.error}</p>
          ) : (
            <p className="red">시각적 활성도: N/A</p>
          )
        ) : (
          <p>분석 이미지가 나타납니다</p>
        )}
      </div>
    </>
  );
};
export default ResultBox;
