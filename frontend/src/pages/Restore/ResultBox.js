const ResultBox = ({ result }) => {
  return (
    <>
      <div className="content-area">
        {result ? (
          result.restored_image_url ? (
            <img
              src={result.restored_image_url}
              alt="Restored Result"
              className="preview"
            />
          ) : result.error ? (
            <p className="red">{result.error}</p>
          ) : (
            <p>복원 이미지가 나타납니다</p>
          )
        ) : (
          <p>복원 이미지가 나타납니다</p>
        )}
      </div>
    </>
  );
};

export default ResultBox;
