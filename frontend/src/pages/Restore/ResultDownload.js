const ResultDownload = ({ result, handleDownload }) => {
  return (
    <>
      <div>
        <button onClick={handleDownload} disabled={!result || result.error}>
          Download
        </button>
      </div>
    </>
  );
};

export default ResultDownload;
