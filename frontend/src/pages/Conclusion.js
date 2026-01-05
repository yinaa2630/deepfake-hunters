import { useState, useEffect } from "react";

export default function Conclusion() {
  const [showOriginal, setShowOriginal] = useState(true);

  useEffect(() => {
    let switchTimeout;

    const cycle = () => {
      // ✅ 1️⃣ 원본 이미지 표시
      setShowOriginal(true);
      switchTimeout = setTimeout(() => {
        // ✅ 2️⃣ 반전 이미지 즉시 전환
        setShowOriginal(false);
      }, 1000);
    };

    cycle();
    const interval = setInterval(cycle, 2000); // 총 주기: 3 + 1초 = 4초
    return () => {
      clearInterval(interval);
      clearTimeout(switchTimeout);
    };
  }, []);

  return (
    <div
      className="container flex-column text-center"
      style={{ justifyContent: "start" }}
    >
      <div>
        <h1 className="conclusion-sub-title">
          AI는 이미지를 분석해 <span className="blue">진실을 찾아내고, </span>
          복원해
          <span className="red"> 잃어버린 기억을 되살립니다</span>
          <br />
        </h1>
        <h1 className="conclusion-main-title">
          우린 그 진실이 사람을 해치지 않도록,
          <br />
          <span className="yellow">오늘도 조용히 — 지켜보고 있습니다</span>
        </h1>
      </div>
      <div className="flex-center conclusion-image">
        <img
          src={
            showOriginal
              ? "/images/watching_reverse.png"
              : "/images/watching_color.png"
          }
          width={380}
          alt="Looking Ahead"
        />
      </div>
    </div>
  );
}
