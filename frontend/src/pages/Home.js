import { useEffect, useState } from "react";

export default function Home() {
  const [showAI, setShowAI] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [alertMode, setAlertMode] = useState(false);
  const [reset, setReset] = useState(false);

  useEffect(() => {
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const cycle = async () => {
      // ① 초기 상태: 원본 표시
      setShowAI(false);
      setShowWarning(false);
      setAlertMode(false);
      setReset(false);
      await wait(2000);

      // ③ AI 이미지 뜨자마자 경고 + 테두리 깜빡임
      setShowAI(true);
      setShowWarning(true);
      setAlertMode(true);
      await wait(2000);

      // ④ 모두 페이드아웃
      setReset(true);
      setShowAI(false);
      setShowWarning(false);
      setAlertMode(false);
      await wait(500);
    };

    cycle();
    const loop = setInterval(cycle, 4500); // 전체 루프 시간
    return () => clearInterval(loop);
  }, []);

  return (
    <div className="container flex-row gap-50">
      {/* 왼쪽 텍스트 영역 */}
      <div className="home-text">
        <h1>
          AI는 <span className="red">거짓</span>을 드러내고,
          <br />
          <span className="blue">진실</span>을 되살리는
          <span className="yellow">기술</span>입니다
        </h1>
        <div className="home-main">
          <div className="home-section">
            <div className="section-number">1</div>
            <div>
              <h3 className="section-title">
                기술의 진화
                <span className="section-subtitle">- 거짓을 드러내는 지능</span>
              </h3>
              <p>
                AI는 이미지를 분석하고, 패턴을 읽어내며,
                <span className="highlight red">보이지 않는 조작의 흔적</span>을
                찾아냅니다 <br />
                우리는 그 기술로 거짓을 드러내고, 진실의 경계를 지켜냅니다
              </p>
            </div>
          </div>

          <div className="home-section">
            <div className="section-number">2</div>
            <div>
              <h3 className="section-title">
                윤리의 기준
                <span className="section-subtitle">- 진실을 되살리는 책임</span>
              </h3>
              <p>
                그러나 기술은 목적이 아니라
                <span className="highlight blue"> 책임의 도구</span>입니다
                <br />
                우리는 그 힘을 사용해
                <span className="highlight blue">
                  사라진 얼굴과 왜곡된 기억을 복원
                </span>
                합니다
                <br />
                진실을 왜곡하지 않는 AI, 그것이 우리가 지켜야 할 기준입니다
              </p>
            </div>
          </div>

          <div className="home-section">
            <div className="section-number">3</div>
            <div>
              <h3 className="section-title">
                Deepfake Hunters의 약속
                <span className="section-subtitle">- 사람을 지켜주는 기술</span>
              </h3>
              <p>
                우리는 기술이 사람을 속이는 시대에
                <span className="highlight yellow">사람을 지켜주는 AI</span>를
                만듭니다
                <br />
                Deepfake Hunters — 거짓을 감별하고, 진실을 복원하는 AI
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 오른쪽 이미지 영역 */}
      <div className={`home-image ${alertMode ? "alert-border" : ""}`}>
        {/* 원본 이미지 */}
        <img
          src="/images/homeImage_origin.jpg"
          alt="Original"
          className={`home-img base ${showAI ? "fade-out" : "fade-in"} ${
            reset ? "reset" : ""
          }`}
        />

        {/* AI 생성 이미지 */}
        <img
          src="/images/homeImage_ai.jpg"
          alt="AI Generated"
          className={`home-img ai ${showAI ? "fade-in" : "fade-out"} ${
            reset ? "reset" : ""
          }`}
        />

        {/* 경고 표시 */}
        {showWarning && (
          <div className={`deepfake-warning ${reset ? "fade-out" : "fade-in"}`}>
            ⚠ Deepfake!
          </div>
        )}
      </div>
    </div>
  );
}
