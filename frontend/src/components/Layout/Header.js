import { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";

export default function Header() {
  const [index, setIndex] = useState(0);
  const [fadeState, setFadeState] = useState("fade-in"); // 'fade-in' | 'fade-out'
  const texts = [
    <span className="special-text">2팀</span>,
    <>
      Deepfake <br /> Hunters
    </>,
    <>
      딥페이크 <br /> 헌 터 스
    </>,
  ];

  useEffect(() => {
    let fadeOutTimeout;
    let switchTimeout;

    const startCycle = () => {
      // ✅ 2초 표시 후 페이드아웃
      fadeOutTimeout = setTimeout(() => {
        setFadeState("fade-out");

        // ✅ 페이드아웃 0.8초 후 다음 텍스트로 전환 + 페이드인
        switchTimeout = setTimeout(() => {
          setIndex((prev) => (prev + 1) % texts.length); // 순환
          setFadeState("fade-in");
        }, 800);
      }, 2000); // 표시시간 (2초)
    };

    startCycle();

    const interval = setInterval(startCycle, 2800); // 2초 표시 + 0.8초 전환
    return () => {
      clearInterval(interval);
      clearTimeout(fadeOutTimeout);
      clearTimeout(switchTimeout);
    };
  }, []);

  return (
    <header className="flex-between">
      <div className="logo-container flex-center gap-40 text-center">
        <img
          src="/images/teamImage.jpg"
          alt="Deepfake Hunters team"
          className="logo"
        />
        {/* ✅ 3단계 텍스트 순환 */}
        <span className={`fade-text ${fadeState}`}>{texts[index]}</span>
      </div>
      <nav>
        {/* ✅ 메뉴 hover 시 한글 설명 추가 */}
        <div className="flex-center  navbar-menu  gap-40">
          <NavLink to="/">
            Home
            <span className="nav-tooltip">홈</span>
          </NavLink>
          <NavLink to="/detect">
            Detect<span className="nav-tooltip">딥페이크 판별</span>
          </NavLink>
          <NavLink to="/restore">
            Restore
            <span className="nav-tooltip">이미지 복원</span>
          </NavLink>
          <NavLink to="/conclusion">
            Conclusion
            <span className="nav-tooltip">엔딩까지 완벽한 서비스!</span>
          </NavLink>
        </div>
      </nav>
    </header>
  );
}
