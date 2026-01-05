import ReactDOM from "react-dom/client";
import App from "./App";

console.log("🔥 ENV CHECK:", process.env.REACT_APP_API_URL);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  // ❌ React.StrictMode 주석 처리 (개발 환경에서만 문제 발생)
  // <React.StrictMode>
  <App />
  // </React.StrictMode>
);
