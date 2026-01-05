import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Detect from "./pages/Detect";
import Restore from "./pages/Restore";
import Conclusion from "./pages/Conclusion";
import Layout from "./components/Layout";

import "./styles/index.css";
import "./styles/header.css";

import "./styles/home.css";
import "./styles/detect.css";
import "./styles/restore.css";
import "./styles/conclusion.css";
import "./styles/global.css";

export default function App() {
  return (
    <Router>
      <Routes>
        {/* Layout을 Route의 element로 지정 */}
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/detect" element={<Detect />} />
          <Route path="/restore" element={<Restore />} />
          <Route path="/conclusion" element={<Conclusion />} />
        </Route>
      </Routes>
    </Router>
  );
}
