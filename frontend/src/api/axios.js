import axios from "axios";
export const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:8000", // 백엔드 주소
  // timeout: 20000, // 요청 제한 시간 10초
});
