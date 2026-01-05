import { api } from "./axios";

export const postReport = async (reportData) => {
  try {
    const res = await api.post(`/detect/report`, reportData, {
      headers: {
        "Content-Type": "application/json",
      },
    });
    return res;
  } catch (err) {
    console.error("API 요청 실패:", err);
    throw err;
  }
};

// ✅ 예측 요청
export const DetectFile = async (formData) => {
  try {
    const res = await api.post(`/detect/result`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res?.data?.message;
  } catch (err) {
    console.error("예측 요청 실패:", err);
    throw err;
  }
};

export const restoreFile = async (formData) => {
  try {
    const res = await api.post(`/restore`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res?.data?.message;
  } catch (err) {
    console.error("복원 요청 실패:", err);
    throw err;
  }
};
