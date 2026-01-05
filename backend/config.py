import os, sys
from dotenv import load_dotenv
from pathlib import Path

backend_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), ".."))
sys.path.append(backend_path)

BASE_DIR = Path(__file__).resolve().parent  # backend/
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

config = {
    "BASE_DIR":BASE_DIR,
    "OPENAI_API": os.getenv("OPENAI_API_KEY"),  
    "DB_URL": os.getenv("DB_URL"),
    "HOST":"0.0.0.0",
    "PORT":8000,
}
print("✅ run path:",backend_path)
print("✅ DB_URL:", os.getenv("DB_URL"))
print("✅ OPENAI_API_KEY 감지됨" if os.getenv("OPENAI_API_KEY") else "⚠️ OPENAI_API_KEY 누락")

