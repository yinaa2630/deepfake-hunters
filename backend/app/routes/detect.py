from fastapi import APIRouter

from backend.app.controllers import detector

router = APIRouter()
router.add_api_route("/result",detector.detection_results, methods=["POST"])
router.add_api_route("/report",detector.generate_report, methods=["POST"])
