from fastapi import APIRouter

from backend.app.controllers import restore

router = APIRouter()
router.add_api_route("/",restore.restoration_results, methods=["POST"])
