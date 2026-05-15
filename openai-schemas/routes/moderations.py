from fastapi import APIRouter, Header

from api.schemas.moderations import ModerationRequest

router = APIRouter(prefix="/v1/moderations", tags=["moderations"])


@router.post("")
async def create_moderation(
    request: ModerationRequest,
    authorization: str = Header(default=""),
):
    pass