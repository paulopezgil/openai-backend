from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/v1/responses", tags=["responses"])


@router.post("")
async def create_response(authorization: str = Header(default="")):
    pass


@router.post("")
async def create_response_stream(authorization: str = Header(default="")):
    pass