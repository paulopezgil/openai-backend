from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from api.schemas.chat import ChatCompletionRequest


router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: str = Header(default=""),
):
    pass