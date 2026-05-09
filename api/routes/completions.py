from fastapi import APIRouter, Header

from api.schemas.completions import CompletionRequest

router = APIRouter(prefix="/v1/completions", tags=["completions"])


@router.post("")
async def create_completion(
    request: CompletionRequest,
    authorization: str = Header(default=""),
):
    pass