from fastapi import APIRouter, Header

from api.schemas.embeddings import EmbeddingRequest

router = APIRouter(prefix="/v1/embeddings", tags=["embeddings"])


@router.post("")
async def create_embedding(
    request: EmbeddingRequest,
    authorization: str = Header(default=""),
):
    pass