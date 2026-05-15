from fastapi import APIRouter, Header, UploadFile, File, Form, Path
from fastapi.responses import FileResponse

router = APIRouter(prefix="/v1/files", tags=["files"])


@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    authorization: str = Header(default=""),
):
    pass


@router.get("")
async def list_files(
    purpose: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    after: str | None = None,
    authorization: str = Header(default=""),
):
    pass


@router.get("/{file_id}")
async def retrieve_file(
    file_id: str = Path(...),
    authorization: str = Header(default=""),
):
    pass


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: str = Path(...),
    authorization: str = Header(default=""),
):
    pass


@router.delete("/{file_id}")
async def delete_file(
    file_id: str = Path(...),
    authorization: str = Header(default=""),
):
    pass