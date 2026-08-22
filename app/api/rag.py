from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import getDb
from app.core.deps import getCurrentUser
from app.models.user import User
from app.schemas.chat import IndexedFileListResponse, IndexedFileOut, IndexFileRequest
from app.services.rag.rag_service import RagService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/index", response_model=IndexedFileOut)
async def indexFile(
    body: IndexFileRequest,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = RagService(db, user.id)
    result = await svc.indexFile(body.fileId, forceReindex=body.forceReindex)
    return IndexedFileOut(**result)


@router.get("/indexed", response_model=IndexedFileListResponse)
async def listIndexed(
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = RagService(db, user.id)
    files = await svc.listIndexed()
    return IndexedFileListResponse(
        files=[IndexedFileOut(**f) for f in files],
        totalCount=len(files),
    )


@router.delete("/index/{fileId}")
async def removeIndexed(
    fileId: str,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = RagService(db, user.id)
    await svc.removeFile(fileId)
    return {"message": "File removed from index", "fileId": fileId}
