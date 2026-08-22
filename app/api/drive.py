from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import getDb
from app.core.deps import getCurrentUser
from app.core.logging import logger
from app.models.user import User
from app.schemas.drive import (
    CreateFolderRequest,
    DriveFileListResponse,
    DriveFileResponse,
    ShareFileRequest,
    UploadResponse,
)
from app.services.google_drive_service import GoogleDriveService
from app.services.rag.rag_service import RagService

router = APIRouter(prefix="/drive", tags=["Google Drive"])


def _driveService(user: User, db: AsyncSession) -> GoogleDriveService:
    return GoogleDriveService(db, user.id)


@router.get("/files", response_model=DriveFileListResponse)
async def listFiles(
    pageSize: int = Query(20, ge=1, le=100),
    pageToken: str | None = Query(None),
    folderId: str | None = Query(None),
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    return await svc.listFiles(pageSize=pageSize, pageToken=pageToken, folderId=folderId)


@router.post("/upload", response_model=UploadResponse)
async def uploadFile(
    file: UploadFile = File(...),
    parentId: str | None = Query(None),
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    content = await file.read()
    result = await svc.uploadFile(
        fileContent=content,
        fileName=file.filename or "untitled",
        mimeType=file.content_type or "application/octet-stream",
        parentId=parentId,
    )

    fileId = result.get("id")
    mimeType = file.content_type or "application/octet-stream"
    if fileId:
        from app.core.config import getSettings
        settings = getSettings()
        if mimeType in settings.RAG_SUPPORTED_MIMETYPES:
            try:
                ragSvc = RagService(db, user.id, drive=svc)
                await ragSvc.indexFile(fileId)
                logger.info("auto_index_on_upload", userId=str(user.id), fileId=fileId)
            except Exception as e:
                logger.warning("auto_index_failed", userId=str(user.id), fileId=fileId, error=str(e))

    return UploadResponse(**result)


@router.get("/download/{fileId}")
async def downloadFile(
    fileId: str,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    content, fileName, mimeType = await svc.downloadFile(fileId)
    return Response(
        content=content,
        media_type=mimeType,
        headers={"Content-Disposition": f'attachment; filename="{quote(fileName)}"'},
    )


@router.get("/download/{fileId}/stream")
async def downloadFileStream(
    fileId: str,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)

    async def _generate():
        async for chunk, _, _ in svc.downloadFileStream(fileId):
            yield chunk

    meta = await svc.getFileMetadata(fileId)
    return StreamingResponse(
        _generate(),
        media_type=meta.get("mimeType", "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{quote(meta.get("name", "download"))}"'},
    )


@router.delete("/file/{fileId}")
async def deleteFile(
    fileId: str,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    await svc.deleteFile(fileId)
    return {"message": "File deleted successfully"}


@router.post("/create-folder", response_model=DriveFileResponse)
async def createFolder(
    body: CreateFolderRequest,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    result = await svc.createFolder(name=body.name, parentId=body.parentId)
    return DriveFileResponse(**result)


@router.get("/search", response_model=DriveFileListResponse)
async def searchFiles(
    q: str = Query(..., min_length=1),
    pageSize: int = Query(20, ge=1, le=100),
    pageToken: str | None = Query(None),
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    return await svc.searchFiles(query=q, pageSize=pageSize, pageToken=pageToken)


@router.post("/share")
async def shareFile(
    body: ShareFileRequest,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = _driveService(user, db)
    result = await svc.shareFile(fileId=body.fileId, email=body.email, role=body.role)
    return {"message": "File shared successfully", "permission": result}
