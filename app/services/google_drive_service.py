import io
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import getSettings
from app.core.exceptions import GoogleDriveException, NotFoundException, TokenExpiredException
from app.core.logging import logger
from app.core.security import decryptValue, encryptValue
from app.models.drive_token import GoogleDriveToken

settings = getSettings()

DRIVE_FIELDS = "id, name, mimeType, size, createdTime, modifiedTime, webViewLink, parents"
FOLDER_MIME = "application/vnd.google-apps.folder"


class GoogleDriveService:
    def __init__(self, db: AsyncSession, userId: uuid.UUID):
        self._db = db
        self._userId = userId
        self._service = None

    async def _getCredentials(self) -> Credentials:
        result = await self._db.execute(
            select(GoogleDriveToken).where(GoogleDriveToken.userId == self._userId)
        )
        token = result.scalar_one_or_none()
        if not token:
            raise TokenExpiredException()

        creds = Credentials(
            token=decryptValue(token.accessToken),
            refresh_token=decryptValue(token.refreshToken),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            scopes=settings.GOOGLE_SCOPES,
        )

        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(GoogleAuthRequest())
                token.accessToken = encryptValue(creds.token)
                token.tokenExpiry = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else datetime.now(timezone.utc) + timedelta(hours=1)
                await self._db.flush()
                logger.info("google_token_refreshed", userId=str(self._userId))
            except Exception as e:
                logger.error("google_token_refresh_failed", userId=str(self._userId), error=str(e))
                raise TokenExpiredException()

        return creds

    async def _getDriveService(self):
        if not self._service:
            creds = await self._getCredentials()
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GoogleDriveException),
        reraise=True,
    )
    async def listFiles(
        self,
        pageSize: int = 20,
        pageToken: str | None = None,
        query: str | None = None,
        folderId: str | None = None,
    ) -> dict[str, Any]:
        service = await self._getDriveService()
        q_parts = ["trashed = false"]
        if folderId:
            q_parts.append(f"'{folderId}' in parents")
        if query:
            q_parts.append(query)

        try:
            results = (
                service.files()
                .list(
                    pageSize=pageSize,
                    pageToken=pageToken,
                    fields=f"nextPageToken, files({DRIVE_FIELDS})",
                    q=" and ".join(q_parts),
                    orderBy="modifiedTime desc",
                )
                .execute()
            )
            return {
                "files": results.get("files", []),
                "nextPageToken": results.get("nextPageToken"),
                "totalCount": len(results.get("files", [])),
            }
        except Exception as e:
            logger.error("drive_list_failed", error=str(e))
            raise GoogleDriveException(f"Failed to list files: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GoogleDriveException),
        reraise=True,
    )
    async def uploadFile(
        self,
        fileContent: bytes,
        fileName: str,
        mimeType: str,
        parentId: str | None = None,
    ) -> dict[str, Any]:
        service = await self._getDriveService()
        metadata: dict[str, Any] = {"name": fileName}
        if parentId:
            metadata["parents"] = [parentId]

        try:
            media = MediaIoBaseUpload(io.BytesIO(fileContent), mimetype=mimeType, resumable=True)
            result = (
                service.files()
                .create(body=metadata, media_body=media, fields=DRIVE_FIELDS)
                .execute()
            )
            logger.info("drive_file_uploaded", fileId=result["id"], name=fileName)
            return result
        except Exception as e:
            logger.error("drive_upload_failed", error=str(e))
            raise GoogleDriveException(f"Failed to upload file: {e}")

    async def downloadFile(self, fileId: str) -> tuple[bytes, str, str]:
        service = await self._getDriveService()
        try:
            fileMeta = service.files().get(fileId=fileId, fields="name, mimeType").execute()
            request = service.files().get_media(fileId=fileId)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buffer.seek(0)
            return buffer.read(), fileMeta.get("name", "download"), fileMeta.get("mimeType", "application/octet-stream")
        except Exception as e:
            logger.error("drive_download_failed", fileId=fileId, error=str(e))
            raise GoogleDriveException(f"Failed to download file: {e}")

    async def downloadFileStream(self, fileId: str) -> AsyncGenerator[tuple[bytes, str, str], None]:
        """Stream download for large files — yields chunks."""
        service = await self._getDriveService()
        try:
            fileMeta = service.files().get(fileId=fileId, fields="name, mimeType").execute()
            request = service.files().get_media(fileId=fileId)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request, chunksize=5 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
                buffer.seek(0)
                chunk = buffer.read()
                if chunk:
                    yield chunk, fileMeta.get("name", "download"), fileMeta.get("mimeType", "application/octet-stream")
                buffer.seek(0)
                buffer.truncate()
        except Exception as e:
            logger.error("drive_stream_download_failed", fileId=fileId, error=str(e))
            raise GoogleDriveException(f"Failed to stream download: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GoogleDriveException),
        reraise=True,
    )
    async def deleteFile(self, fileId: str) -> None:
        service = await self._getDriveService()
        try:
            service.files().delete(fileId=fileId).execute()
            logger.info("drive_file_deleted", fileId=fileId)
        except Exception as e:
            logger.error("drive_delete_failed", fileId=fileId, error=str(e))
            raise GoogleDriveException(f"Failed to delete file: {e}")

    async def createFolder(self, name: str, parentId: str | None = None) -> dict[str, Any]:
        service = await self._getDriveService()
        metadata: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME}
        if parentId:
            metadata["parents"] = [parentId]

        try:
            result = service.files().create(body=metadata, fields=DRIVE_FIELDS).execute()
            logger.info("drive_folder_created", folderId=result["id"], name=name)
            return result
        except Exception as e:
            logger.error("drive_create_folder_failed", error=str(e))
            raise GoogleDriveException(f"Failed to create folder: {e}")

    async def searchFiles(self, query: str, pageSize: int = 20, pageToken: str | None = None) -> dict[str, Any]:
        sanitized = query.replace("'", "\\'")
        driveQuery = f"name contains '{sanitized}' and trashed = false"
        return await self.listFiles(pageSize=pageSize, pageToken=pageToken, query=driveQuery)

    async def shareFile(self, fileId: str, email: str, role: str = "reader") -> dict[str, Any]:
        service = await self._getDriveService()
        try:
            permission = {"type": "user", "role": role, "emailAddress": email}
            result = (
                service.permissions()
                .create(fileId=fileId, body=permission, sendNotificationEmail=True, fields="id, role, emailAddress")
                .execute()
            )
            logger.info("drive_file_shared", fileId=fileId, email=email, role=role)
            return result
        except Exception as e:
            logger.error("drive_share_failed", fileId=fileId, error=str(e))
            raise GoogleDriveException(f"Failed to share file: {e}")

    async def getFileMetadata(self, fileId: str) -> dict[str, Any]:
        service = await self._getDriveService()
        try:
            return service.files().get(fileId=fileId, fields=DRIVE_FIELDS).execute()
        except Exception as e:
            logger.error("drive_get_metadata_failed", fileId=fileId, error=str(e))
            raise NotFoundException("File")

    async def fetchBytes(self, fileId: str, mimeType: str, exportMimeType: str | None = None) -> bytes:
        """Download raw bytes. When `exportMimeType` is provided, uses export_media (for Google Workspace files)."""
        service = await self._getDriveService()
        try:
            if exportMimeType:
                request = service.files().export_media(fileId=fileId, mimeType=exportMimeType)
            else:
                request = service.files().get_media(fileId=fileId)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error("drive_fetch_bytes_failed", fileId=fileId, error=str(e))
            raise GoogleDriveException(f"Failed to fetch file bytes: {e}")
