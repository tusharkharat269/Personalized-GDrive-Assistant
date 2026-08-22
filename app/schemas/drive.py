from datetime import datetime

from pydantic import BaseModel


class DriveFileResponse(BaseModel):
    id: str
    name: str
    mimeType: str | None = None
    size: int | None = None
    createdTime: str | None = None
    modifiedTime: str | None = None
    webViewLink: str | None = None
    parents: list[str] | None = None


class DriveFileListResponse(BaseModel):
    files: list[DriveFileResponse]
    nextPageToken: str | None = None
    totalCount: int


class CreateFolderRequest(BaseModel):
    name: str
    parentId: str | None = None


class ShareFileRequest(BaseModel):
    fileId: str
    email: str
    role: str = "reader"


class SearchQuery(BaseModel):
    query: str
    pageSize: int = 20
    pageToken: str | None = None


class UploadResponse(BaseModel):
    id: str
    name: str
    mimeType: str | None = None
    size: int | None = None
    webViewLink: str | None = None
