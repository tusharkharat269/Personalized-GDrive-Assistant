import uuid
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.services.google_drive_service import GoogleDriveService
from app.services.rag.rag_service import RagService


# ---------- Input schemas (used by the LLM to know tool args) ----------


class ListFilesInput(BaseModel):
    folderId: str | None = Field(default=None, description="Optional Drive folder ID to list children of.")
    pageSize: int = Field(default=20, ge=1, le=100, description="Max files to return (1-100).")
    pageToken: str | None = Field(default=None, description="Opaque page token from a previous response.")


class SearchFilesInput(BaseModel):
    query: str = Field(description="Text to match against file names (substring, case-insensitive).")
    pageSize: int = Field(default=20, ge=1, le=100)


class FileIdInput(BaseModel):
    fileId: str = Field(description="Google Drive file or folder ID.")


class CreateFolderInput(BaseModel):
    name: str = Field(description="Name of the folder to create.")
    parentId: str | None = Field(default=None, description="Optional parent folder ID.")


class ShareFileInput(BaseModel):
    fileId: str = Field(description="Drive file ID to share.")
    email: str = Field(description="Email address of the recipient.")
    role: str = Field(default="reader", description="Permission role: reader | commenter | writer.")


class ReadFileInput(BaseModel):
    fileId: str = Field(description="Drive file ID to read textual content from.")
    maxChars: int = Field(default=6000, ge=500, le=20000, description="Cap on returned characters.")


class IndexFileInput(BaseModel):
    fileId: str = Field(description="Drive file ID to index into the vector store for semantic search / QnA.")
    forceReindex: bool = Field(default=False, description="Re-index even if already indexed with same content.")


class QnaInput(BaseModel):
    question: str = Field(description="Natural-language question to answer from indexed files.")
    fileIds: list[str] | None = Field(
        default=None,
        description="Optional list of Drive file IDs to constrain retrieval. Omit to search across all indexed files.",
    )
    topK: int | None = Field(default=None, ge=1, le=15, description="Top-K chunks to retrieve.")


class EmptyInput(BaseModel):
    pass


# ---------- Toolkit builder ----------


def buildDriveToolkit(
    db: AsyncSession,
    userId: uuid.UUID,
    drive: GoogleDriveService | None = None,
    rag: RagService | None = None,
) -> list[StructuredTool]:
    """Build a user-scoped LangChain toolkit for Google Drive + RAG. Each tool closes
    over the given db session and userId so tool calls stay inside the request context."""

    driveSvc = drive or GoogleDriveService(db, userId)
    ragSvc = rag or RagService(db, userId, drive=driveSvc)

    # ---- Drive CRUD tools ----

    async def listFiles(folderId: str | None = None, pageSize: int = 20, pageToken: str | None = None) -> dict[str, Any]:
        logger.info("tool_listFiles", userId=str(userId), folderId=folderId)
        return await driveSvc.listFiles(pageSize=pageSize, pageToken=pageToken, folderId=folderId)

    async def searchFiles(query: str, pageSize: int = 20) -> dict[str, Any]:
        logger.info("tool_searchFiles", userId=str(userId), query=query)
        return await driveSvc.searchFiles(query=query, pageSize=pageSize)

    async def getFileMetadata(fileId: str) -> dict[str, Any]:
        logger.info("tool_getFileMetadata", userId=str(userId), fileId=fileId)
        return await driveSvc.getFileMetadata(fileId)

    async def createFolder(name: str, parentId: str | None = None) -> dict[str, Any]:
        logger.info("tool_createFolder", userId=str(userId), name=name)
        return await driveSvc.createFolder(name=name, parentId=parentId)

    async def deleteFile(fileId: str) -> dict[str, Any]:
        logger.info("tool_deleteFile", userId=str(userId), fileId=fileId)
        await driveSvc.deleteFile(fileId)
        return {"deleted": True, "fileId": fileId}

    async def shareFile(fileId: str, email: str, role: str = "reader") -> dict[str, Any]:
        logger.info("tool_shareFile", userId=str(userId), fileId=fileId, email=email, role=role)
        return await driveSvc.shareFile(fileId=fileId, email=email, role=role)

    async def readFileContent(fileId: str, maxChars: int = 6000) -> dict[str, Any]:
        logger.info("tool_readFileContent", userId=str(userId), fileId=fileId)
        meta = await driveSvc.getFileMetadata(fileId)
        mimeType = meta.get("mimeType") or "application/octet-stream"
        rawBytes = await ragSvc._fetchBytes(fileId, mimeType)  # noqa: SLF001
        text = ragSvc._loader.parse(rawBytes, mimeType, meta.get("name", ""))  # noqa: SLF001
        truncated = text[:maxChars]
        return {
            "fileId": fileId,
            "fileName": meta.get("name"),
            "mimeType": mimeType,
            "truncated": len(text) > maxChars,
            "content": truncated,
        }

    # ---- RAG tools ----

    async def indexFileForQna(fileId: str, forceReindex: bool = False) -> dict[str, Any]:
        logger.info("tool_indexFileForQna", userId=str(userId), fileId=fileId)
        return await ragSvc.indexFile(fileId, forceReindex=forceReindex)

    async def qnaOverFiles(question: str, fileIds: list[str] | None = None, topK: int | None = None) -> dict[str, Any]:
        logger.info("tool_qnaOverFiles", userId=str(userId), fileCount=len(fileIds or []))
        return await ragSvc.query(question=question, fileIds=fileIds, topK=topK)

    async def listIndexedFiles() -> dict[str, Any]:
        indexed = await ragSvc.listIndexed()
        return {"files": indexed, "totalCount": len(indexed)}

    # ---- Tool definitions with rich descriptions ----

    tools: list[StructuredTool] = [
        StructuredTool.from_function(
            name="listFiles",
            description=(
                "List files and folders from the user's Google Drive. "
                "Use this to browse the drive or list children of a specific folder. "
                "Returns file id, name, mimeType, size, modifiedTime, webViewLink, parents."
            ),
            coroutine=listFiles,
            args_schema=ListFilesInput,
        ),
        StructuredTool.from_function(
            name="searchFiles",
            description=(
                "Search Google Drive for files whose NAME contains the given substring. "
                "Use when the user references a file by partial name (e.g. 'find my invoice')."
            ),
            coroutine=searchFiles,
            args_schema=SearchFilesInput,
        ),
        StructuredTool.from_function(
            name="getFileMetadata",
            description=(
                "Fetch full metadata of a single Drive file or folder by its ID "
                "(name, size, mimeType, modified time, webViewLink, parents)."
            ),
            coroutine=getFileMetadata,
            args_schema=FileIdInput,
        ),
        StructuredTool.from_function(
            name="createFolder",
            description="Create a new folder in Google Drive, optionally inside a parent folder.",
            coroutine=createFolder,
            args_schema=CreateFolderInput,
        ),
        StructuredTool.from_function(
            name="deleteFile",
            description=(
                "Delete a file or folder in Google Drive by its ID. "
                "Only call after the user has explicitly confirmed deletion."
            ),
            coroutine=deleteFile,
            args_schema=FileIdInput,
        ),
        StructuredTool.from_function(
            name="shareFile",
            description=(
                "Share a Drive file with another user by email, granting reader, commenter, or writer role. "
                "Sends a notification email."
            ),
            coroutine=shareFile,
            args_schema=ShareFileInput,
        ),
        StructuredTool.from_function(
            name="readFileContent",
            description=(
                "Read the textual content of a Drive file (supports PDF, DOCX, XLSX, TXT, MD, HTML, JSON, "
                "and Google Docs/Sheets/Slides). Returns up to `maxChars` characters. "
                "Prefer `qnaOverFiles` when the document is large — that uses retrieval."
            ),
            coroutine=readFileContent,
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            name="indexFileForQna",
            description=(
                "Index a Drive file into the vector store so it can be used for semantic search and QnA. "
                "Call before `qnaOverFiles` when the file has not been indexed yet."
            ),
            coroutine=indexFileForQna,
            args_schema=IndexFileInput,
        ),
        StructuredTool.from_function(
            name="qnaOverFiles",
            description=(
                "Answer a natural-language question using Retrieval-Augmented Generation over the user's "
                "indexed Drive files. Pass `fileIds` to constrain to specific files, or omit to search all "
                "indexed files. Returns `status` ('ok' | 'no_matches'), `indexedFileCount`, relevant chunks, "
                "and source metadata. If `status` is 'no_matches', proceed to `searchFiles` to find relevant "
                "files in Drive, then `indexFileForQna` them, and call `qnaOverFiles` again."
            ),
            coroutine=qnaOverFiles,
            args_schema=QnaInput,
        ),
        StructuredTool.from_function(
            name="listIndexedFiles",
            description="List Drive files that have already been indexed for QnA in the vector store.",
            coroutine=listIndexedFiles,
            args_schema=EmptyInput,
        ),
    ]
    return tools
