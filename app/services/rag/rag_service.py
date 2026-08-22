import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import getSettings
from app.core.exceptions import AppException, NotFoundException
from app.core.logging import logger
from app.models.file_embedding import FileEmbedding
from app.services.google_drive_service import GoogleDriveService
from app.services.rag.document_loader import (
    DocumentLoader,
    exportMimeFor,
    isGoogleWorkspaceMime,
)
from app.services.vector_store.vector_store_service import VectorStoreService


class RagService:
    """Indexes Drive files into a per-user vector store and answers retrieval queries."""

    def __init__(self, db: AsyncSession, userId: uuid.UUID, drive: GoogleDriveService | None = None):
        self._db = db
        self._userId = userId
        self._settings = getSettings()
        self._drive = drive or GoogleDriveService(db, userId)
        self._vectorStore = VectorStoreService(userId)
        self._loader = DocumentLoader()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.RAG_CHUNK_SIZE,
            chunk_overlap=self._settings.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ---------- public ----------

    async def indexFile(self, fileId: str, forceReindex: bool = False) -> dict[str, Any]:
        meta = await self._drive.getFileMetadata(fileId)
        mimeType = meta.get("mimeType") or "application/octet-stream"
        fileName = meta.get("name") or fileId
        sizeBytes = int(meta.get("size") or 0)

        self._assertSupported(mimeType, sizeBytes, fileName)

        existing = await self._getEmbeddingRow(fileId)
        contentBytes = await self._fetchBytes(fileId, mimeType)
        contentHash = hashlib.sha256(contentBytes).hexdigest()

        if existing and existing.contentHash == contentHash and not forceReindex and existing.status == "ready":
            logger.info("rag_file_cache_hit", userId=str(self._userId), fileId=fileId)
            return self._rowToDict(existing, cached=True)

        text = self._loader.parse(contentBytes, mimeType, fileName)
        if not text.strip():
            raise AppException(400, "RAG_EMPTY_DOCUMENT", f"No extractable text in '{fileName}'")

        docs = self._chunk(text, fileId=fileId, fileName=fileName, mimeType=mimeType)

        if existing:
            self._vectorStore.deleteByFileId(fileId)
        chunkCount = self._vectorStore.addDocuments(docs, fileId=fileId)

        row = existing or FileEmbedding(userId=self._userId, googleFileId=fileId)
        row.fileName = fileName
        row.mimeType = mimeType
        row.fileSize = sizeBytes or len(contentBytes)
        row.chunkCount = chunkCount
        row.contentHash = contentHash
        row.status = "ready"
        row.errorMessage = None
        row.updatedAt = datetime.now(timezone.utc)
        if not existing:
            self._db.add(row)
        await self._db.flush()

        logger.info("rag_file_indexed", userId=str(self._userId), fileId=fileId, chunks=chunkCount)
        return self._rowToDict(row, cached=False)

    async def removeFile(self, fileId: str) -> None:
        row = await self._getEmbeddingRow(fileId)
        if not row:
            raise NotFoundException("Indexed file")
        self._vectorStore.deleteByFileId(fileId)
        await self._db.delete(row)
        await self._db.flush()
        logger.info("rag_file_removed", userId=str(self._userId), fileId=fileId)

    async def listIndexed(self) -> list[dict[str, Any]]:
        result = await self._db.execute(
            select(FileEmbedding).where(FileEmbedding.userId == self._userId).order_by(FileEmbedding.updatedAt.desc())
        )
        return [self._rowToDict(r) for r in result.scalars().all()]

    async def query(self, question: str, fileIds: list[str] | None = None, topK: int | None = None) -> dict[str, Any]:
        if fileIds:
            for fid in fileIds:
                row = await self._getEmbeddingRow(fid)
                if not row or row.status != "ready":
                    await self.indexFile(fid)

        indexedCount = await self._countIndexed()
        docs = self._vectorStore.similaritySearch(question, k=topK, fileIds=fileIds)

        if not docs:
            reason = "no_indexed_files" if indexedCount == 0 else "no_matches"
            return {
                "status": "no_matches",
                "reason": reason,
                "indexedFileCount": indexedCount,
                "answerContext": "",
                "chunks": [],
                "sources": [],
            }

        chunks = [
            {
                "fileId": d.metadata.get("fileId"),
                "fileName": d.metadata.get("fileName"),
                "chunkIndex": d.metadata.get("chunkIndex"),
                "content": d.page_content,
            }
            for d in docs
        ]
        sources = list({(c["fileId"], c["fileName"]) for c in chunks})
        context = "\n\n---\n\n".join(
            f"[Source: {c['fileName']} | chunk {c['chunkIndex']}]\n{c['content']}" for c in chunks
        )
        return {
            "status": "ok",
            "indexedFileCount": indexedCount,
            "answerContext": context,
            "chunks": chunks,
            "sources": [{"fileId": fid, "fileName": fname} for fid, fname in sources],
        }

    # ---------- internals ----------

    def _assertSupported(self, mimeType: str, sizeBytes: int, fileName: str) -> None:
        limit = self._settings.RAG_MAX_FILE_SIZE_MB * 1024 * 1024
        if sizeBytes and sizeBytes > limit:
            raise AppException(
                413,
                "RAG_FILE_TOO_LARGE",
                f"File '{fileName}' exceeds RAG limit of {self._settings.RAG_MAX_FILE_SIZE_MB} MB",
            )
        if mimeType not in self._settings.RAG_SUPPORTED_MIMETYPES:
            raise AppException(400, "RAG_UNSUPPORTED_MIME", f"Unsupported file type for RAG: {mimeType}")

    async def _fetchBytes(self, fileId: str, mimeType: str) -> bytes:
        exportMime: str | None = None
        if isGoogleWorkspaceMime(mimeType):
            exportMime = exportMimeFor(mimeType)
            if not exportMime:
                raise AppException(400, "RAG_UNSUPPORTED_MIME", f"Cannot export Google file: {mimeType}")
        return await self._drive.fetchBytes(fileId, mimeType, exportMimeType=exportMime)

    def _chunk(self, text: str, fileId: str, fileName: str, mimeType: str) -> list[Document]:
        raw = self._splitter.split_text(text)
        docs: list[Document] = []
        for i, chunk in enumerate(raw):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "fileId": fileId,
                        "fileName": fileName,
                        "mimeType": mimeType,
                        "chunkIndex": i,
                        "userId": str(self._userId),
                    },
                )
            )
        return docs

    async def _getEmbeddingRow(self, fileId: str) -> FileEmbedding | None:
        result = await self._db.execute(
            select(FileEmbedding).where(
                FileEmbedding.userId == self._userId,
                FileEmbedding.googleFileId == fileId,
            )
        )
        return result.scalar_one_or_none()

    async def _countIndexed(self) -> int:
        from sqlalchemy import func
        result = await self._db.execute(
            select(func.count(FileEmbedding.id)).where(
                FileEmbedding.userId == self._userId,
                FileEmbedding.status == "ready",
            )
        )
        return int(result.scalar() or 0)

    def _rowToDict(self, row: FileEmbedding, cached: bool | None = None) -> dict[str, Any]:
        out = {
            "fileId": row.googleFileId,
            "fileName": row.fileName,
            "mimeType": row.mimeType,
            "fileSize": row.fileSize,
            "chunkCount": row.chunkCount,
            "status": row.status,
            "updatedAt": row.updatedAt.isoformat() if row.updatedAt else None,
        }
        if cached is not None:
            out["cached"] = cached
        return out
