import uuid
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import getSettings
from app.core.exceptions import AppException
from app.core.logging import logger
from app.services.vector_store.embedding_service import EmbeddingService


class VectorStoreService:
    """Per-user Chroma collection wrapper.
    Primary: Chroma Cloud (when CHROMA_API_KEY is configured).
    Fallback: local PersistentClient if cloud is unreachable or misconfigured."""

    _clients: dict[str, chromadb.api.ClientAPI] = {}
    _activeMode: str | None = None

    def __init__(self, userId: uuid.UUID):
        self._userId = userId
        self._settings = getSettings()
        self._embeddings = EmbeddingService.build()
        self._store = self._buildStore()

    def _collectionName(self) -> str:
        return f"{self._settings.CHROMA_COLLECTION_PREFIX}{str(self._userId).replace('-', '')}"

    # ---------- client resolution (cloud-primary, local-fallback) ----------

    def _cloudConfigured(self) -> bool:
        return bool(self._settings.CHROMA_API_KEY)

    def _cloudClient(self) -> chromadb.api.ClientAPI:
        key = f"cloud:{self._settings.CHROMA_HOST}:{self._settings.CHROMA_DATABASE}"
        if key not in self._clients:
            kwargs: dict[str, Any] = {"api_key": self._settings.CHROMA_API_KEY}
            if self._settings.CHROMA_TENANT:
                kwargs["tenant"] = self._settings.CHROMA_TENANT
            if self._settings.CHROMA_DATABASE:
                kwargs["database"] = self._settings.CHROMA_DATABASE
            if self._settings.CHROMA_HOST != "api.trychroma.com":
                kwargs["cloud_host"] = self._settings.CHROMA_HOST
                kwargs["cloud_port"] = 443
            self._clients[key] = chromadb.CloudClient(**kwargs)
        return self._clients[key]

    def _localClient(self) -> chromadb.api.ClientAPI:
        key = f"local:{self._settings.CHROMA_PERSIST_DIR}"
        if key not in self._clients:
            Path(self._settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            self._clients[key] = chromadb.PersistentClient(path=self._settings.CHROMA_PERSIST_DIR)
        return self._clients[key]

    def _resolveClient(self) -> chromadb.api.ClientAPI:
        if self._cloudConfigured():
            try:
                client = self._cloudClient()
                client.heartbeat()
                if VectorStoreService._activeMode != "cloud":
                    logger.info("vector_store_mode", mode="cloud", host=self._settings.CHROMA_HOST)
                    VectorStoreService._activeMode = "cloud"
                return client
            except Exception as e:
                logger.warning("chroma_cloud_unreachable", error=str(e))

        if VectorStoreService._activeMode != "local":
            logger.info("vector_store_mode", mode="local", path=self._settings.CHROMA_PERSIST_DIR)
            VectorStoreService._activeMode = "local"
        return self._localClient()

    def _buildStore(self) -> Chroma:
        return Chroma(
            client=self._resolveClient(),
            collection_name=self._collectionName(),
            embedding_function=self._embeddings,
        )

    # ---------- mutations ----------

    def addDocuments(self, docs: list[Document], fileId: str) -> int:
        if not docs:
            return 0
        for d in docs:
            d.metadata.setdefault("fileId", fileId)
            d.metadata.setdefault("userId", str(self._userId))
        ids = [f"{fileId}:{i}" for i in range(len(docs))]
        try:
            self._store.add_documents(docs, ids=ids)
        except Exception as e:
            logger.error("vector_add_failed", userId=str(self._userId), fileId=fileId, error=str(e))
            raise AppException(500, "VECTOR_ADD_FAILED", f"Failed to index documents: {e}")
        logger.info("vector_add_success", userId=str(self._userId), fileId=fileId, count=len(docs))
        return len(docs)

    def deleteByFileId(self, fileId: str) -> None:
        try:
            self._store.delete(where={"fileId": fileId})
            logger.info("vector_delete_success", userId=str(self._userId), fileId=fileId)
        except Exception as e:
            logger.error("vector_delete_failed", fileId=fileId, error=str(e))
            raise AppException(500, "VECTOR_DELETE_FAILED", f"Failed to remove vectors: {e}")

    # ---------- retrieval ----------

    def similaritySearch(
        self,
        query: str,
        k: int | None = None,
        fileIds: list[str] | None = None,
    ) -> list[Document]:
        k = k or self._settings.RAG_TOP_K
        filter_: dict[str, Any] | None = None
        if fileIds:
            filter_ = {"fileId": {"$in": fileIds}} if len(fileIds) > 1 else {"fileId": fileIds[0]}
        try:
            return self._store.similarity_search(query, k=k, filter=filter_)
        except Exception as e:
            logger.error("vector_search_failed", userId=str(self._userId), error=str(e))
            raise AppException(500, "VECTOR_SEARCH_FAILED", f"Vector search failed: {e}")

    def hasFile(self, fileId: str) -> bool:
        try:
            got = self._store.get(where={"fileId": fileId}, limit=1)
            return bool(got and got.get("ids"))
        except Exception:
            return False
