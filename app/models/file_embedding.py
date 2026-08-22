import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column


from app.core.database import Base


class FileEmbedding(Base):
    __tablename__ = "file_embeddings"
    __table_args__ = (UniqueConstraint("userId", "googleFileId", name="uq_user_file_embedding"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    userId: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    googleFileId: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fileName: Mapped[str] = mapped_column(String(1024), nullable=False)
    mimeType: Mapped[str] = mapped_column(String(255), nullable=True)
    fileSize: Mapped[int] = mapped_column(BigInteger, nullable=True)
    chunkCount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contentHash: Mapped[str] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    errorMessage: Mapped[str] = mapped_column(String(1024), nullable=True)
    createdAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
