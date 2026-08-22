from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import auth, chat, drive, health, rag
from app.core.config import getSettings
from app.core.database import Base, engine
from app.core.exceptions import AppException
from app.core.logging import logger, setupLogging
from app.middleware.error_handler import appExceptionHandler, unhandledExceptionHandler
from app.middleware.request_logging import RequestLoggingMiddleware
from app.models import (  # noqa: F401  (registers ORM models with Base)
    Conversation,
    DriveFileMetadata,
    FileEmbedding,
    GoogleDriveToken,
    Message,
    User,
)

settings = getSettings()


_BACKFILL_SEQ_SQL = text(
    """
    WITH ranked AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY "conversationId"
                   ORDER BY "createdAt", id
               ) AS rn
        FROM messages
        WHERE seq IS NULL
    )
    UPDATE messages m
    SET seq = ranked.rn
    FROM ranked
    WHERE m.id = ranked.id
    """
)


async def _runStartupMigrations(conn) -> None:
    """Idempotent inline schema fixes that complement Base.metadata.create_all (which only
    creates missing tables). Add columns/backfills here so existing databases stay healthy
    without a separate migration tool."""
    await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq INTEGER"))
    await conn.execute(_BACKFILL_SEQ_SQL)
    await conn.execute(
        text(
            'CREATE INDEX IF NOT EXISTS ix_messages_conversation_seq '
            'ON messages ("conversationId", seq)'
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setupLogging("DEBUG" if settings.DEBUG else "INFO")
    logger.info("app_starting", env=settings.APP_ENV, debug=settings.DEBUG)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _runStartupMigrations(conn)
    logger.info("app_migrations_applied")
    yield
    logger.info("app_stopping")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.add_exception_handler(AppException, appExceptionHandler)
app.add_exception_handler(Exception, unhandledExceptionHandler)

PREFIX = "/api/v1"
app.include_router(health.router, prefix=PREFIX)
app.include_router(auth.router, prefix=PREFIX)
app.include_router(drive.router, prefix=PREFIX)
app.include_router(rag.router, prefix=PREFIX)
app.include_router(chat.router, prefix=PREFIX)
