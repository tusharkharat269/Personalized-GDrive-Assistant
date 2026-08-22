import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000)
    conversationId: uuid.UUID | None = None
    provider: str | None = Field(default=None, description="openai | anthropic | openrouter")
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    maxTokens: int | None = Field(default=None, ge=64, le=16000)


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    toolName: str | None = None
    toolCallId: str | None = None
    toolCalls: list | None = None
    metadata: dict | None = None
    createdAt: str | None = None


class ChatResponse(BaseModel):
    conversationId: str
    reply: str
    messages: list[ChatMessageOut]


class ConversationOut(BaseModel):
    id: str
    title: str
    modelProvider: str | None = None
    modelName: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]
    totalCount: int


class ConversationDetailResponse(BaseModel):
    conversation: ConversationOut
    messages: list[ChatMessageOut]


class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)


class IndexFileRequest(BaseModel):
    fileId: str
    forceReindex: bool = False


class IndexedFileOut(BaseModel):
    fileId: str
    fileName: str
    mimeType: str | None = None
    fileSize: int | None = None
    chunkCount: int
    status: str
    updatedAt: str | None = None
    cached: bool | None = None


class IndexedFileListResponse(BaseModel):
    files: list[IndexedFileOut]
    totalCount: int
