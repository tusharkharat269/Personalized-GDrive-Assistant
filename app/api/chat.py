import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import getDb
from app.core.deps import getCurrentUser
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationOut,
    RenameConversationRequest,
)
from app.services.chat.chat_service import ChatService
from app.services.chat.conversation_service import ConversationService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def sendChatMessage(
    body: ChatRequest,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    chat = ChatService(db, user.id)
    result = await chat.sendMessage(
        message=body.message,
        conversationId=body.conversationId,
        provider=body.provider,
        model=body.model,
        temperature=body.temperature,
        maxTokens=body.maxTokens,
    )
    return ChatResponse(**result)


@router.get("/conversations", response_model=ConversationListResponse)
async def listConversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = ConversationService(db, user.id)
    items = await svc.listConversations(limit=limit, offset=offset)
    total = await svc.countConversations()
    return ConversationListResponse(
        conversations=[ConversationOut(**ConversationService.conversationToDict(c)) for c in items],
        totalCount=total,
    )


@router.get("/conversations/{conversationId}", response_model=ConversationDetailResponse)
async def getConversation(
    conversationId: uuid.UUID,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = ConversationService(db, user.id)
    conv = await svc.getConversation(conversationId)
    messages = await svc.listMessages(conversationId)
    return ConversationDetailResponse(
        conversation=ConversationOut(**ConversationService.conversationToDict(conv)),
        messages=[ConversationService.messageToDict(m) for m in messages],
    )


@router.patch("/conversations/{conversationId}", response_model=ConversationOut)
async def renameConversation(
    conversationId: uuid.UUID,
    body: RenameConversationRequest,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = ConversationService(db, user.id)
    conv = await svc.renameConversation(conversationId, body.title)
    return ConversationOut(**ConversationService.conversationToDict(conv))


@router.delete("/conversations/{conversationId}")
async def deleteConversation(
    conversationId: uuid.UUID,
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    svc = ConversationService(db, user.id)
    await svc.deleteConversation(conversationId)
    return {"message": "Conversation deleted"}
