import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import getSettings
from app.core.logging import logger
from app.models.conversation import Conversation
from app.services.agent.agent_service import AgentService
from app.services.chat.conversation_service import ConversationService
from app.services.llm.llm_factory import LLMSpec


class ChatService:
    """Orchestrates a chat turn: persist user msg → run agent → persist assistant+tool msgs → return payload."""

    def __init__(self, db: AsyncSession, userId: uuid.UUID):
        self._db = db
        self._userId = userId
        self._settings = getSettings()
        self._conv = ConversationService(db, userId)

    async def sendMessage(
        self,
        message: str,
        conversationId: uuid.UUID | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        maxTokens: int | None = None,
    ) -> dict[str, Any]:
        spec = self._resolveSpec(provider, model, temperature, maxTokens)
        conv = await self._ensureConversation(conversationId, message, spec)

        await self._conv.addUserMessage(conv.id, message)
        history = await self._conv.buildAgentHistory(conv.id)
        history = history[:-1] if history else history  # drop the just-persisted user msg

        agent = AgentService(self._db, self._userId, llmSpec=spec)
        logger.info(
            "chat_turn_start",
            userId=str(self._userId),
            conversationId=str(conv.id),
            provider=spec.provider,
            model=spec.model,
        )
        result = await agent.run(history=history, userInput=message)

        persistedAssistant = await self._conv.addAssistantMessages(
            conv.id,
            result["newMessages"],
        )

        logger.info(
            "chat_turn_complete",
            userId=str(self._userId),
            conversationId=str(conv.id),
            toolCallCount=sum(1 for m in persistedAssistant if m.role == "tool"),
        )
        return {
            "conversationId": str(conv.id),
            "reply": result["reply"],
            "messages": [ConversationService.messageToDict(m) for m in persistedAssistant],
        }

    # ---------- helpers ----------

    def _resolveSpec(
        self,
        provider: str | None,
        model: str | None,
        temperature: float | None,
        maxTokens: int | None,
    ) -> LLMSpec:
        defaults = LLMSpec.default()
        return LLMSpec(
            provider=(provider or defaults.provider).lower(),
            model=model or defaults.model,
            temperature=defaults.temperature if temperature is None else temperature,
            maxTokens=defaults.maxTokens if maxTokens is None else maxTokens,
        )

    async def _ensureConversation(
        self, conversationId: uuid.UUID | None, firstMessage: str, spec: LLMSpec
    ) -> Conversation:
        if conversationId:
            return await self._conv.getConversation(conversationId)
        title = firstMessage.strip().splitlines()[0][: self._settings.CHAT_TITLE_MAX_LEN] or "New chat"
        return await self._conv.createConversation(
            title=title, modelProvider=spec.provider, modelName=spec.model
        )
