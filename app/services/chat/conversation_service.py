import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import getSettings
from app.core.exceptions import NotFoundException
from app.core.logging import logger
from app.models.conversation import Conversation
from app.models.message import Message


class ConversationService:
    """Persists conversations & messages and reconstructs LangChain history for the agent."""

    def __init__(self, db: AsyncSession, userId: uuid.UUID):
        self._db = db
        self._userId = userId
        self._settings = getSettings()

    # ---------- Conversation CRUD ----------

    async def createConversation(
        self,
        title: str | None = None,
        modelProvider: str | None = None,
        modelName: str | None = None,
    ) -> Conversation:
        conv = Conversation(
            userId=self._userId,
            title=title or "New chat",
            modelProvider=modelProvider,
            modelName=modelName,
        )
        self._db.add(conv)
        await self._db.flush()
        logger.info("conversation_created", userId=str(self._userId), conversationId=str(conv.id))
        return conv

    async def getConversation(self, conversationId: uuid.UUID) -> Conversation:
        result = await self._db.execute(
            select(Conversation).where(
                Conversation.id == conversationId,
                Conversation.userId == self._userId,
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise NotFoundException("Conversation")
        return conv

    async def listConversations(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        result = await self._db.execute(
            select(Conversation)
            .where(Conversation.userId == self._userId)
            .order_by(Conversation.updatedAt.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def countConversations(self) -> int:
        result = await self._db.execute(
            select(func.count(Conversation.id)).where(Conversation.userId == self._userId)
        )
        return int(result.scalar() or 0)

    async def renameConversation(self, conversationId: uuid.UUID, title: str) -> Conversation:
        conv = await self.getConversation(conversationId)
        conv.title = title[: self._settings.CHAT_TITLE_MAX_LEN]
        await self._db.flush()
        return conv

    async def deleteConversation(self, conversationId: uuid.UUID) -> None:
        conv = await self.getConversation(conversationId)
        await self._db.delete(conv)
        await self._db.flush()
        logger.info("conversation_deleted", userId=str(self._userId), conversationId=str(conversationId))

    # ---------- Message CRUD ----------

    async def listMessages(self, conversationId: uuid.UUID) -> list[Message]:
        await self.getConversation(conversationId)  # ownership check
        result = await self._db.execute(
            select(Message)
            .where(Message.conversationId == conversationId)
            .order_by(
                Message.seq.asc().nulls_last(),
                Message.createdAt.asc(),
                Message.id.asc(),
            )
        )
        return list(result.scalars().all())

    async def addUserMessage(self, conversationId: uuid.UUID, content: str) -> Message:
        return await self._addMessage(conversationId, role="user", content=content)

    async def addAssistantMessages(
        self,
        conversationId: uuid.UUID,
        newMessages: list[BaseMessage],
        finalAssistantMeta: dict | None = None,
    ) -> list[Message]:
        """Persist the agent's turn. `finalAssistantMeta` (if provided) is attached as `metaData`
        on the LAST assistant AIMessage that has no tool_calls — i.e., the user-facing reply."""
        finalIndex = -1
        for i, m in enumerate(newMessages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                finalIndex = i

        persisted: list[Message] = []
        for i, m in enumerate(newMessages):
            if isinstance(m, AIMessage):
                toolCalls = None
                if m.tool_calls:
                    toolCalls = [
                        {"id": c.get("id"), "name": c.get("name"), "args": c.get("args")}
                        for c in m.tool_calls
                    ]
                meta = finalAssistantMeta if (finalAssistantMeta and i == finalIndex) else None
                persisted.append(
                    await self._addMessage(
                        conversationId,
                        role="assistant",
                        content=m.content if isinstance(m.content, str) else json.dumps(m.content, default=str),
                        toolCalls=toolCalls,
                        metaData=meta,
                    )
                )
            elif isinstance(m, ToolMessage):
                persisted.append(
                    await self._addMessage(
                        conversationId,
                        role="tool",
                        content=m.content if isinstance(m.content, str) else json.dumps(m.content, default=str),
                        toolName=m.name,
                        toolCallId=m.tool_call_id,
                    )
                )
        return persisted

    async def _addMessage(
        self,
        conversationId: uuid.UUID,
        role: str,
        content: str,
        toolCalls: list | None = None,
        toolName: str | None = None,
        toolCallId: str | None = None,
        metaData: dict | None = None,
    ) -> Message:
        nextSeq = await self._db.scalar(
            select(func.coalesce(func.max(Message.seq), 0) + 1).where(
                Message.conversationId == conversationId
            )
        )
        msg = Message(
            conversationId=conversationId,
            seq=int(nextSeq or 1),
            role=role,
            content=content or "",
            toolCalls=toolCalls,
            toolName=toolName,
            toolCallId=toolCallId,
            metaData=metaData,
        )
        self._db.add(msg)
        await self._db.flush()
        return msg

    # ---------- History for the agent ----------

    async def buildAgentHistory(self, conversationId: uuid.UUID) -> list[BaseMessage]:
        """Return the last N messages as LangChain BaseMessages, deterministically ordered by
        per-conversation `seq` and sanitized so the sequence is acceptable to OpenAI/Anthropic
        (every tool message must follow an assistant turn whose tool_calls contains its id;
        every assistant tool_call must be answered)."""
        result = await self._db.execute(
            select(Message)
            .where(Message.conversationId == conversationId)
            .order_by(
                Message.seq.desc().nulls_last(),
                Message.createdAt.desc(),
                Message.id.desc(),
            )
            .limit(self._settings.CHAT_HISTORY_WINDOW)
        )
        rows = list(reversed(list(result.scalars().all())))

        converted: list[BaseMessage] = []
        for m in rows:
            if m.role == "user":
                converted.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                toolCalls = None
                if m.toolCalls:
                    toolCalls = [
                        {"id": c.get("id"), "name": c.get("name"), "args": c.get("args") or {}}
                        for c in m.toolCalls
                        if c.get("id") and c.get("name")
                    ]
                converted.append(
                    AIMessage(content=m.content or "", tool_calls=toolCalls or [])
                )
            elif m.role == "tool":
                converted.append(
                    ToolMessage(
                        content=m.content or "",
                        tool_call_id=m.toolCallId or "",
                        name=m.toolName or "tool",
                    )
                )

        return self._sanitizeHistory(converted)

    @staticmethod
    def _sanitizeHistory(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Drop orphan ToolMessages and AIMessages whose tool_calls aren't fully answered.
        This guarantees the LLM API never receives a tool message that is not paired with
        a preceding assistant tool_call entry (avoids the 400 'must be a response to a
        preceeding message with tool_calls' error)."""
        out: list[BaseMessage] = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]

            if isinstance(m, ToolMessage):
                # Orphan tool message; drop it.
                i += 1
                continue

            if isinstance(m, AIMessage) and m.tool_calls:
                expectedIds = {tc.get("id") for tc in m.tool_calls if tc.get("id")}
                collected: list[ToolMessage] = []
                seenIds: set[str] = set()
                j = i + 1
                while j < n and isinstance(messages[j], ToolMessage):
                    tm = messages[j]  # type: ignore[assignment]
                    if tm.tool_call_id in expectedIds and tm.tool_call_id not in seenIds:
                        collected.append(tm)
                        seenIds.add(tm.tool_call_id)
                    j += 1

                if expectedIds and expectedIds.issubset(seenIds):
                    out.append(m)
                    out.extend(collected)
                else:
                    # Tool calls weren't fully answered (truncation, crash, max_tokens).
                    # Drop the assistant turn AND any partial tool replies.
                    pass
                i = j
                continue

            out.append(m)
            i += 1

        # Conversations should not start with an orphan AIMessage in the first slot only if
        # it has tool_calls (already handled above). Leading clean AIMessage / HumanMessage
        # are both valid first messages for OpenAI.
        return out

    # ---------- Utilities ----------

    async def purgeEmptyConversations(self) -> int:
        """Delete conversations for this user that have no messages. Returns count deleted."""
        subquery = (
            select(Conversation.id)
            .outerjoin(Message, Message.conversationId == Conversation.id)
            .where(Conversation.userId == self._userId)
            .group_by(Conversation.id)
            .having(func.count(Message.id) == 0)
        )
        ids = [row[0] for row in (await self._db.execute(subquery)).all()]
        if not ids:
            return 0
        await self._db.execute(delete(Conversation).where(Conversation.id.in_(ids)))
        await self._db.flush()
        return len(ids)

    @staticmethod
    def messageToDict(m: Message) -> dict[str, Any]:
        return {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "toolName": m.toolName,
            "toolCallId": m.toolCallId,
            "toolCalls": m.toolCalls,
            "metadata": m.metaData,
            "createdAt": m.createdAt.isoformat() if m.createdAt else None,
        }

    @staticmethod
    def conversationToDict(c: Conversation) -> dict[str, Any]:
        return {
            "id": str(c.id),
            "title": c.title,
            "modelProvider": c.modelProvider,
            "modelName": c.modelName,
            "createdAt": c.createdAt.isoformat() if c.createdAt else None,
            "updatedAt": c.updatedAt.isoformat() if c.updatedAt else None,
        }
