import json
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import getSettings
from app.core.exceptions import AppException
from app.core.logging import logger
from app.services.agent.prompts import SYSTEM_PROMPT
from app.services.llm.llm_factory import LLMFactory, LLMSpec
from app.services.tools.drive_toolkit import buildDriveToolkit


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class AgentService:
    """LangGraph-based ReAct agent wired with Drive + RAG tools for a specific user."""

    def __init__(self, db: AsyncSession, userId: uuid.UUID, llmSpec: LLMSpec | None = None):
        self._db = db
        self._userId = userId
        self._settings = getSettings()
        self._llmSpec = llmSpec or LLMSpec.default()
        self._tools = buildDriveToolkit(db, userId)
        self._toolsByName = {t.name: t for t in self._tools}

        baseLlm = LLMFactory.build(self._llmSpec)
        self._llm = baseLlm.bind_tools(self._tools)
        self._graph = self._buildGraph()

    # ---------- public API ----------

    async def run(self, history: list[BaseMessage], userInput: str) -> dict[str, Any]:
        initialMessages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=userInput)]
        try:
            result = await self._graph.ainvoke(
                {"messages": initialMessages},
                config={"recursion_limit": self._settings.AGENT_MAX_ITERATIONS * 2 + 4},
            )
        except Exception as e:
            logger.error("agent_run_failed", userId=str(self._userId), error=str(e))
            raise AppException(500, "AGENT_FAILED", f"Agent execution failed: {e}")

        newMessages = result["messages"][len(initialMessages):]
        finalAi = next((m for m in reversed(newMessages) if isinstance(m, AIMessage) and not m.tool_calls), None)
        if finalAi is None:
            finalAi = AIMessage(content="I couldn't produce a final response. Please try rephrasing.")

        return {
            "reply": finalAi.content if isinstance(finalAi.content, str) else str(finalAi.content),
            "newMessages": newMessages,
            "finalMessage": finalAi,
        }

    # ---------- graph ----------

    def _buildGraph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agentNode)
        graph.add_node("tools", self._toolsNode)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._shouldContinue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    async def _agentNode(self, state: AgentState) -> AgentState:
        response = await self._llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def _shouldContinue(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    async def _toolsNode(self, state: AgentState) -> AgentState:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        outputs: list[ToolMessage] = []
        for call in last.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            callId = call.get("id") or ""
            tool = self._toolsByName.get(name)
            if not tool:
                outputs.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=callId, name=name or "unknown"))
                continue
            try:
                raw = await tool.ainvoke(args)
                content = self._serializeToolResult(raw)
            except AppException as e:
                content = json.dumps({"error": e.detail.get("message"), "code": e.detail.get("errorCode")})
                logger.warning("tool_invocation_app_error", tool=name, error=content)
            except Exception as e:
                content = json.dumps({"error": str(e), "code": "TOOL_ERROR"})
                logger.error("tool_invocation_failed", tool=name, error=str(e))
            outputs.append(ToolMessage(content=content, tool_call_id=callId, name=name))
        return {"messages": outputs}

    # ---------- helpers ----------

    @staticmethod
    def _serializeToolResult(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)
