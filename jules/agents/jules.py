"""Heavyweight Jules responder that optionally consumes a cheat-sheet."""

from __future__ import annotations

import datetime
from pathlib import Path
import os
from typing import List

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]
# Optional langchain import with local stubs.

try:
    from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage  # type: ignore
except ImportError:  # pragma: no cover – dependency not present
    class _BareMsg(str):
        @property
        def content(self):
            return str(self)

    AIMessage = HumanMessage = SystemMessage = BaseMessage = _BareMsg  # type: ignore

from jules.config import JulesCfg, get_agent_cfg


class JulesAgent:
    """Final response generator.

    The agent places the user message *first* and, when available, appends a
    ``### Background`` section followed by the cheat-sheet.  This keeps
    alignment with system-prompt guidelines while ensuring the LLM has quick
    access to the freshly distilled context.
    """

    def __init__(self, cfg: JulesCfg | None = None, *, llm: ChatOpenAI | None = None):
        self.cfg = cfg or get_agent_cfg("jules")  # type: ignore[arg-type]
        if llm is not None:
            self.llm = llm
        elif ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(  # type: ignore[call-arg]
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
        else:
            # Offline echo style
            from langchain.schema import AIMessage  # locally scoped to avoid heavy import

            class _EchoLLM:
                async def ainvoke(self, _msgs, **_):
                    return AIMessage("[offline] Jules response TBD")  # type: ignore[arg-type]

            self.llm = _EchoLLM()

        # Root system prompt lives next to backend prompts for consistency
        root_path = Path(__file__).parents[3] / "backend" / "app" / "prompts" / "root.system.md"
        try:
            self._system_text = root_path.read_text(encoding="utf-8").strip()
        except Exception:
            self._system_text = ""

    async def __call__(self, user_message: str, *, cheat_sheet: str = "") -> str:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        user_block = user_message.strip()
        if cheat_sheet:
            prompt = f"{user_block}\n\n### Background\n{cheat_sheet}\n\n### Reply"
        else:
            prompt = f"{user_block}\n\n### Reply"

        msgs: List[BaseMessage] = []
        if self._system_text:
            msgs.append(SystemMessage(content=self._system_text))
        msgs.append(HumanMessage(content=prompt, additional_kwargs={"timestamp": now}))

        ai_msg: AIMessage = await self.llm.ainvoke(msgs)
        return ai_msg.content
