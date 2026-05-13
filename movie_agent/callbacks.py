"""
LangChain callback handlers: streaming output, tool debugging, and token tracking.
"""

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Streaming — collects tokens as the LLM generates them
# ---------------------------------------------------------------------------

class StreamingHandler(BaseCallbackHandler):
    """Captures tokens via ``on_llm_new_token`` for progressive UI updates.

    Tokens are appended to an internal list; use the ``text`` property to get
    the concatenated output so far, or ``consume()`` to drain tokens.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = []

    # -- ignore everything except token generation for perf -----------------
    @property
    def ignore_llm(self) -> bool:
        return False

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return True

    @property
    def ignore_tool(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    # -- the one hook we care about -----------------------------------------

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    def reset(self) -> None:
        self.tokens.clear()


# ---------------------------------------------------------------------------
# 2. Tool debugging — logs every tool invocation with name & parameters
# ---------------------------------------------------------------------------

class ToolDebugHandler(BaseCallbackHandler):
    """Prints every tool invocation (name + args + truncated output) for debugging.

    Set ``verbose`` to True to also print the full tool output.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.calls: list[dict[str, Any]] = []  # record of all tool calls

    # -- only listen to tool events ----------------------------------------
    @property
    def ignore_llm(self) -> bool:
        return True

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    # -- hooks --------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "unknown")
        msg = f"[TOOL START] {name}"
        logger.info("%s  Args: %s", msg, input_str[:500])
        print(f"\n{msg}")
        print(f"  Args: {input_str[:500]}{'...' if len(input_str) > 500 else ''}")
        self.calls.append({"name": name, "input": input_str, "output": None, "error": None})

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        output_str = str(output)
        if self.calls:
            self.calls[-1]["output"] = output_str
        truncated = output_str[:300] + ("..." if len(output_str) > 300 else "")
        print(f"[TOOL END]   Output ({len(output_str)} chars): {truncated}")
        if self.verbose:
            print(f"  Full output: {output_str}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self.calls:
            self.calls[-1]["error"] = str(error)
        logger.error("[TOOL ERROR] %s", error)
        print(f"[TOOL ERROR] {error}")


# ---------------------------------------------------------------------------
# 3. Token tracking — accumulates per-invocation & cumulative token usage
# ---------------------------------------------------------------------------

class TokenTracker(BaseCallbackHandler):
    """Tracks prompt / completion / total tokens across LLM calls.

    Probes multiple locations for token usage because different providers
    (OpenAI, MiniMax, etc.) and different modes (streaming vs non-streaming)
    store the data in different places:

    * ``llm_output["token_usage"]`` — standard non-streaming path
    * ``message.usage_metadata``  — streaming path (LangChain >= 0.3)
    * ``message.response_metadata["token_usage"]`` — OpenAI-format fallback

    Usage::

        tracker = TokenTracker()
        agent.ainvoke(..., config={"callbacks": [tracker]})
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.last_call: dict[str, int] = {}

    # -- only listen to LLM events -----------------------------------------
    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return True

    @property
    def ignore_tool(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Public API — allows external callers (e.g. chat_stream) to push
    # usage data that was extracted from astream_events metadata
    # ------------------------------------------------------------------

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Explicitly record token usage (used as a fallback when the
        callback hook doesn't fire or ``llm_output`` is empty)."""
        if prompt_tokens == 0 and completion_tokens == 0:
            return
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_count += 1
        self.last_call = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        logger.info(
            "[TOKEN] call #%d (explicit)  prompt=%d  completion=%d  total=%d",
            self.call_count,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict[str, int] | None:
        """Try every known location for token usage data.

        Returns a dict with keys ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``, or ``None`` if nothing was found.
        """
        # 1) llm_output["token_usage"] — standard non-streaming path
        if response.llm_output:
            tu = response.llm_output.get("token_usage")
            if tu:
                return {
                    "prompt_tokens": tu.get("prompt_tokens", 0),
                    "completion_tokens": tu.get("completion_tokens", 0),
                    "total_tokens": tu.get("total_tokens", 0),
                }

        # 2) message.usage_metadata — streaming path (LangChain >= 0.3)
        try:
            msg = response.generations[0][0].message  # type: ignore[index]
            um = getattr(msg, "usage_metadata", None)
            if um:
                return {
                    "prompt_tokens": um.get("input_tokens", 0),
                    "completion_tokens": um.get("output_tokens", 0),
                    "total_tokens": um.get("total_tokens", 0),
                }
        except (IndexError, AttributeError):
            pass

        # 3) message.response_metadata["token_usage"] — OpenAI-format fallback
        try:
            msg = response.generations[0][0].message  # type: ignore[index]
            rm = getattr(msg, "response_metadata", None)
            if rm:
                tu = rm.get("token_usage")
                if tu:
                    return {
                        "prompt_tokens": tu.get("prompt_tokens", 0),
                        "completion_tokens": tu.get("completion_tokens", 0),
                        "total_tokens": tu.get("total_tokens", 0),
                    }
        except (IndexError, AttributeError):
            pass

        return None

    # ------------------------------------------------------------------
    # Callback hook
    # ------------------------------------------------------------------

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        if usage is None:
            # Log a one-shot diagnostic so we can see what IS available
            logger.debug(
                "[TOKEN] on_llm_end fired but no usage found. "
                "llm_output=%s",
                response.llm_output,
            )
            return

        self.total_prompt_tokens += usage["prompt_tokens"]
        self.total_completion_tokens += usage["completion_tokens"]
        self.call_count += 1
        self.last_call = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }

        logger.info(
            "[TOKEN] call #%d  prompt=%d  completion=%d  total=%d",
            self.call_count,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
        )

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def summary(self) -> str:
        return (
            f"LLM calls: {self.call_count}  |  "
            f"Prompt tokens: {self.total_prompt_tokens}  |  "
            f"Completion tokens: {self.total_completion_tokens}  |  "
            f"Total tokens: {self.total_tokens}"
        )

    def reset(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.last_call = {}
