"""
LangChain callback handlers: streaming output, tool debugging, and token tracking.
"""

import logging
import time
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

    Each recorded call carries ``duration_ms``. It is what gets persisted into
    ``tool_invocations``: a slow MCP round-trip is the usual reason a reply feels
    stuck, and without a per-tool timing there is no way to tell that apart from
    a slow model.

    Records are keyed by ``run_id``, not by arrival order. The model fires tool
    calls in **parallel batches** (three ``on_tool_start`` before the first
    ``on_tool_end`` is routine), so pairing each end event with "the call that
    started last" silently mis-attributes it: the earlier calls of a batch keep
    ``output=None`` forever, and the last one gets overwritten by another tool's
    output with a duration timed from an unrelated start.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.calls: list[dict[str, Any]] = []  # record of all tool calls
        # run_id -> (record, start time). Entries are popped on end/error, so a
        # batch that finishes leaves nothing behind.
        self._in_flight: dict[Any, tuple[dict[str, Any], float]] = {}

    # -- only listen to tool events ----------------------------------------
    @property
    def ignore_llm(self) -> bool:
        return True

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        # NOTE: langchain-core gates on_tool_start / on_tool_end / on_tool_error
        # behind ``ignore_agent`` — there is no ``ignore_tool`` flag. Returning
        # True here would silence every tool event this handler exists to record.
        return False

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
        record = {
            "name": name,
            "input": input_str,
            "output": None,
            "error": None,
            "duration_ms": None,
        }
        self.calls.append(record)
        self._in_flight[kwargs.get("run_id")] = (record, time.perf_counter())

    def _finish(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
        """Claim the record this end/error event belongs to, plus its elapsed ms.

        Falls back to the most recent call when ``run_id`` is absent or unknown:
        a mis-attributed timing is still better than dropping the invocation, and
        the sequential case (which is what a missing run_id implies) is correct.
        """
        entry = self._in_flight.pop(kwargs.get("run_id"), None)
        if entry is None:
            record = self.calls[-1] if self.calls else None
            return record, None
        record, started = entry
        return record, int((time.perf_counter() - started) * 1000)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        # langchain >= 1.0 hands back a ToolMessage; unwrap it so the recorded
        # output is the tool's text rather than the message repr.
        output_str = str(getattr(output, "content", output))
        record, elapsed = self._finish(kwargs)
        if record is not None:
            record["output"] = output_str
            record["duration_ms"] = elapsed
        name = record["name"] if record else "unknown"
        truncated = output_str[:300] + ("..." if len(output_str) > 300 else "")
        print(f"[TOOL END]   {name} ({len(output_str)} chars, {elapsed} ms): {truncated}")
        if self.verbose:
            print(f"  Full output: {output_str}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        record, elapsed = self._finish(kwargs)
        if record is not None:
            record["error"] = str(error)
            record["duration_ms"] = elapsed
        logger.error("[TOOL ERROR] %s", error)
        print(f"[TOOL ERROR] {error}")


# ---------------------------------------------------------------------------
# 3. Token tracking — accumulates per-invocation & cumulative token usage
# ---------------------------------------------------------------------------

class TokenTracker(BaseCallbackHandler):
    """Tracks prompt / completion / total tokens across LLM calls.

    Probes multiple locations for token usage because different providers
    (OpenAI, DeepSeek, etc.) and different modes (streaming vs non-streaming)
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
