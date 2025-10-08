"""OpenAI GPT-5 mini client wrapper using the Responses API."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
    from openai.types.responses import Response
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    Response = Any  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_RETRIES = 3
TOKENS_PER_DOLLAR_INPUT = 1_000_000  # placeholder; update with official pricing
TOKENS_PER_DOLLAR_OUTPUT = 1_000_000

if load_dotenv is not None:  # pragma: no cover - environment bootstrap
    load_dotenv()


@dataclass
class LLMStats:
    """Track usage statistics for monitoring costs and latency."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_sec: float = 0.0

    def record(self, response: Response, latency: float) -> None:
        self.total_requests += 1
        usage = getattr(response, "usage", None)
        if usage:
            self.total_input_tokens += getattr(usage, "input_tokens", 0)
            self.total_output_tokens += getattr(usage, "output_tokens", 0)
        self.total_latency_sec += latency

    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_sec / self.total_requests

    def estimated_cost_usd(self) -> float:
        input_cost = self.total_input_tokens / TOKENS_PER_DOLLAR_INPUT
        output_cost = self.total_output_tokens / TOKENS_PER_DOLLAR_OUTPUT
        return input_cost + output_cost


class OpenAIPlannerClient:
    """Thin wrapper around OpenAI Responses API with retries and logging."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout: int = 30,
        api_key: Optional[str] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover - runtime guard
            raise RuntimeError("openai package is not installed. `pip install openai`.")
        client_kwargs: Dict[str, Any] = {"timeout": request_timeout}
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it in the environment or pass api_key."
            )
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_retries = max_retries
        self.stats = LLMStats()

    def invoke(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Call Responses API with retries."""
        payload = {
            "model": self.model,
            "input": _format_messages(messages),
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if metadata:
            payload["metadata"] = metadata

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self.client.responses.create(**payload)
                latency = time.perf_counter() - start
                self.stats.record(response, latency)
                logger.debug(
                    "LLM call succeeded (attempt %d) in %.2fs", attempt, latency
                )
                return response
            except Exception as exc:  # pragma: no cover - network failures
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(2**attempt * 0.1)
        raise RuntimeError("LLM call failed") from last_error

    def invoke_text(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience wrapper returning concatenated text output."""
        response = self.invoke(messages, metadata=metadata)
        return self.extract_text(response)

    def decode_tool_call(self, response: Response) -> Dict[str, Any]:
        """Extract JSON tool call payload from the response."""
        output = response.output
        if not output:
            raise ValueError("LLM response missing output.")
        content = output[0].content  # type: ignore[index]
        if not content:
            raise ValueError("LLM response missing content.")

        for block in content:
            if block.type == "tool_call":
                try:
                    arguments = json.loads(block.arguments)
                except json.JSONDecodeError as err:
                    raise ValueError("Tool call is not valid JSON.") from err
                return {"name": block.name, "arguments": arguments}
            if block.type == "output_text":
                text = getattr(block, "text", "").strip().lower()
                if text == "done":
                    return {"name": "done", "arguments": {}}
        raise ValueError("No tool call found in LLM response.")

    @staticmethod
    def extract_text(response: Response) -> str:
        output = response.output
        if not output:
            raise ValueError("LLM response missing output.")
        text_chunks: List[str] = []
        for item in output:  # type: ignore[assignment]
            for block in getattr(item, "content", []) or []:
                if block.type == "output_text":
                    text_chunks.append(getattr(block, "text", ""))
        if not text_chunks:
            raise ValueError("No text content in LLM response.")
        return "\n".join(chunk.strip() for chunk in text_chunks if chunk)


def _format_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, str):
            formatted.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content,
                        }
                    ],
                }
            )
        else:
            formatted.append(message)
    return formatted

    def asdict(self) -> Dict[str, Any]:
        stats = self.stats
        return {
            "total_requests": stats.total_requests,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "avg_latency_sec": stats.avg_latency,
            "estimated_cost_usd": stats.estimated_cost_usd(),
        }
