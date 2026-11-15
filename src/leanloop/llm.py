"""Thin Ollama client. Returns the reply plus exact token counts reported by Ollama."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"


class LLMError(RuntimeError):
    """A problem talking to Ollama, worded so the user knows what to do about it."""


def _request(path: str, body: dict | None, model: str, timeout: int) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{OLLAMA}{path}", data, {"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", errors="replace")
        if e.code == 404:
            raise LLMError(f"model '{model}' was not found in Ollama. See `ollama list`, or switch with /model NAME.") from None
        raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach Ollama at {OLLAMA} ({e.reason}). Start it with `ollama serve`.") from None
    except (TimeoutError, OSError) as e:      # socket.timeout is a TimeoutError; a dropped connection is an OSError
        raise LLMError(f"Ollama did not answer within {timeout}s ({type(e).__name__}).") from None
    except json.JSONDecodeError:
        raise LLMError("Ollama sent a reply that is not valid JSON.") from None


_CALL_AS_TEXT = re.compile(r"(?:<tool_call>\s*)?(?:```(?:json)?\s*)?(\{.*\})(?:\s*```)?(?:\s*</tool_call>)?", re.DOTALL)


def _salvage_tool_call(msg: dict, tools: list[dict] | None) -> None:
    """Some models (Qwen2.5-Coder through Ollama, for one) print a tool call as JSON text instead of using the
    structured tool_calls field, so a harness that trusts the field sees plain chat and never runs the tool. If tools
    were offered and the whole reply is exactly a call to one of them, turn it into a real tool call."""
    if msg.get("tool_calls") or not tools:
        return
    m = _CALL_AS_TEXT.fullmatch((msg.get("content") or "").strip())
    if not m:
        return
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return
    if not isinstance(d, dict):
        return
    name, args = d.get("name"), d.get("arguments", d.get("parameters", {}))
    if name in {t["function"]["name"] for t in tools} and isinstance(args, (dict, str)):
        msg["tool_calls"] = [{"function": {"name": name, "arguments": args}}]
        msg["content"] = ""


def chat(model: str, messages: list[dict], tools: list[dict] | None = None, *, num_ctx: int = 8192,
         num_predict: int = 1024, temperature: float = 0.0, think: bool = False, timeout: int = 600) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}}
    if tools:
        body["tools"] = tools
    d = _request("/api/chat", body, model, timeout)
    if d.get("error"):
        raise LLMError(f"Ollama error: {d['error']}")
    msg = d.get("message")
    if not isinstance(msg, dict):
        raise LLMError("Ollama sent a reply without a message.")
    msg["content"] = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.DOTALL).strip()
    _salvage_tool_call(msg, tools)
    return {"message": msg, "prompt_tokens": d.get("prompt_eval_count", 0), "completion_tokens": d.get("eval_count", 0)}


def list_models(timeout: int = 5) -> list[str]:
    """Names of the models Ollama has (raises LLMError if it is not reachable)."""
    d = _request("/api/tags", None, "", timeout)
    return [m.get("name", "") for m in d.get("models", [])]
