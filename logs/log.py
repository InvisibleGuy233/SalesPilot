"""Simple, dependency-free project logging."""

from __future__ import annotations

import inspect
import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_LOG_DIR = Path(__file__).resolve().parent
_LOG_LOCK = threading.Lock()
_SEPARATOR = "=" * 41


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _get_last_message(state: Any, last_message: Any) -> Any:
    if last_message is not None:
        return last_message
    if isinstance(state, Mapping):
        messages = state.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) and messages:
            return messages[-1]
    return None


def _message_content(message: Any) -> Any:
    if message is None:
        return ""
    if isinstance(message, Mapping):
        return message.get("content", message)
    return getattr(message, "content", message)


def _tool_calls(message: Any, explicit_tool_call: Any) -> Any:
    if explicit_tool_call is not None:
        return explicit_tool_call
    if message is None:
        return ""
    if isinstance(message, Mapping):
        return message.get("tool_calls", "")
    return getattr(message, "tool_calls", "") or ""


def _format_exception(error: Any) -> str:
    if error is None:
        return ""
    if isinstance(error, BaseException):
        return "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
    return _to_text(error)


def Log(
    message: Any,
    *,
    state: Any = None,
    last_message: Any = None,
    tool_call: Any = None,
    level: str = "INFO",
    error: Any = None,
    context: Any = None,
) -> Path:
    """Append one log entry and return the generated log file path.

    Basic usage: ``Log("想记录的内容")``.
    Pass ``state=state`` or ``last_message=message`` to record the latest
    LangGraph/LangChain message and automatically extract its tool calls.
    ``tool_call`` can be supplied explicitly when the call is not attached to
    the latest message.
    
    call methods: 
    Log(
        "RAG 检索完成",
        level="DEBUG",
        context={"result_count": 5, "query": query},
    )

    """
    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None

    function_name = caller.f_code.co_name if caller is not None else "<unknown>"
    source_file = Path(caller.f_code.co_filename).resolve() if caller is not None else ""
    source_line = caller.f_lineno if caller is not None else ""
    now = datetime.now().astimezone()
    current_message = _get_last_message(state, last_message)
    calls = _tool_calls(current_message, tool_call)

    fields = (
        ("function name", function_name),
        ("time", now.isoformat(timespec="milliseconds")),
        ("level", str(level).upper()),
        ("print", message),
        ("last message", _message_content(current_message)),
        ("tool_call", calls),
        ("source", f"{source_file}:{source_line}" if source_file else ""),
        ("thread", f"{threading.current_thread().name} ({threading.get_ident()})"),
        ("context", context),
        ("exception", _format_exception(error)),
    )
    entry = "\n".join(f"[{name}] {_to_text(value)}" for name, value in fields)
    entry += f"\n{_SEPARATOR}\n"

    log_path = _LOG_DIR / f"{now:%Y-%m-%d}.log"
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(entry)

    return log_path


log = Log

__all__ = ["Log", "log"]
