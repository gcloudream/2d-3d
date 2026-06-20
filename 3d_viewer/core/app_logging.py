"""Application logging helpers for viewer operations and diagnostics."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np


LOGGER_NAME = "3d_viewer"
_ACTIVE_WORKSPACE: Path | None = None


def logs_dir(workspace: Path) -> Path:
    return Path(workspace) / "out" / "logs"


def operation_log_path(workspace: Path) -> Path:
    return logs_dir(workspace) / "viewer.log"


def operation_events_path(workspace: Path) -> Path:
    return logs_dir(workspace) / "viewer_events.jsonl"


def configure_operation_logging(workspace: Path) -> logging.Logger:
    global _ACTIVE_WORKSPACE
    root = Path(workspace).expanduser()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    active_handlers = [
        handler for handler in logger.handlers
        if getattr(handler, "_viewer_operation_handler", False)
    ]
    if _ACTIVE_WORKSPACE == root and active_handlers:
        return logger

    for handler in active_handlers:
        logger.removeHandler(handler)
        handler.close()

    log_path = operation_log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    ))
    handler._viewer_operation_handler = True
    logger.addHandler(handler)
    _ACTIVE_WORKSPACE = root
    return logger


def reset_operation_logging():
    global _ACTIVE_WORKSPACE
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_viewer_operation_handler", False):
            logger.removeHandler(handler)
            handler.close()
    _ACTIVE_WORKSPACE = None


def log_operation(
    workspace: Path,
    event: str,
    *,
    component: str = "viewer",
    level: str = "info",
    **fields,
) -> Path:
    """Append one operation event to JSONL and emit a text log line.

    Logging must never become the reason an interactive operation fails, so
    callers keep the event fields simple and this helper normalizes numpy/path
    values into JSON-safe primitives.
    """
    root = Path(workspace).expanduser()
    logger = configure_operation_logging(root)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "component": str(component),
        **_json_safe(fields),
    }
    event_path = operation_events_path(root)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        fh.write("\n")

    log_method = getattr(logger, str(level).lower(), logger.info)
    log_method("%s %s", event, json.dumps(_json_safe(fields), ensure_ascii=False, sort_keys=True))
    return event_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
