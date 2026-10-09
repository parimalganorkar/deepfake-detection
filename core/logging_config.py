"""Structured logging setup.

Replaces the old ``st.warning`` / ``print`` error handling with real logging that works
identically in the Streamlit app, the FastAPI service, and tests. JSON output is the default
so logs are machine-parseable in production (e.g. in Docker).
"""

from __future__ import annotations

import json
import logging
import sys

from .config import settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter — one object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = (level or settings.log_level).upper()
    json_logs = settings.json_logs if json_logs is None else json_logs

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
