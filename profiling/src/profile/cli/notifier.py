"""Notification helpers for profiling sweeps."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROFILE_ROOT = Path(__file__).resolve().parent.parent
SECRET_CONFIG_PATH = PROFILE_ROOT / "secret.yaml"


class Notifier:
    """Base notifier interface."""

    def send(self, message: str) -> None:  # pragma: no cover - interface stub
        raise NotImplementedError


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.api_url = (
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            if self.bot_token
            else None
        )

    def send(self, message: str) -> None:
        if not message or not self.api_url or not self.chat_id:
            return
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "disable_web_page_preview": "true",
        }
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=data)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                resp.read(0)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"[WARN] Telegram notifier failed: {exc}")


class SlackNotifier(Notifier):
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = (webhook_url or "").strip()

    def send(self, message: str) -> None:
        if not message or not self.webhook_url:
            return
        payload = json.dumps({"text": message}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(self.webhook_url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                resp.read(0)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"[WARN] Slack notifier failed: {exc}")


def load_secret_config(path: Path = SECRET_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to read secret config {path}: {exc}")
        return {}


def build_default_notifier(path: Path = SECRET_CONFIG_PATH) -> Optional[Notifier]:
    secrets = load_secret_config(path)
    if not isinstance(secrets, dict):
        return None

    telegram = secrets.get("telegram")
    if isinstance(telegram, dict):
        bot_token = telegram.get("bot_token")
        chat_id = telegram.get("chat_id")
        if bot_token and chat_id:
            return TelegramNotifier(str(bot_token), str(chat_id))

    slack = secrets.get("slack")
    if isinstance(slack, dict):
        webhook_url = slack.get("webhook_url")
        if webhook_url:
            return SlackNotifier(str(webhook_url))
    return None


def get_secret_value(key: str, default: Any = None, *, path: Path = SECRET_CONFIG_PATH) -> Any:
    """Retrieve a nested value from the secret config using dot notation."""
    secrets = load_secret_config(path)
    if not isinstance(secrets, dict):
        return default
    parts = [part for part in key.split(".") if part]
    current: Any = secrets
    for part in parts:
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current if current is not None else default


__all__ = [
    "Notifier",
    "TelegramNotifier",
    "SlackNotifier",
    "load_secret_config",
    "build_default_notifier",
    "get_secret_value",
    "SECRET_CONFIG_PATH",
]
