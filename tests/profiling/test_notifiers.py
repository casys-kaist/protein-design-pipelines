from __future__ import annotations

import pytest

from profile.cli import notifier as notifier_mod
from profile.cli import run_sweeps as rs


def test_telegram_notifier_builds_payload(monkeypatch):
    sent = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, *_args, **_kwargs):  # pragma: no cover - unused
            return b""

    def fake_urlopen(request, timeout):  # noqa: D401 - helper stub
        sent["url"] = request.full_url
        sent["data"] = request.data
        sent["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(notifier_mod.urllib.request, "urlopen", fake_urlopen)

    notifier = rs.TelegramNotifier("123:abc", "555")
    notifier.send("hello world")

    assert "sendMessage" in sent["url"]
    assert b"hello+world" in sent["data"]
    assert sent["timeout"] == 10


def test_telegram_notifier_logs_failure(monkeypatch, capsys):
    def fail_urlopen(*_args, **_kwargs):
        raise notifier_mod.urllib.error.URLError("boom")

    monkeypatch.setattr(notifier_mod.urllib.request, "urlopen", fail_urlopen)

    notifier = rs.TelegramNotifier("123:abc", "555")
    notifier.send("hello world")

    captured = capsys.readouterr().out
    assert "[WARN] Telegram notifier failed" in captured


def test_slack_notifier_builds_payload(monkeypatch):
    sent = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, *_args, **_kwargs):  # pragma: no cover - unused
            return b""

    def fake_urlopen(request, timeout):  # noqa: D401 - helper stub
        sent["url"] = request.full_url
        sent["data"] = request.data
        sent["timeout"] = timeout
        sent["content_type"] = request.headers.get("Content-type")
        return DummyResponse()

    monkeypatch.setattr(notifier_mod.urllib.request, "urlopen", fake_urlopen)

    notifier = rs.SlackNotifier("https://hooks.slack.com/services/test")
    notifier.send("hello slack")

    assert sent["url"].startswith("https://hooks.slack.com/")
    assert b"hello slack" in sent["data"]
    assert sent["content_type"] == "application/json"
    assert sent["timeout"] == 10


def test_slack_notifier_logs_failure(monkeypatch, capsys):
    def fail_urlopen(*_args, **_kwargs):
        raise notifier_mod.urllib.error.URLError("boom")

    monkeypatch.setattr(notifier_mod.urllib.request, "urlopen", fail_urlopen)

    notifier = rs.SlackNotifier("https://hooks.slack.com/services/test")
    notifier.send("hello slack")

    captured = capsys.readouterr().out
    assert "[WARN] Slack notifier failed" in captured


@pytest.mark.skip(reason="Notifier integration tests disabled to avoid sending real alerts during development")
def test_telegram_notifier_uses_real_secret_when_available(capsys):
    telegram = _require_secret_section("telegram")
    bot_token = telegram.get("bot_token")
    chat_id = telegram.get("chat_id")
    if not bot_token or not chat_id:
        pytest.skip("telegram bot credentials incomplete")

    notifier = rs.TelegramNotifier(str(bot_token), str(chat_id))
    notifier.send("[profiling] Telegram notifier integration test")

    captured = capsys.readouterr().out
    assert "[WARN] Telegram notifier failed" not in captured


@pytest.mark.skip(reason="Notifier integration tests disabled to avoid sending real alerts during development")
def test_slack_notifier_uses_real_secret_when_available(capsys):
    slack = _require_secret_section("slack")
    webhook_url = slack.get("webhook_url")
    if not webhook_url:
        pytest.skip("slack webhook_url missing in secret.yaml")

    notifier = rs.SlackNotifier(str(webhook_url))
    notifier.send("[profiling] Slack notifier integration test")

    captured = capsys.readouterr().out
    assert "[WARN] Slack notifier failed" not in captured


def _require_secret_section(section: str) -> dict:
    secret_path = rs.SECRET_CONFIG_PATH
    if not secret_path.exists():
        pytest.skip("secret.yaml missing")
    cfg = rs.load_secret_config()
    if not isinstance(cfg, dict):
        pytest.skip("secret config invalid")
    section_cfg = cfg.get(section)
    if not isinstance(section_cfg, dict):
        pytest.skip(f"{section} section missing in secret.yaml")
    return section_cfg
