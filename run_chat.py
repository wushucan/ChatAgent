"""Chat Agent Desktop — Entry point.

Usage:
    python run_chat.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Force UTF-8 for stdout/stderr
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Git Bash env vars that break httpx/OpenAI
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import webview

# ── PyInstaller 兼容路径 ──────────────────────────────────────────

def _app_root() -> Path:
    """Return the writable app directory (next to the .exe or script)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def _bundle_root() -> Path:
    """Return the bundle extraction directory (PyInstaller tmp or script dir)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

APP_ROOT = _app_root()
BUNDLE_ROOT = _bundle_root()
sys.path.insert(0, str(BUNDLE_ROOT))

from agent.api import AgentAPI


def _load_config() -> dict:
    """Load config.json — user config next to exe, fallback to bundled default."""
    # User-editable config next to .exe
    user_config = APP_ROOT / "config.json"
    if user_config.is_file():
        with open(user_config, "r", encoding="utf-8") as f:
            return json.load(f)

    # Bundled default
    bundled = BUNDLE_ROOT / "config.json"
    if bundled.is_file():
        with open(bundled, "r", encoding="utf-8") as f:
            return json.load(f)

    # Last resort: .env
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "system_prompt": os.getenv("SYSTEM_PROMPT", ""),
    }


def _load_html() -> str:
    html_path = BUNDLE_ROOT / "frontend" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    config = _load_config()
    api = AgentAPI(config, data_dir=str(APP_ROOT / "sessions"))

    html = _load_html()

    window = webview.create_window(
        title="Chat Agent",
        html=html,
        js_api=api,
        width=1000,
        height=700,
        min_size=(800, 500),
    )

    webview.start(debug=False, http_server=False)


if __name__ == "__main__":
    main()
