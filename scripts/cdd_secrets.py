from __future__ import annotations

import os
from pathlib import Path

CDD_API_TOKEN_ENV_VAR = "CDD_API_TOKEN"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _read_dotenv_value(name: str) -> str:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return _strip_quotes(value)
    except Exception:
        return ""

    return ""


def get_required_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value

    value = _read_dotenv_value(name).strip()
    if value:
        return value

    raise RuntimeError(
        f"Missing required secret {name}. Set it in the environment or project .env file."
    )


def get_cdd_api_token() -> str:
    return get_required_secret(CDD_API_TOKEN_ENV_VAR)
