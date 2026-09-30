"""Persistent MCP server configuration — user-defined, not source-defined.

`data/mcp.json` is the source of truth for which MCP servers Athena can use:
transport, command/args or URL, enabled state, and per-tool disables. Nothing
here constrains the *set* of servers — Athena ships no bundled MCP server, so a
server whose name appears nowhere in Athena's source can be added, connected and
used at runtime.

Two deliberate departures from Odysseus, which stores `McpServer.env` as
plaintext JSON and returns it raw from `GET /api/mcp/servers`:

* environment values are encrypted at rest with the same `SecretBox` that
  protects provider API keys, and
* every API-facing row masks them (`••••`).

The UI can *write* a secret; it can never read one back. A client that PATCHes a
masked value back unchanged keeps the stored secret instead of overwriting it
with the mask.

Schema (version 1)::

    {
      "version": 1,
      "servers": {
        "<id>": {
          "id", "name", "transport", "command", "args": [...],
          "env": {"KEY": "enc:..."}, "url",
          "enabled", "disabled_tools": [...],
          "created_at", "updated_at"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.security.secrets import SecretBox, is_encrypted

if TYPE_CHECKING:  # avoid a config <-> mcp import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_FILENAME = "mcp.json"
# Reuses the provider key file: one Fernet key protects every secret Athena
# stores, so losing `data/` loses them together and there is no second key to
# back up.
KEY_FILENAME = ".provider_key"

TRANSPORTS = ("stdio", "sse", "http")

# What the API shows instead of a stored secret.
MASK = "••••"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "server"


def mask_env(env: Dict[str, str]) -> Dict[str, str]:
    """The API-facing view of an env map: keys are real, values never are."""
    return {str(key): (MASK if value else "") for key, value in (env or {}).items()}


class McpStore:
    def __init__(self, settings: "Settings", *, box: Optional[SecretBox] = None) -> None:
        self.settings = settings
        data_dir = settings.resolved_data_dir()
        self.path = data_dir / STORE_FILENAME
        self.box = box or SecretBox(data_dir / KEY_FILENAME)

    # ------------------------------------------------------------------
    # raw persistence
    # ------------------------------------------------------------------
    def _empty(self) -> Dict[str, Any]:
        return {"version": STORE_VERSION, "servers": {}}

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", self.path, exc)
            return self._empty()
        if not isinstance(data, dict):
            return self._empty()
        servers = data.get("servers")
        data["servers"] = servers if isinstance(servers, dict) else {}
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data["version"] = STORE_VERSION
        fd, temp_name = tempfile.mkstemp(prefix="mcp-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            Path(temp_name).replace(self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def records(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.load().get("servers", {}))

    def record(self, server_id: str) -> Dict[str, Any]:
        return dict(self.records().get(server_id, {}))

    def has(self, server_id: str) -> bool:
        return server_id in self.records()

    def ids(self) -> List[str]:
        return list(self.records())

    def _unique_id(self, data: Dict[str, Any], name: str) -> str:
        base = slugify(name)
        taken = set(data["servers"])
        if base not in taken:
            return base
        for n in range(2, 1000):
            candidate = f"{base}-{n}"
            if candidate not in taken:
                return candidate
        raise ValueError("Could not allocate a unique server id")

    # ------------------------------------------------------------------
    # validation + secrets
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_transport(transport: str) -> str:
        value = (transport or "").strip().lower()
        if value not in TRANSPORTS:
            raise ValueError(f"Transport must be one of: {', '.join(TRANSPORTS)}")
        return value

    @staticmethod
    def _clean_args(args: Any) -> List[str]:
        if args is None or args == "":
            return []
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as exc:
                raise ValueError("Args must be a JSON array of strings") from exc
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError("Args must be a JSON array of strings")
        return [str(item) for item in args]

    @staticmethod
    def _clean_env(env: Any) -> Dict[str, str]:
        if env is None or env == "":
            return {}
        if isinstance(env, str):
            try:
                env = json.loads(env)
            except ValueError as exc:
                raise ValueError("Env must be a JSON object of strings") from exc
        if not isinstance(env, dict):
            raise ValueError("Env must be a JSON object of strings")
        out: Dict[str, str] = {}
        for key, value in env.items():
            if value is None:
                continue
            out[str(key)] = str(value)
        return out

    def _encrypt_env(self, env: Dict[str, str], *, existing: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Encrypt values on the way in, keeping stored secrets for masked input."""
        existing = existing or {}
        out: Dict[str, str] = {}
        for key, value in env.items():
            if value == MASK:
                # The UI echoed the mask back — keep whatever is already stored.
                if key in existing:
                    out[key] = existing[key]
                continue
            out[key] = self.box.encrypt(value) if value else ""
        return out

    def env_for(self, server_id: str) -> Dict[str, str]:
        """Decrypted environment for connecting (never exposed over HTTP)."""
        stored = self.record(server_id).get("env") or {}
        out: Dict[str, str] = {}
        for key, value in stored.items():
            if not value:
                out[str(key)] = ""
                continue
            decrypted = self.box.decrypt(value) if is_encrypted(value) else value
            out[str(key)] = decrypted
        return out

    # ------------------------------------------------------------------
    # mutations
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        name: str = "",
        transport: str = "stdio",
        command: str = "",
        args: Any = None,
        env: Any = None,
        url: str = "",
        enabled: bool = True,
    ) -> str:
        data = self.load()
        transport = self._clean_transport(transport)
        display = (name or "").strip() or slugify(command or url or "server")
        clean_args = self._clean_args(args)
        clean_env = self._encrypt_env(self._clean_env(env))

        candidate_command = (command or "").strip()
        candidate_url = (url or "").strip()
        if transport == "stdio":
            if not candidate_command:
                raise ValueError("A stdio server needs a command")
            candidate_url = ""
        else:
            if not candidate_url:
                raise ValueError(f"A {transport} server needs a URL")
            if not candidate_url.lower().startswith(("http://", "https://")):
                raise ValueError("Server URL must start with http:// or https://")
            candidate_command = ""
            clean_args = []

        server_id = self._unique_id(data, display)
        now = time.time()
        data["servers"][server_id] = {
            "id": server_id,
            "name": display,
            "transport": transport,
            "command": candidate_command,
            "args": clean_args,
            "env": clean_env,
            "url": candidate_url,
            "enabled": bool(enabled),
            "disabled_tools": [],
            "created_at": now,
            "updated_at": now,
        }
        self.save(data)
        logger.info("Added MCP server id=%s name=%r transport=%s", server_id, display, transport)
        return server_id

    def update(self, server_id: str, **fields: Any) -> Dict[str, Any]:
        data = self.load()
        record = data["servers"].get(server_id)
        if record is None:
            raise KeyError(server_id)

        if "name" in fields and fields["name"] is not None:
            record["name"] = str(fields["name"]).strip() or record["name"]
        if "transport" in fields and fields["transport"] is not None:
            record["transport"] = self._clean_transport(fields["transport"])
        if "command" in fields and fields["command"] is not None:
            record["command"] = str(fields["command"]).strip()
        if "url" in fields and fields["url"] is not None:
            record["url"] = str(fields["url"]).strip()
        if "args" in fields and fields["args"] is not None:
            record["args"] = self._clean_args(fields["args"])
        if "env" in fields and fields["env"] is not None:
            record["env"] = self._encrypt_env(self._clean_env(fields["env"]), existing=record.get("env") or {})
        if "enabled" in fields and fields["enabled"] is not None:
            record["enabled"] = bool(fields["enabled"])
        if "disabled_tools" in fields and fields["disabled_tools"] is not None:
            disabled = fields["disabled_tools"]
            if not isinstance(disabled, list):
                raise ValueError("disabled_tools must be a list of tool names")
            record["disabled_tools"] = [str(name) for name in disabled]

        # Keep the record self-consistent with its transport.
        if record["transport"] == "stdio":
            if not record["command"]:
                raise ValueError("A stdio server needs a command")
            record["url"] = ""
        elif not record["url"]:
            raise ValueError(f"A {record['transport']} server needs a URL")

        record["updated_at"] = time.time()
        self.save(data)
        return dict(record)

    def delete(self, server_id: str) -> bool:
        data = self.load()
        existed = data["servers"].pop(server_id, None) is not None
        if existed:
            self.save(data)
            logger.info("Removed MCP server id=%s (credential deleted)", server_id)
        return existed

    def set_disabled_tools(self, server_id: str, disabled: List[str]) -> List[str]:
        record = self.update(server_id, disabled_tools=disabled)
        return list(record.get("disabled_tools") or [])

    # ------------------------------------------------------------------
    # API-facing views (never contain a secret)
    # ------------------------------------------------------------------
    def public_row(self, server_id: str) -> Dict[str, Any]:
        record = self.record(server_id)
        return {
            "id": record.get("id") or server_id,
            "name": record.get("name") or server_id,
            "transport": record.get("transport") or "stdio",
            "command": record.get("command") or "",
            "args": list(record.get("args") or []),
            "env": mask_env(record.get("env") or {}),
            "url": record.get("url") or "",
            "enabled": bool(record.get("enabled", True)),
            "disabled_tools": list(record.get("disabled_tools") or []),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    def public_rows(self) -> List[Dict[str, Any]]:
        return [self.public_row(server_id) for server_id in self.ids()]
