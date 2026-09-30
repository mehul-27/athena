"""MCP client manager — connect to user-configured MCP servers and call their tools.

Behavioural port of Odysseus's `src/mcp_manager.py`, re-shaped for Athena: no
SQLAlchemy, no app/session coupling, no built-in servers, no prompt cache. What
is deliberately identical is the part that matters — the transport set, the
per-server status model, the discovery call, and the `mcp__{server}__{tool}`
namespacing (Odysseus's convention, kept so a tool name means the same thing in
both projects).

Differences worth knowing:

* **Sync API, async engine.** Athena's backend is synchronous, so a dedicated
  daemon thread runs one asyncio loop and every public method submits a coroutine
  to it. Sessions and their exit stacks stay alive on that loop for the process
  lifetime.
* **No bundled servers.** Only servers the user configured are connected.
* **No OAuth.** A server that needs authorization is *detected* and reported as
  `needs_auth`; the interactive flow is not implemented here.
* `connect_all_enabled()` never blocks startup — `main.py` runs it in a thread.

Statuses: `connected · connecting · disconnected · error · timeout · needs_auth
· disabled · unavailable`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.mcp.store import McpStore

if TYPE_CHECKING:  # avoid a config <-> mcp import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)

try:  # the SDK is optional at import time so Athena still boots without it
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    MCP_AVAILABLE = False

# How long a single MCP tool call may take. Odysseus has no explicit call
# timeout; a chat message is blocked on this, so it is bounded.
CALL_TIMEOUT = 120.0

# Verb prefixes used when a tool declares no read-only hint (fail-closed: an
# unknown verb is treated as destructive). Same heuristic as Odysseus.
_READONLY_PREFIXES = (
    "get", "list", "read", "search", "find", "fetch", "query", "describe",
    "show", "lookup", "stat", "head", "count", "resolve",
)
_DESTRUCTIVE_PREFIXES = (
    "write", "create", "delete", "remove", "update", "edit", "send", "move",
    "copy", "exec", "run", "kill", "drop", "patch", "put", "post", "set",
    "insert", "append", "rename", "mkdir", "rmdir", "truncate", "install",
)


def _split_qualified(name: str) -> Optional[tuple[str, str]]:
    """`mcp__server__tool` -> `("server", "tool")`, else None."""
    parts = (name or "").split("__", 2)
    if len(parts) != 3 or parts[0] != "mcp" or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def mcp_tool_is_readonly(tool: Dict[str, Any]) -> bool:
    """True only when the tool looks safe to run without confirmation.

    Prefers the MCP annotations (`readOnlyHint` / `destructiveHint`); falls back
    to a verb heuristic. Anything unrecognised is **not** read-only.
    """
    annotations = tool.get("annotations") or {}
    if isinstance(annotations, dict):
        if annotations.get("destructiveHint") is True:
            return False
        if annotations.get("readOnlyHint") is True:
            return True

    name = str(tool.get("name") or "").lower()
    verb = name.split("_", 1)[0].split("-", 1)[0]
    if verb in _DESTRUCTIVE_PREFIXES:
        return False
    if verb in _READONLY_PREFIXES:
        return True
    return False


def _classify_error(exc: BaseException) -> tuple[str, str]:
    """Map an SDK exception onto our status vocabulary + a user-readable line."""
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", "Connection timed out."
    if "unauthor" in lowered or "401" in lowered or "oauth" in lowered or "forbidden" in lowered:
        return "needs_auth", "This server requires authorization, which Athena does not support yet."
    return "error", text[:300]


class McpManager:
    def __init__(self, settings: "Settings", *, store: Optional[McpStore] = None) -> None:
        self.settings = settings
        self.store = store or McpStore(settings)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._guard = threading.RLock()
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._tools: Dict[str, List[Dict[str, Any]]] = {}
        self._sessions: Dict[str, Any] = {}
        self._stacks: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # event loop plumbing
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.settings.mcp_enabled) and MCP_AVAILABLE

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._guard:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=_run, name="athena-mcp", daemon=True)
            thread.start()
            self._loop = loop
            self._loop_thread = thread
            return loop

    def _run(self, coro, *, timeout: float) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def _set_status(self, server_id: str, status: str, **extra: Any) -> None:
        with self._guard:
            row = self._connections.setdefault(server_id, {})
            row.update({"status": status, "at": time.time(), **extra})

    def status(self, server_id: str) -> Dict[str, Any]:
        with self._guard:
            row = dict(self._connections.get(server_id) or {})
        row.setdefault("status", "disconnected")
        row["tool_count"] = len(self._tools.get(server_id) or [])
        return row

    def statuses(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for server_id in self.store.ids():
            out[server_id] = self.status(server_id)
        return out

    def tools(self, server_id: str) -> List[Dict[str, Any]]:
        return list(self._tools.get(server_id) or [])

    def all_tools(self) -> List[Dict[str, Any]]:
        """Every discovered tool across connected servers, qualified."""
        out: List[Dict[str, Any]] = []
        for server_id in self.store.ids():
            if self.status(server_id).get("status") != "connected":
                continue
            for tool in self._tools.get(server_id) or []:
                out.append({**tool, "server_id": server_id, "qualified": f"mcp__{server_id}__{tool['name']}"})
        return out

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------
    def connect(self, server_id: str, *, record: Optional[Dict[str, Any]] = None) -> bool:
        """Connect one server and discover its tools. Never raises."""
        record = record or self.store.record(server_id)
        if not record:
            self._set_status(server_id, "error", error="Unknown server.")
            return False
        if not self.enabled:
            self._set_status(
                server_id,
                "disabled" if not self.settings.mcp_enabled else "unavailable",
                error="MCP is disabled." if not self.settings.mcp_enabled else "The mcp package is not installed.",
            )
            return False

        self.disconnect(server_id, quiet=True)
        self._set_status(server_id, "connecting")

        env = self.store.env_for(server_id)
        timeout = float(self.settings.mcp_connect_timeout or 20.0)
        try:
            tools = self._run(self._connect(server_id, record, env), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - the status IS the error channel
            status, message = _classify_error(exc)
            self._set_status(server_id, status, error=message)
            logger.warning("MCP server id=%s connect failed status=%s", server_id, status)
            return False

        with self._guard:
            self._tools[server_id] = tools
        self._set_status(server_id, "connected", error="")
        logger.info("MCP server id=%s connected tools=%d", server_id, len(tools))
        return True

    async def _connect(self, server_id: str, record: Dict[str, Any], env: Dict[str, str]) -> List[Dict[str, Any]]:
        from contextlib import AsyncExitStack

        stack = AsyncExitStack()
        transport = (record.get("transport") or "stdio").lower()
        try:
            if transport == "stdio":
                # The child inherits Athena's environment (so `python`, `npx` and
                # PATH resolve) with the configured values layered on top — the
                # same behaviour as Odysseus.
                params = StdioServerParameters(
                    command=record.get("command") or "",
                    args=list(record.get("args") or []),
                    env={**os.environ, **{k: v for k, v in (env or {}).items() if v}},
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif transport == "sse":
                read, write = await stack.enter_async_context(sse_client(record.get("url") or ""))
            elif transport == "http":
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(record.get("url") or "")
                )
            else:  # pragma: no cover - store validates transports
                raise ValueError(f"Unknown MCP transport: {transport}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            tools = [self._normalise_tool(tool) for tool in getattr(listed, "tools", []) or []]
        except BaseException:
            await stack.aclose()
            raise

        with self._guard:
            self._sessions[server_id] = session
            self._stacks[server_id] = stack
        return tools

    @staticmethod
    def _normalise_tool(tool: Any) -> Dict[str, Any]:
        schema = getattr(tool, "inputSchema", None)
        annotations = getattr(tool, "annotations", None)
        if annotations is not None and not isinstance(annotations, dict):
            annotations = annotations.model_dump(exclude_none=True) if hasattr(annotations, "model_dump") else None
        return {
            "name": getattr(tool, "name", "") or "",
            "description": (getattr(tool, "description", "") or "").strip(),
            "input_schema": schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
            "annotations": annotations if isinstance(annotations, dict) else {},
        }

    def disconnect(self, server_id: str, *, quiet: bool = False) -> None:
        stack = self._stacks.pop(server_id, None)
        with self._guard:
            self._sessions.pop(server_id, None)
            self._tools.pop(server_id, None)
        if stack is not None:
            try:
                self._run(stack.aclose(), timeout=10.0)
            except Exception as exc:  # noqa: BLE001 - teardown must not break the caller
                if not quiet:
                    logger.warning("MCP server id=%s disconnect failed: %s", server_id, exc)
        if not quiet:
            self._set_status(server_id, "disconnected")
            logger.info("MCP server id=%s disconnected", server_id)

    def disconnect_all(self) -> None:
        for server_id in list(self._stacks):
            self.disconnect(server_id)

    def connect_all_enabled(self) -> None:
        """Connect every enabled server. Call in a background thread."""
        if not self.enabled:
            logger.info("MCP disabled — no servers connected")
            return
        for server_id in self.store.ids():
            record = self.store.record(server_id)
            if not record.get("enabled", True):
                self._set_status(server_id, "disabled")
                continue
            self.connect(server_id, record=record)

    def start_background_connect(self) -> threading.Thread:
        """Kick `connect_all_enabled()` off the startup path."""
        thread = threading.Thread(target=self.connect_all_enabled, name="athena-mcp-connect", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        try:
            self.disconnect_all()
        finally:
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            self._loop = None

    # ------------------------------------------------------------------
    # schemas + execution
    # ------------------------------------------------------------------
    def get_all_openai_schemas(self, disabled_map: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """Every discovered tool as an OpenAI function-calling schema.

        Same shape as Odysseus's method of the same name: the MCP JSON Schema is
        passed through as `parameters` untouched.
        """
        disabled_map = disabled_map or {}
        schemas: List[Dict[str, Any]] = []
        for row in self.all_tools():
            server_id = row["server_id"]
            if row["name"] in set(disabled_map.get(server_id) or []):
                continue
            label = self.store.record(server_id).get("name") or server_id
            schemas.append({
                "type": "function",
                "function": {
                    "name": row["qualified"],
                    "description": f"[MCP:{label}] {row.get('description') or row['name']}",
                    "parameters": row.get("input_schema") or {"type": "object", "properties": {}},
                },
            })
        return schemas

    def call_tool(self, qualified: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a discovered tool. Returns a result dict, never raises."""
        parsed = _split_qualified(qualified)
        if not parsed:
            return {"error": f"Invalid MCP tool name: {qualified}", "exit_code": 1}
        server_id, tool_name = parsed

        session = self._sessions.get(server_id)
        if session is None:
            return {"error": f"MCP server not connected: {server_id}", "exit_code": 1}

        known = {tool["name"] for tool in (self._tools.get(server_id) or [])}
        if tool_name not in known:
            return {"error": f"Unknown tool {tool_name!r} on MCP server {server_id!r}", "exit_code": 1}

        try:
            return self._run(self._call(session, tool_name, arguments or {}), timeout=CALL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
            status, message = _classify_error(exc)
            if status == "needs_auth":
                self._set_status(server_id, "needs_auth", error=message)
            logger.warning("MCP tool call failed server=%s tool=%s", server_id, tool_name)
            return {"error": message, "exit_code": 1}

    @staticmethod
    async def _call(session: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = await session.call_tool(tool_name, arguments)
        chunks: List[str] = []
        images: List[Dict[str, Any]] = []
        for block in getattr(result, "content", None) or []:
            block_type = getattr(block, "type", "") or ""
            if block_type == "text":
                chunks.append(getattr(block, "text", "") or "")
            elif block_type == "image":
                images.append({
                    "mime_type": getattr(block, "mimeType", "") or "",
                    "data": getattr(block, "data", "") or "",
                })
        text = "\n".join(chunk for chunk in chunks if chunk).strip()

        if getattr(result, "isError", False):
            return {"error": text or "The tool reported an error.", "exit_code": 1, "images": images}
        return {"content": text, "exit_code": 0, "images": images}
