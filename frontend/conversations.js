"use strict";

/* Athena left sidebar — persistent chat history.

   Lists the real conversations from GET /api/conversations and offers the same
   actions as the chat header (Rename / Compact / Copy Chat / PDF / Save to
   Documents / Delete Chat), each bound to that row's conversation id through the
   id-aware methods on window.AthenaChat. Nothing here is a placeholder: every
   item calls the same endpoint the header does. */

(() => {
  const q = (sel) => document.querySelector(sel);
  const chat = () => window.AthenaChat;

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  let refreshing = false;

  /* ─────────────── row menu ─────────────── */
  function closeMenus() {
    document.querySelectorAll(".conv-menu").forEach((m) => { m.hidden = true; });
    document.querySelectorAll(".conv-menu-btn").forEach((b) => b.setAttribute("aria-expanded", "false"));
  }

  function toggleMenu(row) {
    const menu = row.querySelector(".conv-menu");
    const btn = row.querySelector(".conv-menu-btn");
    if (!menu || !btn) return;
    const willOpen = menu.hidden;
    closeMenus();
    if (!willOpen) return;
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  }

  /* ─────────────── row rendering ─────────────── */
  function rowFor(id) {
    return Array.from(document.querySelectorAll(".conv-row")).find((r) => r.dataset.id === id) || null;
  }

  function toMillis(value) {
    if (value == null || value === "") return null;
    if (typeof value === "number") return value > 1e12 ? value : value * 1000;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
    const n = Number(value);
    return Number.isFinite(n) ? (n > 1e12 ? n : n * 1000) : null;
  }

  function relativeTime(value) {
    const ms = toMillis(value);
    if (ms == null) return "";
    const secs = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (secs < 60) return "just now";
    const mins = Math.round(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(ms).toLocaleDateString();
  }

  function rowMeta(conv) {
    const bits = [];
    const when = relativeTime(conv.updated_at || conv.created_at);
    if (when) bits.push(when);
    if (conv.total_tokens) bits.push(`${Number(conv.total_tokens).toLocaleString()} tok`);
    return bits.join(" · ");
  }

  const MENU_ITEMS = [
    ["rename", "Rename"],
    ["compact", "Compact"],
    ["copy", "Copy Chat"],
    ["pdf", "PDF"],
    ["save", "Save to Documents"],
    ["delete", "Delete Chat"],
  ];

  function buildMenu(conv) {
    const menu = el("div", "conv-menu");
    menu.setAttribute("role", "menu");
    menu.hidden = true;
    menu.addEventListener("click", (event) => event.stopPropagation());
    MENU_ITEMS.forEach(([action, label]) => {
      const button = el("button", action === "delete" ? "danger" : null, label);
      button.type = "button";
      button.setAttribute("role", "menuitem");
      button.dataset.action = action;
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        closeMenus();
        runAction(action, conv);
      });
      menu.appendChild(button);
    });
    return menu;
  }

  function buildRow(conv, currentId) {
    const row = el("div", "conv-row");
    row.dataset.id = conv.id;
    if (conv.id === currentId) row.classList.add("is-active");

    const title = el("button", "conv-title");
    title.type = "button";
    const meta = rowMeta(conv);
    title.title = meta ? `${conv.title || "New Chat"} · ${meta}` : (conv.title || "New Chat");
    title.appendChild(el("span", "conv-title-text", conv.title || "New Chat"));
    title.addEventListener("click", () => selectConversation(conv.id));
    row.appendChild(title);

    const menuBtn = el("button", "conv-menu-btn", "⋯");
    menuBtn.type = "button";
    menuBtn.setAttribute("aria-haspopup", "true");
    menuBtn.setAttribute("aria-expanded", "false");
    menuBtn.setAttribute("aria-label", "Chat actions");
    menuBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleMenu(row);
    });
    row.appendChild(menuBtn);

    row.appendChild(buildMenu(conv));
    return row;
  }

  function render(convs) {
    const list = q("#conv-list");
    if (!list) return;
    list.replaceChildren();

    if (!convs.length) {
      list.appendChild(el("div", "state", "No chats yet. Start one below."));
      return;
    }

    const api = chat();
    const currentId = api && api.getConversationId ? api.getConversationId() : null;
    convs.forEach((conv) => list.appendChild(buildRow(conv, currentId)));
  }

  /* ─────────────── actions ─────────────── */
  async function selectConversation(id) {
    const api = chat();
    if (!api) return;
    closeMenus();
    if (id === api.getConversationId()) return;
    try {
      await api.open(id);
    } catch (err) {
      toast(`Could not open chat: ${err.message}`, "error");
    }
  }

  function runAction(action, conv) {
    const handlers = {
      rename: renameInline,
      compact: compactConv,
      copy: copyConv,
      pdf: pdfConv,
      save: saveConv,
      delete: deleteConv,
    };
    if (handlers[action]) handlers[action](conv);
  }

  async function renameInline(conv) {
    const api = chat();
    const row = rowFor(conv.id);
    if (!api || !row) return;
    const titleBtn = row.querySelector(".conv-title");
    if (!titleBtn) return;

    const original = conv.title || "";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "conv-rename-input";
    input.maxLength = 200;
    input.value = original;
    titleBtn.replaceWith(input);
    input.focus();
    input.select();

    let done = false;
    const commit = async () => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      if (!value || value === original) { refresh(); return; }
      try {
        await api.renameConversation(conv.id, value);
        toast("Renamed", "success");
      } catch (err) {
        toast(`Rename failed: ${err.message}`, "error");
      }
      refresh();
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); input.blur(); }
      if (event.key === "Escape") { event.preventDefault(); input.value = original; input.blur(); }
    });
  }

  async function compactConv(conv) {
    const api = chat();
    if (!api) return;
    if (!window.confirm(
      "Compact this chat?\n\nThe older messages are summarised into a single summary message and " +
      "archived (kept, not deleted). The recent messages stay as they are."
    )) return;
    try {
      toast("Compacting…", "info");
      const res = await api.compactConversation(conv.id);
      toast(`Compacted ${res.summarized} message(s)` + (res.archived ? ` (${res.archived} archived)` : ""), "success");
    } catch (err) {
      toast(`Compact failed: ${err.message}`, "error");
    }
  }

  async function copyConv(conv) {
    const api = chat();
    if (!api) return;
    try {
      const text = await api.transcriptOf(conv.id);
      if (!text.trim()) { toast("Nothing to copy yet.", "warn"); return; }
      await navigator.clipboard.writeText(text);
      toast("Conversation copied", "success");
    } catch (err) {
      toast(`Copy failed: ${err.message}`, "error");
    }
  }

  async function pdfConv(conv) {
    const api = chat();
    if (!api) return;
    try {
      const text = await api.transcriptOf(conv.id);
      const win = window.open("", "_blank");
      if (!win) { toast("Allow pop-ups to export a PDF.", "warn"); return; }
      win.document.write(
        `<!doctype html><html><head><meta charset="utf-8"><title>${esc(conv.title || "Athena Chat")}</title>` +
        "<style>body{font:13px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}" +
        "pre{white-space:pre-wrap;word-wrap:break-word;font:inherit}</style></head>" +
        `<body><pre>${esc(text)}</pre></body></html>`
      );
      win.document.close();
      win.focus();
      win.print();
    } catch (err) {
      toast(`PDF export failed: ${err.message}`, "error");
    }
  }

  async function saveConv(conv) {
    const api = chat();
    if (!api) return;
    try {
      const res = await api.saveConversation(conv.id);
      toast(`Saved “${res.title}” to documents`, "success");
      if (window.AthenaSettings && window.AthenaSettings.reload) window.AthenaSettings.reload();
    } catch (err) {
      toast(`Save failed: ${err.message}`, "error");
    }
  }

  async function deleteConv(conv) {
    const api = chat();
    if (!api) return;
    if (!window.confirm(
      "Delete this chat?\n\nIts messages and usage records are removed. Saved documents and " +
      "research reports are kept, and your RAG documents are untouched."
    )) return;
    try {
      await api.deleteConversationById(conv.id);
      toast("Chat deleted", "success");
    } catch (err) {
      toast(`Delete failed: ${err.message}`, "error");
    }
  }

  /* ─────────────── list ─────────────── */
  async function refresh() {
    const list = q("#conv-list");
    const api = chat();
    if (!list || !api || !api.list || refreshing) return;
    refreshing = true;
    try {
      render(await api.list(100));
    } catch (err) {
      // Backend offline: keep whatever list is already shown rather than clearing it.
      if (!list.childElementCount) list.appendChild(el("div", "state", "Could not load chats."));
    } finally {
      refreshing = false;
    }
  }

  /* ─────────────── wiring ─────────────── */
  function init() {
    const newBtn = q("#new-chat");
    if (newBtn) {
      newBtn.addEventListener("click", () => {
        const api = chat();
        if (!api) return;
        api.startNewConversation();
        refresh();
        const input = q("#chat-input");
        if (input) input.focus();
      });
    }

    const toggle = q("#sidebar-toggle");
    if (toggle) {
      toggle.addEventListener("click", (event) => {
        event.stopPropagation();
        const open = document.body.classList.toggle("sidebar-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }

    // Outside click closes the row menu, and the chat-list overlay on narrow screens.
    document.addEventListener("click", (event) => {
      closeMenus();
      const sidebar = q("#sidebar");
      if (sidebar && document.body.classList.contains("sidebar-open") && !sidebar.contains(event.target)) {
        document.body.classList.remove("sidebar-open");
        if (toggle) toggle.setAttribute("aria-expanded", "false");
      }
    });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenus(); });

    refresh();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.AthenaConversations = { refresh };
})();
