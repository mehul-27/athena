"use strict";

/* Athena chat header + dropdown menu.
   Modelled on Odysseus's `#current-meta` header and `#export-dropdown-menu`
   (Title · N msgs · $cost · tokens ▼), backed by the conversation the chat
   actually persisted. Every action does real work — no placeholder buttons. */

(() => {
  const q = (sel) => document.querySelector(sel);

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  let conversation = null;

  const chat = () => window.AthenaChat;

  /* ─────────────── header ─────────────── */
  function updateHeader(conv) {
    conversation = conv || conversation;
    if (!conversation) return;
    const title = q("#chat-title");
    const count = q("#chat-count");
    const cost = q("#chat-cost");
    const tokens = q("#chat-tokens");
    if (title) title.textContent = conversation.title || "New Chat";

    const msgs = conversation.message_count || 0;
    if (count) count.textContent = msgs ? `· ${msgs} msg${msgs === 1 ? "" : "s"}` : "";

    // Cost is only shown when it is actually known; unknown pricing renders $—.
    if (cost) {
      const display = conversation.cost_display || "$—";
      cost.textContent = `· ${display}`;
      const unknown = !conversation.cost_known;
      cost.classList.toggle("is-unknown", unknown);
      cost.title = unknown
        ? "Pricing is unknown for one or more models used here — tokens are still tracked."
        : "Estimated from published list prices.";
    }

    if (tokens) {
      const n = conversation.total_tokens || 0;
      tokens.textContent = n ? `· ${n.toLocaleString()} tok` : "";
      const models = (conversation.by_model || [])
        .filter((m) => m.provider || m.model)
        .map((m) => `${m.provider || "?"} · ${m.model || "?"} (${m.requests} call${m.requests === 1 ? "" : "s"}, ${(m.total_tokens || 0).toLocaleString()} tok)`)
        .join("\n");
      tokens.title = models ? `Usage by provider/model:\n${models}` : "No LLM calls recorded yet.";
    }
  }

  function setBusyLabel(text) {
    const tokens = q("#chat-tokens");
    if (tokens && text !== undefined) tokens.dataset.busy = text || "";
  }

  /* ─────────────── menu ─────────────── */
  function menu() { return q("#chat-menu"); }
  function titleBtn() { return q("#chat-title-btn"); }

  function closeMenu() {
    const m = menu();
    if (m) m.hidden = true;
    const b = titleBtn();
    if (b) b.setAttribute("aria-expanded", "false");
  }

  function openMenu() {
    const m = menu();
    const b = titleBtn();
    if (!m || !b) return;
    m.hidden = false;
    b.setAttribute("aria-expanded", "true");
  }

  function wireMenu() {
    const b = titleBtn();
    if (!b) return;
    b.addEventListener("click", (event) => {
      event.stopPropagation();
      if (menu() && menu().hidden) openMenu(); else closeMenu();
    });
    document.addEventListener("click", () => closeMenu());
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
    menu().addEventListener("click", (event) => event.stopPropagation());
    menu().addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      closeMenu();
      run(button.dataset.action);
    });
  }

  function run(action) {
    if (action === "rename") return rename();
    if (action === "compact") return compact();
    if (action === "copy") return copyChat();
    if (action === "pdf") return exportPdf();
    if (action === "save") return saveToDocuments();
    if (action === "delete") return deleteChat();
  }

  /* ─────────────── actions ─────────────── */
  function rename() {
    const el = q("#chat-title");
    if (!el || !chat()) return;
    const original = el.textContent;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "chat-title-input";
    input.value = original;
    input.maxLength = 200;
    el.replaceWith(input);
    input.focus();
    input.select();

    const restore = (text) => {
      const span = document.createElement("span");
      span.className = "chat-title";
      span.id = "chat-title";
      span.textContent = text;
      input.replaceWith(span);
    };
    const commit = async () => {
      const value = input.value.trim();
      if (!value || value === original) { restore(original); return; }
      try {
        const res = await chat().rename(value);
        restore(res?.title || value);
        updateHeader(res);
        toast("Renamed", "success");
      } catch (err) {
        restore(original);
        toast(`Rename failed: ${err.message}`, "error");
      }
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
      if (e.key === "Escape") { e.preventDefault(); restore(original); }
    });
  }

  async function compact() {
    if (!chat()) return;
    const id = chat().getConversationId();
    if (!id) { toast("Nothing to compact yet.", "warn"); return; }
    const msgs = (conversation && conversation.message_count) || 0;
    if (msgs < 6) { toast("Not enough messages to compact.", "warn"); return; }
    if (!window.confirm(
      "Compact this chat?\n\nThe older messages are summarised into a single summary message and " +
      "archived (kept, not deleted). The recent messages stay as they are."
    )) return;
    try {
      toast("Compacting…", "info");
      const res = await chat().compact();
      updateHeader(res);
      toast(`Compacted ${res.summarized} message(s)` + (res.archived ? ` (${res.archived} archived)` : ""), "success");
    } catch (err) {
      toast(`Compact failed: ${err.message}`, "error");
    }
  }

  async function copyChat() {
    if (!chat()) return;
    try {
      const text = await chat().transcript();
      if (!text.trim()) { toast("Nothing to copy yet.", "warn"); return; }
      await navigator.clipboard.writeText(text);
      toast("Conversation copied", "success");
    } catch (err) {
      toast(`Copy failed: ${err.message}`, "error");
    }
  }

  async function exportPdf() {
    if (!chat()) return;
    try {
      const text = await chat().transcript();
      const title = (conversation && conversation.title) || "Athena Chat";
      // Same approach as Odysseus (print the transcript); a dedicated print
      // window keeps the app chrome out of the PDF.
      const win = window.open("", "_blank");
      if (!win) { toast("Allow pop-ups to export a PDF.", "warn"); return; }
      win.document.write(
        `<!doctype html><html><head><meta charset="utf-8"><title>${esc(title)}</title>` +
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

  async function saveToDocuments() {
    if (!chat()) return;
    try {
      const res = await chat().saveDocument();
      toast(`Saved “${res.title}” to documents`, "success");
      if (window.AthenaSettings && window.AthenaSettings.reload) window.AthenaSettings.reload();
    } catch (err) {
      toast(`Save failed: ${err.message}`, "error");
    }
  }

  async function deleteChat() {
    if (!chat()) return;
    const id = chat().getConversationId();
    if (!id) { toast("No chat to delete.", "warn"); return; }
    if (!window.confirm(
      "Delete this chat?\n\nIts messages and usage records are removed. Saved documents and " +
      "research reports are kept, and your RAG documents are untouched."
    )) return;
    try {
      await chat().deleteConversation();
      toast("Chat deleted", "success");
      chat().startNewConversation();
    } catch (err) {
      toast(`Delete failed: ${err.message}`, "error");
    }
  }

  /* ─────────────── init ─────────────── */
  function init() {
    wireMenu();
    if (chat() && chat().getConversation()) updateHeader(chat().getConversation());
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.AthenaChatHeader = { updateHeader, closeMenu, setBusyLabel };
})();
