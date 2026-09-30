"use strict";

/* Athena UI.
   One Chat conversation with optional capabilities (Web Search, RAG). The
   conversation lives here as a message list so turns can be re-sent after an
   edit, and an in-flight request can be stopped. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/* ─────────────── toasts (no alert() dialogs) ─────────────── */
function toast(message, kind = "info", ttl = 4200) {
  const node = el("div", `toast ${kind}`, message);
  $("#toasts").appendChild(node);
  setTimeout(() => {
    node.style.transition = "opacity .2s";
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 220);
  }, ttl);
}

/* ─────────────── api ─────────────── */
async function api(path, options) {
  let res;
  try {
    res = await fetch(path, { cache: "no-store", ...(options || {}) });
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new Error("backend unavailable");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  }
  return body;
}

/* ─────────────── health ─────────────── */
async function refreshHealth() {
  const box = $("#status");
  const text = $("#status-text");
  try {
    await api("/api/health");
    text.textContent = "online";
    box.title = "backend online";
    box.className = "status is-ok";
  } catch (err) {
    text.textContent = "offline";
    box.title = "backend offline";
    box.className = "status is-bad";
  }
}

/* ─────────────── navigation ─────────────── */
function setMode(mode) {
  const tab = document.querySelector(`.tab[data-mode="${mode}"]`);
  if (!tab || tab.classList.contains("is-disabled")) return;

  $$(".tab").forEach((t) => {
    t.classList.remove("is-active");
    t.setAttribute("aria-selected", "false");
  });
  $$(".panel").forEach((p) => p.classList.remove("is-active"));

  tab.classList.add("is-active");
  tab.setAttribute("aria-selected", "true");
  const panel = $(`#panel-${mode}`);
  if (panel) panel.classList.add("is-active");
}

$$(".tab[data-mode]").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

/* ─────────────── chat capabilities ─────────────── */
/* RAG and Web Search are capabilities of Chat. They are mutually exclusive for
   now: the backend has no combined-retrieval path, and Athena does not fake one. */
const CAPABILITY_LABELS = { web_search: "Web Search", rag: "RAG" };
const CAPABILITY_STATE = { web_search: false, rag: false };

function activeCapabilities() {
  return Object.keys(CAPABILITY_STATE).filter((cap) => CAPABILITY_STATE[cap]);
}

function capabilityBadge(caps) {
  if (!caps || !caps.length) return "";
  return " · " + caps.map((c) => CAPABILITY_LABELS[c] || c).join(" + ");
}

function syncCapabilityUI({ announce = "" } = {}) {
  Object.keys(CAPABILITY_STATE).forEach((cap) => {
    const button = $(`#cap-${cap}`);
    if (!button) return;
    const on = CAPABILITY_STATE[cap];
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.classList.toggle("is-on", on);
    button.textContent = on ? `${CAPABILITY_LABELS[cap]} ✓` : CAPABILITY_LABELS[cap];
  });

  const ragOn = CAPABILITY_STATE.rag;
  const panel = $("#docs-panel");
  if (panel) panel.hidden = !ragOn;
  const workspace = $("#chat-workspace");
  if (workspace) workspace.classList.toggle("has-docs", ragOn);

  const input = $("#chat-input");
  if (input) {
    input.placeholder = ragOn
      ? "Ask about your documents…"
      : (CAPABILITY_STATE.web_search ? "Search the web and answer…" : "Ask anything…");
  }
  if (announce) toast(announce, "warn", 5200);
}

function toggleCapability(cap) {
  const turningOn = !CAPABILITY_STATE[cap];
  let notice = "";
  if (turningOn) {
    // Enabling one turns the other off — combined retrieval isn't implemented.
    for (const other of Object.keys(CAPABILITY_STATE)) {
      if (other !== cap && CAPABILITY_STATE[other]) {
        CAPABILITY_STATE[other] = false;
        notice = `${CAPABILITY_LABELS[other]} and ${CAPABILITY_LABELS[cap]} cannot be combined yet — ${CAPABILITY_LABELS[other]} was turned off.`;
      }
    }
  }
  CAPABILITY_STATE[cap] = turningOn;
  syncCapabilityUI({ announce: notice });
  if (CAPABILITY_STATE.rag) refreshDocuments();
}

$$(".cap-btn[data-cap]").forEach((button) => {
  button.addEventListener("click", () => toggleCapability(button.dataset.cap));
});

/* ─────────────── conversation model ─────────────── */
/* The conversation is a list, not just DOM: editing a turn means re-sending it,
   which requires knowing what came before. It is also persisted server-side so
   the chat header, rename, compact, usage and Deep Research links all survive a
   reload. */
const conversation = [];
let pending = null;        // { controller, label } while a request is in flight
let editingIndex = null;

/* Server-side conversation this thread belongs to. */
let conversationId = (() => {
  try { return localStorage.getItem("athena-conversation") || null; } catch (_) { return null; }
})();
let conversationSummary = null;

function rememberConversation(id) {
  conversationId = id;
  try {
    if (id) localStorage.setItem("athena-conversation", id);
    else localStorage.removeItem("athena-conversation");
  } catch (_) { /* ignore */ }
}

function setConversationSummary(conv) {
  if (!conv) return;
  conversationSummary = conv;
  if (window.AthenaChatHeader) window.AthenaChatHeader.updateHeader(conv);
}

function historyPayload() {
  return conversation
    .filter((m) => (m.role === "user" || m.role === "assistant") && (m.text || "").trim())
    .map((m) => ({ role: m.role, content: m.text }));
}

function pushNote(text, kind = "note") {
  conversation.push({ role: kind, text });
  renderStream();
}

function sourcesBlock(sources) {
  const wrap = el("div", "sources");
  const scored = sources.filter((source) => Number.isFinite(source.score));

  if (scored.length) {
    const heading = el("div", "relevance-heading");
    heading.appendChild(el("h4", null, "Source relevance"));
    heading.appendChild(el("span", "relevance-scale", "relative match · 0–1"));
    wrap.appendChild(heading);

    wrap.appendChild(relevanceSpiderChart(scored));
    wrap.appendChild(el(
      "p",
      "relevance-note",
      "Hybrid semantic and keyword match to the question; this is a relative score, not a probability."
    ));
  }

  const unscored = sources.filter((source) => !Number.isFinite(source.score));
  if (unscored.length) {
    if (!scored.length) wrap.appendChild(el("h4", null, "Sources"));
    const ul = el("ul");
    unscored.forEach((source) => {
      const li = el("li");
      li.appendChild(el("span", "src-name", source.filename || source.title || "document"));
      if (source.chunk_id !== undefined && source.chunk_id !== null) {
        li.appendChild(el("span", "src-meta", `chunk ${source.chunk_id}`));
      }
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
  }
  return wrap;
}

function relevanceSpiderChart(sources) {
  const NS = "http://www.w3.org/2000/svg";
  const width = 420;
  const height = 320;
  const cx = 210;
  const cy = 151;
  const radius = 104;
  // A radar polygon needs at least three axes. Empty axes remain at zero when
  // only one or two chunks clear the threshold, making the sparse result clear.
  const axisCount = Math.max(3, sources.length);
  const values = Array.from({ length: axisCount }, (_, index) => (
    index < sources.length ? Math.max(0, Math.min(1, Number(sources[index].score))) : 0
  ));

  const svgEl = (name, attrs = {}) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const pointAt = (index, scale = 1) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / axisCount;
    return [cx + Math.cos(angle) * radius * scale, cy + Math.sin(angle) * radius * scale];
  };
  const pointsAt = (scale) => Array.from({ length: axisCount }, (_, index) => (
    pointAt(index, scale).map((n) => n.toFixed(1)).join(",")
  )).join(" ");

  const figure = el("figure", "relevance-spider");
  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Source relevance spiderweb",
  });
  const title = svgEl("title");
  title.textContent = "Source relevance spiderweb";
  svg.appendChild(title);
  const desc = svgEl("desc");
  desc.textContent = sources.map((source, index) => (
    `S${index + 1}, ${source.filename || source.title || "document"}, score ${Number(source.score).toFixed(2)}`
  )).join("; ");
  svg.appendChild(desc);

  [0.25, 0.5, 0.75, 1].forEach((level) => {
    svg.appendChild(svgEl("polygon", {
      class: level === 1 ? "spider-ring spider-ring-outer" : "spider-ring",
      points: pointsAt(level),
    }));
    const [lx, ly] = pointAt(0, level);
    const label = svgEl("text", { class: "spider-scale-label", x: lx + 4, y: ly + 3 });
    label.textContent = level.toFixed(2);
    svg.appendChild(label);
  });

  Array.from({ length: axisCount }, (_, index) => index).forEach((index) => {
    const [x, y] = pointAt(index);
    svg.appendChild(svgEl("line", { class: "spider-axis", x1: cx, y1: cy, x2: x, y2: y }));
    if (index < sources.length) {
      const [labelX, labelY] = pointAt(index, 1.18);
      const label = svgEl("text", {
        class: "spider-axis-label",
        x: labelX,
        y: labelY,
        "text-anchor": labelX < cx - 8 ? "end" : (labelX > cx + 8 ? "start" : "middle"),
      });
      label.textContent = `S${index + 1}`;
      svg.appendChild(label);
    }
  });

  const dataPoints = values.map((value, index) => (
    pointAt(index, value).map((n) => n.toFixed(1)).join(",")
  )).join(" ");
  svg.appendChild(svgEl("polygon", { class: "spider-data", points: dataPoints }));

  values.forEach((value, index) => {
    if (index >= sources.length) return;
    const [x, y] = pointAt(index, value);
    const dot = svgEl("circle", {
      class: "spider-dot",
      cx: x,
      cy: y,
      r: 4,
      tabindex: 0,
      "aria-label": `Source ${index + 1}, relevance ${value.toFixed(2)}`,
    });
    const tooltip = svgEl("title");
    tooltip.textContent = `${sources[index].filename || sources[index].title || "document"} · ${value.toFixed(2)}`;
    dot.appendChild(tooltip);
    svg.appendChild(dot);
  });
  figure.appendChild(svg);

  const legend = el("ol", "spider-legend");
  sources.forEach((source, index) => {
    const item = el("li");
    item.appendChild(el("span", "spider-key", `S${index + 1}`));
    const label = el("span", "spider-source", source.filename || source.title || "document");
    label.title = source.filename || source.title || "document";
    item.appendChild(label);
    if (source.chunk_id !== undefined && source.chunk_id !== null) {
      item.appendChild(el("span", "src-meta", `chunk ${source.chunk_id}`));
    }
    item.appendChild(el("strong", "spider-score", Number(source.score).toFixed(2)));
    legend.appendChild(item);
  });
  figure.appendChild(legend);
  return figure;
}

function webSourcesBlock(sources) {
  const wrap = el("div", "sources");
  wrap.appendChild(el("h4", null, "Sources"));
  const ul = el("ul");
  sources.forEach((s) => {
    const li = el("li");
    const link = el("a", "src-name", s.title || s.url);
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    li.appendChild(link);
    ul.appendChild(li);
  });
  wrap.appendChild(ul);
  return wrap;
}

function providerMeta(res) {
  if (!res) return "";
  if (res.fallback && res.requested_provider) {
    return `via ${res.provider_label || res.provider} · ${res.requested_provider} could not answer`;
  }
  return res.provider_label ? `via ${res.provider_label}` : "";
}

/* A Deep Research result posted back into the conversation it came from. */
function researchCard(msg) {
  const meta = msg.meta || {};
  const wrap = el("div", "research-card");
  wrap.appendChild(el("div", "research-card-title", msg.text || "Deep Research"));
  const facts = [];
  // Real metadata only — never invented counts.
  if (meta.rounds) facts.push(`${meta.rounds} round${meta.rounds === 1 ? "" : "s"}`);
  if (meta.sources) facts.push(`${meta.sources} source${meta.sources === 1 ? "" : "s"}`);
  if (meta.duration) facts.push(String(meta.duration));
  if (meta.status && meta.status !== "done") facts.push(String(meta.status));
  if (facts.length) wrap.appendChild(el("div", "research-card-meta", facts.join(" · ")));
  if (meta.report_url) {
    const link = el("a", "research-card-link", "Open visual report ↗");
    link.href = meta.report_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    wrap.appendChild(link);
  }
  return wrap;
}

/* Warn when the model you picked couldn't answer and another one did. */
function noteFallback(res) {
  if (res && res.fallback && res.requested_provider) {
    toast(
      `${res.requested_provider} could not answer — replied via ${res.provider_label || res.provider} instead. ` +
      `Check the key/model for ${res.requested_provider} in Settings.`,
      "warn",
      7000
    );
  }
}

function renderMessage(msg, index) {
  const node = el("div", `msg ${msg.role}`);

  if (msg.role === "user") {
    node.appendChild(el("span", "role", "You"));
    if (editingIndex === index) {
      const area = el("textarea", "msg-edit");
      area.value = msg.text || "";
      node.appendChild(area);
      const actions = el("div", "msg-actions");
      const save = el("button", "btn-sm", "Save & send");
      save.type = "button";
      save.addEventListener("click", () => saveEdit(index, area.value));
      const cancel = el("button", "btn-ghost", "Cancel");
      cancel.type = "button";
      cancel.addEventListener("click", () => { editingIndex = null; renderStream(); });
      actions.append(save, cancel);
      node.appendChild(actions);
      setTimeout(() => area.focus(), 0);
      return node;
    }
    node.appendChild(document.createTextNode(msg.text ?? ""));
    if (msg.capabilities && msg.capabilities.length) {
      node.appendChild(el("span", "msg-meta", capabilityBadge(msg.capabilities).trim()));
    }
    const actions = el("div", "msg-actions");
    const edit = el("button", "msg-action", "Edit");
    edit.type = "button";
    edit.addEventListener("click", () => startEdit(index));
    actions.appendChild(edit);
    node.appendChild(actions);
    return node;
  }

  if (msg.role === "assistant") {
    node.appendChild(el("span", "role", "Athena"));
    const kind = msg.meta && msg.meta.kind;
    if (kind === "research") {
      node.appendChild(researchCard(msg));
      return node;
    }
    node.appendChild(document.createTextNode(msg.text ?? ""));
    const meta = (msg.meta && msg.meta.text ? msg.meta.text : "") + capabilityBadge(msg.capabilities);
    if (meta) node.appendChild(el("span", "msg-meta", meta));
    if (msg.sources && msg.sources.length) {
      node.appendChild(msg.sourcesAreLinks ? webSourcesBlock(msg.sources) : sourcesBlock(msg.sources));
    }
    if (msg.note) node.appendChild(el("span", "msg-meta", msg.note));
    return node;
  }

  node.appendChild(document.createTextNode(msg.text ?? ""));
  return node;
}

function typingNode(label) {
  const node = el("div", "msg assistant");
  node.appendChild(el("span", "role", label));
  const dots = el("span", "typing");
  dots.append(el("i"), el("i"), el("i"));
  node.appendChild(dots);
  return node;
}

function renderStream() {
  const stream = $("#chat-stream");
  if (!stream) return;
  stream.replaceChildren();
  conversation.forEach((msg, i) => stream.appendChild(renderMessage(msg, i)));
  if (pending) stream.appendChild(typingNode(pending.label));
  stream.scrollTop = stream.scrollHeight;
}

function setBusy(busy) {
  const input = $("#chat-input");
  const send = $("#chat-send");
  const stop = $("#chat-stop");
  if (input) input.disabled = busy;
  if (send) send.hidden = busy;
  if (stop) stop.hidden = !busy;
  if (!busy && input) input.focus();
}

/* ─────────────── edit → resend ─────────────── */
function startEdit(index) {
  if (pending) {
    toast("Stop the current reply first.", "warn");
    return;
  }
  editingIndex = index;
  renderStream();
}

function saveEdit(index, value) {
  const text = (value || "").trim();
  if (!text) return;
  const capabilities = conversation[index].capabilities || activeCapabilities();
  // Drop the edited turn and everything after it, then ask again.
  conversation.splice(index);
  editingIndex = null;
  renderStream();
  send(text, capabilities);
}

/* ─────────────── send / stop ─────────────── */
async function send(message, capabilities = activeCapabilities()) {
  conversation.push({ role: "user", text: message, capabilities: [...capabilities] });
  const history = historyPayload().slice(0, -1);  // everything before this turn
  renderStream();

  const ragOn = capabilities.includes("rag");
  const searchOn = capabilities.includes("web_search");
  const label = ragOn ? "Athena · searching documents"
    : (searchOn ? "Athena · searching the web" : "Athena · generating");

  const controller = new AbortController();
  pending = { controller, label };
  setBusy(true);
  renderStream();

  try {
    const res = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, capabilities, history, conversation_id: conversationId }),
      signal: controller.signal,
    });
    if (res.conversation) {
      rememberConversation(res.conversation.id);
      setConversationSummary(res.conversation);
    }
    conversation.push(assistantMessage(res, { ragOn, searchOn }));
    noteFallback(res);
  } catch (err) {
    if (err && err.name === "AbortError") {
      conversation.push({ role: "note", text: "Stopped." });
    } else {
      conversation.push({ role: "error", text: `Chat failed — ${err.message}` });
      toast(`Chat failed: ${err.message}`, "error");
    }
  } finally {
    pending = null;
    setBusy(false);
    renderStream();
    refreshConversationList();
  }
}

function assistantMessage(res, { ragOn, searchOn }) {
  const msg = {
    role: "assistant",
    text: res.answer || "",
    meta: {
      text: providerMeta(res),
      fallback: !!res.fallback,
      requested_provider: res.requested_provider || null,
      kind: "answer",
    },
    capabilities: res.capabilities || [],
    sources: [],
    note: "",
  };
  if (searchOn) {
    if (!res.answer) msg.text = "No search results were found for that query.";
    msg.sources = res.sources || [];
    msg.sourcesAreLinks = true;
    return msg;
  }
  msg.sources = res.sources || [];
  if (ragOn && res.grounded === false) {
    msg.note = "Not found in your documents — RAG refuses to guess.";
    toast("No sufficiently relevant document context was found.", "warn");
  }
  return msg;
}

$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (pending) return;
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  send(message);
});

$("#chat-stop").addEventListener("click", () => {
  // Aborting stops the UI waiting immediately. The upstream provider call that
  // is already in flight is not cancelled server-side (it is a blocking HTTP
  // request), so its quota may still be spent — it just never reaches the UI.
  if (pending && pending.controller) {
    pending.controller.abort();
    toast("Stopped.", "warn", 2500);
  }
});

/* ─────────────── RAG documents panel ─────────────── */
async function refreshDocuments() {
  const list = $("#docs-list");
  if (!list) return;
  try {
    const data = await api("/api/rag/documents");
    renderDocuments(data.documents || []);
  } catch (err) {
    list.replaceChildren(el("li", "state", `Could not load documents — ${err.message}`));
  }
}

function renderDocuments(docs) {
  const list = $("#docs-list");
  if (!list) return;
  list.replaceChildren();

  if (!docs.length) {
    list.appendChild(el("li", "state", "No documents yet. Upload a PDF to get started."));
    return;
  }

  docs.forEach((doc) => {
    const li = el("li");

    const info = el("div", "doc-info");
    info.appendChild(el("span", "doc-name", doc.filename));
    const meta = el("span", "doc-meta");
    meta.appendChild(el("span", "ok", "Indexed"));
    meta.appendChild(document.createTextNode(` · ${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`));
    info.appendChild(meta);
    li.appendChild(info);

    const actions = el("div", "doc-actions");
    const del = el("button", "btn-ghost danger", "Delete");
    del.type = "button";
    del.dataset.id = doc.document_id;
    del.addEventListener("click", () => armDelete(del, doc.document_id));
    actions.appendChild(del);
    li.appendChild(actions);

    list.appendChild(li);
  });
}

/* Two-step inline confirm — avoids a browser confirm() dialog. */
function armDelete(btn, documentId) {
  if (btn.dataset.armed === "1") {
    deleteDocument(documentId, btn);
    return;
  }
  btn.dataset.armed = "1";
  btn.textContent = "Confirm?";
  btn.classList.add("confirm");
  setTimeout(() => {
    if (btn.isConnected && btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      btn.textContent = "Delete";
      btn.classList.remove("confirm");
    }
  }, 3000);
}

async function deleteDocument(documentId, btn) {
  btn.disabled = true;
  btn.textContent = "Deleting…";
  try {
    const res = await api(`/api/rag/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    toast(`Removed ${res.removed_chunks} chunk${res.removed_chunks === 1 ? "" : "s"}.`, "success");
    await refreshDocuments();
    refreshHealth();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, "error");
    btn.disabled = false;
    btn.dataset.armed = "";
    btn.textContent = "Delete";
    btn.classList.remove("confirm");
  }
}

$("#upload-input").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!file) return;

  const label = $("#upload-label");
  const labelText = $("#upload-label-text");
  label.classList.add("is-busy");
  labelText.textContent = "Uploading…";

  const stateRow = el("li", "state", `Uploading ${file.name}…`);
  $("#docs-list").prepend(stateRow);
  const phase = setTimeout(() => {
    if (stateRow.isConnected) stateRow.textContent = `Indexing ${file.name}…`;
  }, 500);

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await api("/api/rag/documents", { method: "POST", body: form });
    const doc = res.document || {};
    if (res.duplicate) {
      toast(`Already indexed: ${doc.filename}`, "warn");
    } else {
      toast(`Indexed ${doc.filename} · ${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`, "success");
      pushNote(`Indexed ${doc.filename} — ready to query with RAG.`);
    }
    await refreshDocuments();
    refreshHealth();
  } catch (err) {
    stateRow.remove();
    toast(`Upload failed: ${err.message}`, "error");
  } finally {
    clearTimeout(phase);
    label.classList.remove("is-busy");
    labelText.textContent = "Upload PDF";
  }
});

const docsClose = $("#docs-close");
if (docsClose) {
  docsClose.addEventListener("click", () => {
    CAPABILITY_STATE.rag = false;
    syncCapabilityUI();
  });
}

/* ─────────────── server conversation lifecycle ─────────────── */
function serverMessageToLocal(m) {
  const meta = m.meta || {};
  if (m.role === "system") {
    return { role: "note", text: m.content };
  }
  if (meta.kind === "research") {
    return {
      role: "assistant",
      text: m.content,
      meta,
      capabilities: [],
      sources: m.sources || [],
    };
  }
  return {
    role: m.role === "user" ? "user" : "assistant",
    text: m.content,
    meta: {
      text: m.provider ? `via ${m.provider}${m.model ? ` · ${m.model}` : ""}` : "",
      kind: "answer",
    },
    capabilities: m.capabilities || [],
    sources: m.sources || [],
  };
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
  conversation.length = 0;
  for (const m of data.messages || []) conversation.push(serverMessageToLocal(m));
  rememberConversation(data.conversation.id);
  setConversationSummary(data.conversation);
  renderStream();
  refreshConversationList();
  return data.conversation;
}

async function refreshConversation() {
  if (!conversationId) return null;
  try {
    return await loadConversation(conversationId);
  } catch (_) {
    return null;
  }
}

function startNewConversation() {
  conversation.length = 0;
  rememberConversation(null);
  conversationSummary = null;
  renderStream();
  if (window.AthenaChatHeader) {
    window.AthenaChatHeader.updateHeader({
      title: "New Chat", message_count: 0, cost_display: "$—", cost_known: false,
      total_tokens: 0, by_model: [],
    });
  }
  refreshConversationList();
}

/* The left sidebar (conversations.js) owns its own rendering but re-runs itself
   through here whenever a conversation is created, opened, renamed or removed. */
function refreshConversationList() {
  if (window.AthenaConversations && window.AthenaConversations.refresh) {
    window.AthenaConversations.refresh();
  }
}

function conversationPath(id, suffix = "") {
  return `/api/conversations/${encodeURIComponent(id)}${suffix}`;
}

async function listConversations(limit = 100) {
  const res = await api(`/api/conversations?limit=${encodeURIComponent(limit)}`);
  return res.conversations || [];
}

/* The id-aware forms back the per-conversation sidebar menu; with no id they act
   on the conversation currently open in the chat. */
async function renameConversation(id, title) {
  const target = id || conversationId;
  if (!target) throw new Error("no conversation");
  const res = await send2(conversationPath(target), "PATCH", { title });
  if (target === conversationId) setConversationSummary(res.conversation);
  refreshConversationList();
  return res.conversation;
}

async function compactConversation(id) {
  const target = id || conversationId;
  if (!target) throw new Error("no conversation");
  const res = await send2(conversationPath(target, "/compact"), "POST", {});
  if (target === conversationId) {
    setConversationSummary(res.conversation);
    await refreshConversation();
  }
  refreshConversationList();
  return res;
}

async function transcriptOf(id) {
  const target = id || conversationId;
  if (!target) throw new Error("no conversation");
  const res = await fetch(conversationPath(target, "/transcript"), { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

async function saveConversation(id) {
  const target = id || conversationId;
  if (!target) throw new Error("no conversation");
  const res = await send2(conversationPath(target, "/save"), "POST", {});
  return res.document;
}

async function deleteConversationById(id) {
  const target = id || conversationId;
  if (!target) return;
  await send2(conversationPath(target), "DELETE");
  if (target === conversationId) startNewConversation();
  refreshConversationList();
}

/* Exposed to chatHeader.js (header actions), conversations.js (sidebar) and
   research.js (result refresh). The no-argument forms act on the current chat. */
window.AthenaChat = {
  getConversationId: () => conversationId,
  getConversation: () => conversationSummary,
  refresh: refreshConversation,
  startNewConversation,
  list: listConversations,
  open: (id) => loadConversation(id),
  renameConversation,
  compactConversation,
  transcriptOf,
  saveConversation,
  deleteConversationById,
  rename: (title) => renameConversation(conversationId, title),
  compact: () => compactConversation(conversationId),
  transcript: () => transcriptOf(conversationId),
  saveDocument: () => saveConversation(conversationId),
  deleteConversation: () => deleteConversationById(conversationId),
};

async function send2(path, method, body) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body === undefined ? {} : body),
  });
}

/* ─────────────── init ─────────────── */
syncCapabilityUI();
setBusy(false);
refreshHealth();
refreshDocuments();
setInterval(refreshHealth, 30000);
// Resume the conversation this browser was last in (chat header, usage and any
// Deep Research result cards come back with it).
if (conversationId) {
  loadConversation(conversationId).catch(() => {
    rememberConversation(null);
    startNewConversation();
  });
} else {
  startNewConversation();
}
