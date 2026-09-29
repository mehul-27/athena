"use strict";

/* Athena Settings — modal panel.
   Interaction model follows Odysseus's Settings (modal + sidebar categories +
   "Add Models" / "Added Models" split). Providers are fully user-defined: any
   OpenAI-compatible endpoint can be added here without touching Athena's source.
   Secrets never leak: keys are only ever submitted, never rendered. */

(() => {
  const q = (sel, root = document) => root.querySelector(sel);
  const qa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const modal = q("#settings-modal");
  const pane = q("#settings-pane");
  const navItems = q("#settings-nav-items");
  const summary = q("#settings-active-summary");

  let state = null;
  let section = "add-models";
  let editingId = null;
  let draftModels = [];
  let addForm = { preset: "custom", name: "", base_url: "", api_key: "", model: "", api_key_required: true, auth_type: "bearer" };

  const STATUS_TEXT = {
    available: "Available",
    rate_limited: "Rate limited",
    auth_failed: "Authentication failed",
    endpoint_unavailable: "Endpoint unavailable",
    model_unavailable: "Model unavailable",
    timeout: "Timeout",
    error: "Error",
    disabled: "Disabled",
    not_configured: "API key required",
  };
  const STATUS_CLASS = {
    available: "is-ok",
    rate_limited: "is-warn",
    not_configured: "is-off",
    disabled: "is-off",
    auth_failed: "is-error",
    endpoint_unavailable: "is-error",
    model_unavailable: "is-error",
    timeout: "is-warn",
    error: "is-error",
  };

  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, { cache: "no-store", ...(options || {}) });
    } catch (_) {
      throw new Error("backend unavailable");
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
    return body;
  }

  const send = (path, method, body) => api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  /* ─────────────── modal shell ─────────────── */
  function openSettings() {
    modal.hidden = false;
    document.body.classList.add("modal-open");
    load().then(() => q(".settings-find", modal)?.focus());
  }

  function closeSettings() {
    modal.hidden = true;
    document.body.classList.remove("modal-open");
    editingId = null;
  }

  function selectSection(next) {
    section = next;
    qa(".settings-nav-item", navItems).forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.section === next);
    });
    render();
  }

  /* ─────────────── data ─────────────── */
  async function load() {
    try {
      state = await api("/api/providers");
    } catch (err) {
      pane.innerHTML = `<div class="pv-error">Could not load settings: ${esc(err.message)}</div>`;
      return;
    }
    render();
  }

  function render() {
    if (summary && state) {
      summary.textContent = state.active
        ? `active: ${state.providers.find((p) => p.id === state.active)?.name || state.active} · ${state.active_model || ""}`
        : "no provider active";
    }
    if (section === "add-models") renderAddModels();
    else if (section === "added-models") renderAddedModels();
    else if (section === "ai-defaults") renderAIDefaults();
    else if (section === "library") renderLibrary();
  }

  function refreshAll() {
    load();
    if (typeof refreshHealth === "function") refreshHealth();
    if (window.AthenaModels && window.AthenaModels.reload) window.AthenaModels.reload();
  }

  /* ─────────────── Add Models ─────────────── */
  function renderAddModels() {
    const presets = (state && state.presets) || [];
    const options = presets.map((p) =>
      `<option value="${esc(p.key)}"${p.key === addForm.preset ? " selected" : ""}>${esc(p.label)}</option>`
    ).join("");
    const modelOptions = draftModels.map((m) => `<option value="${esc(m)}"></option>`).join("");

    pane.innerHTML = `
      <div class="pane-head">
        <h3>Add API Model</h3>
        <p>Connect any OpenAI-compatible endpoint. Pick a preset to pre-fill the fields, or
           choose <strong>Custom</strong> and type everything yourself — no provider needs to
           exist in Athena already.</p>
      </div>
      <div class="form-grid">
        <label class="field span-2">
          <span>Provider type</span>
          <select id="add-preset">${options}</select>
        </label>
        <label class="field">
          <span>Provider name</span>
          <input id="add-name" type="text" placeholder="My New Provider" value="${esc(addForm.name)}" />
        </label>
        <label class="field">
          <span>Endpoint (base URL)</span>
          <input id="add-base-url" type="text" spellcheck="false"
                 placeholder="https://api.example.com/v1" value="${esc(addForm.base_url)}" />
        </label>
        <label class="field">
          <span>API key</span>
          <input id="add-api-key" type="password" autocomplete="off" spellcheck="false"
                 placeholder="paste the key (never shown again)" value="${esc(addForm.api_key)}" />
        </label>
        <label class="field">
          <span>Model</span>
          <input id="add-model" type="text" spellcheck="false" list="add-model-list"
                 placeholder="model id" value="${esc(addForm.model)}" />
          <datalist id="add-model-list">${modelOptions}</datalist>
        </label>
        <label class="check span-2">
          <input id="add-key-required" type="checkbox" ${addForm.api_key_required ? "checked" : ""} />
          <span>This endpoint requires an API key (untick for local servers such as Ollama / LM Studio)</span>
        </label>
      </div>
      <div class="form-actions">
        <button class="btn-sm" id="add-discover" type="button">Discover models</button>
        <button class="btn-sm" id="add-test" type="button">Test connection</button>
        <span class="test-slot" id="add-test-slot"></span>
        <button class="btn-primary" id="add-save" type="button">Add provider</button>
      </div>
      <p class="settings-hint" id="add-hint"></p>
    `;

    q("#add-preset", pane).addEventListener("change", (event) => applyPreset(event.target.value));
    ["name", "base_url", "api_key", "model"].forEach((field) => {
      const el = q(`#add-${field.replace("_", "-")}`, pane);
      el.addEventListener("input", () => { addForm[field] = el.value; });
    });
    q("#add-key-required", pane).addEventListener("change", (e) => { addForm.api_key_required = e.target.checked; });
    q("#add-discover", pane).addEventListener("click", discoverDraft);
    q("#add-test", pane).addEventListener("click", testDraft);
    q("#add-save", pane).addEventListener("click", saveNewProvider);
  }

  function applyPreset(key) {
    const preset = (state.presets || []).find((p) => p.key === key);
    addForm.preset = key;
    // Keep a name the user typed; otherwise take the preset label.
    if (preset) {
      addForm.name = preset.key === "custom" ? addForm.name : preset.label;
      addForm.base_url = preset.base_url || "";
      addForm.model = preset.default_model || "";
      addForm.api_key_required = preset.api_key_required;
      addForm.auth_type = preset.auth_type;
    }
    draftModels = [];
    renderAddModels();
  }

  function draftPayload() {
    return {
      preset: addForm.preset,
      name: addForm.name,
      base_url: addForm.base_url,
      api_key: addForm.api_key,
      model: addForm.model,
      auth_type: addForm.auth_type,
      api_key_required: addForm.api_key_required,
    };
  }

  async function discoverDraft() {
    const slot = q("#add-hint", pane);
    slot.textContent = "Discovering models…";
    try {
      const result = await send("/api/providers/models", "POST", draftPayload());
      draftModels = result.models || [];
      slot.textContent = result.supported
        ? `Found ${result.count} models — pick one in the Model field.`
        : "Model discovery unavailable for this endpoint — type the model id manually.";
      renderAddModels();
    } catch (err) {
      slot.textContent = `Discovery failed: ${err.message}`;
    }
  }

  async function testDraft() {
    const slot = q("#add-test-slot", pane);
    slot.className = "test-slot is-pending";
    slot.textContent = "testing…";
    try {
      const result = await send("/api/providers/test", "POST", draftPayload());
      slot.className = `test-slot ${result.ok ? "is-ok" : "is-error"}`;
      slot.textContent = result.ok
        ? `OK · ${result.model}`
        : `${STATUS_TEXT[result.status] || result.status}${result.error ? ` — ${result.error}` : ""}`;
    } catch (err) {
      slot.className = "test-slot is-error";
      slot.textContent = err.message;
    }
  }

  async function saveNewProvider() {
    try {
      const result = await send("/api/providers", "POST", draftPayload());
      toast(`Added ${result.provider.name}`, "success");
      addForm = { preset: "custom", name: "", base_url: "", api_key: "", model: "", api_key_required: true, auth_type: "bearer" };
      draftModels = [];
      await load();
      selectSection("added-models");
      refreshAll();
    } catch (err) {
      toast(`Could not add provider: ${err.message}`, "error");
    }
  }

  /* ─────────────── Added Models ─────────────── */
  function modelListFor(row) {
    const models = new Set();
    if (row.model) models.add(row.model);
    (row.models || []).forEach((m) => models.add(m));
    return Array.from(models);
  }

  function providerRow(row, index, total) {
    const statusClass = STATUS_CLASS[row.status] || "is-off";
    const statusText = STATUS_TEXT[row.status] || row.status;
    const keyText = row.api_key_required
      ? (row.has_key ? row.key_masked : "no key")
      : "not required";
    const editing = editingId === row.id;
    const modelOptions = modelListFor(row)
      .map((m) => `<option value="${esc(m)}"${m === row.model ? " selected" : ""}>${esc(m)}</option>`)
      .join("");

    return `
      <article class="provider-card" data-id="${esc(row.id)}">
        <header class="pv-head">
          <span class="pv-priority">${index + 1}</span>
          <span class="pv-name">${esc(row.name)}</span>
          <span class="pv-status ${statusClass}">${esc(statusText)}</span>
          ${state.active === row.id ? '<span class="pv-active-tag">active</span>' : ""}
          <span class="pv-move">
            <button class="icon-btn" data-action="up" title="Higher priority" ${index === 0 ? "disabled" : ""}>▲</button>
            <button class="icon-btn" data-action="down" title="Lower priority" ${index === total - 1 ? "disabled" : ""}>▼</button>
          </span>
        </header>
        <div class="pv-detail">
          <span class="pv-endpoint" title="${esc(row.base_url)}">${esc(row.base_url)}</span>
          <span class="pv-model">${esc(row.model || "no model set")}</span>
        </div>
        <div class="pv-meta">key ${esc(keyText)}${row.key_fingerprint ? ` · ${esc(row.key_fingerprint)}` : ""} · ${esc(row.type)}</div>
        ${row.last_error ? `<div class="pv-error" title="${esc(row.last_error)}">${esc(row.last_error)}</div>` : ""}
        ${editing ? `
          <div class="form-grid pv-edit">
            <label class="field"><span>Name</span><input data-edit="name" type="text" value="${esc(row.name)}" /></label>
            <label class="field"><span>Endpoint</span><input data-edit="base_url" type="text" spellcheck="false" value="${esc(row.base_url)}" /></label>
            <label class="field"><span>Model</span>
              <input data-edit="model" type="text" spellcheck="false" list="models-${esc(row.id)}" value="${esc(row.model)}" />
              <datalist id="models-${esc(row.id)}">${modelOptions}</datalist>
            </label>
            <label class="field"><span>API key</span><input data-edit="api_key" type="password" autocomplete="off" placeholder="leave blank to keep current" /></label>
          </div>
          <div class="form-actions">
            <button class="btn-sm" data-action="refresh-models" type="button">Refresh models</button>
            <button class="btn-sm" data-action="test" type="button">Test</button>
            <button class="btn-primary" data-action="save" type="button">Save</button>
            <button class="btn-ghost" data-action="cancel" type="button">Cancel</button>
          </div>
        ` : `
          <div class="form-actions">
            <button class="btn-sm" data-action="test" type="button">Test</button>
            <button class="btn-sm" data-action="edit" type="button">Edit</button>
            <button class="btn-sm" data-action="toggle" type="button">${row.enabled ? "Disable" : "Enable"}</button>
            <button class="btn-sm danger" data-action="remove" type="button">Remove</button>
          </div>
        `}
      </article>`;
  }

  function renderAddedModels() {
    const providers = (state && state.providers) || [];
    if (!providers.length) {
      pane.innerHTML = `
        <div class="pane-head"><h3>Added Models</h3>
          <p>No providers configured yet.</p></div>
        <div class="empty-state">Use <strong>Add Models</strong> to connect a provider — OpenAI,
          DeepSeek, Google Gemini, Groq, NVIDIA, OpenRouter, a local server, or any other
          OpenAI-compatible endpoint.</div>`;
      return;
    }
    pane.innerHTML = `
      <div class="pane-head">
        <h3>Added Models</h3>
        <p>Priority order is the fallback order: the top provider is tried first, then the next
           one if it fails or is rate limited.</p>
      </div>
      <div class="provider-cards">
        ${providers.map((row, i) => providerRow(row, i, providers.length)).join("")}
      </div>`;
  }

  async function patchProvider(id, body) {
    await send(`/api/providers/${encodeURIComponent(id)}`, "PATCH", body);
    await load();
    refreshAll();
  }

  async function handleRowAction(button, card) {
    const id = card.dataset.id;
    const action = button.dataset.action;
    const row = state.providers.find((p) => p.id === id);
    try {
      if (action === "edit") { editingId = id; render(); return; }
      if (action === "cancel") { editingId = null; render(); return; }
      if (action === "save") {
        const body = {};
        qa("[data-edit]", card).forEach((input) => {
          const value = input.value.trim();
          if (input.dataset.edit === "api_key") { if (value) body.api_key = value; return; }
          body[input.dataset.edit] = value;
        });
        await patchProvider(id, body);
        editingId = null;
        toast("Provider saved", "success");
        return;
      }
      if (action === "toggle") { await patchProvider(id, { enabled: !row.enabled }); return; }
      if (action === "remove") {
        if (!window.confirm(`Remove ${row.name} and its stored credential?`)) return;
        await send(`/api/providers/${encodeURIComponent(id)}`, "DELETE");
        toast(`${row.name} removed`, "success");
        await load();
        refreshAll();
        return;
      }
      if (action === "test") {
        button.disabled = true; button.textContent = "Testing…";
        const result = await send(`/api/providers/${encodeURIComponent(id)}/test`, "POST", {});
        button.disabled = false; button.textContent = "Test";
        toast(result.ok ? `${row.name}: OK (${result.model})`
                        : `${row.name}: ${STATUS_TEXT[result.status] || result.status}`,
              result.ok ? "success" : "error", 6000);
        await load();
        return;
      }
      if (action === "refresh-models") {
        button.disabled = true; button.textContent = "Discovering…";
        const result = await send(`/api/providers/${encodeURIComponent(id)}/models`, "POST", {});
        button.disabled = false; button.textContent = "Refresh models";
        toast(result.supported ? `${result.count} models found` : "Model discovery unavailable — enter the model id manually",
              result.supported ? "success" : "warn");
        await load();
        return;
      }
      if (action === "up" || action === "down") {
        const order = state.providers.map((p) => p.id);
        const from = order.indexOf(id);
        const to = action === "up" ? from - 1 : from + 1;
        if (to < 0 || to >= order.length) return;
        [order[from], order[to]] = [order[to], order[from]];
        await send("/api/providers/order", "PUT", { order });
        await load();
        refreshAll();
      }
    } catch (err) {
      toast(`Action failed: ${err.message}`, "error");
      await load();
    }
  }

  /* ─────────────── AI Defaults ─────────────── */
  function providerModelOptions() {
    const options = ['<option value="">Auto · fallback chain</option>'];
    (state.providers || []).filter((p) => p.enabled).forEach((p) => {
      const models = modelListFor(p);
      if (!models.length) return;
      options.push(`<optgroup label="${esc(p.name)}">`);
      models.forEach((m) => options.push(`<option value="${esc(p.id)}::${esc(m)}">${esc(p.name)} · ${esc(m)}</option>`));
      options.push("</optgroup>");
    });
    return options.join("");
  }

  function modeSelect(mode, label) {
    const selection = (state.mode_models || {})[mode] || { provider: "", model: "" };
    const current = selection.provider ? `${selection.provider}::${selection.model}` : "";
    const html = providerModelOptions().replace(
      `value="${esc(current)}"`,
      `value="${esc(current)}" selected`
    );
    return `<label class="field"><span>${esc(label)}</span>
      <select data-mode="${esc(mode)}">${html}</select></label>`;
  }

  function renderAIDefaults() {
    const roles = state.research_models || {};
    const allModels = (state.available_models || []).map((m) => `<option value="${esc(m)}"></option>`).join("");
    pane.innerHTML = `
      <div class="pane-head">
        <h3>AI Defaults</h3>
        <p>Which configured provider/model each mode uses. "Auto" follows the priority chain.
           RAG and Web Search run inside Chat, so they use the Chat model.</p>
      </div>
      <div class="form-grid">
        ${modeSelect("chat", "Default Chat model")}
        ${modeSelect("research", "Research model (default)")}
      </div>
      <div class="pane-head">
        <h3>Research roles</h3>
        <p><strong>FAST</strong> runs control calls (planning, queries, page extraction, stop
           decisions). <strong>STRONG</strong> writes the report. Blank = the active provider's model.</p>
      </div>
      <div class="form-grid">
        <label class="field"><span>FAST model</span>
          <input data-role="fast" type="text" list="all-models" value="${esc(roles.fast || "")}" /></label>
        <label class="field"><span>STRONG model</span>
          <input data-role="strong" type="text" list="all-models" value="${esc(roles.strong || "")}" /></label>
        <datalist id="all-models">${allModels}</datalist>
      </div>
    `;
  }

  /* ─────────────── Library (research history + saved documents) ─────────────── */
  /* Deep Research and "Save to Documents" both live in Chat now; this is where
     their results remain reachable without a separate research/ documents page. */
  async function renderLibrary() {
    pane.innerHTML = `
      <div class="pane-head">
        <h3>Library</h3>
        <p>Research reports and documents saved out of your chats.</p>
      </div>
      <div class="pane-head"><h3>Saved documents</h3></div>
      <div class="library-list" id="library-docs">Loading…</div>
      <div class="pane-head"><h3>Research history</h3></div>
      <div class="library-list" id="library-research">Loading…</div>`;

    try {
      const docs = await api("/api/saved-documents");
      const list = q("#library-docs", pane);
      list.replaceChildren();
      const items = docs.documents || [];
      if (!items.length) {
        list.appendChild(el("div", "empty-state", "No saved documents yet. Use Chat → ▼ → Save to Documents."));
      } else {
        items.forEach((doc) => {
          const row = el("div", "library-row");
          const info = el("div", "library-info");
          info.appendChild(el("span", "library-title", doc.title));
          info.appendChild(el("span", "library-meta", `${doc.chars} chars · ${String(doc.created_at || "").slice(0, 19)}`));
          row.appendChild(info);
          const actions = el("div", "library-actions");
          const open = el("a", "btn-sm", "Open");
          open.href = `/api/saved-documents/${encodeURIComponent(doc.id)}`;
          open.target = "_blank";
          open.rel = "noopener noreferrer";
          const del = el("button", "btn-sm danger", "Delete");
          del.type = "button";
          del.addEventListener("click", async () => {
            if (!window.confirm(`Delete “${doc.title}”?`)) return;
            try {
              await send(`/api/saved-documents/${encodeURIComponent(doc.id)}`, "DELETE");
              toast("Document deleted", "success");
              renderLibrary();
            } catch (err) { toast(`Delete failed: ${err.message}`, "error"); }
          });
          actions.append(open, del);
          row.appendChild(actions);
          list.appendChild(row);
        });
      }
    } catch (err) {
      q("#library-docs", pane).textContent = `Could not load documents: ${err.message}`;
    }

    try {
      const data = await api("/api/research/library?limit=30");
      const list = q("#library-research", pane);
      list.replaceChildren();
      const rows = data.research || [];
      if (!rows.length) {
        list.appendChild(el("div", "empty-state", "No research yet. Use Chat → Deep Research."));
      } else {
        rows.forEach((row) => {
          const item = el("div", "library-row");
          const info = el("div", "library-info");
          info.appendChild(el("span", "library-title", row.query || "Research"));
          const bits = [String(row.status || "")];
          if (row.round_count) bits.push(`${row.round_count} rounds`);
          if ((row.sources || []).length) bits.push(`${row.sources.length} sources`);
          if (row.conversation_id) bits.push("from chat");
          info.appendChild(el("span", "library-meta", bits.filter(Boolean).join(" · ")));
          item.appendChild(info);
          const actions = el("div", "library-actions");
          const open = el("a", "btn-sm", "Visual report");
          open.href = `/api/research/report/${encodeURIComponent(row.research_id)}`;
          open.target = "_blank";
          open.rel = "noopener noreferrer";
          actions.appendChild(open);
          item.appendChild(actions);
          list.appendChild(item);
        });
      }
    } catch (err) {
      q("#library-research", pane).textContent = `Could not load research: ${err.message}`;
    }
  }

  /* ─────────────── wiring ─────────────── */
  navItems.addEventListener("click", (event) => {
    const button = event.target.closest(".settings-nav-item[data-section]");
    if (button) selectSection(button.dataset.section);
  });

  pane.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    const card = event.target.closest(".provider-card");
    if (button && card) handleRowAction(button, card);
  });

  pane.addEventListener("change", async (event) => {
    const target = event.target;
    if (target.matches("select[data-mode]")) {
      const value = target.value;
      const [provider, model] = value ? [value.slice(0, value.indexOf("::")), value.slice(value.indexOf("::") + 2)] : ["", ""];
      try {
        await send("/api/providers/mode-models", "PUT", { mode: target.dataset.mode, provider, model });
        toast(provider ? `${target.dataset.mode} → ${model}` : `${target.dataset.mode}: automatic`, "success");
        await load();
        refreshAll();
      } catch (err) {
        toast(`Could not save: ${err.message}`, "error");
      }
    } else if (target.matches("input[data-role]")) {
      try {
        await send("/api/providers/research-models", "PUT", { [target.dataset.role]: target.value.trim() });
        toast("Research model saved", "success");
        refreshAll();
      } catch (err) {
        toast(`Could not save: ${err.message}`, "error");
      }
    }
  });

  q("#open-settings").addEventListener("click", openSettings);
  q("#settings-close").addEventListener("click", closeSettings);
  modal.addEventListener("click", (event) => { if (event.target === modal) closeSettings(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.hidden) closeSettings();
  });

  // "Find settings" filters the sidebar entries.
  q("#settings-find").addEventListener("input", (event) => {
    const needle = event.target.value.trim().toLowerCase();
    qa(".settings-nav-item", navItems).forEach((btn) => {
      const match = !needle || btn.textContent.toLowerCase().includes(needle);
      btn.style.display = match ? "" : "none";
    });
  });

  window.AthenaSettings = { open: openSettings, reload: load };
})();
