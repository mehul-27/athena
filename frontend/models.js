"use strict";

/* Athena per-mode model selection.
   Each of the four working tabs (Chat, RAG Chat, Search, Deep Research) has a
   Model dropdown. The choice is persisted in Athena's provider store, so it
   survives restarts and is read by the backend at request time. */

(() => {
  const SELECTS = ["#model-chat", "#model-research"]
    .map((sel) => document.querySelector(sel))
    .filter(Boolean);

  let lastData = null;

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  async function api(path, options) {
    let res;
    try {
      // `no-store`: the provider/model list is live configuration, not a
      // cacheable asset — a stale copy looks like a provider "vanished".
      res = await fetch(path, { cache: "no-store", ...(options || {}) });
    } catch (_) {
      throw new Error("backend unavailable");
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
    return body;
  }

  function configuredProviders(data) {
    return (data.providers || []).filter((p) => p.configured && p.enabled);
  }

  function optionsFor(providers, current) {
    const html = ['<option value="">Auto · fallback chain</option>'];
    let found = current === "";
    for (const p of providers) {
      const models = [];
      if (p.model) models.push(p.model);
      for (const m of (p.models || [])) if (!models.includes(m)) models.push(m);
      if (!models.length) continue;
      html.push(`<optgroup label="${esc(p.name)}">`);
      for (const m of models) {
        const value = `${p.id}::${m}`;
        const selected = value === current;
        if (selected) found = true;
        const suffix = m === p.model ? " (current)" : "";
        html.push(`<option value="${esc(value)}"${selected ? " selected" : ""}>${esc(m)}${suffix}</option>`);
      }
      html.push("</optgroup>");
    }
    if (!found && current) {
      // Keep a saved selection visible even if the provider is momentarily
      // unlisted (e.g. just disabled) instead of silently showing "Auto".
      const provider = current.slice(0, current.indexOf("::"));
      const model = current.slice(current.indexOf("::") + 2);
      const label = (providers.find((p) => p.id === provider) || {}).name || provider;
      html.push(`<option value="${esc(current)}" selected>${esc(label)} · ${esc(model)}</option>`);
    }
    return html.join("");
  }

  function fill(select, data) {
    const mode = select.dataset.mode;
    const selection = (data.mode_models || {})[mode] || { provider: "", model: "" };
    const current = selection.provider ? `${selection.provider}::${selection.model}` : "";
    select.innerHTML = optionsFor(configuredProviders(data), current);
    select.disabled = false;
  }

  async function load() {
    let data;
    try {
      data = await api("/api/providers");
    } catch (_) {
      return;
    }
    lastData = data;
    SELECTS.forEach((select) => fill(select, data));
    return data;
  }

  async function choose(select) {
    const value = select.value;
    let provider = "";
    let model = "";
    if (value) {
      const index = value.indexOf("::");
      provider = index === -1 ? value : value.slice(0, index);
      model = index === -1 ? "" : value.slice(index + 2);
    }
    try {
      await api("/api/providers/mode-models", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: select.dataset.mode, provider, model }),
      });
      const label = select.dataset.mode === "rag" ? "RAG" : select.dataset.mode;
      toast(provider ? `${label} model → ${model}` : `${label}: automatic (fallback chain)`, "success");
    } catch (err) {
      toast(`Could not set model: ${err.message}`, "error");
    }
    await load();
  }

  SELECTS.forEach((select) => select.addEventListener("change", () => choose(select)));
  // Keep the dropdowns in sync when the Settings tab changes providers/models.
  document.querySelectorAll('.tab[data-mode]').forEach((tab) => {
    tab.addEventListener("click", () => load());
  });
  // ...and after a server restart or a return to the tab.
  window.addEventListener("focus", () => load());
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) load();
  });

  window.AthenaModels = {
    reload: load,
    onChange: () => load(),
    get data() {
      return lastData;
    },
  };

  load();
})();
