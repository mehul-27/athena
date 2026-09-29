"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const jobsEl = $("#research-jobs");
  const currentEl = $("#research-current");
  const emptyEl = $("#research-empty");
  const form = $("#research-form");
  const jobs = new Map();
  let selectedId = null;
  let libraryLoaded = false;

  function escapeText(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  async function request(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (_) {
      throw new Error("Backend unavailable");
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
    return body;
  }

  function selectedProviderLabel(provider) {
    if (provider === "groq") return "Groq · Qwen 3.8 27B";
    if (provider === "nvidia") return "NVIDIA · Nemotron 3.5 Lightning";
    if (provider === "openrouter") return "OpenRouter · Nemotron 3.5 Lightning";
    return provider ? provider.toUpperCase() : "Search provider";
  }

  function phaseLabel(progress = {}) {
    const round = progress.round ? `Round ${progress.round} · ` : "";
    const maxRounds = $("#research-rounds").value;
    switch (progress.phase) {
      case "probing": return "Checking model…";
      case "planning": return "Planning research strategy…";
      case "searching": return `${round}Searching · ${progress.queries || 0} queries`;
      case "reading": return `${round}Reading ${progress.total_sources || 0} sources`;
      case "analyzing": return `${round}Analyzing ${progress.total_findings || 0} findings`;
      case "writing": return `Synthesizing report · ${progress.total_sources || 0} sources`;
      case "warning": return progress.message || "Continuing with partial results";
      case "error": return progress.message || "Research error";
      default: return maxRounds ? `${round}Research in progress` : `${round}Auto research in progress`;
    }
  }

  function jobCard(id, row) {
    let card = jobs.get(id)?.card;
    if (!card) {
      card = document.createElement("article");
      card.className = "research-job-card";
      card.dataset.researchId = id;
      card.innerHTML = `
        <button class="research-job-select" type="button">
          <span class="research-job-state"><i></i><span></span></span>
          <span class="research-job-title"></span>
          <span class="research-job-progress"></span>
        </button>
        <div class="research-job-actions">
          <button class="btn-ghost research-cancel" type="button">Cancel</button>
          <button class="btn-ghost research-delete" type="button">Delete</button>
        </div>`;
      $(".research-job-select", card).addEventListener("click", () => selectJob(id));
      $(".research-cancel", card).addEventListener("click", () => cancelJob(id));
      $(".research-delete", card).addEventListener("click", () => deleteJob(id));
      jobsEl.prepend(card);
      const previous = jobs.get(id) || {};
      jobs.set(id, { ...previous, card });
    }
    card.classList.toggle("is-selected", id === selectedId);
    card.classList.toggle("is-running", row.status === "running");
    card.classList.toggle("is-done", row.status === "done");
    card.classList.toggle("is-error", row.status === "error");
    card.classList.toggle("is-cancelled", row.status === "cancelled");
    $(".research-job-state span", card).textContent = row.status === "done" ? "Complete" : row.status === "running" ? phaseLabel(row.progress) : row.status;
    $(".research-job-title", card).textContent = row.query || "Research job";
    const roundText = row.progress?.round ? `Round ${row.progress.round}` : row.round_count ? `${row.round_count} rounds` : "";
    $(".research-job-progress", card).textContent = roundText;
    $(".research-cancel", card).hidden = row.status !== "running";
    $(".research-delete", card).hidden = row.status === "running";
  }

  function upsert(id, patch) {
    const previous = jobs.get(id) || { id };
    const row = { ...previous, ...patch, id };
    jobs.set(id, row);
    jobCard(id, row);
    if (id === selectedId && row.status !== "running") loadResult(id);
    else if (!selectedId) selectJob(id);
  }

  function setDetail(html) {
    emptyEl.hidden = true;
    currentEl.hidden = false;
    currentEl.innerHTML = html;
  }

  function safeLink(url) {
    try {
      const parsed = new URL(url);
      return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
    } catch (_) {
      return "";
    }
  }

  function renderSources(sources = []) {
    if (!sources.length) return "";
    const list = sources.map((source) => {
      const href = safeLink(source.url);
      const title = escapeText(source.title || source.url || "Source");
      const domain = (() => { try { return new URL(source.url).hostname; } catch (_) { return ""; } })();
      return `<a class="research-source" href="${href || "#"}" ${href ? 'target="_blank" rel="noopener noreferrer"' : ""}>
        <span class="source-number">${sources.indexOf(source) + 1}</span>
        <span class="source-title">${title}</span><span class="source-domain">${escapeText(domain)}</span>
      </a>`;
    }).join("");
    return `<section class="research-source-list"><h4>Sources <span>${sources.length}</span></h4>${list}</section>`;
  }

  function renderResult(record = {}) {
    const report = escapeText(record.result || "No report text was saved.");
    const providerText = record.providers_used?.length ? record.providers_used.join(", ") : "Automatic provider routing";
    const reportUrl = `/api/research/report/${encodeURIComponent(record.research_id || selectedId)}`;
    setDetail(`
      <header class="research-result-head">
        <div><span class="research-eyebrow">Completed investigation</span><h3>${escapeText(record.query)}</h3></div>
        <a class="research-report-open" href="${reportUrl}" target="_blank" rel="noopener">Open visual report ↗</a>
      </header>
      <div class="research-result-meta">
        <span>${escapeText(record.stats?.Duration || "")}</span>
        <span>${escapeText(record.stats?.Rounds ?? record.round_count ?? "")} rounds</span>
        <span>${escapeText(record.stats?.URLs ?? record.sources?.length ?? 0)} sources</span>
        <span>${escapeText(providerText)}</span>
      </div>
      <details class="research-report-preview" open><summary>Research report</summary><pre>${report}</pre></details>
      ${renderSources(record.sources || [])}
    `);
  }

  async function selectJob(id) {
    selectedId = id;
    $$(".research-job-card", jobsEl).forEach((card) => card.classList.toggle("is-selected", card.dataset.researchId === id));
    const row = jobs.get(id);
    if (!row) return;
    if (row.status === "running") {
      const progress = row.progress || {};
      setDetail(`<div class="research-progress-view">
        <span class="research-live-dot"></span><span class="research-eyebrow">Investigation in progress</span>
        <h3>${escapeText(row.query)}</h3><p>${escapeText(phaseLabel(progress))}</p>
        <div class="research-progress-track"><i></i></div>
        ${progress.query_preview ? `<p class="research-query-preview">Current query · ${escapeText(progress.query_preview)}</p>` : ""}
        <div class="research-live-stats"><span>${progress.total_sources || 0} sources</span><span>${progress.total_findings || 0} findings</span></div>
      </div>`);
      return;
    }
    if (row.result) renderResult(row);
    else await loadResult(id);
  }

  async function loadResult(id) {
    try {
      const record = await request(`/api/research/detail/${encodeURIComponent(id)}`);
      record.research_id = id;
      jobs.set(id, { ...(jobs.get(id) || {}), ...record, id });
      renderResult(record);
    } catch (err) {
      setDetail(`<div class="research-error-state">Could not load research result: ${escapeText(err.message)}</div>`);
    }
  }

  function listen(job) {
    if (!job || job.eventSource || job.status !== "running") return;
    const source = new EventSource(`/api/research/stream/${encodeURIComponent(job.id)}`);
    job.eventSource = source;
    source.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch (_) { return; }
      if (data.status === "not_found") {
        source.close();
        job.eventSource = null;
        upsert(job.id, { status: "error", progress: { phase: "error", message: "Research job no longer exists" } });
        return;
      }
      const status = data.final ? data.status : (data.status || "running");
      const progress = data.final ? (job.progress || {}) : data;
      upsert(job.id, { status, progress, ...(data.error ? { error: data.error } : {}) });
      if (data.final) {
        source.close();
        job.eventSource = null;
        loadResult(job.id);
        refreshLibrary();
        // The finished research was posted into the conversation — show its card.
        if (window.AthenaChat && window.AthenaChat.refresh) window.AthenaChat.refresh();
      }
    };
    source.onerror = () => {
      source.close();
      job.eventSource = null;
      poll(job.id);
    };
  }

  async function poll(id) {
    const job = jobs.get(id);
    if (!job || job.status !== "running") return;
    try {
      const data = await request(`/api/research/status/${encodeURIComponent(id)}`);
      upsert(id, data);
      if (data.status === "running") setTimeout(() => poll(id), 2000);
      else { loadResult(id); refreshLibrary(); }
    } catch (err) {
      upsert(id, { status: "error", progress: { phase: "error", message: err.message } });
    }
  }

  async function startResearch(event) {
    event.preventDefault();
    const query = $("#research-query").value.trim();
    if (!query) return;
    const button = $("#research-start");
    button.disabled = true;
    button.textContent = "Starting…";
    try {
      const payload = {
        query,
        max_rounds: $("#research-rounds").value ? Number($("#research-rounds").value) : null,
        category: $("#research-category").value || null,
        search_provider: $("#research-provider").value || null,
        // Research belongs to the chat it was launched from.
        conversation_id: window.AthenaChat ? window.AthenaChat.getConversationId() : null,
      };
      const result = await request("/api/research/start", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const id = result.research_id || result.session_id;
      upsert(id, { ...result, status: "running", progress: {} });
      const job = jobs.get(id);
      listen(job);
      selectedId = id;
      selectJob(id);
      toast("Research started", "success");
    } catch (err) {
      toast(`Could not start research: ${err.message}`, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Start research";
    }
  }

  async function cancelJob(id) {
    try {
      await request(`/api/research/cancel/${encodeURIComponent(id)}`, { method: "POST" });
      const job = jobs.get(id);
      if (job?.eventSource) job.eventSource.close();
      upsert(id, { status: "cancelled", progress: { phase: "error", message: "Research cancelled" } });
      toast("Research cancelled", "warn");
    } catch (err) { toast(`Could not cancel: ${err.message}`, "error"); }
  }

  async function deleteJob(id) {
    if (!window.confirm("Delete this research report and its source data?")) return;
    try {
      await request(`/api/research/${encodeURIComponent(id)}`, { method: "DELETE" });
      jobs.get(id)?.eventSource?.close();
      jobs.delete(id);
      $(`[data-research-id="${CSS.escape(id)}"]`, jobsEl)?.remove();
      if (selectedId === id) {
        selectedId = null;
        currentEl.hidden = true;
        emptyEl.hidden = false;
      }
      toast("Research deleted", "success");
      refreshLibrary();
    } catch (err) { toast(`Could not delete: ${err.message}`, "error"); }
  }

  async function refreshLibrary() {
    const data = await request("/api/research/library?limit=30").catch((err) => {
      jobsEl.replaceChildren();
      const error = document.createElement("div");
      error.className = "research-error-state";
      error.textContent = `Research library unavailable: ${err.message}`;
      jobsEl.appendChild(error);
      return null;
    });
    if (!data) return;
    const known = new Set(jobs.keys());
    for (const row of (data.research || [])) {
      const id = row.research_id;
      if (!id) continue;
      upsert(id, row);
      const job = jobs.get(id);
      if (row.status === "running") listen(job);
      known.delete(id);
    }
    libraryLoaded = true;
  }

  form.addEventListener("submit", startResearch);
  $("#research-refresh").addEventListener("click", refreshLibrary);

  // Deep Research is launched from Chat as a modal, not a top-level page.
  const modal = $("#research-modal");
  const openBtn = $("#open-research");
  const closeBtn = $("#close-research");
  const openModal = () => {
    if (!modal) return;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    if (!libraryLoaded) refreshLibrary();
    $("#research-query")?.focus();
  };
  const closeModal = () => {
    if (!modal) return;
    modal.hidden = true;
    document.body.classList.remove("modal-open");
  };
  if (openBtn) openBtn.addEventListener("click", openModal);
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (modal) {
    modal.addEventListener("click", (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !modal.hidden) closeModal();
    });
  }
  refreshLibrary();
})();
