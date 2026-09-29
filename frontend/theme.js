"use strict";

/* Athena theme system.
   Ported from Odysseus's static/js/theme.js (presets, colour maths, harmony
   generator, apply/persist/reset behaviour). The preset palettes are used
   verbatim; the derived variables are mapped onto ATHENA's own CSS tokens
   (--panel-2/--card/--fg-dim/--muted/--accent/--accent-2/--border-strong)
   instead of Odysseus's markup-specific ones.

   Apply = inline custom properties on <html>, exactly like Odysseus.
   Persist = localStorage first, mirrored to /api/prefs/theme so it also
   survives a different browser or a cleared profile. */

(() => {
  const THEMES = {
    dark:       { bg: "#282c34", fg: "#9cdef2", panel: "#111111", border: "#355a66", red: "#e06c75" },
    light:      { bg: "#f0ebe3", fg: "#5a5248", panel: "#faf6f0", border: "#d4cdc2", red: "#c47d5a" },
    midnight:   { bg: "#0d1117", fg: "#c9d1d9", panel: "#161b22", border: "#30363d", red: "#f85149" },
    paper:      { bg: "#faf8f5", fg: "#3b3836", panel: "#ffffff", border: "#d5d0c8", red: "#c5ac4a" },
    cyberpunk:  { bg: "#0a0a0f", fg: "#0ff0fc", panel: "#12101a", border: "#9b30ff", red: "#e040fb" },
    retrowave:  { bg: "#1a1a2e", fg: "#e94560", panel: "#16213e", border: "#533483", red: "#e94560" },
    forest:     { bg: "#1b2a1b", fg: "#a8d5a2", panel: "#142414", border: "#3d6b3d", red: "#7cb871" },
    ocean:      { bg: "#0b1a2c", fg: "#64d2ff", panel: "#091422", border: "#1e5074", red: "#4facfe" },
    ume:        { bg: "#2b1b2e", fg: "#f5c2e7", panel: "#1e1420", border: "#6c4675", red: "#f5a0c0" },
    copper:     { bg: "#1c1410", fg: "#e8c39e", panel: "#140f0a", border: "#7a5533", red: "#d4764e" },
    terminal:   { bg: "#000000", fg: "#00ff41", panel: "#0a0a0a", border: "#003b00", red: "#00ff41" },
    organs:     { bg: "#0a0406", fg: "#efe1c8", panel: "#15080a", border: "#3a1519", red: "#c83240" },
    lavender:   { bg: "#f3eef8", fg: "#3d3551", panel: "#faf7ff", border: "#cec3de", red: "#9b6dcc" },
    gpt:        { bg: "#212121", fg: "#ececec", panel: "#171717", border: "#424242", red: "#949494",
                  advanced: { userBubbleBg: "#2f2f2f", aiBubbleBg: "#171717", inputBg: "#2f2f2f" } },
    claude:     { bg: "#262624", fg: "#f5f4f0", panel: "#30302e", border: "#4a4a47", red: "#c6613f" },
    cute:       { bg: "#fff0f5", fg: "#d4608a", panel: "#fff8fa", border: "#f0c0d0", red: "#ff6b9d" },
  };

  // Display labels — Odysseus shows "original" for `dark` and "GPT" for `gpt`.
  const THEME_LABELS = { dark: "original", gpt: "GPT" };
  const themeLabel = (name) => THEME_LABELS[name] || name;

  const DEFAULT_THEME = "dark";
  const LS_KEY = "athena-theme";
  const CUSTOM_THEMES_KEY = "athena-custom-themes";
  const MAX_CUSTOM_THEMES = 8;

  // Extra controls, mapped to Athena's real tokens.
  const ADV_KEYS = [
    { key: "panel2",      css: "--panel-2",       label: "Panel 2",        group: "Surfaces" },
    { key: "card",        css: "--card",          label: "Card",           group: "Surfaces" },
    { key: "borderStrong", css: "--border-strong", label: "Border (strong)", group: "Surfaces" },
    { key: "fgDim",       css: "--fg-dim",        label: "Text (dim)",     group: "Text" },
    { key: "muted",       css: "--muted",         label: "Text (muted)",   group: "Text" },
    { key: "accent2",     css: "--accent-2",      label: "Accent 2",       group: "Accent" },
  ];

  const BASE_PICKERS = [
    { key: "bg", label: "Background" },
    { key: "fg", label: "Text" },
    { key: "panel", label: "Panel" },
    { key: "border", label: "Border" },
    { key: "red", label: "Accent" },
  ];

  /* ─────────────── colour maths (from Odysseus) ─────────────── */
  function hexToRgb(hex) {
    let h = (hex || "").trim().replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    if (!/^[0-9a-f]{6}$/i.test(h)) return null;
    return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
  }

  function hexToHSL(hex) {
    const rgb = hexToRgb(hex) || { r: 0, g: 0, b: 0 };
    const r = rgb.r / 255, g = rgb.g / 255, b = rgb.b / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;
    if (max === min) { h = s = 0; }
    else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      else if (max === g) h = ((b - r) / d + 2) / 6;
      else h = ((r - g) / d + 4) / 6;
    }
    return [h * 360, s * 100, l * 100];
  }

  function hslToHex(h, s, l) {
    h = ((h % 360) + 360) % 360;
    s = Math.max(0, Math.min(100, s)) / 100;
    l = Math.max(0, Math.min(100, l)) / 100;
    const a = s * Math.min(l, 1 - l);
    const f = (n) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    const toHex = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
    return "#" + toHex(f(0)) + toHex(f(8)) + toHex(f(4));
  }

  /* Derive Athena's full token set from a theme's five base colours. */
  function deriveTokens(colors) {
    const [bgH, bgS, bgL] = hexToHSL(colors.bg);
    const [fgH, fgS, fgL] = hexToHSL(colors.fg);
    const [bdH, bdS, bdL] = hexToHSL(colors.border);
    const [acH, acS, acL] = hexToHSL(colors.red || "#e06c75");
    const isDark = bgL < 50;

    return {
      "--panel-2": hslToHex(bgH, bgS, isDark ? Math.min(bgL + 6, 100) : Math.max(bgL - 5, 0)),
      "--card": hslToHex(bgH, bgS, isDark ? Math.min(bgL + 3, 100) : Math.max(bgL - 2, 0)),
      "--border-strong": hslToHex(bdH, bdS, isDark ? Math.min(bdL + 12, 100) : Math.max(bdL - 12, 0)),
      "--fg-dim": hslToHex(fgH, fgS, isDark ? Math.min(fgL + 6, 100) : Math.max(fgL - 12, 0)),
      "--muted": hslToHex(fgH, Math.max(fgS - 20, 5), fgL * 0.5 + bgL * 0.5),
      "--accent": colors.red || "#e06c75",
      "--accent-2": hslToHex(acH, acS, isDark ? Math.min(acL + 15, 85) : Math.max(acL - 12, 25)),
    };
  }

  function computeAdvancedDefaults(colors) {
    const t = deriveTokens(colors);
    return {
      panel2: t["--panel-2"],
      card: t["--card"],
      borderStrong: t["--border-strong"],
      fgDim: t["--fg-dim"],
      muted: t["--muted"],
      accent2: t["--accent-2"],
    };
  }

  /* Ported verbatim from Odysseus's generateHarmonyColors. */
  function generateHarmonyColors(accentHex, harmonyType, mode) {
    const [h, s] = hexToHSL(accentHex);
    const isDark = mode === "dark";
    let bgH, bgS, bgL, fgS, fgL, panelL, borderH, borderS, borderL;

    if (harmonyType === "complementary") {
      bgH = h; bgS = Math.max(s * 0.15, 3);
      bgL = isDark ? 13 : 95; fgL = isDark ? 85 : 15; fgS = Math.max(s * 0.2, 5);
      panelL = isDark ? 8 : 98;
      borderH = h; borderS = Math.max(s * 0.25, 8); borderL = isDark ? 28 : 75;
    } else if (harmonyType === "analogous") {
      bgH = (h - 30 + 360) % 360; bgS = Math.max(s * 0.12, 3);
      bgL = isDark ? 14 : 95; fgL = isDark ? 84 : 18; fgS = Math.max(s * 0.15, 5);
      panelL = isDark ? 9 : 97;
      borderH = (h + 30) % 360; borderS = Math.max(s * 0.3, 10); borderL = isDark ? 30 : 72;
    } else if (harmonyType === "triadic") {
      bgH = (h + 240) % 360; bgS = Math.max(s * 0.1, 2);
      bgL = isDark ? 13 : 96; fgL = isDark ? 86 : 14; fgS = Math.max(s * 0.18, 5);
      panelL = isDark ? 8 : 99;
      borderH = (h + 120) % 360; borderS = Math.max(s * 0.2, 8); borderL = isDark ? 28 : 74;
    } else {
      bgH = h; bgS = Math.max(s * 0.08, 2);
      bgL = isDark ? 12 : 96; fgL = isDark ? 87 : 13; fgS = Math.max(s * 0.15, 5);
      panelL = isDark ? 7 : 99;
      borderH = h; borderS = Math.max(s * 0.2, 6); borderL = isDark ? 26 : 76;
    }

    return {
      bg: hslToHex(bgH, bgS, bgL),
      fg: hslToHex(h, fgS, fgL),
      panel: hslToHex(bgH, bgS * 0.6, panelL),
      border: hslToHex(borderH, borderS, borderL),
      red: accentHex,
    };
  }

  /* ─────────────── apply ─────────────── */
  function applyColors(colors) {
    const s = document.documentElement.style;
    if (colors.bg) s.setProperty("--bg", colors.bg);
    if (colors.fg) s.setProperty("--fg", colors.fg);
    if (colors.panel) s.setProperty("--panel", colors.panel);
    if (colors.border) s.setProperty("--border", colors.border);
    if (colors.red) s.setProperty("--red", colors.red);

    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta && colors.bg) meta.setAttribute("content", colors.bg);

    const tokens = deriveTokens(colors);
    const adv = colors.advanced || {};
    const defaults = computeAdvancedDefaults(colors);
    for (const { key, css } of ADV_KEYS) {
      s.setProperty(css, adv[key] || defaults[key] || tokens[css] || "");
    }
    for (const [css, value] of Object.entries(tokens)) {
      if (!ADV_KEYS.some((a) => a.css === css)) s.setProperty(css, value);
    }
    if (colors.font) s.setProperty("--font", colors.font);
    applyFavicon(colors.red || "#e06c75");
  }

  function applyFavicon(accent) {
    const link = document.querySelector('link[rel="icon"]');
    if (!link) return;
    const svg =
      "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>" +
      `<rect width='32' height='32' rx='7' fill='${accent}'/>` +
      "<path d='M16 6 L26 26 H21.4 L16 14.4 L10.6 26 H6 Z' fill='#0b0e12'/></svg>";
    link.setAttribute("href", "data:image/svg+xml;utf8," + encodeURIComponent(svg));
  }

  /* ─────────────── persistence ─────────────── */
  function getSaved() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  }

  function _writeLocal(name, colors, opts) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({ name, colors, ...(opts || {}) }));
    } catch (_) { /* storage disabled — the server mirror still works */ }
  }

  function _syncToServer(obj) {
    try {
      fetch("/api/prefs/theme", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: obj }),
      }).catch(() => {});
    } catch (_) { /* ignore */ }
  }

  function save(name, colors, opts) {
    const payload = { name, colors, ...(opts || {}) };
    _writeLocal(name, colors, opts);
    _syncToServer(payload);
    return payload;
  }

  async function loadFromServer() {
    try {
      const res = await fetch("/api/prefs/theme", { cache: "no-store" });
      if (!res.ok) return null;
      const body = await res.json();
      const value = body && body.value;
      return value && value.colors ? value : null;
    } catch (_) { return null; }
  }

  function _loadCustomThemes() {
    try {
      const raw = localStorage.getItem(CUSTOM_THEMES_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (_) { return {}; }
  }

  function _saveCustomThemes(obj) {
    try { localStorage.setItem(CUSTOM_THEMES_KEY, JSON.stringify(obj)); } catch (_) { /* ignore */ }
    try {
      fetch("/api/prefs/custom-themes", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: obj }),
      }).catch(() => {});
    } catch (_) { /* ignore */ }
  }

  function saveCustomTheme(name, colors, opts) {
    const ct = _loadCustomThemes();
    if (!ct[name] && Object.keys(ct).length >= MAX_CUSTOM_THEMES) return "limit";
    ct[name] = { ...colors, ...(opts || {}) };
    _saveCustomThemes(ct);
    render();
    return "ok";
  }

  function deleteCustomTheme(name) {
    const ct = _loadCustomThemes();
    delete ct[name];
    _saveCustomThemes(ct);
    render();
  }

  /* ─────────────── state ─────────────── */
  let activeName = DEFAULT_THEME;
  let working = { ...THEMES[DEFAULT_THEME] };   // colours currently on screen
  let refColors = { ...THEMES[DEFAULT_THEME] };  // reference for per-row reset
  let tab = "browse";

  function selectTheme(name, colors) {
    activeName = name;
    working = { ...colors };
    refColors = { ...colors };
    applyColors(working);
    save(name, working, {});
    render();
  }

  /* ─────────────── UI ─────────────── */
  const $ = (sel) => document.querySelector(sel);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  function swatchCard(name, colors, { custom = false } = {}) {
    const dots = ["bg", "panel", "fg", "red"]
      .map((k) => `<span style="background:${esc(colors[k] || "#000")}"></span>`).join("");
    return `<div class="theme-swatch${name === activeName ? " active" : ""}" data-theme="${esc(name)}">
      <div class="theme-swatch-colors">${dots}</div>
      <span class="theme-swatch-name">${esc(themeLabel(name))}</span>
      ${custom ? `<button class="theme-delete-btn" data-delete="${esc(name)}" title="Delete theme">×</button>` : ""}
    </div>`;
  }

  function renderBrowse() {
    const grid = $("#themeGrid");
    const userGrid = $("#themeUserGrid");
    const userCard = $("#themeUserCard");
    if (!grid) return;
    grid.innerHTML = Object.entries(THEMES).map(([n, c]) => swatchCard(n, c)).join("");

    const custom = _loadCustomThemes();
    const names = Object.keys(custom);
    if (userCard) userCard.hidden = names.length === 0;
    if (userGrid) userGrid.innerHTML = names.map((n) => swatchCard(n, custom[n], { custom: true })).join("");
  }

  function colorRow(label, value, key, { advanced = false } = {}) {
    const attr = advanced ? `data-adv="${key}"` : `data-color="${key}"`;
    const ref = advanced ? computeAdvancedDefaults(refColors)[key] : refColors[key];
    const changed = (value || "").toLowerCase() !== (ref || "").toLowerCase();
    return `<div class="theme-color-row">
      <span class="theme-color-label">${esc(label)}</span>
      <input type="color" ${attr} value="${esc(value || "#000000")}" />
      <button class="color-reset-btn${changed ? " changed" : ""}" data-reset${advanced ? "-adv" : ""}="${esc(key)}" title="Reset">↺</button>
    </div>`;
  }

  function renderCustomize() {
    const adv = working.advanced || {};
    const defaults = computeAdvancedDefaults(working);
    const groups = ["Surfaces", "Text", "Accent"];
    const advRows = groups.map((group) => {
      const rows = ADV_KEYS.filter((k) => k.group === group).map((k) =>
        colorRow(k.label, adv[k.key] || defaults[k.key], k.key, { advanced: true })).join("");
      return rows ? `<div class="theme-color-group">${esc(group)}</div>${rows}` : "";
    }).join("");

    $("#theme-tab-customize").innerHTML = `
      <div class="admin-card">
        <h2>Colors</h2>
        <div class="theme-color-list">
          ${BASE_PICKERS.map((p) => colorRow(p.label, working[p.key], p.key)).join("")}
        </div>
        <h2 class="theme-subhead">More Colors</h2>
        <div class="theme-color-list">${advRows}</div>
        <div class="theme-actions">
          <button class="btn-sm" id="theme-adv-clear" type="button">Clear Advanced Overrides</button>
          <button class="btn-sm" id="theme-export-btn" type="button">Export</button>
          <button class="btn-sm" id="theme-import-btn" type="button">Import</button>
        </div>
      </div>
      <div class="admin-card">
        <h2>Color Harmony</h2>
        <div class="theme-harmony">
          <label>Accent <input type="color" id="harmony-accent" value="${esc(working.red || "#e06c75")}" /></label>
          <label>Type
            <select id="harmony-type">
              <option value="analogous">Analogous</option>
              <option value="complementary">Complementary</option>
              <option value="triadic">Triadic</option>
              <option value="monochromatic">Monochromatic</option>
            </select>
          </label>
          <label>Mode
            <select id="harmony-mode">
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
          </label>
          <button class="btn-sm" id="harmony-generate-btn" type="button">Generate</button>
        </div>
      </div>
      <div class="admin-card">
        <h2>Your Themes</h2>
        <div class="theme-save-row">
          <input id="theme-custom-name" type="text" placeholder="Theme name" maxlength="40" />
          <button class="btn-primary" id="theme-save-btn" type="button">Save as theme</button>
          <button class="btn-sm" id="theme-reset-btn" type="button">Reset to default</button>
        </div>
        <p class="settings-hint">Custom themes are limited to ${MAX_CUSTOM_THEMES}.</p>
      </div>`;
    wireCustomize();
  }

  function render() {
    renderBrowse();
    if (tab === "customize") renderCustomize();
  }

  /* ─────────────── wiring ─────────────── */
  function setTab(next) {
    tab = next;
    document.querySelectorAll("#theme-tabs .admin-tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === next);
    });
    const browse = $("#theme-tab-browse");
    const customize = $("#theme-tab-customize");
    if (browse) browse.hidden = next !== "browse";
    if (customize) customize.hidden = next !== "customize";
    render();
  }

  function wireBrowse() {
    const grid = $("#themeGrid");
    const userGrid = $("#themeUserGrid");
    const onClick = (event) => {
      const del = event.target.closest("[data-delete]");
      if (del) {
        deleteCustomTheme(del.dataset.delete);
        if (activeName === del.dataset.delete) selectTheme(DEFAULT_THEME, THEMES[DEFAULT_THEME]);
        return;
      }
      const card = event.target.closest(".theme-swatch");
      if (!card) return;
      const name = card.dataset.theme;
      const colors = THEMES[name] || _loadCustomThemes()[name];
      if (colors) selectTheme(name, colors);
    };
    if (grid) grid.addEventListener("click", onClick);
    if (userGrid) userGrid.addEventListener("click", onClick);
  }

  function wireCustomize() {
    const list = $("#theme-tab-customize");
    if (!list) return;

    list.addEventListener("input", (event) => {
      const input = event.target;
      if (input.matches('input[type="color"][data-color]')) {
        working[input.dataset.color] = input.value;
        applyColors(working);
        save(activeName, working, {});
      } else if (input.matches('input[type="color"][data-adv]')) {
        working.advanced = { ...(working.advanced || {}), [input.dataset.adv]: input.value };
        applyColors(working);
        save(activeName, working, {});
      }
    });

    list.addEventListener("click", (event) => {
      const reset = event.target.closest(".color-reset-btn");
      if (reset) {
        if (reset.dataset.resetAdv) {
          const key = reset.dataset.resetAdv;
          if (working.advanced) delete working.advanced[key];
        } else {
          const key = reset.dataset.reset;
          working[key] = refColors[key];
        }
        applyColors(working);
        save(activeName, working, {});
        renderCustomize();
        return;
      }
      if (event.target.id === "theme-adv-clear") {
        delete working.advanced;
        applyColors(working);
        save(activeName, working, {});
        renderCustomize();
        return;
      }
      if (event.target.id === "theme-reset-btn") {
        try { localStorage.removeItem(LS_KEY); } catch (_) { /* ignore */ }
        selectTheme(DEFAULT_THEME, THEMES[DEFAULT_THEME]);
        return;
      }
      if (event.target.id === "theme-save-btn") {
        const name = ($("#theme-custom-name").value || "").trim();
        if (!name) { toast("Enter a theme name first.", "warn"); return; }
        const result = saveCustomTheme(name, working, {});
        toast(result === "limit" ? `Custom theme limit (${MAX_CUSTOM_THEMES}) reached.` : `Saved “${name}”.`,
              result === "limit" ? "warn" : "success");
        if (result === "ok") setTab("browse");
        activeName = name;
        render();
        return;
      }
      if (event.target.id === "harmony-generate-btn") {
        working = generateHarmonyColors($("#harmony-accent").value, $("#harmony-type").value, $("#harmony-mode").value);
        applyColors(working);
        save(activeName, working, {});
        renderCustomize();
        return;
      }
      if (event.target.id === "theme-export-btn") {
        const blob = new Blob([JSON.stringify({ name: activeName, colors: working }, null, 2)], { type: "application/json" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${activeName || "theme"}.json`;
        link.click();
        URL.revokeObjectURL(link.href);
        return;
      }
      if (event.target.id === "theme-import-btn") {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "application/json,.json";
        input.addEventListener("change", async () => {
          const file = input.files && input.files[0];
          if (!file) return;
          try {
            const parsed = JSON.parse(await file.text());
            if (!parsed || !parsed.colors) throw new Error("missing colors");
            activeName = parsed.name || "imported";
            working = { ...parsed.colors };
            refColors = { ...working };
            applyColors(working);
            save(activeName, working, {});
            setTab("browse");
          } catch (err) {
            toast(`Import failed: ${err.message}`, "error");
          }
        });
        input.click();
      }
    });
  }

  function open() {
    const modal = $("#theme-modal");
    if (!modal) return;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    setTab(tab);
  }

  function close() {
    const modal = $("#theme-modal");
    if (modal) modal.hidden = true;
    document.body.classList.remove("modal-open");
  }

  function initThemeUI() {
    const tabs = $("#theme-tabs");
    if (tabs) {
      tabs.addEventListener("click", (event) => {
        const btn = event.target.closest(".admin-tab");
        if (btn) setTab(btn.dataset.tab);
      });
    }
    const openBtn = $("#open-theme");
    if (openBtn) openBtn.addEventListener("click", open);
    const closeBtn = $("#close-theme");
    if (closeBtn) closeBtn.addEventListener("click", close);
    const modal = $("#theme-modal");
    if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal && !modal.hidden) close();
    });
    wireBrowse();
    render();
  }

  /* ─────────────── boot ─────────────── */
  async function boot() {
    const saved = getSaved();
    if (saved && saved.colors) {
      activeName = saved.name || DEFAULT_THEME;
      working = { ...saved.colors };
      refColors = { ...working };
      applyColors(working);
    } else {
      const server = await loadFromServer();
      if (server) {
        activeName = server.name || DEFAULT_THEME;
        working = { ...server.colors };
        refColors = { ...working };
        _writeLocal(activeName, working, {});
        applyColors(working);
      } else {
        applyColors(working);
      }
    }
    try {
      const res = await fetch("/api/prefs/custom-themes", { cache: "no-store" });
      if (res.ok) {
        const body = await res.json();
        if (body && body.value && typeof body.value === "object" && !Object.keys(_loadCustomThemes()).length) {
          _saveCustomThemes(body.value);
        }
      }
    } catch (_) { /* ignore */ }
    initThemeUI();
  }

  // Apply the saved theme before paint where possible (avoids a flash), then
  // wire the UI once the DOM exists.
  const early = getSaved();
  if (early && early.colors) applyColors(early.colors);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { boot(); });
  } else {
    boot();
  }

  window.AthenaTheme = {
    THEMES,
    themeLabel,
    applyColors,
    generateHarmonyColors,
    saveCustomTheme,
    deleteCustomTheme,
    getCustomThemes: _loadCustomThemes,
    getSaved,
    selectTheme,
    open,
    close,
    initThemeUI,
    _state: () => ({ activeName, working }),
  };
})();
