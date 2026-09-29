"use strict";

/* Athena Calendar.
   Behaviour ported from Odysseus's static/js/calendar.js: the same toolbar
   (prev / Today / next, Week-Month-Year-Agenda toggle, settings, refresh, New),
   the same quick-add line backed by /api/calendar/quick-parse, the same month
   grid with a day detail panel, event create/edit/delete, per-calendar colours,
   week-start preference and ICS import/export.

   Rendered as a modal so Athena keeps its existing navigation (the calendar is a
   tool you open, not a replacement for Chat). */

(() => {
  const q = (sel, root = document) => root.querySelector(sel);
  const qa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"];
  const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const WEEKDAYS_SUN = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  const REPEAT_OPTIONS = [
    ["", "Does not repeat"],
    ["FREQ=DAILY", "Daily"],
    ["FREQ=WEEKLY", "Weekly"],
    ["FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", "Weekdays"],
    ["FREQ=MONTHLY", "Monthly"],
    ["FREQ=YEARLY", "Yearly"],
  ];

  let currentDate = new Date();
  let view = "month";
  let selectedDay = null;
  let events = [];
  let calendars = [];
  let weekStartSun = localStorage.getItem("cal-week-start") === "sun";
  let editing = null;
  let loading = false;

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
  const pad = (n) => String(n).padStart(2, "0");
  const ds = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const todayStr = () => ds(new Date());

  function parseLocalDate(value) {
    // Event strings are either "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS" (naive) or
    // with a trailing Z (absolute UTC). Treat naive as local, Z as UTC.
    if (!value) return new Date(NaN);
    if (value.length === 10) return new Date(`${value}T00:00:00`);
    if (value.endsWith("Z")) return new Date(value);
    return new Date(value);
  }

  const localDateOf = (value) => ds(parseLocalDate(value));

  function fmtTime(value) {
    const d = parseLocalDate(value);
    if (isNaN(d)) return "";
    return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  }

  function fmtDate(dateStr) {
    const d = new Date(`${dateStr}T00:00:00`);
    return d.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
  }

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

  const calColor = (ev) => ev.color || "#5b8abf";

  /* ─────────────── ranges ─────────────── */
  function monthRange(d) {
    const first = new Date(d.getFullYear(), d.getMonth(), 1);
    const dow = weekStartSun ? first.getDay() : (first.getDay() + 6) % 7;
    const start = new Date(first);
    start.setDate(first.getDate() - dow);
    const end = new Date(start);
    end.setDate(start.getDate() + 42);
    return [ds(start), ds(end)];
  }

  function weekRange(d) {
    const start = new Date(d);
    const dow = weekStartSun ? start.getDay() : (start.getDay() + 6) % 7;
    start.setDate(start.getDate() - dow);
    const end = new Date(start);
    end.setDate(start.getDate() + 7);
    return [ds(start), ds(end)];
  }

  function viewRange() {
    if (view === "week") return weekRange(currentDate);
    if (view === "year") return [`${currentDate.getFullYear()}-01-01`, `${currentDate.getFullYear() + 1}-01-01`];
    if (view === "agenda") return [todayStr(), ds(new Date(Date.now() + 60 * 86400000))];
    return monthRange(currentDate);
  }

  async function fetchCalendars() {
    try {
      const data = await api("/api/calendar/calendars");
      calendars = data.calendars || [];
    } catch (_) {
      calendars = [];
    }
  }

  async function fetchEvents() {
    const [start, end] = viewRange();
    loading = true;
    try {
      const data = await api(`/api/calendar/events?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`);
      events = data.events || [];
    } catch (err) {
      events = [];
      toast(`Calendar load failed: ${err.message}`, "error");
    } finally {
      loading = false;
    }
  }

  const eventsForDay = (dateStr) => events.filter((e) => {
    const start = e.all_day ? e.dtstart : localDateOf(e.dtstart);
    const end = e.all_day ? e.dtend : localDateOf(e.dtend);
    return start <= dateStr && dateStr <= end;
  });

  /* ─────────────── toolbar ─────────────── */
  function toolbarHTML() {
    const title = view === "agenda"
      ? "Upcoming"
      : `${MONTHS[currentDate.getMonth()]} ${currentDate.getFullYear()}`;
    const views = ["week", "month", "year", "agenda"].map((v) =>
      `<button class="cal-view-btn${view === v ? " active" : ""}" data-view="${v}" type="button">${v[0].toUpperCase() + v.slice(1)}</button>`
    ).join("");
    return `<div class="cal-toolbar">
      <div class="cal-toolbar-nav">
        <button class="cal-nav" id="cal-prev" type="button" title="Previous">←</button>
        <button class="cal-nav cal-today-btn" id="cal-today" type="button">Today</button>
        <span class="cal-title">${esc(title)}</span>
        <button class="cal-nav" id="cal-next" type="button" title="Next">→</button>
      </div>
      <div class="cal-toolbar-right">
        <div class="cal-view-toggle">${views}</div>
        <button class="cal-nav" id="cal-settings" type="button" title="Calendar settings">⚙</button>
        <button class="cal-nav" id="cal-refresh" type="button" title="Refresh">⟳</button>
        <button class="cal-add-btn" id="cal-add" type="button"><span class="cal-add-plus">+</span> New</button>
      </div>
    </div>
    <div class="cal-quickadd-row">
      <input type="text" id="cal-quickadd" class="cal-quickadd-input" autocomplete="off"
             placeholder="Quick add — return home to Ithaca 1pm tmrw" />
      <span class="cal-quickadd-status" id="cal-quickadd-status"></span>
    </div>`;
  }

  /* ─────────────── month view ─────────────── */
  function monthHTML() {
    const y = currentDate.getFullYear();
    const m = currentDate.getMonth();
    const today = todayStr();
    const [startStr] = monthRange(currentDate);
    const start = new Date(`${startStr}T00:00:00`);

    let html = '<div class="cal-grid">';
    html += '<div class="cal-week-headers">';
    (weekStartSun ? WEEKDAYS_SUN : WEEKDAYS).forEach((w) => { html += `<div class="cal-weekday">${w}</div>`; });
    html += "</div>";

    for (let row = 0; row < 6; row++) {
      html += '<div class="cal-week-row">';
      for (let col = 0; col < 7; col++) {
        const cell = new Date(start);
        cell.setDate(start.getDate() + row * 7 + col);
        const dateStr = ds(cell);
        const classes = ["cal-day"];
        if (cell.getMonth() !== m) classes.push("cal-other");
        if (dateStr === today) classes.push("cal-today");
        if (dateStr === selectedDay) classes.push("cal-selected");
        html += `<div class="${classes.join(" ")}" data-date="${dateStr}">
          <span class="cal-day-num">${cell.getDate()}</span>`;
        const dayEvents = eventsForDay(dateStr).slice(0, 3);
        for (const ev of dayEvents) {
          const time = ev.all_day ? "" : fmtTime(ev.dtstart);
          html += `<div class="cal-event-row" draggable="false" data-uid="${esc(ev.uid)}" title="${esc(ev.summary)}">
            <span class="cal-event-row-dot" style="background:${esc(calColor(ev))}"></span>
            ${time ? `<span class="cal-event-row-time">${esc(time)}</span>` : ""}
            <span class="cal-event-row-name">${esc(ev.summary)}</span>
          </div>`;
        }
        const more = eventsForDay(dateStr).length - dayEvents.length;
        if (more > 0) html += `<span class="cal-more">+${more} more</span>`;
        html += "</div>";
      }
      html += "</div>";
    }
    return html + "</div>";
  }

  /* ─────────────── week view ─────────────── */
  function weekHTML() {
    const [startStr] = weekRange(currentDate);
    const start = new Date(`${startStr}T00:00:00`);
    let html = '<div class="cal-week-grid">';
    for (let i = 0; i < 7; i++) {
      const day = new Date(start);
      day.setDate(start.getDate() + i);
      const dateStr = ds(day);
      const classes = ["cal-week-col"];
      if (dateStr === todayStr()) classes.push("cal-today");
      if (dateStr === selectedDay) classes.push("cal-selected");
      html += `<div class="${classes.join(" ")}" data-date="${dateStr}">
        <div class="cal-week-col-head">${esc(day.toLocaleDateString(undefined, { weekday: "short", day: "numeric" }))}</div>`;
      const dayEvents = eventsForDay(dateStr).sort((a, b) => (a.dtstart || "").localeCompare(b.dtstart || ""));
      if (!dayEvents.length) html += '<div class="cal-empty cal-empty-sm">—</div>';
      for (const ev of dayEvents) {
        const time = ev.all_day ? "All day" : fmtTime(ev.dtstart);
        html += `<div class="cal-event-item" data-uid="${esc(ev.uid)}">
          <div class="cal-event-dot" style="background:${esc(calColor(ev))}"></div>
          <div class="cal-event-info">
            <div class="cal-event-name">${esc(ev.summary)}</div>
            <div class="cal-event-time">${esc(time)}</div>
          </div>
        </div>`;
      }
      html += "</div>";
    }
    return html + "</div>";
  }

  /* ─────────────── year view ─────────────── */
  function yearHTML() {
    const year = currentDate.getFullYear();
    const today = todayStr();
    let html = '<div class="cal-year-grid">';
    for (let m = 0; m < 12; m++) {
      html += `<div class="cal-year-month"><h4>${MONTHS[m]}</h4><div class="cal-year-days">`;
      const first = new Date(year, m, 1);
      const dow = weekStartSun ? first.getDay() : (first.getDay() + 6) % 7;
      for (let b = 0; b < dow; b++) html += '<span class="cal-year-blank"></span>';
      const daysInMonth = new Date(year, m + 1, 0).getDate();
      for (let d = 1; d <= daysInMonth; d++) {
        const dateStr = `${year}-${pad(m + 1)}-${pad(d)}`;
        const count = eventsForDay(dateStr).length;
        const classes = ["cal-year-day"];
        if (dateStr === today) classes.push("cal-today");
        if (count) classes.push("cal-has-events");
        html += `<button class="${classes.join(" ")}" data-date="${dateStr}" type="button">${d}</button>`;
      }
      html += "</div></div>";
    }
    return html + "</div>";
  }

  /* ─────────────── agenda view ─────────────── */
  function agendaHTML() {
    const groups = new Map();
    const sorted = [...events].sort((a, b) => (a.dtstart || "").localeCompare(b.dtstart || ""));
    for (const ev of sorted) {
      const date = ev.all_day ? ev.dtstart : localDateOf(ev.dtstart);
      if (!groups.has(date)) groups.set(date, []);
      groups.get(date).push(ev);
    }
    if (!groups.size) return '<div class="cal-empty">Nothing scheduled in the next 60 days.</div>';
    let html = '<div class="cal-agenda">';
    for (const [date, dayEvents] of groups) {
      html += `<div class="cal-agenda-day"><div class="cal-agenda-date">${esc(fmtDate(date))}</div>`;
      for (const ev of dayEvents) {
        const time = ev.all_day ? "All day" : `${fmtTime(ev.dtstart)} – ${fmtTime(ev.dtend)}`;
        html += `<div class="cal-event-item" data-uid="${esc(ev.uid)}">
          <div class="cal-event-dot" style="background:${esc(calColor(ev))}"></div>
          <div class="cal-event-info">
            <div class="cal-event-name">${esc(ev.summary)}</div>
            <div class="cal-event-time">${esc(time)}</div>
            ${ev.location ? `<div class="cal-event-loc">${esc(ev.location)}</div>` : ""}
          </div>
        </div>`;
      }
      html += "</div>";
    }
    return html + "</div>";
  }

  /* ─────────────── day detail ─────────────── */
  function dayDetailHTML() {
    const dateStr = selectedDay || todayStr();
    const isToday = dateStr === todayStr();
    const dayEvents = eventsForDay(dateStr);
    let html = `<div class="cal-day-detail">
      <div class="cal-detail-header">
        <span>${esc(fmtDate(dateStr))}${isToday ? ' <span class="cal-today-tag">(Today)</span>' : ""}</span>
        <button class="cal-add-btn" id="cal-add-day" type="button"><span class="cal-add-plus">+</span> New</button>
      </div>`;
    if (!dayEvents.length) html += '<div class="cal-empty">No events</div>';
    for (const ev of dayEvents) {
      const time = ev.all_day ? "All day" : `${fmtTime(ev.dtstart)} – ${fmtTime(ev.dtend)}`;
      html += `<div class="cal-event-item" data-uid="${esc(ev.uid)}">
        <div class="cal-event-dot" style="background:${esc(calColor(ev))}"></div>
        <div class="cal-event-info">
          <div class="cal-event-name">${esc(ev.summary)}${ev.is_recurrence ? ' <span class="cal-recur-tag">↻</span>' : ""}</div>
          <div class="cal-event-time">${esc(time)}</div>
          ${ev.location ? `<div class="cal-event-loc">${esc(ev.location)}</div>` : ""}
        </div>
      </div>`;
    }
    return html + "</div>";
  }

  /* ─────────────── render ─────────────── */
  function render() {
    const body = q("#cal-body");
    if (!body) return;
    const grid = view === "week" ? weekHTML()
      : view === "year" ? yearHTML()
        : view === "agenda" ? agendaHTML()
          : monthHTML();
    body.innerHTML = toolbarHTML() + '<div class="cal-split">' + grid + dayDetailHTML() + "</div>";
    wire();
  }

  async function refresh() {
    await fetchCalendars();
    await fetchEvents();
    render();
  }

  /* ─────────────── wiring ─────────────── */
  function step(delta) {
    if (view === "week") currentDate.setDate(currentDate.getDate() + delta * 7);
    else if (view === "year") currentDate.setFullYear(currentDate.getFullYear() + delta);
    else if (view === "agenda") currentDate.setDate(currentDate.getDate() + delta * 30);
    else currentDate.setMonth(currentDate.getMonth() + delta);
    refresh();
  }

  function wire() {
    const body = q("#cal-body");
    q("#cal-prev", body).addEventListener("click", () => step(-1));
    q("#cal-next", body).addEventListener("click", () => step(1));
    q("#cal-today", body).addEventListener("click", () => {
      currentDate = new Date();
      selectedDay = todayStr();
      refresh();
    });
    q("#cal-refresh", body).addEventListener("click", () => refresh());
    q("#cal-settings", body).addEventListener("click", openSettings);
    q("#cal-add", body).addEventListener("click", () => openEventForm(null, selectedDay || todayStr()));
    const addDay = q("#cal-add-day", body);
    if (addDay) addDay.addEventListener("click", () => openEventForm(null, selectedDay || todayStr()));

    qa(".cal-view-btn", body).forEach((btn) => {
      btn.addEventListener("click", () => { view = btn.dataset.view; refresh(); });
    });

    qa("[data-date]", body).forEach((cell) => {
      cell.addEventListener("click", (event) => {
        if (event.target.closest(".cal-event-row, .cal-event-item, .cal-year-day")) return;
        selectedDay = cell.dataset.date;
        render();
      });
      cell.addEventListener("dblclick", () => openEventForm(null, cell.dataset.date));
    });

    qa("[data-uid]", body).forEach((item) => {
      item.addEventListener("click", (event) => {
        event.stopPropagation();
        const uid = item.dataset.uid;
        const ev = events.find((e) => e.uid === uid);
        if (ev) openEventForm(ev);
      });
    });

    const quick = q("#cal-quickadd", body);
    if (quick) {
      quick.addEventListener("keydown", (event) => {
        if (event.key === "Enter") { event.preventDefault(); quickAdd(quick.value); }
      });
    }
  }

  /* ─────────────── quick add ─────────────── */
  async function quickAdd(text) {
    const value = (text || "").trim();
    const status = q("#cal-quickadd-status");
    if (!value) return;
    if (status) { status.textContent = "Parsing…"; status.className = "cal-quickadd-status"; }
    try {
      const parsed = await send("/api/calendar/quick-parse", "POST", { text: value });
      if (!parsed.ok) {
        if (status) { status.textContent = parsed.error || "Could not parse"; status.className = "cal-quickadd-status is-error"; }
        return;
      }
      const event = parsed.event;
      await send("/api/calendar/events", "POST", event);
      const input = q("#cal-quickadd");
      if (input) input.value = "";
      if (status) { status.textContent = `Added “${event.summary}”`; status.className = "cal-quickadd-status is-ok"; }
      selectedDay = (event.all_day ? event.dtstart : localDateOf(event.dtstart));
      currentDate = new Date(`${selectedDay}T00:00:00`);
      await refresh();
    } catch (err) {
      if (status) { status.textContent = err.message; status.className = "cal-quickadd-status is-error"; }
    }
  }

  /* ─────────────── event form ─────────────── */
  function toLocalInput(value) {
    if (!value) return "";
    const d = parseLocalDate(value);
    if (isNaN(d)) return "";
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function toDateInput(value) {
    if (!value) return "";
    return value.slice(0, 10);
  }

  function openEventForm(existing, defaultDate) {
    editing = existing || null;
    const allDay = existing ? !!existing.all_day : false;
    const startValue = existing
      ? (allDay ? toDateInput(existing.dtstart) : toLocalInput(existing.dtstart))
      : `${defaultDate || todayStr()}T09:00`;
    const endValue = existing
      ? (allDay ? toDateInput(existing.dtend) : toLocalInput(existing.dtend))
      : `${defaultDate || todayStr()}T10:00`;
    const calOptions = calendars.map((c) =>
      `<option value="${esc(c.href)}"${existing && existing.calendar_href === c.href ? " selected" : ""}>${esc(c.name)}</option>`
    ).join("");

    showOverlay(`
      <div class="cal-form-modal" role="dialog" aria-modal="true" aria-label="Event">
        <header class="settings-modal-head">
          <h2>${existing ? "Edit event" : "New event"}</h2>
          <button class="modal-close" id="cal-form-close" type="button" aria-label="Close">×</button>
        </header>
        <div class="cal-form-body">
          <label class="field span-2"><span>Title</span>
            <input id="ev-summary" type="text" value="${esc(existing ? existing.summary : "")}" placeholder="Event title" /></label>
          <label class="check span-2">
            <input id="ev-allday" type="checkbox" ${allDay ? "checked" : ""} />
            <span>All day</span>
          </label>
          <label class="field"><span>Starts</span>
            <input id="ev-start" type="${allDay ? "date" : "datetime-local"}" value="${esc(startValue)}" /></label>
          <label class="field"><span>Ends</span>
            <input id="ev-end" type="${allDay ? "date" : "datetime-local"}" value="${esc(endValue)}" /></label>
          <label class="field"><span>Calendar</span><select id="ev-calendar">${calOptions}</select></label>
          <label class="field"><span>Repeat</span><select id="ev-repeat">
            ${REPEAT_OPTIONS.map(([v, label]) =>
              `<option value="${esc(v)}"${existing && existing.rrule === v ? " selected" : ""}>${esc(label)}</option>`).join("")}
          </select></label>
          <label class="field"><span>Location</span>
            <input id="ev-location" type="text" value="${esc(existing ? existing.location : "")}" /></label>
          <label class="field"><span>Color</span>
            <input id="ev-color" type="color" value="${esc(existing && existing.color ? existing.color : "#5b8abf")}" /></label>
          <label class="field span-2"><span>Description</span>
            <textarea id="ev-description" rows="3">${esc(existing ? existing.description : "")}</textarea></label>
        </div>
        <div class="cal-form-actions">
          ${existing ? '<button class="btn-sm danger" id="ev-delete" type="button">Delete</button>' : ""}
          <span class="cal-form-spacer"></span>
          <button class="btn-ghost" id="ev-cancel" type="button">Cancel</button>
          <button class="btn-primary" id="ev-save" type="button">Save</button>
        </div>
      </div>`);

    q("#cal-form-close").addEventListener("click", closeOverlay);
    q("#ev-cancel").addEventListener("click", closeOverlay);
    q("#ev-allday").addEventListener("change", (e) => {
      const type = e.target.checked ? "date" : "datetime-local";
      q("#ev-start").type = type;
      q("#ev-end").type = type;
    });
    q("#ev-save").addEventListener("click", saveEvent);
    const del = q("#ev-delete");
    if (del) del.addEventListener("click", deleteEvent);
  }

  function eventPayload() {
    const allDay = q("#ev-allday").checked;
    const startRaw = q("#ev-start").value;
    const endRaw = q("#ev-end").value;
    return {
      summary: q("#ev-summary").value.trim() || "Untitled",
      all_day: allDay,
      // Date-only values are midnight; timed values are the user's wall clock.
      dtstart: allDay ? startRaw : startRaw,
      dtend: endRaw || startRaw,
      location: q("#ev-location").value.trim(),
      description: q("#ev-description").value.trim(),
      rrule: q("#ev-repeat").value,
      color: q("#ev-color").value,
      calendar_href: q("#ev-calendar") ? q("#ev-calendar").value : null,
    };
  }

  async function saveEvent() {
    const payload = eventPayload();
    if (!payload.dtstart) { toast("A start date is required.", "warn"); return; }
    try {
      if (editing) {
        const baseUid = editing.series_uid || editing.uid;
        await send(`/api/calendar/events/${encodeURIComponent(baseUid)}`, "PUT", payload);
      } else {
        await send("/api/calendar/events", "POST", payload);
      }
      closeOverlay();
      toast(editing ? "Event updated" : "Event created", "success");
      editing = null;
      await refresh();
    } catch (err) {
      toast(`Save failed: ${err.message}`, "error");
    }
  }

  async function deleteEvent() {
    if (!editing) return;
    const isRecurring = !!(editing.rrule && editing.rrule.trim());
    const scope = isRecurring
      ? (window.confirm("Delete just this occurrence? OK = this occurrence, Cancel = the whole series.") ? "occurrence" : "series")
      : "series";
    try {
      await send(`/api/calendar/events/${encodeURIComponent(editing.uid)}?scope=${scope}`, "DELETE");
      closeOverlay();
      toast("Event deleted", "success");
      editing = null;
      await refresh();
    } catch (err) {
      toast(`Delete failed: ${err.message}`, "error");
    }
  }

  /* ─────────────── calendar settings ─────────────── */
  function openSettings() {
    const rows = calendars.map((c) => `
      <div class="cal-cal-row" data-href="${esc(c.href)}">
        <input type="color" class="cal-cal-color" value="${esc(c.color || "#5b8abf")}" />
        <input type="text" class="cal-cal-name" value="${esc(c.name)}" />
        <button class="btn-sm cal-cal-save" type="button">Save</button>
        <a class="btn-sm" href="/api/calendar/export/${encodeURIComponent(c.href)}">Export .ics</a>
        <button class="btn-sm danger cal-cal-delete" type="button">Delete</button>
      </div>`).join("");

    showOverlay(`
      <div class="cal-form-modal" role="dialog" aria-modal="true" aria-label="Calendar settings">
        <header class="settings-modal-head">
          <h2>Calendar settings</h2>
          <button class="modal-close" id="cal-settings-close" type="button" aria-label="Close">×</button>
        </header>
        <div class="cal-form-body">
          <div class="pane-head"><h3>Calendars</h3></div>
          <div class="cal-cal-list">${rows || '<div class="cal-empty">No calendars</div>'}</div>
          <div class="cal-settings-add">
            <input id="cal-new-name" type="text" placeholder="New calendar name" />
            <input id="cal-new-color" type="color" value="#5b8abf" />
            <button class="btn-sm" id="cal-new-add" type="button">Add calendar</button>
          </div>
          <div class="pane-head"><h3>Week starts on</h3></div>
          <div class="cal-weekstart">
            <button class="btn-sm${weekStartSun ? "" : " is-on"}" data-ws="mon" type="button">Monday</button>
            <button class="btn-sm${weekStartSun ? " is-on" : ""}" data-ws="sun" type="button">Sunday</button>
          </div>
          <div class="pane-head"><h3>Import</h3></div>
          <label class="btn-upload">
            <input id="cal-import-input" type="file" accept=".ics,text/calendar" hidden />
            <span>Import .ics file</span>
          </label>
          <p class="settings-hint" id="cal-import-status"></p>
        </div>
        <div class="cal-form-actions">
          <span class="cal-form-spacer"></span>
          <button class="btn-primary" id="cal-settings-done" type="button">Done</button>
        </div>
      </div>`);

    q("#cal-settings-close").addEventListener("click", closeOverlay);
    q("#cal-settings-done").addEventListener("click", closeOverlay);

    qa(".cal-cal-save").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const row = btn.closest(".cal-cal-row");
        try {
          await send(`/api/calendar/calendars/${encodeURIComponent(row.dataset.href)}`, "PUT", {
            name: q(".cal-cal-name", row).value.trim() || "Calendar",
            color: q(".cal-cal-color", row).value,
          });
          toast("Calendar updated", "success");
          await refresh();
          openSettings();
        } catch (err) { toast(`Update failed: ${err.message}`, "error"); }
      });
    });

    qa(".cal-cal-delete").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const row = btn.closest(".cal-cal-row");
        if (!window.confirm(`Delete calendar “${q(".cal-cal-name", row).value}” and all its events?`)) return;
        try {
          await send(`/api/calendar/calendars/${encodeURIComponent(row.dataset.href)}`, "DELETE");
          toast("Calendar deleted", "success");
          await refresh();
          openSettings();
        } catch (err) { toast(`Delete failed: ${err.message}`, "error"); }
      });
    });

    q("#cal-new-add").addEventListener("click", async () => {
      const name = q("#cal-new-name").value.trim();
      if (!name) { toast("Enter a calendar name.", "warn"); return; }
      try {
        await send("/api/calendar/calendars", "POST", { name, color: q("#cal-new-color").value });
        toast(`Added “${name}”`, "success");
        await refresh();
        openSettings();
      } catch (err) { toast(`Add failed: ${err.message}`, "error"); }
    });

    qa("[data-ws]").forEach((btn) => {
      btn.addEventListener("click", () => {
        weekStartSun = btn.dataset.ws === "sun";
        localStorage.setItem("cal-week-start", weekStartSun ? "sun" : "mon");
        openSettings();
        render();
      });
    });

    q("#cal-import-input").addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      event.target.value = "";
      if (!file) return;
      const status = q("#cal-import-status");
      status.textContent = "Importing…";
      try {
        const form = new FormData();
        form.append("file", file);
        const result = await api("/api/calendar/import", { method: "POST", body: form });
        status.textContent = `Imported ${result.imported} event(s) into “${result.calendar}” (${result.skipped} skipped).`;
        await refresh();
      } catch (err) {
        status.textContent = `Import failed: ${err.message}`;
      }
    });
  }

  /* ─────────────── overlay plumbing ─────────────── */
  function showOverlay(html) {
    let overlay = q("#cal-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "cal-overlay";
      overlay.className = "modal-backdrop cal-overlay";
      document.body.appendChild(overlay);
      overlay.addEventListener("click", (event) => { if (event.target === overlay) closeOverlay(); });
    }
    overlay.innerHTML = html;
    overlay.hidden = false;
    document.body.classList.add("modal-open");
  }

  function closeOverlay() {
    const overlay = q("#cal-overlay");
    if (overlay) overlay.hidden = true;
    document.body.classList.remove("modal-open");
    editing = null;
  }

  /* ─────────────── open / close ─────────────── */
  async function open() {
    const modal = q("#calendar-modal");
    if (!modal) return;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    if (!selectedDay) selectedDay = todayStr();
    await refresh();
  }

  function close() {
    const modal = q("#calendar-modal");
    if (modal) modal.hidden = true;
    document.body.classList.remove("modal-open");
  }

  function init() {
    const openBtn = q("#open-calendar");
    if (openBtn) openBtn.addEventListener("click", open);
    const closeBtn = q("#close-calendar");
    if (closeBtn) closeBtn.addEventListener("click", close);
    const modal = q("#calendar-modal");
    if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const overlay = q("#cal-overlay");
      if (overlay && !overlay.hidden) { closeOverlay(); return; }
      if (modal && !modal.hidden) close();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.AthenaCalendar = { open, close, refresh, _state: () => ({ view, currentDate, events, calendars }) };
})();
