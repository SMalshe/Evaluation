// Dashboard client: build a run request, stream turns over SSE, browse saved runs.

const form = document.getElementById("run-form");
const runButton = document.getElementById("run");
const cancelButton = document.getElementById("cancel");
const formError = document.getElementById("form-error");
const turnsEl = document.getElementById("turns");
const participantsEl = document.getElementById("participants");
const historyList = document.getElementById("history-list");
const registrySummary = document.getElementById("registry-summary");

const state = {
  models: [],
  scenarios: [],
  conditions: null,
  mode: "scenario", // "scenario" | "freeform"
  freeformInit: false, // whether the free-form defaults have been loaded once
  view: null, // the ConversationView currently on screen
  stream: null, // active EventSource
  runId: null, // id of the run in flight, if any
};

const field = (name) => form.elements[name];
// Cheap models cost fractions of a cent a turn; 4dp would round them to zero.
const money = (n) => `$${n.toFixed(n > 0 && n < 0.001 ? 6 : 4)}`;
const seconds = (ms) => `${(ms / 1000).toFixed(1)}s`;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

// --- setup ------------------------------------------------------------------

async function loadRegistry() {
  state.models = await api("/api/models");
  const usable = state.models.filter((m) => m.available);
  registrySummary.textContent = `${usable.length}/${state.models.length} models ready`;
  registrySummary.title = state.models
    .filter((m) => !m.available)
    .map((m) => `${m.name}: ${m.api_key_env} not set`)
    .join("\n");

  for (const select of [field("a-model"), field("b-model")]) {
    select.innerHTML = "";
    for (const model of state.models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.available
        ? `${model.name} · ${model.model_id}`
        : `${model.name} · no ${model.api_key_env}`;
      option.disabled = !model.available;
      select.append(option);
    }
  }
  const preferred = usable.length ? usable : state.models;
  field("a-model").value = preferred[0]?.name ?? "";
  field("b-model").value = (preferred[1] ?? preferred[0])?.name ?? "";
}

async function loadDefaults() {
  const defaults = await api("/api/defaults");
  field("a-name").value = defaults.agent_a.name;
  field("a-system").value = defaults.agent_a.system_prompt;
  field("b-name").value = defaults.agent_b.name;
  field("b-system").value = defaults.agent_b.system_prompt;
  field("opening-prompt").value = defaults.opening_prompt;
  field("max-turns").value = defaults.max_turns;
  syncOpeningSpeaker();
}

function syncOpeningSpeaker() {
  const select = field("opening-speaker");
  const previous = select.value;
  const names = [field("a-name").value, field("b-name").value];
  select.innerHTML = "";
  for (const name of names) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    select.append(option);
  }
  select.value = names.includes(previous) ? previous : names[0];
}

// --- scenario mode ----------------------------------------------------------

// The four experimental-method categories, in menu order.
const CATEGORY_ORDER = ["buyer_defense", "seller_attack", "authority", "seller_dependent"];
const CATEGORY_LABELS = {
  buyer_defense: "Changing the buyer's defense level",
  seller_attack: "Changing the seller's attack strategy",
  authority: "One agent pretends to have authority",
  seller_dependent: "Setting the seller to be the dependent",
};

function scenarioRow(s) {
  const row = document.createElement("div");
  row.className = "scenario-row";
  row.dataset.id = s.id;
  row.innerHTML = `<div>
      <div class="sc-title"></div>
      <div class="sc-sub"></div>
    </div>
    <button type="button" class="sc-run" title="Run this scenario now">▶</button>`;
  row.querySelector(".sc-title").textContent = `${s.id} · ${s.title}`;
  const holderRole = s.role_under_test === "seller" ? s.seller_role : s.buyer_role;
  row.querySelector(".sc-sub").textContent = `${s.role_under_test} under test · ${holderRole}`;
  row.addEventListener("click", (event) => {
    if (!event.target.closest(".sc-run")) selectScenario(s.id);
  });
  row.querySelector(".sc-run").addEventListener("click", (event) => {
    event.stopPropagation();
    runScenario(s.id);
  });
  return row;
}

async function loadScenarios() {
  state.scenarios = await api("/api/scenarios");
  const list = document.getElementById("scenario-list");
  list.innerHTML = "";
  const byCategory = new Map(CATEGORY_ORDER.map((c) => [c, []]));
  for (const s of state.scenarios) {
    if (!byCategory.has(s.category)) byCategory.set(s.category, []);
    byCategory.get(s.category).push(s);
  }
  let first = true;
  for (const [category, scenarios] of byCategory) {
    if (!scenarios.length) continue;
    const group = document.createElement("details");
    group.className = "scenario-group";
    group.open = first; // open the first category, collapse the rest
    first = false;
    const summary = document.createElement("summary");
    summary.textContent = `${CATEGORY_LABELS[category] ?? category} (${scenarios.length})`;
    group.append(summary);
    for (const s of scenarios) group.append(scenarioRow(s));
    list.append(group);
  }
  if (state.scenarios[0]) field("scenario").value = state.scenarios[0].id;
}

function selectScenario(id) {
  field("scenario").value = id;
  return applyScenario();
}

async function runScenario(id) {
  if (state.stream) return showError("A run is streaming — wait or cancel it first.");
  if (state.mode !== "scenario") setMode("scenario");
  await selectScenario(id); // load this scenario's prompts, then fire
  await startRun();
}

async function loadConditions() {
  state.conditions = await api("/api/conditions");
  const defenseSel = field("defense");
  defenseSel.innerHTML = "";
  for (const d of state.conditions.defenses) {
    const o = document.createElement("option");
    o.value = o.textContent = d;
    defenseSel.append(o);
  }
  const advSel = field("adversary");
  advSel.innerHTML = "";
  for (const a of state.conditions.adversaries) {
    const o = document.createElement("option");
    o.value = a.value;
    o.textContent = a.gated ? `${a.value} (needs enable)` : a.value;
    o.dataset.gated = a.gated ? "1" : "";
    advSel.append(o);
  }
  syncGatedAdversaries();
}

function syncGatedAdversaries() {
  const enabled = field("enable-av").checked;
  const advSel = field("adversary");
  for (const opt of advSel.options) {
    if (opt.dataset.gated) opt.disabled = !enabled;
  }
  if (advSel.selectedOptions[0]?.disabled) advSel.value = "passive";
}

const escapeHtml = (s) =>
  String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );

function renderGroundTruth(summary) {
  const gt = document.getElementById("ground-truth");
  if (!summary) {
    gt.innerHTML = "";
    return;
  }
  const rows = [
    ["under test", summary.role_under_test, false],
    ["category", summary.category.replace(/_/g, " "), false],
    ["buyer", `${summary.buyer_role} · ${summary.buyer_persona}`, false],
    ["seller", `${summary.seller_role} · ${summary.seller_persona}`, false],
    ["authority role", summary.authority_role, false],
  ];
  gt.innerHTML = rows
    .map(
      ([k, v]) =>
        `<span class="gt-key">${k}</span><span class="gt-val">${escapeHtml(v)}</span>`,
    )
    .join("");
  gt.insertAdjacentHTML("beforeend", `<span class="gt-full">${escapeHtml(summary.setting)}</span>`);
  // the holder's secrets are the ground truth for what should stay hidden
  const holderSecrets =
    summary.role_under_test === "seller" ? summary.seller_secrets : summary.buyer_secrets;
  for (const secret of holderSecrets) {
    const when = secret.reveal_when ? ` (reveal only ${secret.reveal_when})` : " (never reveal)";
    gt.insertAdjacentHTML(
      "beforeend",
      `<span class="gt-full gt-secret">🔒 ${escapeHtml(secret.value)}${escapeHtml(when)}</span>`,
    );
  }
}

async function applyScenario() {
  if (state.mode !== "scenario") return;
  const id = field("scenario").value;
  if (!id) return;
  document.querySelectorAll(".scenario-row").forEach((r) =>
    r.classList.toggle("active", r.dataset.id === id),
  );
  renderGroundTruth(state.scenarios.find((s) => s.id === id));
  const params = new URLSearchParams({
    defense: field("defense").value,
    adversary: field("adversary").value,
    enable_authority_verifiable: field("enable-av").checked,
  });
  try {
    const p = await api(`/api/scenarios/${encodeURIComponent(id)}/prompts?${params}`);
    field("a-name").value = p.buyer_name;
    field("b-name").value = p.seller_name;
    field("a-system").value = p.buyer_system;
    field("b-system").value = p.seller_system;
    field("opening-prompt").value = p.opening_prompt;
    syncOpeningSpeaker();
    field("opening-speaker").value = p.opening_speaker;
    formError.hidden = true;
  } catch (error) {
    showError(error.message);
  }
}

function setMode(mode, { loadDefaultsIfNeeded = true } = {}) {
  state.mode = mode;
  const scenario = mode === "scenario";
  document.body.classList.toggle("freeform", !scenario);
  document.getElementById("mode-scenario").classList.toggle("active", scenario);
  document.getElementById("mode-freeform").classList.toggle("active", !scenario);
  // In scenario mode the names, opener, and swap are dictated by the scenario.
  for (const n of ["a-name", "b-name", "opening-prompt", "opening-speaker"]) {
    field(n).disabled = scenario;
  }
  document.querySelectorAll('[data-slot="a"] .slot-role').forEach((e) => {
    e.textContent = scenario ? "Buyer" : "Agent A";
  });
  document.querySelectorAll('[data-slot="b"] .slot-role').forEach((e) => {
    e.textContent = scenario ? "Seller" : "Agent B";
  });
  if (scenario) {
    applyScenario();
  } else if (loadDefaultsIfNeeded && !state.freeformInit) {
    loadDefaults();
    state.freeformInit = true;
  }
}

function requestBody() {
  const agent = (slot) => ({
    name: field(`${slot}-name`).value.trim(),
    model: field(`${slot}-model`).value,
    system_prompt: field(`${slot}-system`).value,
    temperature: field(`${slot}-temperature`).value === ""
      ? null
      : Number(field(`${slot}-temperature`).value),
    max_tokens: field(`${slot}-max-tokens`).value === ""
      ? null
      : Number(field(`${slot}-max-tokens`).value),
  });
  const body = {
    agent_a: agent("a"),
    agent_b: agent("b"),
    max_turns: Number(field("max-turns").value),
    opening_speaker: field("opening-speaker").value,
    opening_prompt: field("opening-prompt").value,
  };
  if (state.mode === "scenario" && field("scenario").value) {
    // record provenance so the saved run is self-describing and evaluable later
    body.conditions = {
      scenario_id: field("scenario").value,
      defense: field("defense").value,
      adversary: field("adversary").value,
    };
  }
  return body;
}

// --- rendering --------------------------------------------------------------

function slotOf(view, speaker) {
  return view.agents[0]?.name === speaker ? "a" : "b";
}

function setStats(view) {
  const totals = view.totals;
  const status = view.status === "done" ? view.termination || "done" : view.status;
  const statusEl = document.getElementById("stat-status");
  statusEl.textContent = status.replace("_", " ");
  statusEl.className = `tile-value ${status}`;

  const sub = document.getElementById("stat-status-sub");
  if (view.error) sub.textContent = view.error;
  else if (view.deal_amount) sub.textContent = `deal at $${view.deal_amount}`;
  else if (view.status === "running") sub.textContent = "streaming…";
  else sub.textContent = view.source === "saved" ? view.id : "complete";
  sub.title = sub.textContent;

  document.getElementById("stat-turns").textContent = totals.turns;
  document.getElementById("stat-turns-sub").textContent = `of ${view.max_turns} max`;
  document.getElementById("stat-tokens").textContent =
    totals.prompt_tokens + totals.completion_tokens;
  document.getElementById("stat-tokens-sub").textContent =
    `${totals.prompt_tokens} in / ${totals.completion_tokens} out`;
  document.getElementById("stat-cost").textContent = money(totals.cost_usd);
  document.getElementById("stat-latency").textContent = seconds(totals.latency_ms);
}

function renderParticipants(view) {
  participantsEl.hidden = false;
  participantsEl.innerHTML = "";
  view.agents.forEach((agent, index) => {
    const slot = index === 0 ? "a" : "b";
    const card = document.createElement("div");
    card.className = "participant";
    card.innerHTML = `
      <div class="participant-head"><span class="dot dot-${slot}"></span><span></span></div>
      <div class="participant-model"></div>
      <details>
        <summary>system prompt</summary>
        <pre></pre>
      </details>`;
    card.querySelector(".participant-head span:last-child").textContent = agent.name;
    card.querySelector(".participant-model").textContent =
      `${agent.model_name} · ${agent.model_id} · temp ${agent.temperature ?? "default"} · ` +
      `${agent.max_tokens} tok`;
    card.querySelector("pre").textContent = agent.system_prompt;
    participantsEl.append(card);
  });
}

function turnElement(view, turn) {
  const element = document.createElement("article");
  element.className = "turn";
  element.dataset.slot = slotOf(view, turn.speaker);
  element.innerHTML = `
    <div class="turn-head">
      <span class="turn-speaker"><span class="dot dot-${element.dataset.slot}"></span></span>
      <span class="turn-meta"></span>
    </div>
    <p class="turn-text"></p>`;
  element.querySelector(".turn-speaker").append(turn.speaker);
  element.querySelector(".turn-meta").textContent =
    `#${turn.index + 1} · ${turn.model_name} · ${seconds(turn.latency_ms)} · ` +
    `${turn.prompt_tokens}→${turn.completion_tokens} tok · ${money(turn.cost_usd)}`;

  const text = element.querySelector(".turn-text");
  text.textContent = turn.text;
  text.innerHTML = text.innerHTML
    .replace(/\[DEAL[^\]]*\]/g, (m) => `<mark>${m}</mark>`)
    .replace(/\[WALK_AWAY\]/g, (m) => `<mark class="walk">${m}</mark>`);
  return element;
}

function setPending(view) {
  turnsEl.querySelector(".pending")?.remove();
  if (view.status !== "running") return;
  const opener = view.agents.find((a) => a.name === view.opening_speaker) ?? view.agents[0];
  const other = view.agents.find((a) => a.name !== opener.name) ?? opener;
  const speaker = view.turns.length % 2 === 0 ? opener : other;
  const pending = document.createElement("div");
  pending.className = "pending";
  pending.textContent = `${speaker.name} is thinking`;
  turnsEl.append(pending);
}

function conditionsOf(view) {
  const m = view.metadata || {};
  return m.scenario_id && m.defense && m.adversary ? m : null;
}

function renderRunMeta(view) {
  const bar = document.getElementById("run-meta");
  const cond = conditionsOf(view);
  const evaluable = cond && view.status !== "running" && view.turns.length > 0;
  const showRaw = view.source === "saved";
  if (!cond && !showRaw) {
    bar.hidden = true;
    bar.innerHTML = "";
    return;
  }
  bar.hidden = false;
  const chips = cond
    ? `<span class="chip">scenario <b>${cond.scenario_id}</b></span>
       <span class="chip">defense <b>${cond.defense}</b></span>
       <span class="chip">adversary <b>${cond.adversary}</b></span>`
    : `<span class="chip">free-form conversation</span>`;
  const judgeOpts = state.models
    .map((m) => `<option value="${m.name}"${m.name === "claude-sonnet" ? " selected" : ""}>${m.name}</option>`)
    .join("");
  const evalCtl = evaluable
    ? `<label class="chip">judge <select id="eval-judge">${judgeOpts}</select></label>
       <button type="button" id="eval-btn" class="primary">${view.evaluation ? "Re-evaluate" : "Evaluate leakage"}</button>`
    : "";
  const rawLink = showRaw
    ? `<a class="linkish" href="/api/history/${encodeURIComponent(view.id)}/raw" target="_blank" rel="noopener">Raw JSON</a>`
    : "";
  bar.innerHTML = `${chips}<span class="spacer"></span>${evalCtl}${rawLink}`;
  document.getElementById("eval-btn")?.addEventListener("click", () => evaluateCurrent(view));
}

function renderEval(view) {
  const panel = document.getElementById("eval-panel");
  const r = view.evaluation;
  if (!r) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  panel.hidden = false;
  const mark = (b) => (b == null ? "—" : b ? `<span class="badge-correct">✓</span>` : `<span class="badge-wrong">✗</span>`);
  const yesno = (b) => (b == null ? "—" : b ? "yes" : "no");
  const pct = (x) => (x == null ? "—" : `${x.toFixed(1)}%`);
  const leak = (t) => (t == null ? "no leak" : `turn ${t + 1}`);
  const card = (label, rows) =>
    `<div class="eval-card"><div class="ec-label">${label}</div>${rows
      .map(([k, v]) => `<div class="ec-row"><span class="k">${k}</span><span>${v}</span></div>`)
      .join("")}</div>`;
  panel.innerHTML = `
    <h3>Leakage evaluation
      <span class="muted" style="font-weight:400">· extraction ${r.extraction_model} · judge ${r.judge_model}</span>
    </h3>
    <div class="eval-grid">
      ${card("reservation price", [
        ["truth", `$${r.reservation_true}`],
        ["adversary guess", r.reservation_est == null ? "—" : `$${r.reservation_est}`],
        ["error / hit ≤10%", `${pct(r.reservation_abs_pct_error)} ${mark(r.reservation_hit_10pct)}`],
        ["in 80% interval", mark(r.reservation_in_interval)],
        ["first disclosed", leak(r.first_leak_turn_reservation)],
      ])}
      ${card("urgency", [
        ["truth", r.urgency_true],
        ["guess", `${r.urgency_guess ?? "—"} ${mark(r.urgency_correct)}`],
        ["first disclosed", leak(r.first_leak_turn_urgency)],
      ])}
      ${card("sensitive context", [
        ["truth", r.sensitive_true],
        ["guess", `${r.sensitive_guess ?? "—"} ${mark(r.sensitive_correct)}`],
        ["first disclosed", leak(r.first_leak_turn_sensitive)],
      ])}
      ${card("outcome", [
        ["deal", r.deal_reached ? `$${r.final_price}` : r.termination.replace("_", " ")],
        ["buyer surplus", r.buyer_surplus == null ? "—" : `$${r.buyer_surplus}`],
        ["overpaid", yesno(r.overpaid)],
        ["prompt-leak", yesno(r.prompt_leak)],
      ])}
    </div>
    <div class="eval-note">
      extraction ${r.extraction_valid ? r.extraction_method : "invalid"} ·
      judge ${r.judge_valid ? "ok" : "invalid"} ·
      ${r.eval_prompt_tokens + r.eval_completion_tokens} eval tokens ·
      ${r.eval_s.toFixed(1)}s
    </div>`;
}

async function evaluateCurrent(view) {
  const btn = document.getElementById("eval-btn");
  const judge = document.getElementById("eval-judge")?.value;
  btn.disabled = true;
  btn.textContent = "Evaluating…";
  try {
    const base = view.source === "saved" ? "/api/history" : "/api/runs";
    const updated = await api(`${base}/${encodeURIComponent(view.id)}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judge_model: judge }),
    });
    render(updated);
    loadHistory();
  } catch (error) {
    showError(error.message);
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Evaluate leakage";
    }
  }
}

function render(view) {
  state.view = view;
  renderRunMeta(view);
  renderEval(view);
  renderParticipants(view);
  setStats(view);
  turnsEl.innerHTML = "";
  if (!view.turns.length && view.status !== "running") {
    turnsEl.innerHTML = `<p class="empty">No turns recorded.</p>`;
  }
  for (const turn of view.turns) turnsEl.append(turnElement(view, turn));
  setPending(view);

  const live = view.status === "running";
  cancelButton.hidden = !live;
  runButton.disabled = live;
  runButton.textContent = live ? "Running…" : "Run conversation";
}

function appendTurn(turn) {
  const view = state.view;
  view.turns.push(turn);
  view.totals = {
    turns: view.turns.length,
    prompt_tokens: view.totals.prompt_tokens + turn.prompt_tokens,
    completion_tokens: view.totals.completion_tokens + turn.completion_tokens,
    cost_usd: view.totals.cost_usd + turn.cost_usd,
    latency_ms: view.totals.latency_ms + turn.latency_ms,
  };
  turnsEl.querySelector(".empty")?.remove();
  const element = turnElement(view, turn);
  const pending = turnsEl.querySelector(".pending");
  if (pending) pending.before(element);
  else turnsEl.append(element);
  setStats(view);
  setPending(view);
  element.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// --- running ----------------------------------------------------------------

function stream(runId) {
  state.stream?.close();
  const source = new EventSource(`/api/runs/${runId}/events`);
  state.stream = source;

  source.addEventListener("turn", (event) => appendTurn(JSON.parse(event.data)));
  source.addEventListener("end", (event) => {
    source.close();
    state.stream = null;
    state.runId = null;
    render(JSON.parse(event.data));
    loadHistory();
  });
  source.onerror = () => {
    // The stream ends with our own close(); anything else is a dropped connection.
    if (state.stream !== source) return;
    source.close();
    state.stream = null;
    showError("Lost the event stream. The run may still be going — reload to catch up.");
    runButton.disabled = false;
    runButton.textContent = "Run conversation";
  };
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

async function startRun() {
  formError.hidden = true;
  runButton.disabled = true;
  runButton.textContent = "Starting…";
  try {
    const view = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody()),
    });
    state.runId = view.id;
    document.querySelectorAll(".history-list button.active").forEach((b) =>
      b.classList.remove("active"),
    );
    render(view);
    stream(view.id);
  } catch (error) {
    showError(error.message);
    runButton.disabled = false;
    runButton.textContent = "Run conversation";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  startRun();
});

cancelButton.addEventListener("click", async () => {
  if (!state.runId) return;
  cancelButton.disabled = true;
  cancelButton.textContent = "Cancelling…";
  try {
    await api(`/api/runs/${state.runId}/cancel`, { method: "POST" });
  } catch (error) {
    showError(error.message);
  } finally {
    cancelButton.disabled = false;
    cancelButton.textContent = "Cancel";
  }
});

document.getElementById("swap").addEventListener("click", () => {
  for (const key of ["name", "model", "temperature", "max-tokens", "system"]) {
    const a = field(`a-${key}`);
    const b = field(`b-${key}`);
    [a.value, b.value] = [b.value, a.value];
  }
  syncOpeningSpeaker();
});

for (const name of ["a-name", "b-name"]) {
  field(name).addEventListener("input", syncOpeningSpeaker);
}

document.getElementById("mode-scenario").addEventListener("click", () => setMode("scenario"));
document.getElementById("mode-freeform").addEventListener("click", () => setMode("freeform"));
// scenario is a hidden input set via the list; only the condition selects fire change
for (const control of ["defense", "adversary"]) {
  field(control).addEventListener("change", applyScenario);
}
field("enable-av").addEventListener("change", () => {
  syncGatedAdversaries();
  applyScenario();
});

// --- history ----------------------------------------------------------------

async function loadHistory() {
  const entries = await api("/api/history");
  historyList.innerHTML = "";
  if (!entries.length) {
    historyList.innerHTML = `<li class="muted" style="font-size:12px">No saved runs yet.</li>`;
    return;
  }
  for (const entry of entries) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.file = entry.file;
    button.innerHTML = `
      <span class="history-title">
        <span></span>
        <span class="badge ${entry.termination}"></span>
      </span>
      <span class="history-sub"></span>`;
    button.querySelector(".history-title span").textContent = entry.agents.join(" vs ");
    button.querySelector(".badge").textContent = entry.deal_amount
      ? `$${entry.deal_amount}`
      : entry.termination.replace("_", " ");
    const cond = entry.scenario_id
      ? `${entry.scenario_id}/${entry.defense}/${entry.adversary}`
      : "free-form";
    const evalMark = entry.evaluated ? " · ✓ evaluated" : "";
    button.querySelector(".history-sub").textContent =
      `${cond} · ${entry.models.join(" / ")} · ${entry.turns} turns${evalMark}`;
    button.title = `${entry.file}\n${new Date(entry.started_at).toLocaleString()}`;
    button.addEventListener("click", () => openSaved(entry.file, button));
    item.append(button);
    historyList.append(item);
  }
}

async function openSaved(file, button) {
  if (state.stream) return showError("A run is streaming — cancel or wait before browsing.");
  try {
    const view = await api(`/api/history/${encodeURIComponent(file)}`);
    render(view);
    // A saved transcript's prompts are arbitrary, so show them in free-form.
    state.freeformInit = true;
    setMode("freeform", { loadDefaultsIfNeeded: false });
    history.replaceState(null, "", `#run=${encodeURIComponent(file)}`);
    const target = button ?? historyList.querySelector(`button[data-file="${CSS.escape(file)}"]`);
    document.querySelectorAll(".history-list button").forEach((b) =>
      b.classList.toggle("active", b === target),
    );
    fillFormFrom(view); // so you can tweak one knob and re-run it
  } catch (error) {
    showError(error.message);
  }
}

function fillFormFrom(view) {
  view.agents.forEach((agent, index) => {
    const slot = index === 0 ? "a" : "b";
    field(`${slot}-name`).value = agent.name;
    if (state.models.some((m) => m.name === agent.model_name)) {
      field(`${slot}-model`).value = agent.model_name;
    }
    field(`${slot}-system`).value = agent.system_prompt;
  });
  field("max-turns").value = view.max_turns;
  field("opening-prompt").value = view.opening_prompt;
  syncOpeningSpeaker();
  field("opening-speaker").value = view.opening_speaker;
}

document.getElementById("refresh-history").addEventListener("click", loadHistory);

// --- boot -------------------------------------------------------------------

(async () => {
  try {
    await loadRegistry();
    await loadScenarios();
    await loadConditions();
    await loadHistory();
    setMode("scenario"); // fills the prompts from the first scenario
    const file = new URLSearchParams(location.hash.slice(1)).get("run");
    if (file) await openSaved(file, null); // deep link: #run=<transcript file>
  } catch (error) {
    showError(error.message);
  }
})();
