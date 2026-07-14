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
  return {
    agent_a: agent("a"),
    agent_b: agent("b"),
    max_turns: Number(field("max-turns").value),
    opening_speaker: field("opening-speaker").value,
    opening_prompt: field("opening-prompt").value,
  };
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

function render(view) {
  state.view = view;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
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
    button.querySelector(".history-sub").textContent =
      `${entry.models.join(" / ")} · ${entry.turns} turns`;
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
    await loadDefaults();
    await loadHistory();
    const file = new URLSearchParams(location.hash.slice(1)).get("run");
    if (file) await openSaved(file, null); // deep link: #run=<transcript file>
  } catch (error) {
    showError(error.message);
  }
})();
