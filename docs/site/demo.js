// Cartograph demo page: talks to a live backend through an SSH tunnel, or replays recorded real responses.
const FRESH = "2026-10-01T04:00:00+00:00"; // 2 h after the feature snapshot watermark
const STALE = "2026-10-01T10:00:00+00:00"; // 8 h after: past the 6 h freshness limit
const PRESETS = [
  ["Quiet keyboard + mouse, $100", "Find a quiet wireless keyboard and mouse bundle under $100. Only include available products and explain the tradeoffs."],
  ["Webcam + microphone, $150", "Find a webcam and microphone bundle under $150."],
  ["Headphones + mic, $60", "Find a gaming headphones and microphone bundle under $60."],
  ["Impossible: $5", "Find a wireless keyboard and mouse bundle under $5."],
  ["Vague request", "I want something nice for my desk."],
];
const DECISIONS = {
  verified_bundle: ["good", "✓", "Verified bundle"],
  abstain_with_reason: ["info", "⊘", "Can't be satisfied"],
  withheld_unverified: ["warn", "!", "Answer withheld"],
  stale_features: ["warn", "⏱", "Features too old"],
  features_unavailable: ["bad", "✕", "Feature store unavailable"],
  invalid_request: ["neutral", "?", "Needs more detail"],
};
const ERRORS = {
  unknown_product: "Every product exists in the retrieved evidence",
  duplicate_product: "No duplicate products",
  not_one_per_category: "Exactly one product per required category",
  missing_category: "Every required category covered",
  out_of_stock: "All products in stock (live features)",
  incorrect_total: "Stated total equals the sum of live prices",
  over_budget: "Total within budget",
  invalid_schema: "Answer is a well-formed JSON decision",
};
const STAGES = [["validate", "#7b7a76"], ["retrieve", "#2a78d6"], ["features", "#1baf7a"], ["solve", "#eda100"], ["generate_and_verify", "#eb6834"]];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (cents) => (cents == null ? "—" : "$" + (cents / 100).toFixed(2));
let live = false, recorded = null, busy = false;

function setConn(state, text) {
  const el = $("conn");
  el.className = "conn " + state;
  $("conn-text").textContent = text;
}

async function connect() {
  const url = $("backend").value.replace(/\/$/, "");
  try { localStorage.setItem("cartograph-backend", url); } catch (e) { /* storage unavailable */ }
  setConn("", "Checking " + url + " …");
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    const res = await fetch(url + "/v1/info", { signal: ctrl.signal });
    clearTimeout(timer);
    const info = await res.json();
    live = true;
    setConn("live", `Live · release ${info.release} · features as of ${info.feature_watermark?.as_of?.slice(0, 16).replace("T", " ")} UTC`);
    $("send").disabled = false;
  } catch (e) {
    live = false;
    setConn("off", "Backend offline: showing recorded real responses");
    $("send").disabled = true;
  }
}

async function send() {
  if (!live || busy) return;
  busy = true;
  $("send").disabled = true;
  $("result").innerHTML = `<div class="card"><span class="spinner"></span> Retrieving, fetching live features, generating, verifying…</div>`;
  const url = $("backend").value.replace(/\/$/, "");
  const started = performance.now();
  try {
    const res = await fetch(url + "/v1/bundle", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: $("request").value, now: $("clock").value }),
    });
    const trace = await res.json();
    render(trace, res.status, { source: "live", clientMs: performance.now() - started });
  } catch (e) {
    $("result").innerHTML = `<div class="callout bad">Request failed: ${esc(e.message)}. Is the tunnel still open?</div>`;
    connect();
  } finally {
    busy = false;
    $("send").disabled = !live;
  }
}

function checksHtml(score, decision) {
  if (!score) return "";
  if (decision === "abstain_with_reason") {
    return `<ul class="checks"><li>✓ Answer is a well-formed JSON decision</li><li>✓ No products recommended, so nothing unverified is shown</li></ul>`;
  }
  const failed = new Set(score.errors || []);
  return `<ul class="checks">${Object.entries(ERRORS).filter(([k]) => k !== "not_one_per_category" || failed.has(k) || score.valid_bundle)
    .map(([k, label]) => `<li>${failed.has(k) ? '<span style="color:var(--bad)">✕</span>' : '<span style="color:var(--good)">✓</span>'} ${esc(label)}</li>`).join("")}</ul>`;
}

function render(t, status, meta) {
  const decision = t.decision || "invalid_request";
  const [tone, icon, label] = DECISIONS[decision] || ["neutral", "·", decision];
  const evidence = t.evidence || [];
  const byId = Object.fromEntries(evidence.map((e) => [e.product_id, e]));
  const parsed = t.parsed || {};
  const answer = t.answer || {};
  const finalScore = (t.attempts || []).slice(-1)[0]?.score;
  const oracle = t.oracle || null;
  let html = `<div class="card"><div class="row" style="justify-content:space-between">
    <span class="badge ${tone}">${icon} ${esc(label)}</span>
    <span class="small muted">${meta.source === "live" ? "Live response" : "Recorded real response"} · HTTP ${esc(status)}${t.timings_ms?.total ? ` · ${Math.round(t.timings_ms.total)} ms server` : ""}</span></div>
    <p style="margin:12px 0 4px"><strong>Request:</strong> ${esc(t.request)}</p>`;
  if (parsed.required_categories) {
    html += `<p class="small muted" style="margin:0">Understood as: ${parsed.required_categories.map((c) => `<code>${esc(c)}</code>`).join(" + ")}
      · budget ${money(parsed.budget_cents)}${parsed.adjective ? ` · “${esc(parsed.adjective)}”` : ""} · in stock only</p>`;
  }
  html += `</div>`;

  if (decision === "verified_bundle" && answer.product_ids) {
    const total = answer.total_cents;
    const pct = Math.min(100, (total / parsed.budget_cents) * 100);
    html += `<div class="card"><h3 style="margin-top:0">Recommended bundle</h3><div class="products">${answer.product_ids.map((id) => {
      const e = byId[id] || {};
      return `<div class="product"><div><div class="title">${esc(e.title || id)}</div>
        <div class="meta">${esc(e.category)} · <span class="mono">${esc(id)}</span> · ${e.in_stock ? "in stock" : "out of stock"}</div></div>
        <div class="price">${money(e.price_cents)}</div></div>`;
    }).join("")}</div>
    <div class="budget"><div class="row small" style="justify-content:space-between"><span>Total ${money(total)}</span><span class="muted">Budget ${money(parsed.budget_cents)}</span></div>
    <div class="bar"><div class="fill" style="width:${pct}%"></div></div></div>
    ${answer.explanation ? `<p style="margin:12px 0 0"><strong>Model's explanation:</strong> ${esc(answer.explanation)}</p>` : ""}</div>`;
  } else if (decision === "abstain_with_reason") {
    html += `<div class="card"><h3 style="margin-top:0">Why no bundle</h3><p>${esc(answer.explanation || "The model found no valid bundle in the evidence.")}</p>
      ${oracle && oracle.feasible === false ? `<div class="callout">The exact solver agrees: ${esc((oracle.reasons || []).join(", ").replace(/_/g, " "))}.</div>` : ""}
      ${oracle && oracle.feasible === true ? `<div class="callout warn"><strong>Known failure mode:</strong> the model refused, but the exact solver found a valid bundle
        (${(oracle.bundle?.product_ids || []).map((id) => esc(byId[id]?.title?.slice(0, 40) || id)).join(" + ")}, ${money(oracle.bundle?.total_cents)}).
        This is a false abstention; the system errs toward saying no rather than showing something wrong.</div>` : ""}</div>`;
  } else if (decision === "withheld_unverified") {
    html += `<div class="card"><h3 style="margin-top:0">Answer withheld</h3>
      <p>The model proposed a bundle ${(t.attempts || []).length} time(s), and each failed verification
      (${esc([...new Set((t.attempts || []).flatMap((a) => a.score?.errors || []))].join(", ").replace(/_/g, " "))}).
      An unverified bundle is never shown to the user.</p>
      ${oracle?.feasible ? `<div class="callout">A valid bundle did exist (the exact solver found one), so this counts as a miss, but a safe one.</div>` : ""}</div>`;
  } else if (decision === "stale_features") {
    html += `<div class="card"><div class="callout warn">The feature snapshot is ${Math.round((t.feature_snapshot_age_s || 0) / 3600)} h old, past the 6 h limit.
      Cartograph makes no price or stock claims on stale data. It retrieved ${evidence.length} candidates but didn't recommend any.</div></div>`;
  } else if (decision === "features_unavailable") {
    html += `<div class="card"><div class="callout bad">The online feature store (Redis) is unreachable. Cartograph fails closed:
      HTTP 503, and no prices or stock are invented.<br><span class="mono small">${esc(t.error)}</span></div></div>`;
  } else if (decision === "invalid_request") {
    html += `<div class="card"><div class="callout">${esc(t.error || "The request couldn't be parsed.")} Try naming product types and a budget, e.g. “keyboard and mouse under $80”.</div></div>`;
  }

  if (finalScore) {
    html += `<div class="grid two"><div class="card"><h3 style="margin-top:0">Verifier checks</h3>${checksHtml(finalScore, decision)}
      <p class="small muted" style="margin:6px 0 0">${(t.attempts || []).length} model call(s)${(t.attempts || []).length > 1 ? "; the first answer failed and the verifier's errors were fed back for one retry" : ""}.
      ${t.correct_vs_oracle === true ? "Decision matches the exact solver." : t.correct_vs_oracle === false ? "Decision does <strong>not</strong> match the exact solver." : ""}</p></div>`;
    html += `<div class="card"><h3 style="margin-top:0">Ground truth (exact solver)</h3>${oracle ? (oracle.feasible
      ? `<p>A valid bundle exists. Solver optimum: ${(oracle.bundle?.product_ids || []).map((id) => `<code>${esc(id)}</code>`).join(" + ")} at ${money(oracle.bundle?.total_cents)}.</p>`
      : `<p>No valid bundle: ${esc((oracle.reasons || []).join(", ").replace(/_/g, " "))}.</p>`) : "<p>—</p>"}
      <p class="small muted" style="margin:0">The solver checks every combination of the retrieved, in-stock candidates. The model doesn't see its answer.</p></div></div>`;
  }

  const timings = t.timings_ms || {};
  const total = STAGES.reduce((s, [k]) => s + (timings[k] || 0), 0);
  if (total > 0) {
    html += `<div class="card"><h3 style="margin-top:0">Where the time went</h3><div class="timing">${STAGES.filter(([k]) => timings[k] > 0)
      .map(([k, color]) => `<span title="${k}: ${timings[k].toFixed(1)} ms" style="width:${Math.max(0.6, (timings[k] / total) * 100)}%;background:${color}"></span>`).join("")}</div>
      <div class="legend">${STAGES.filter(([k]) => timings[k] != null).map(([k, color]) => `<span><i style="background:${color}"></i>${k.replace(/_/g, " ")} ${timings[k].toFixed(1)} ms</span>`).join("")}</div></div>`;
  }

  if (evidence.length) {
    const chosen = new Set(decision === "verified_bundle" ? answer.product_ids || [] : []);
    html += `<div class="card"><h3 style="margin-top:0">Evidence the model saw (${evidence.length} candidates, live features)</h3><div class="table-wrap"><table>
      <thead><tr><th>Product</th><th>Category</th><th>Price</th><th>Stock</th><th>Match</th></tr></thead><tbody>${evidence.map((e) => `
      <tr${chosen.has(e.product_id) ? ' style="background:var(--good-soft)"' : ""}><td>${chosen.has(e.product_id) ? "★ " : ""}${esc((e.title || "").slice(0, 70))}</td>
      <td>${esc(e.category)}</td><td>${money(e.price_cents)}</td><td>${e.in_stock == null ? "—" : e.in_stock ? "in stock" : '<span style="color:var(--bad)">out</span>'}</td>
      <td class="mono">${e.retrieval_score ?? ""}</td></tr>`).join("")}</tbody></table></div></div>`;
  }

  html += `<div class="card"><h3 style="margin-top:0">Trace</h3><dl class="kv">
    <dt>request id</dt><dd>${esc(t.request_id || "—")}</dd><dt>release</dt><dd>${esc(t.release_id || "—")}</dd>
    <dt>model</dt><dd>${esc((t.model || "").split("/").slice(-1)[0] || "—")} (Qwen2.5-1.5B-Instruct + SFT LoRA)</dd>
    <dt>feature snapshot</dt><dd>${esc(t.feature_watermark?.export_tag || "—")} · as of ${esc(t.feature_watermark?.as_of || "—")}</dd>
    <dt>simulated now</dt><dd>${esc(t.now || "—")}</dd><dt>snapshot age</dt><dd>${t.feature_snapshot_age_s != null ? (t.feature_snapshot_age_s / 3600).toFixed(1) + " h" : "—"}</dd>
    </dl><details style="margin-top:10px"><summary>Raw JSON response</summary><pre>${esc(JSON.stringify(t, null, 2))}</pre></details></div>`;
  $("result").innerHTML = html;
}

async function loadRecorded() {
  try {
    recorded = await (await fetch("site/recorded.json")).json();
  } catch (e) { return; }
  $("recorded-meta").textContent = `Recorded ${recorded.recorded_at_utc.slice(0, 10)} from release ${recorded.backend.release}.`;
  $("recorded-list").innerHTML = recorded.examples.map((ex, i) => `<button class="chip" data-i="${i}">${esc(ex.label)}</button>`).join("");
  $("recorded-list").querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    $("recorded-list").querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    const ex = recorded.examples[+chip.dataset.i];
    render(ex.trace, ex.status, { source: "recorded" });
  }));
  const wanted = /#example=(\d+)/.exec(location.hash);
  if (wanted) $("recorded-list").querySelector(`.chip[data-i="${wanted[1]}"]`)?.click();
  else if (!live) $("recorded-list").querySelector(".chip")?.click();
}

function init() {
  try { const saved = localStorage.getItem("cartograph-backend"); if (saved) $("backend").value = saved; } catch (e) { /* ignore */ }
  $("presets").innerHTML = PRESETS.map(([label], i) => `<button class="chip" data-i="${i}">${esc(label)}</button>`).join("");
  $("presets").querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => { $("request").value = PRESETS[+chip.dataset.i][1]; }));
  $("clock").innerHTML = `<option value="${FRESH}">Fresh: 2 h after the feature snapshot</option><option value="${STALE}">Stale: 8 h after (over the 6 h limit)</option>`;
  $("connect").addEventListener("click", connect);
  $("send").addEventListener("click", send);
  $("request").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") send(); });
  // Shareable links: #example=N opens a recorded response; #ask sends the current request to the live backend.
  connect().then(loadRecorded).then(() => { if (location.hash === "#ask" && live) send(); });
}
document.addEventListener("DOMContentLoaded", init);
