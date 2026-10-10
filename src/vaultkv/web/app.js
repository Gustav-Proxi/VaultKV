"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const state = {
  overview: null,
  traces: [],
  selected: null,
  trace: null,
  view: "workspace",
  editor: null,
  accessDirectory: null,
  busy: false,
};
const access = new URLSearchParams(location.hash.slice(1)).get("access");
if (access) {
  sessionStorage.setItem("vaultkv-access", access);
  history.replaceState(null, "", location.pathname);
}
function notice(text = "") {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}
async function api(path, body) {
  const response = await fetch(path, {
    headers: {
      "X-VaultKV-Token": sessionStorage.getItem("vaultkv-access") || "",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    method: body ? "POST" : "GET",
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401) {
      state.trace = null;
      renderDetail();
    }
    throw Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}
const elapsed = (value) =>
  value == null
    ? "—"
    : value < 1000
      ? `${Math.round(value)} ms`
      : `${(value / 1000).toFixed(2)} s`;
const num = (value) => (value == null ? "—" : Number(value).toFixed(4));
const parseNames = (value) => [
  ...new Set(
    value
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
  ),
];
const status = (value) =>
  `<span class="status ${esc(value)}">${esc(value)}</span>`;
const table = (heads, rows) =>
  `<div class="table-wrap"><table><thead><tr>${heads.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
function view(name) {
  state.view = name;
  document
    .querySelectorAll(".view")
    .forEach((el) => (el.hidden = el.id !== name));
  document
    .querySelectorAll("nav button")
    .forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  $("page-title").textContent = {
    workspace: "Workspace",
    requests: "Requests",
    knowledge: "Documents & access",
    caches: "Performance & safety",
    vectorlens: "VectorLens",
  }[name];
}
document
  .querySelectorAll("nav button")
  .forEach((el) => el.addEventListener("click", () => view(el.dataset.view)));
async function loadAccessDirectory() {
  const workspace = $("access-workspace").value.trim();
  state.accessDirectory = await api(
    "/api/access?" + new URLSearchParams({ tenant: workspace }),
  );
  renderAccessDirectory();
}
function renderAccessDirectory() {
  const directory = state.accessDirectory;
  if (!directory) return;
  $("access-users").innerHTML = directory.users.length
    ? directory.users
        .map(
          (user, index) =>
            `<div class="directory-item"><div><strong>${esc(user.user)}</strong><small>${user.enabled ? "Active" : "Disabled"}</small></div><button data-user="${index}">${user.enabled ? "Disable" : "Enable"}</button></div>`,
        )
        .join("")
    : '<div class="empty compact">No users configured.</div>';
  $("access-users")
    .querySelectorAll("[data-user]")
    .forEach((button) => {
      button.onclick = async () => {
        const user = directory.users[Number(button.dataset.user)];
        try {
          await api("/api/users", {
            tenant: directory.workspace,
            user: user.user,
            enabled: !user.enabled,
            expected_epoch: user.access_version,
          });
          notice(
            `${user.user} is now ${user.enabled ? "disabled" : "active"}.`,
          );
          await loadAccessDirectory();
        } catch (error) {
          notice(error.message);
        }
      };
    });
  $("access-groups").innerHTML = directory.groups.length
    ? directory.groups
        .map(
          (group, index) =>
            `<div class="directory-item"><div><strong>${esc(group.group)}</strong><small>${group.members.length ? esc(group.members.join(", ")) : "No members"} · ${group.enabled ? "Active" : "Disabled"}</small></div><button data-group="${index}">Edit</button></div>`,
        )
        .join("")
    : '<div class="empty compact">No groups configured.</div>';
  $("access-groups")
    .querySelectorAll("[data-group]")
    .forEach((button) => {
      button.onclick = () => {
        const group = directory.groups[Number(button.dataset.group)];
        $("group-name").value = group.group;
        $("group-name").readOnly = true;
        $("group-members").value = group.members.join(", ");
        $("group-enabled").checked = group.enabled;
        $("group-version").value = group.access_version;
        $("group-name").focus();
      };
    });
}
function clearGroupForm() {
  $("group-form").reset();
  $("group-name").readOnly = false;
  $("group-version").value = "0";
}
$("show-access").onclick = async () => {
  $("access-manager").hidden = false;
  $("import-form").hidden = true;
  $("access-workspace").value = $("tenant").value;
  try {
    await loadAccessDirectory();
  } catch (error) {
    notice(error.message);
  }
};
$("close-access").onclick = () => {
  $("access-manager").hidden = true;
  state.accessDirectory = null;
};
$("load-access").onclick = async () => {
  try {
    await loadAccessDirectory();
  } catch (error) {
    notice(error.message);
  }
};
$("user-form").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const user = $("new-user").value.trim();
    await api("/api/users", {
      tenant: $("access-workspace").value.trim(),
      user,
      enabled: true,
      expected_epoch: 1,
    });
    notice(`${user} was added.`);
    $("user-form").reset();
    await loadAccessDirectory();
  } catch (error) {
    notice(error.message);
  }
};
$("new-group").onclick = clearGroupForm;
$("group-form").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const group = $("group-name").value.trim();
    await api("/api/groups", {
      tenant: $("access-workspace").value.trim(),
      group,
      members: parseNames($("group-members").value),
      enabled: $("group-enabled").checked,
      expected_epoch: Number($("group-version").value),
    });
    notice(`${group} was saved.`);
    clearGroupForm();
    await loadAccessDirectory();
    state.documentKey = null;
    await refresh();
  } catch (error) {
    notice(error.message);
  }
};
$("show-import").onclick = () => {
  $("import-form").hidden = false;
  $("access-manager").hidden = true;
  $("import-workspace").value = $("tenant").value;
  $("import-users").value = $("principal").value;
  $("document-files").focus();
};
$("cancel-import").onclick = () => {
  $("import-form").hidden = true;
  $("import-form").reset();
};
$("import-form").onsubmit = async (event) => {
  event.preventDefault();
  notice();
  const selected = [...$("document-files").files];
  if (!selected.length) return notice("Choose at least one .md or .txt file.");
  if (selected.length > 20)
    return notice("Choose no more than 20 files at once.");
  $("import-submit").disabled = true;
  try {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const files = [];
    let totalBytes = 0;
    for (const file of selected) {
      if (!/\.(md|txt)$/i.test(file.name))
        throw Error(`${file.name} must be a .md or .txt file.`);
      totalBytes += file.size;
      if (file.size > 2_000_000 || totalBytes > 2_000_000)
        throw Error("Selected files must total no more than 2 MB.");
      files.push({
        name: file.name,
        text: decoder.decode(await file.arrayBuffer()),
      });
    }
    const principals = parseNames($("import-users").value);
    const groups = parseNames($("import-groups").value);
    if (!principals.length && !groups.length)
      throw Error(
        "Add at least one user or group that can use these documents.",
      );
    const result = await api("/api/import", {
      tenant: $("import-workspace").value,
      principals,
      groups,
      files,
    });
    notice(
      `${result.documents} ${result.documents === 1 ? "document" : "documents"} imported successfully.`,
    );
    $("import-form").hidden = true;
    $("import-form").reset();
    state.documentKey = null;
    await refresh();
  } catch (error) {
    notice(
      error instanceof TypeError
        ? "A selected file is not valid UTF-8 text."
        : error.message,
    );
  } finally {
    $("import-submit").disabled = false;
  }
};
function documentAccess(doc) {
  if (doc.scope === "public") return "Everyone";
  const values = [
    ...doc.principals,
    ...(doc.groups || []).map((group) => `${group} group`),
  ];
  return values.join(", ") || "Nobody";
}
function renderOverview() {
  const o = state.overview,
    c = o.request_counts,
    total = Object.values(c).reduce((a, b) => a + b, 0);
  const metrics = [
    [
      "Content version",
      `Version ${o.knowledge_epoch}`,
      `${o.documents.length} documents available`,
    ],
    [
      "Questions",
      total,
      `${c.emitted || 0} answered · ${c.blocked ? `${c.blocked} blocked` : "none blocked"}`,
    ],
    ["Typical answer time", elapsed(o.p50_ms), "Based on answered questions"],
    [
      "Saved search data",
      o.embedding_entries,
      "Used to find relevant sources faster",
    ],
    ["Answer protection", "On", "Access checked before every answer"],
  ];
  $("metrics").innerHTML = metrics
    .map(
      ([label, value, detail]) =>
        `<div class="metric"><label>${label}</label><div class="value">${value}</div><small>${detail}</small></div>`,
    )
    .join("");
  const documentKey = JSON.stringify(o.documents);
  if (documentKey !== state.documentKey) {
    state.documentKey = documentKey;
    $("documents").innerHTML = table(
      ["Document", "Workspace", "Content", "Who can use it", ""],
      o.documents.map(
        (d, i) =>
          `<tr><td>${esc(d.doc_id)}</td><td>${esc(d.tenant_ns)}</td><td>${d.chunks} ${d.chunks === 1 ? "section" : "sections"}</td><td>${esc(documentAccess(d))}</td><td><button data-document="${i}">Open</button></td></tr>`,
      ),
    );
    $("documents")
      .querySelectorAll("[data-document]")
      .forEach(
        (b) =>
          (b.onclick = () =>
            editDocument(o.documents[Number(b.dataset.document)]).catch((e) =>
              notice(e.message),
            )),
      );
  }
  const cacheEvents = o.events.filter((e) => e.kind === "embedding.cache");
  const answerEvents = o.events.filter((e) => e.kind === "answer.cache");
  const hits = cacheEvents.reduce((n, e) => n + e.details.hits, 0),
    misses = cacheEvents.reduce((n, e) => n + e.details.misses, 0);
  const answerHits = answerEvents.filter((e) => e.details.hit).length;
  $("cache-cards").innerHTML =
    `<article class="panel"><h2>Document search</h2><div class="cache-value">${o.embedding_entries} ready</div><p class="muted">${hits} quick ${hits === 1 ? "reuse" : "reuses"} and ${misses === 0 ? "no rebuilds" : `${misses} ${misses === 1 ? "rebuild" : "rebuilds"}`} in recent activity.</p><span class="tag">WORKING</span></article><article class="panel"><h2>Saved answers</h2><div class="cache-value">${o.answer_entries} ready</div><p class="muted">${answerHits ? `${answerHits} repeated ${answerHits === 1 ? "question was" : "questions were"} answered instantly in recent activity.` : "Repeated questions can reuse an answer after fresh access checks."}</p><span class="tag">ACCESS CHECKED</span></article><article class="panel"><h2>Built-in safety test</h2><div class="cache-value">${o.synthetic_kv.entries ? `${o.synthetic_kv.entries} saved` : "Not in use"}</div><p class="muted">This storage is only for automated CPU safety checks. Ollama answers do not use it.</p><span class="tag">TEST ONLY</span></article><article class="panel"><h2>Optional GPU acceleration</h2><div class="cache-value">${o.tensor_kv.loaded_tokens == null ? "Not active" : `${o.tensor_kv.loaded_tokens} tokens`}</div><p class="muted">${o.tensor_kv.load_events ? `${o.tensor_kv.load_events} GPU transfers have been recorded.` : "No GPU transfers have been recorded on this computer."}</p><span class="tag">${o.tensor_kv.loaded_tokens == null ? "UNAVAILABLE HERE" : "TRANSFER HISTORY"}</span></article>`;
  $("connector-status").innerHTML = o.connector
    ? `<h2>GPU acceleration setup</h2><div class="callout">${o.connector.reasons.length ? "Optional GPU acceleration is unavailable on this computer. The local Ollama dashboard still works normally." : "The required GPU software was found. Run the validation check before relying on acceleration."}</div><details><summary>Technical requirements</summary><p class="muted">Detected system: ${esc(o.connector.platform)}</p>${o.connector.reasons.length ? `<ul>${o.connector.reasons.map((reason) => `<li>${esc(reason)}</li>`).join("")}</ul>` : ""}<p class="muted">Software versions: vLLM ${esc(o.connector.pins.vllm)} and LMCache ${esc(o.connector.pins.lmcache)}. Run <code>uv run vaultkv connector-doctor</code> for diagnostics.</p></details>`
    : "";
  $("events").innerHTML =
    cacheEvents
      .slice(0, 10)
      .map(
        (e) =>
          `<div class="event"><time>${new Date(e.created_at * 1000).toLocaleTimeString()}</time><div><strong>Search data reused</strong><p>${e.details.hits || 0} saved ${(e.details.hits || 0) === 1 ? "entry" : "entries"} reused · ${e.details.misses || 0} rebuilt · ${elapsed(e.details.elapsed_ms)}</p></div></div>`,
      )
      .join("") || '<div class="empty">No recent search activity.</div>';
  $("updated").textContent =
    `Updated ${new Date().toLocaleTimeString()} · refresh 2s`;
}
function renderRequests() {
  const key = JSON.stringify([state.traces, state.selected]);
  if (key === state.requestKey) return;
  state.requestKey = key;
  $("request-list").innerHTML = state.traces.length
    ? table(
        ["Question", "Workspace / user", "State", "Duration", ""],
        state.traces.map(
          (t, i) =>
            `<tr><td>${esc(t.question)}<div class="muted">${new Date(t.created_at * 1000).toLocaleString()}</div></td><td>${esc(t.principal.tenant_ns)} / ${esc(t.principal.principal_id)} · K${t.knowledge_epoch}</td><td>${status(t.status)}</td><td>${elapsed(t.elapsed_ms)}</td><td><button data-trace="${i}">${t.request_id === state.selected ? "Selected" : "Inspect"}</button></td></tr>`,
        ),
      )
    : '<div class="empty">No traced requests yet. Run a question in Workspace.</div>';
  $("request-list")
    .querySelectorAll("[data-trace]")
    .forEach(
      (b) =>
        (b.onclick = async () => {
          state.selected = state.traces[Number(b.dataset.trace)].request_id;
          try {
            await refreshDetail();
            renderRequests();
          } catch (e) {
            notice(e.message);
          }
        }),
    );
  renderLensPicker();
}
function renderLensPicker() {
  const picker = $("lens-request");
  const key = JSON.stringify([
    state.selected,
    state.traces.map((trace) => [
      trace.request_id,
      trace.question,
      trace.created_at,
    ]),
  ]);
  if (key === state.lensPickerKey) return;
  state.lensPickerKey = key;
  picker.innerHTML = state.traces.length
    ? state.traces
        .map(
          (trace) =>
            `<option value="${esc(trace.request_id)}">${esc(trace.question)} · ${new Date(trace.created_at * 1000).toLocaleString()}</option>`,
        )
        .join("")
    : '<option value="">No answered questions yet</option>';
  picker.disabled = !state.traces.length;
  if (state.selected) picker.value = state.selected;
}
$("lens-request").onchange = async (event) => {
  state.selected = event.target.value;
  try {
    await refreshDetail();
    renderRequests();
  } catch (error) {
    notice(error.message);
  }
};
function sources(result) {
  return `<div class="sources">${result.sources.map((s) => `<div class="source"><span class="tag">${esc(s.id)}</span> <strong>${esc(s.source.resource.doc_id)}</strong><div class="muted">${esc(s.source.chunk_id)} · v${s.source.doc_version} · A${s.source.resource_auth_epoch}</div><p>${esc(s.excerpt)}</p><div class="score">BM25 ${num(s.scores?.bm25)} · cosine ${num(s.scores?.cosine)} · RRF ${num(s.scores?.rrf)}</div></div>`).join("")}</div>`;
}
function sectionName(chunkId) {
  const match = /^chunk-(\d+)$/.exec(chunkId);
  return match ? `Section ${Number(match[1]) + 1}` : chunkId;
}
function lensSources(result) {
  const strongest = Math.max(
    ...result.sources.map((source) => source.scores?.rrf || 0),
    1e-9,
  );
  return `<div class="lens-sources">${result.sources
    .map((source, index) => {
      const strength = Math.round(
        ((source.scores?.rrf || 0) / strongest) * 100,
      );
      return `<article class="lens-source"><div class="lens-source-head"><div><span class="source-rank">${index + 1}</span><strong>${esc(source.source.resource.doc_id)}</strong><div class="muted">${esc(sectionName(source.source.chunk_id))} · Content version ${source.source.doc_version}</div></div><span class="tag">${esc(source.id)}</span></div><div class="rank-label"><span>Relative rank score</span><strong>${strength}%</strong></div><div class="rank-bar"><i style="width:${strength}%"></i></div><p class="source-excerpt">${esc(source.excerpt)}</p><details><summary>View technical scores</summary><div class="score-grid"><span>Keyword match (BM25)<strong>${num(source.scores?.bm25)}</strong></span><span>Meaning match (cosine)<strong>${num(source.scores?.cosine)}</strong></span><span>Combined rank (RRF)<strong>${num(source.scores?.rrf)}</strong></span></div></details></article>`;
    })
    .join("")}</div>`;
}
function renderDetail() {
  renderLensPicker();
  const key = JSON.stringify(state.trace);
  if (key === state.detailKey) return;
  state.detailKey = key;
  const expanded = $("request-detail").querySelector("details")?.open;
  const t = state.trace;
  if (!t) {
    $("journey").innerHTML =
      '<div class="empty">Run a query to inspect each stage.<small>Retrieve → context → generate → authorize</small></div>';
    $("answer").innerHTML =
      '<div class="empty">Your answer and supporting evidence will appear here.</div>';
    $("request-detail").innerHTML = "";
    $("lens").innerHTML = '<div class="empty">No request selected.</div>';
    $("export").disabled = true;
    return;
  }
  $("journey-status").textContent = t.withheld
    ? "WITHHELD"
    : t.status.toUpperCase();
  const milestones = t.events.filter((e) =>
    [
      "request.started",
      "retrieval.completed",
      "context.assembled",
      "generation.completed",
      "finality.accepted",
      "request.failed",
      "request.blocked",
    ].includes(e.kind),
  );
  const end =
    t.elapsed_ms ||
    Math.max(...t.events.map((e) => e.details.elapsed_ms || 0), 1);
  const journey = milestones
    .map((e, i) => {
      const labels = {
        "request.started": "Question received",
        "retrieval.completed": "Sources found",
        "context.assembled": "Evidence prepared",
        "generation.completed": e.details.answer_cache_hit
          ? "Saved answer reused"
          : "Answer written",
        "finality.accepted": "Access approved",
        "request.failed": "Request failed",
        "request.blocked": "Answer blocked",
      };
      return `<div class="stage"><b>${String(i + 1).padStart(2, "0")}</b><span>${esc(labels[e.kind] || e.kind)}</span><time>${elapsed(e.details.elapsed_ms || 0)}</time><div class="bar"><i style="width:${Math.max(1, Math.min(100, ((e.details.elapsed_ms || 0) / end) * 100))}%"></i></div></div>`;
    })
    .join("");
  $("journey").innerHTML = journey;
  const gen =
    t.events.find((e) => e.kind === "generation.completed")?.details || {};
  const usage = gen.answer_cache_hit
    ? "saved answer reused"
    : `${gen.prompt_eval_count ?? "—"} prompt tokens / ${gen.eval_count ?? "—"} output tokens`;
  const metrics = `<p class="muted">${esc(t.principal.tenant_ns)} / ${esc(t.principal.principal_id)} · K${t.knowledge_epoch} · ${elapsed(t.elapsed_ms)} · ${usage}</p>`;
  const answer = t.result
    ? `<div class="panel-head"><h2>Released answer</h2><span class="tag">ACCESS CHECKED</span></div>${metrics}<div class="answer-text">${esc(t.result.answer)}</div>${sources(t.result)}`
    : `<div class="empty">${esc(t.withheld ? `Content withheld: ${t.withheld}` : t.error ? `Request ${t.status}: ${t.error}` : "Request in progress. Output remains private until finality.")}</div>`;
  $("answer").innerHTML = answer;
  $("request-detail").innerHTML =
    `<h2>${esc(t.question)}</h2><p class="muted">Request ${esc(t.request_id)}</p>${answer}<details ${expanded ? "open" : ""}><summary>Trace events & contract</summary><pre>${esc(JSON.stringify({ contract: t.contract, events: t.events }, null, 2))}</pre></details>`;
  $("lens").innerHTML = t.result
    ? `<div class="lens-title"><div><span class="tag">ACCESS CHECKED</span><h3>${esc(t.question)}</h3><p class="muted">${esc(t.principal.tenant_ns)} workspace · ${esc(t.principal.principal_id)} · Content version ${t.knowledge_epoch}</p></div></div><div class="lens-summary"><div><span>Status</span><strong>Answer released</strong><small>Access checked before display</small></div><div><span>Sources</span><strong>${t.result.sources.length}</strong><small>Ranked for this question</small></div><div><span>Answer time</span><strong>${elapsed(t.elapsed_ms)}</strong><small>Search through final answer</small></div><div><span>Model</span><strong>${esc(t.result.model || "Local model")}</strong><small>${t.result.retrieval === "hybrid" ? "Keyword + meaning search" : "Keyword search"}</small></div></div><section class="lens-answer"><h3>Answer</h3><p>${esc(t.result.answer)}</p></section><div class="lens-section-head"><div><h3>Why these sources were selected</h3><p class="muted">Sources are shown in retrieval order. A relative score of 100% marks the strongest source in this request.</p></div></div>${lensSources(t.result)}<p class="muted lens-export-note">Download the report for the answer, sources, exact ranking scores, and model usage in JSON format.</p>`
    : `<div class="empty">${esc(t.withheld ? "This request is no longer authorized for content inspection." : "A successfully released request is required.")}</div>`;
  $("export").disabled = !t.result;
}
async function refreshDetail() {
  if (state.selected) {
    state.trace = await api("/api/traces/" + state.selected);
    renderDetail();
  }
}
async function refresh() {
  const [o, t] = await Promise.all([api("/api/overview"), api("/api/traces")]);
  state.overview = o;
  state.traces = t;
  renderOverview();
  renderRequests();
  if (!state.selected && !state.busy && t.length)
    state.selected = t[0].request_id;
  await refreshDetail();
}
$("ask-form").onsubmit = async (event) => {
  event.preventDefault();
  notice();
  $("run").disabled = true;
  state.busy = true;
  state.selected = null;
  state.trace = null;
  renderDetail();
  $("journey-status").textContent = "RUNNING";
  try {
    const { job_id } = await api("/api/ask", {
      question: $("question").value,
      tenant: $("tenant").value,
      principal: $("principal").value,
      model: $("model").value,
      retrieval: $("retrieval").value,
      top_k: Number($("top-k").value),
    });
    for (;;) {
      const job = await api("/api/jobs/" + job_id);
      if (job.request_id) {
        state.selected = job.request_id;
        await refreshDetail();
      }
      if (job.state === "failed") throw Error(job.error);
      if (job.state === "done") break;
      await new Promise((resolve) => setTimeout(resolve, 600));
    }
    await refresh();
  } catch (e) {
    notice(e.message);
  } finally {
    $("run").disabled = false;
    state.busy = false;
  }
};
async function editDocument(doc) {
  const data = await api(
    "/api/document?" +
      new URLSearchParams({ tenant: doc.tenant_ns, doc: doc.doc_id }),
  );
  state.editor = { doc, data };
  $("editor").hidden = false;
  $("editor").innerHTML =
    `<div class="panel-head"><h2>Edit ${esc(doc.doc_id)}</h2><button id="close-editor">Close</button></div><p class="muted">Update the document or change who can use it. If another person saves first, close and reopen this editor before trying again.</p><form id="content-form">${data.chunks.map((c, i) => `<label>Section ${i + 1}<textarea data-chunk="${i}" rows="5">${esc(c.text)}</textarea></label>`).join("")}<button class="primary">Save document changes</button></form><hr><form id="acl-form"><div class="form-grid"><label>Who can use this document?<select id="acl-scope"><option value="tenant-shared">Listed users and groups</option><option value="private">Listed users, kept separate</option><option value="public">Everyone</option></select></label><label>Users to allow<input id="acl-principals" value="${esc(doc.principals.join(", "))}" placeholder="alice, bob"></label><label>Groups to allow<input id="acl-groups" value="${esc((doc.groups || []).join(", "))}" placeholder="staff, managers"></label></div><p class="muted">Enter users and groups separated by commas. Leave both lists empty to block access. The lists are ignored when “Everyone” is selected.</p><button>Save access settings</button></form>`;
  $("acl-scope").value = doc.scope;
  $("close-editor").onclick = () => {
    $("editor").hidden = true;
    state.editor = null;
  };
  $("content-form").onsubmit = async (e) => {
    e.preventDefault();
    try {
      const chunks = {};
      $("content-form")
        .querySelectorAll("[data-chunk]")
        .forEach(
          (el) =>
            (chunks[data.chunks[Number(el.dataset.chunk)].chunk_id] = el.value),
        );
      await api("/api/publish", {
        tenant: doc.tenant_ns,
        doc: doc.doc_id,
        expected_epoch: data.knowledge_epoch,
        chunks,
      });
      notice(
        "Document changes saved. New questions will use the updated text.",
      );
      $("editor").hidden = true;
      await refresh();
    } catch (error) {
      notice(error.message);
    }
  };
  $("acl-form").onsubmit = async (e) => {
    e.preventDefault();
    try {
      await api("/api/permissions", {
        tenant: doc.tenant_ns,
        doc: doc.doc_id,
        expected_epoch: doc.auth_epoch,
        scope: $("acl-scope").value,
        principals:
          $("acl-scope").value === "public"
            ? []
            : parseNames($("acl-principals").value),
        groups:
          $("acl-scope").value === "public"
            ? []
            : parseNames($("acl-groups").value),
      });
      notice("Access settings saved and applied to future answers.");
      $("editor").hidden = true;
      await refresh();
    } catch (error) {
      notice(error.message);
    }
  };
}
$("export").onclick = async () => {
  try {
    const data = await api("/api/traces/" + state.selected + "/vectorlens");
    if (!data.llm_responses.length)
      throw Error("Content is no longer authorized for export.");
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `vectorlens-${state.selected}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    notice(e.message);
  }
};
$("race-run").onclick = async () => {
  $("race-run").disabled = true;
  try {
    $("race-result").innerHTML =
      '<p class="muted">Running safety checks with sample data…</p>';
    const result = await api("/api/races", {});
    $("race-result").innerHTML =
      `<div class="callout">VaultKV blocked every protected answer after access was removed. The other rows show why the final access check is necessary.</div>${table(
        ["Protection method", "Answers after access removal", "Result"],
        Object.entries(result.post_revocation_emissions).map(
          ([strategy, count]) =>
            `<tr><td>${esc({ retrieval_only: "Check during search", retrieval_and_kv: "Check during search and reuse", txnrag: "Check again before answering" }[strategy] || strategy)}</td><td>${count}</td><td>${count === 0 ? '<span class="status protected">Passed</span>' : '<span class="status blocked">Needs final check</span>'}</td></tr>`,
        ),
      )}<details><summary>View all ${result.cases.length} technical test cases</summary><pre>${esc(JSON.stringify(result.cases, null, 2))}</pre></details>`;
  } catch (e) {
    notice(e.message);
  } finally {
    $("race-run").disabled = false;
  }
};
async function runtime() {
  try {
    const data = await api("/api/runtime");
    $("runtime").textContent = data.online
      ? data.warmup?.state === "starting"
        ? "◌ Ollama online · warming models"
        : data.warmup?.state === "ready"
          ? "● Ollama online · ready"
          : "● Ollama online"
      : "○ Ollama offline";
    const current = $("model").value;
    const models = data.models.filter((m) => !m.name.includes("embed"));
    if (models.length) {
      $("model").innerHTML = models
        .map((m) => `<option>${esc(m.name)}</option>`)
        .join("");
      if (models.some((m) => m.name === current)) $("model").value = current;
    }
  } catch (e) {
    notice(e.message);
  }
}
let refreshing = false;
async function tick() {
  if (refreshing) return;
  refreshing = true;
  try {
    await refresh();
  } catch (e) {
    notice(e.message);
  } finally {
    refreshing = false;
  }
}
renderDetail();
tick();
runtime();
setInterval(tick, 2000);
setInterval(runtime, 30000);
