# Local observatory

Start the operator dashboard against an existing corpus:

```sh
uv run vaultkv serve --db data/local-demo.sqlite3 --port 8765 --open
```

Open the complete URL printed by the command. It contains a per-process access
secret in its fragment; the page moves it into tab-scoped session storage and
clears the fragment. Restarting the server creates a new secret. The server binds
only to `127.0.0.1`, validates Host/Origin, and requires the secret on every API.
Static assets are packaged locally; no frontend build or CDN is required.

In the interface, **Workspace** means the organization or project that owns the
documents, and **User** means the person or service asking the question. Internal
APIs retain the older `tenant` and `principal` field names for compatibility.

## Views and controls

- **Workspace:** query local Ollama with a selected workspace, user, model, retrieval mode
  and source count. Inspect cumulative stage times, released answers and citations.
- **Requests:** persisted status, elapsed time, snapshot/model contract and events.
  Prompt/output token counts come from Ollama when available.
- **Documents & access:** import `.md` and `.txt` files, edit document sections,
  manage users, and create reusable groups. Documents can allow multiple users and
  groups. Disabling a user or changing a group is applied to current and historical
  reads; requests already being prepared are checked again before output. Existing
  filenames keep their access settings when their text is updated.
- **Performance & safety:** readable search reuse, optional GPU status, recent
  activity, and isolated safety checks that do not mutate the displayed corpus.
- **VectorLens:** view answers, request metrics, ranked evidence cards and score
  bars in the dashboard, or download compatible session JSON.
  The adapter was checked against `~/Code/vectorlens/vectorlens/types.py`. It uses
  that session/event contract without importing its automatic interceptors or
  transmitting sessions to its server. It exports selected excerpts, citation
  links, BM25/cosine/RRF scores, released text and available model usage. Full
  prompts and embedding vectors are omitted; attribution arrays remain empty.

Embedding reuse, synthetic context bytes and model tensor KV are separate.
**The experimental vLLM/LMCache adapter needs GPU validation.** The view reports
local prerequisites and actual recorded loaded-token counts. Tensor memory and hit
rate remain unavailable. See [connector setup](connectors.md).
Source scores explain ranking; they do not prove that every answer claim is correct.
Metrics cover requests traced after this feature was installed; prior requests
are not backfilled. The cache activity window is the last 80 events.

## CLI inspection

```sh
uv run vaultkv status --db data/local-demo.sqlite3
uv run vaultkv traces --db data/local-demo.sqlite3
uv run vaultkv inspect REQUEST_ID --db data/local-demo.sqlite3
uv run vaultkv inspect REQUEST_ID --db data/local-demo.sqlite3 --vectorlens
```

## Trust and persistence

This is a **privileged local operator console**, not a multi-user login service.
The token grants administration, including full document editing and selection
of any local identity. Do not publish the token or proxy the server publicly.

Questions, request metadata and authorized released results persist in SQLite.
Telemetry events exclude raw model output and chunk text. Blocked/failed traces
have no stored result. The underlying trusted artifact store can retain internal
pipeline content, including content from rejected requests; no forensic erasure
or automatic retention cleanup is provided. Protect the database accordingly.

Every historical answer/excerpt read and export rechecks the current user,
group membership, policy and source authorization while holding the lock through delivery. Content
updates preserve historical pinned snapshots; authorization changes can withhold
previously released content. Browser polling refreshes the view every two seconds.
Already delivered answers and downloaded exports cannot be recalled. A process
crash may leave a trace marked running; this prototype has no job recovery queue.
