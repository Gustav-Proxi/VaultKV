"""Prometheus text exposition derived from persisted trace tables.

Computed from SQLite on each scrape, so counters survive restarts and every
replica sharing a database reports the same values. Only counts, epochs and
durations are exported; never questions, chunks or answers.
"""

from __future__ import annotations

import json

from vaultkv.txnrag import TxnRAG

# Seconds; covers cached answers (ms) through cold CPU generation (tens of seconds).
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
PHASES = ("retrieval.completed", "context.assembled", "generation.completed")


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**labels: object) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in labels.items()) + "}"


class _Exposition:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def family(self, name: str, kind: str, help_text: str) -> None:
        self.lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]

    def sample(self, name: str, value: float, **labels: object) -> None:
        self.lines.append(f"{name}{_labels(**labels)} {value:g}")

    def histogram(self, name: str, values: list[float], **labels: object) -> None:
        for bound in LATENCY_BUCKETS:
            self.sample(f"{name}_bucket", sum(v <= bound for v in values), le=bound, **labels)
        self.sample(f"{name}_bucket", len(values), le="+Inf", **labels)
        self.sample(f"{name}_sum", sum(values), **labels)
        self.sample(f"{name}_count", len(values), **labels)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def render(store: TxnRAG) -> str:
    out = _Exposition()
    with store._ledger.transaction() as db:
        meta = db.execute("SELECT knowledge_epoch, policy_epoch FROM txn_meta").fetchone()
        statuses = dict(db.execute("SELECT status, count(*) FROM txn_traces GROUP BY status"))
        blocked = db.execute(
            "SELECT error, count(*) FROM txn_traces WHERE status = 'blocked' GROUP BY error"
        ).fetchall()
        latencies = [
            row[0] / 1000
            for row in db.execute(
                "SELECT elapsed_ms FROM txn_traces WHERE status = 'emitted' "
                "AND elapsed_ms IS NOT NULL"
            )
        ]
        events = db.execute("SELECT kind, count(*) FROM txn_events GROUP BY kind").fetchall()
        phase_rows = db.execute(
            f"SELECT kind, details_json FROM txn_events WHERE kind IN "
            f"({','.join('?' * len(PHASES))})",
            PHASES,
        ).fetchall()
        documents = db.execute(
            "SELECT count(*) FROM txn_snapshot_docs WHERE epoch = ?", (meta["knowledge_epoch"],)
        ).fetchone()[0]
        chunks = db.execute(
            "SELECT count(*) FROM txn_snapshot_docs s JOIN txn_chunks c "
            "ON s.tenant_ns = c.tenant_ns AND s.doc_id = c.doc_id "
            "AND s.doc_version = c.doc_version WHERE s.epoch = ?",
            (meta["knowledge_epoch"],),
        ).fetchone()[0]
        principals = db.execute("SELECT enabled, count(*) FROM txn_principals GROUP BY enabled")
        principals = {bool(row[0]): row[1] for row in principals}
        cached_answers = db.execute("SELECT count(*) FROM txn_answers").fetchone()[0]

    out.family("vaultkv_requests_total", "counter", "Answer requests by final status.")
    for status in ("emitted", "blocked", "failed", "running"):
        out.sample("vaultkv_requests_total", statuses.get(status, 0), status=status)

    out.family(
        "vaultkv_blocked_requests_total",
        "counter",
        "Requests stopped by the authorization/finality checks, by reason.",
    )
    for reason, count in blocked:
        out.sample("vaultkv_blocked_requests_total", count, reason=reason or "unknown")

    out.family(
        "vaultkv_request_duration_seconds",
        "histogram",
        "End-to-end latency of released answers.",
    )
    out.histogram("vaultkv_request_duration_seconds", latencies)

    # Per-phase cumulative elapsed time recorded by RequestTrace events.
    phase_values: dict[str, list[float]] = {phase: [] for phase in PHASES}
    for kind, details in phase_rows:
        elapsed = json.loads(details).get("elapsed_ms")
        if isinstance(elapsed, int | float):
            phase_values[kind].append(elapsed / 1000)
    out.family(
        "vaultkv_phase_elapsed_seconds",
        "histogram",
        "Time from request start until each pipeline phase completed.",
    )
    for phase, values in phase_values.items():
        out.histogram("vaultkv_phase_elapsed_seconds", values, phase=phase.split(".")[0])

    out.family("vaultkv_events_total", "counter", "Persisted pipeline events by kind.")
    for kind, count in events:
        out.sample("vaultkv_events_total", count, kind=kind)

    out.family("vaultkv_knowledge_epoch", "gauge", "Current published knowledge snapshot.")
    out.sample("vaultkv_knowledge_epoch", meta["knowledge_epoch"])
    out.family("vaultkv_policy_epoch", "gauge", "Current global policy epoch.")
    out.sample("vaultkv_policy_epoch", meta["policy_epoch"])
    out.family("vaultkv_documents", "gauge", "Documents in the current snapshot.")
    out.sample("vaultkv_documents", documents)
    out.family("vaultkv_chunks", "gauge", "Chunks in the current snapshot.")
    out.sample("vaultkv_chunks", chunks)
    out.family("vaultkv_principals", "gauge", "Known principals by enabled state.")
    out.sample("vaultkv_principals", principals.get(True, 0), enabled="true")
    out.sample("vaultkv_principals", principals.get(False, 0), enabled="false")
    out.family("vaultkv_cached_answers", "gauge", "Answers stored in the answer cache.")
    out.sample("vaultkv_cached_answers", cached_answers)
    return out.text()
