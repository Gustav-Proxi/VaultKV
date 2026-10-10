"""Optional OpenTelemetry tracing.

Active only when the `otel` extra is installed and OTEL_EXPORTER_OTLP_ENDPOINT is
set; otherwise every helper is a no-op, so the core keeps zero runtime deps.
Span attributes carry ids, statuses and epochs, never questions or content.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

_tracer = None


def init_tracing(service_name: str = "vaultkv") -> bool:
    global _tracer
    if _tracer is not None:
        return True
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("vaultkv")
    return True


class _NullSpan:
    def set_attribute(self, key: str, value: object) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None, timestamp=None) -> None:
        pass


@contextmanager
def span(name: str, **attributes: object) -> Iterator[object]:
    if _tracer is None:
        yield _NullSpan()
        return
    with _tracer.start_as_current_span(name, attributes=attributes) as current:
        yield current
