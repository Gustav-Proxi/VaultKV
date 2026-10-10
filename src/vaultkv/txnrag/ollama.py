"""Small loopback-only Ollama client; all responses are buffered and validated."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ModelError(RuntimeError):
    pass


KEEP_ALIVE = "30m"
CONTEXT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 48


@dataclass(frozen=True)
class ModelRef:
    name: str
    digest: str

    @property
    def revision(self) -> str:
        return f"ollama:{self.name}@{self.digest}"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("Ollama redirects are disabled")


class OllamaClient:
    def __init__(self, url: str = "http://127.0.0.1:11434", *, timeout: float = 180) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "Ollama URL must be an HTTP loopback address without credentials or path"
            )
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.last_generation_stats: dict = {}
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise ModelError(f"Ollama HTTP {exc.code}; check the model and local server") from exc
        except (URLError, OSError) as exc:
            raise ModelError("Cannot reach local Ollama; start it with `ollama serve`") from exc
        except (ValueError, UnicodeError) as exc:
            raise ModelError("Ollama returned invalid JSON") from exc
        if not isinstance(result, dict) or "error" in result:
            raise ModelError("Ollama returned an error or malformed response")
        return result

    def resolve(self, name: str) -> ModelRef:
        full_name = name if ":" in name else name + ":latest"
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list):
            raise ModelError("Ollama returned an invalid model listing")
        for item in models:
            if not isinstance(item, dict):
                raise ModelError("Ollama returned an invalid model entry")
            if (
                item.get("name") == full_name
                and isinstance(item.get("digest"), str)
                and item["digest"]
            ):
                return ModelRef(full_name, item["digest"])
        raise ModelError(f"Model {full_name!r} is not installed; run `ollama pull {full_name}`")

    def verify(self, model: ModelRef) -> None:
        if self.resolve(model.name) != model:
            raise ModelError(
                "Model revision changed during the request; retry with a fresh request"
            )

    def embed(self, model: ModelRef, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.verify(model)
        data = self._request(
            "/api/embed",
            {
                "model": model.name,
                "input": texts,
                "truncate": False,
                "keep_alive": KEEP_ALIVE,
            },
        )
        self.verify(model)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ModelError("Ollama returned the wrong number of embeddings")
        dimensions = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise ModelError("Ollama returned an empty embedding")
            if dimensions is not None and len(vector) != dimensions:
                raise ModelError("Embedding dimensions are inconsistent")
            dimensions = len(vector)
            if any(type(value) not in (float, int) or not math.isfinite(value) for value in vector):
                raise ModelError("Ollama returned an invalid embedding")
            if sum(value * value for value in vector) == 0:
                raise ModelError("Ollama returned a zero embedding")
        return vectors

    def warm(self, model: ModelRef) -> None:
        """Load a model without generating text so the first question avoids cold-start I/O."""
        self.verify(model)
        data = self._request(
            "/api/generate",
            {
                "model": model.name,
                "prompt": "",
                "stream": False,
                "options": {"num_ctx": CONTEXT_TOKENS},
                "keep_alive": KEEP_ALIVE,
            },
        )
        if data.get("done") is not True:
            raise ModelError("Ollama did not finish warming the model")
        self.verify(model)

    def generate(self, model: ModelRef, system: str, prompt: str) -> str:
        self.last_generation_stats = {}
        # Conservative byte bound leaves room for chat framing and output in 8192 tokens.
        if len((system + prompt).encode()) > 6500:
            raise ModelError("Prompt exceeds the local generation budget")
        self.verify(model)
        data = self._request(
            "/api/chat",
            {
                "model": model.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": CONTEXT_TOKENS,
                    "num_predict": MAX_OUTPUT_TOKENS,
                },
                "keep_alive": KEEP_ALIVE,
            },
        )
        self.verify(model)
        message = data.get("message", {})
        answer = message.get("content") if isinstance(message, dict) else None
        if data.get("done") is not True or not isinstance(answer, str) or not answer.strip():
            raise ModelError("Ollama did not return a complete, nonempty answer")
        if data.get("done_reason") == "length":
            raise ModelError("Ollama exhausted the output budget; no partial answer was released")
        self.last_generation_stats = {
            key: data[key]
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "prompt_eval_cached_count",
            )
            if type(data.get(key)) in (int, float)
        }
        return answer.strip()
