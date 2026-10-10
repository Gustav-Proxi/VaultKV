"""Deterministic no-model generator for CI, smoke tests and model-free deployments.

Implements the subset of the OllamaClient interface LocalRAG uses. The answer is
the opening words of the top-ranked source with its citation, so every released
answer still passes LocalRAG's citation checks and the finality gate.
"""

from __future__ import annotations

import re

from vaultkv.txnrag.ollama import ModelRef

MAX_WORDS = 20


class ExtractiveClient:
    backend = "extractive"
    kv_format = "extractive:no-kv"

    def __init__(self) -> None:
        self.last_generation_stats: dict = {}

    def resolve(self, name: str) -> ModelRef:
        return ModelRef("extractive", "v1")

    def verify(self, model: ModelRef) -> None:
        pass

    def generate(self, model: ModelRef, system: str, prompt: str) -> str:
        self.last_generation_stats = {}
        match = re.search(r"^\[S1\] (.*?)(?=\n\n\[S\d+\] |\Z)", prompt, re.MULTILINE | re.DOTALL)
        if match is None:
            return "I do not know."
        body = [line for line in match.group(1).splitlines() if not line.lstrip().startswith("#")]
        words = " ".join(body).split()
        if not words:
            return "I do not know."
        return " ".join(words[:MAX_WORDS]).rstrip(".,;:") + " [S1]"
