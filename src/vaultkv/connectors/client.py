"""Optional in-process vLLM producer. Output still passes through LocalRAG finality."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from vaultkv.connectors.guard import TransferGuard
from vaultkv.connectors.runtime import engine_config, require_versions
from vaultkv.txnrag.ollama import ModelError, ModelRef


@dataclass(frozen=True)
class VLLMModelRef(ModelRef):
    @property
    def revision(self) -> str:
        return f"vllm:{self.name}@{self.digest}"


class VLLMClient:
    backend = "vllm-lmcache"
    kv_format = "vllm-0.12.0:lmcache-0.3.9:fp16:single-gpu:prefix-v1"

    def __init__(self, path: Path, model: str, revision: str) -> None:
        settings = engine_config(path, model, revision)
        require_versions()
        from vllm import LLM

        self.model = VLLMModelRef(model, revision)
        self.engine = LLM(**settings)
        self.last_generation_stats = {}
        self._bound = None

    def resolve(self, model: str) -> ModelRef:
        if model != self.model.name:
            raise ValueError("This vLLM engine serves only its pinned generation model")
        return self.model

    def bind_request(self, store, token, context) -> None:
        self._bound = (store, token, context)

    def generate(self, model: ModelRef, system: str, prompt: str) -> str:
        from vllm import SamplingParams

        if self._bound is None or model != self.model:
            raise ModelError("vLLM generation requires a registered TxnRAG context")
        store, token, context = self._bound
        self._bound = None
        tokenizer = self.engine.get_tokenizer()
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(ids) + 384 > 8192:
            raise ModelError("Prompt exceeds the pinned vLLM context budget")
        guard = TransferGuard(store)
        configs = guard.issue(token, context, ids)
        try:
            started = time.perf_counter()
            results = self.engine.generate(
                {"prompt_token_ids": ids},
                SamplingParams(
                    temperature=0,
                    seed=42,
                    max_tokens=384,
                    extra_args={"kv_transfer_params": configs},
                ),
                use_tqdm=False,
            )
            if len(results) != 1 or not results[0].finished or len(results[0].outputs) != 1:
                raise ModelError("vLLM returned an incomplete response")
            output = results[0].outputs[0]
            if output.finish_reason != "stop" or not output.text.strip():
                raise ModelError("vLLM did not finish a nonempty answer within its budget")
            with store._ledger.transaction() as db:
                loaded = sum(
                    json.loads(row[0])["tokens"]
                    for row in db.execute(
                        "SELECT details_json FROM txn_events WHERE request_id = ? "
                        "AND kind = 'connector.loaded'",
                        (token.request_id,),
                    )
                )
            self.last_generation_stats = {
                "prompt_eval_count": len(ids),
                "eval_count": len(output.token_ids),
                "connector_loaded_tokens": loaded,
                "total_duration": int((time.perf_counter() - started) * 1e9),
            }
            return output.text.strip()
        finally:
            guard.close(configs)
