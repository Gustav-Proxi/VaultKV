"""Opt-in CUDA test: cold/warm prefix transfers and output-time revocation.

Run on the pinned Linux GPU environment; never part of ordinary CPU pytest.
Uses a temporary corpus and real vLLM/LMCache, not fixture transfer counts.
"""

import json
import tempfile
from pathlib import Path

from vaultkv import Identity
from vaultkv.connectors import MODEL, MODEL_REVISION
from vaultkv.connectors.client import VLLMClient
from vaultkv.connectors.runtime import require_versions
from vaultkv.txnrag import Resource, TxnRAG, TxnRejected
from vaultkv.txnrag.local import LocalRAG


def main() -> None:
    require_versions()
    identity = Identity("smoke", "alice")
    resource = Resource("smoke", "hotel-policy")
    with tempfile.TemporaryDirectory(prefix="vaultkv-gpu-") as directory:
        path = Path(directory) / "smoke.sqlite3"
        with TxnRAG(path) as store:
            store.publish(
                {
                    resource: {
                        "c": (
                            "The nightly hotel reimbursement limit is $180 excluding taxes. "
                            "Employees must retain itemized receipts, obtain manager approval "
                            "for exceptions before booking, and submit expenses within thirty days."
                        )
                    }
                },
                expected_epoch=0,
            )
            store.set_permissions(resource, expected_epoch=1, principals=[identity])
            client = VLLMClient(path, MODEL, MODEL_REVISION)
            rag = LocalRAG(store, client, model=MODEL, embedding_model=None)
            outputs, runs = [], []
            for _ in range(2):
                rag.ask(identity, "What is the nightly hotel limit?", outputs.append, top_k=1)
                runs.append(dict(client.last_generation_stats))
            assert len(outputs) == 2
            assert runs[1]["connector_loaded_tokens"] > 0, "No real warm-prefix transfer observed"
            assert all("180" in json.loads(value)["answer"] for value in outputs)
            blocked = []

            def revoke(phase, token):
                if phase == "after_generation":
                    store.set_permissions(resource, expected_epoch=2)

            try:
                rag.ask(
                    identity,
                    "What is the nightly hotel limit?",
                    blocked.append,
                    top_k=1,
                    barrier=revoke,
                )
            except TxnRejected:
                pass
            else:
                raise AssertionError("Revoked output was not rejected")
            assert not blocked
            print(
                json.dumps(
                    {
                        "mode": "real-cuda-vllm-lmcache",
                        "model_revision": MODEL_REVISION,
                        "runs": runs,
                        "post_revocation_outputs": len(blocked),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
