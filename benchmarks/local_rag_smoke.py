"""Opt-in real Ollama smoke test: normal answer, then revoke during generation.

Run after installing qwen2.5:1.5b and nomic-embed-text. This checks the output
boundary, not model quality or GPU KV reuse. The fixture uses invented data.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from vaultkv import Identity
from vaultkv.txnrag import Resource, TxnRAG, TxnRejected
from vaultkv.txnrag.local import LocalRAG
from vaultkv.txnrag.ollama import OllamaClient


def main() -> None:
    started, revoked = Event(), Event()

    class ObservedOllama(OllamaClient):
        racing = False
        completed_calls = 0

        def generate(self, model, system, prompt):
            if self.racing:
                started.set()
            answer = super().generate(model, system, prompt)
            self.completed_calls += 1
            if self.racing and not revoked.wait(timeout=10):
                raise AssertionError("Revocation failed to commit during the generation phase")
            return answer

    principal = Identity("smoke", "alice")
    resource = Resource("smoke", "launch")
    client = ObservedOllama()
    with TemporaryDirectory(prefix="txnrag-live-") as directory:
        path = Path(directory) / "test.sqlite3"
        with TxnRAG(path) as reader, TxnRAG(path) as writer:
            reader.publish(
                {resource: {"c": "The internal project launch code is ORION-742."}},
                expected_epoch=0,
            )
            reader.set_permissions(resource, expected_epoch=1, principals=[principal])
            rag = LocalRAG(reader, client)
            output = []
            question = "What is the internal project launch code?"
            rag.ask(principal, question, output.append, top_k=1)
            baseline = json.loads(output[0])
            if "ORION-742" not in baseline["answer"]:
                raise AssertionError("The model failed to answer the grounded control question")
            client.racing = True
            with ThreadPoolExecutor(max_workers=2) as pool:
                request = pool.submit(rag.ask, principal, question, output.append, top_k=1)
                if not started.wait(timeout=180):
                    request.result(timeout=1)
                    raise AssertionError("Real generation never started")
                mutation = pool.submit(writer.set_permissions, resource, expected_epoch=2)
                try:
                    mutation.result(timeout=10)
                finally:
                    revoked.set()
                try:
                    request.result(timeout=180)
                except TxnRejected as exc:
                    reason = exc.reason
                else:
                    raise AssertionError("Revoked generation was not blocked")
            if len(output) != 1 or client.completed_calls != 2:
                raise AssertionError("Unexpected output count or incomplete model execution")
            print(
                json.dumps(
                    {
                        "mode": "real-local-ollama-smoke",
                        "model": rag.model.name,
                        "control_answer": baseline["answer"],
                        "real_generation_calls": client.completed_calls,
                        "revoked_request": reason,
                        "protected_outputs_after_revocation": len(output) - 1,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
