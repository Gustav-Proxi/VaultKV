"""Compare authorization boundaries with deterministic revocation injection."""

import json

from vaultkv.txnrag.races import run_race_matrix

if __name__ == "__main__":
    print(json.dumps(run_race_matrix(), indent=2))
