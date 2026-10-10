"""Run a deterministic synthetic mutation stream; no GPU performance claims."""

import argparse
import json

from vaultkv.workload import run_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_workload(steps=args.steps, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
