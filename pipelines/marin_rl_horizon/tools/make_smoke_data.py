"""Write a tiny arithmetic dataset in the SkyRL row format for the `aime` environment."""

import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

INSTRUCTION = (
    " Please reason step by step. At the very end, output your final answer on its "
    "own line in the exact format: 'Answer: \\boxed{ANSWER}'."
)


def problems(count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        kind = rng.choice(("plus", "minus", "times"))
        answer = {"plus": a + b, "minus": a - b, "times": a * b}[kind]
        rows.append((f"What is {a} {kind} {b}?", str(answer)))
    return rows


def main(path: str) -> None:
    rows = problems(64, seed=0)
    table = pa.table(
        {
            "data_source": ["smoke-arithmetic"] * len(rows),
            "prompt": [[{"role": "user", "content": question + INSTRUCTION}] for question, _ in rows],
            "env_class": ["aime"] * len(rows),
            "reward_spec": [{"method": "rule", "ground_truth": answer} for _, answer in rows],
        }
    )
    pq.write_table(table, path)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main(sys.argv[1])
