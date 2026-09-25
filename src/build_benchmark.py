import json
import random
import csv
from pathlib import Path

SOURCE = Path("data/processed/cpt/validation.jsonl")
OUTPUT = Path("data/benchmarks/hinglish_bench_1000.csv")

SEED = 42
N = 1000


def main():
    rows = []

    with SOURCE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            text = (
                obj.get("text")
                or obj.get("utterance")
                or obj.get("source_text")
                or obj.get("sentence")
            )

            if text is None:
                continue

            text = str(text).strip()

            if text:
                rows.append({
                    "row_index": i,
                    "source_text": text
                })

    # Remove duplicate utterances
    seen = set()
    unique_rows = []

    for row in rows:
        key = row["source_text"].strip().lower()

        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    print(f"Usable rows: {len(rows)}")
    print(f"Unique rows: {len(unique_rows)}")

    if len(unique_rows) < N:
        raise RuntimeError(
            f"Only {len(unique_rows)} unique rows available; need {N}."
        )

    # Deterministic sample
    rng = random.Random(SEED)
    selected = rng.sample(unique_rows, N)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "row_index", "source_text"]
        )

        writer.writeheader()

        for prompt_id, row in enumerate(selected, start=1):
            writer.writerow({
                "prompt_id": prompt_id,
                "row_index": row["row_index"],
                "source_text": row["source_text"]
            })

    print()
    print(f"Created: {OUTPUT}")
    print(f"Rows written: {N}")

    print("\nFirst 5 rows:")
    for row in selected[:5]:
        print(row)


if __name__ == "__main__":
    main()