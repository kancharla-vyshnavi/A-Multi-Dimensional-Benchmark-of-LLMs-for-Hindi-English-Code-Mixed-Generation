import csv
from pathlib import Path

p = Path(r"data\benchmarks\hinglish_bench_1000.csv")
tmp = p.with_suffix(".tmp.csv")

rows = []

with p.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for r in reader:
        source = r["source_text"].strip()

        prompt = (
            "Respond naturally to the following real Hinglish dataset utterance. "
            "Preserve the intended meaning, use natural Hindi-English code-mixing, "
            "and avoid unnecessary English-only or Hindi-only rewriting.\n\n"
            "Source utterance:\n"
            f"{source}"
        )

        rows.append({
            "prompt_id": r["prompt_id"],
            "row_index": r["row_index"],
            "prompt": prompt,
            "source_text": source,
        })

with tmp.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["prompt_id", "row_index", "prompt", "source_text"]
    )
    writer.writeheader()
    writer.writerows(rows)

tmp.replace(p)

print("Updated benchmark:", p)
print("Rows:", len(rows))
print("Columns: prompt_id, row_index, prompt, source_text")
