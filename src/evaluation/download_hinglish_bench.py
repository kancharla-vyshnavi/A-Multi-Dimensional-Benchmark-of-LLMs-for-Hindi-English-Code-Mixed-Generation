import os
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "data/benchmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("HINGLISH-BENCH DOWNLOAD")
print("=" * 60)

print("Loading Hinglish-Bench...")

dataset = load_dataset("saidutta69/hinglish-bench")

test = dataset["test"]

print(f"Number of prompts: {len(test)}")
print(f"Columns: {test.column_names}")

df = test.to_pandas()

output_file = os.path.join(
    OUTPUT_DIR,
    "hinglish_bench_test.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print()
print("Categories:")
print(df["category_name"].value_counts())

print()
print(f"Saved: {output_file}")
print("DONE.")