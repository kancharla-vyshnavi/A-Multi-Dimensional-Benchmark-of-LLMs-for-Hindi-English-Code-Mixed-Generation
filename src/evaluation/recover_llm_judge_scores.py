import pandas as pd
import json
import re
import os

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results.csv"
OUTPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered.csv"

df = pd.read_csv(INPUT)

score_columns = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

recovered = 0
already_valid = 0
invalid = 0


def extract_scores(raw):
    if pd.isna(raw):
        return None

    text = str(raw)

    # Find every JSON-like block containing the six judge fields
    pattern = r'\{.*?\}'

    matches = re.findall(pattern, text, flags=re.DOTALL)

    candidates = []

    for match in matches:
        try:
            obj = json.loads(match)

            if all(field in obj for field in score_columns):
                values = []

                for field in score_columns:
                    value = float(obj[field])

                    if value < 1 or value > 5:
                        raise ValueError

                    values.append(value)

                candidates.append(values)

        except Exception:
            continue

    if candidates:
        # Use the first valid complete score set
        return candidates[0]

    return None


for idx, row in df.iterrows():

    # If overall already exists, keep it
    if pd.notna(row["overall"]):
        already_valid += 1
        continue

    scores = extract_scores(row["judge_raw_output"])

    if scores is not None:

        for col, value in zip(score_columns, scores):
            df.at[idx, col] = value

        recovered += 1

    else:
        invalid += 1


df.to_csv(OUTPUT, index=False)

print("=" * 75)
print("LLM JUDGE SCORE RECOVERY")
print("=" * 75)
print()

print("Original rows:", len(df))
print("Already valid:", already_valid)
print("Recovered:", recovered)
print("Still invalid:", invalid)
print("Final rows:", len(df))
print()

print("Model-wise valid scores:")
print(
    df.groupby("model")["overall"]
      .apply(lambda x: x.notna().sum())
)

print()
print("Model-wise invalid scores:")
print(
    df.groupby("model")["overall"]
      .apply(lambda x: x.isna().sum())
)

print()
print("Output:")
print(OUTPUT)
print()
print("DONE ✅")