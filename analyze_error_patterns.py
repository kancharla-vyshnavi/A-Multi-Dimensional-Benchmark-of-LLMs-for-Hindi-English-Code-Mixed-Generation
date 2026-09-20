import pandas as pd
import os

INPUT = r"outputs\independent_judge\independent_mistral_judge_v3_results.csv"
OUTDIR = r"outputs\error_analysis"

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

# --------------------------------------------------
# 1. Missing judge responses
# --------------------------------------------------
missing = (
    df.assign(missing=df["overall"].isna())
      .groupby("model")["missing"]
      .agg(["sum", "count"])
      .rename(columns={"sum": "missing_count", "count": "total"})
)

missing["missing_rate"] = missing["missing_count"] / missing["total"]

print("\n========================================")
print("MISSING JUDGE RESPONSES")
print("========================================")
print(missing.round(4).to_string())

missing.to_csv(
    os.path.join(OUTDIR, "v3_missing_judge_summary.csv")
)

# --------------------------------------------------
# 2. Error flags
# --------------------------------------------------
for col in ["E6_prompt_misunderstanding", "E10_irrelevant_response"]:
    if col not in df.columns:
        print(f"Column not found: {col}")
        continue

    temp = (
        df.groupby("model")[col]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "valid_count",
            "sum": "flag_count",
            "mean": "flag_rate"
        })
    )

    print(f"\n========================================")
    print(col)
    print("========================================")
    print(temp.round(4).to_string())

    temp.to_csv(
        os.path.join(
            OUTDIR,
            f"v3_{col}.csv"
        )
    )

# --------------------------------------------------
# 3. Overall score distribution
# --------------------------------------------------
score_dist = (
    df.dropna(subset=["overall"])
      .groupby(["model", "overall"])
      .size()
      .reset_index(name="count")
)

print("\n========================================")
print("OVERALL SCORE DISTRIBUTION")
print("========================================")
print(score_dist.to_string(index=False))

score_dist.to_csv(
    os.path.join(
        OUTDIR,
        "v3_overall_score_distribution.csv"
    ),
    index=False
)

# --------------------------------------------------
# 4. Low-score responses
# --------------------------------------------------
low_scores = (
    df[df["overall"].notna() & (df["overall"] <= 2)]
    [["model", "prompt", "overall"]]
)

print("\n========================================")
print("LOW-SCORE RESPONSES (OVERALL <= 2)")
print("========================================")
print(
    low_scores.groupby("model")
    .size()
    .rename("low_score_count")
    .to_string()
)

low_scores.to_csv(
    os.path.join(
        OUTDIR,
        "v3_low_score_responses.csv"
    ),
    index=False
)

# --------------------------------------------------
# 5. Dimension-wise low scores
# --------------------------------------------------
dimensions = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

dim_rows = []

for model in models:
    sub = df[df["model"] == model]

    for dim in dimensions:
        valid = sub[dim].dropna()

        dim_rows.append({
            "model": model,
            "dimension": dim,
            "valid_count": len(valid),
            "mean": valid.mean(),
            "score_1_or_2_count": (valid <= 2).sum(),
            "score_1_or_2_rate": (valid <= 2).mean()
        })

dim_df = pd.DataFrame(dim_rows)

print("\n========================================")
print("LOW-SCORE RATE BY DIMENSION")
print("========================================")
print(
    dim_df[
        [
            "model",
            "dimension",
            "mean",
            "score_1_or_2_count",
            "score_1_or_2_rate"
        ]
    ].round(4).to_string(index=False)
)

dim_df.to_csv(
    os.path.join(
        OUTDIR,
        "v3_dimension_error_summary.csv"
    ),
    index=False
)

print("\n========================================")
print("FILES CREATED")
print("========================================")

for f in os.listdir(OUTDIR):
    print(os.path.join(OUTDIR, f))
