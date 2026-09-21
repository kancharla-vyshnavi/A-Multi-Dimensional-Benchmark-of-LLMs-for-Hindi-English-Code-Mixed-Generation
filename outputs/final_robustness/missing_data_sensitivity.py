import pandas as pd

INPUT_FILE = "outputs/independent_judge/independent_mistral_judge_v3_results.csv"
OUTPUT_FILE = "outputs/final_robustness_v3/missing_data_sensitivity_v3.csv"

df = pd.read_csv(INPUT_FILE)

score_cols = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Original: use only available judge scores
original = (
    df.groupby("model")["overall"]
    .mean()
    .rename("overall_original")
)

# Conservative sensitivity: treat every missing score as 1/5
sensitivity_df = df.copy()
sensitivity_df["overall"] = sensitivity_df["overall"].fillna(1)

worst_case = (
    sensitivity_df.groupby("model")["overall"]
    .mean()
    .rename("overall_worst_case")
)

result = pd.concat([original, worst_case], axis=1)

result["change"] = (
    result["overall_worst_case"]
    - result["overall_original"]
)

result["missing_rows"] = (
    df.groupby("model")["overall"]
    .apply(lambda x: x.isna().sum())
)

result = result.reset_index()

print("\n=== MISSING-DATA SENSITIVITY ===")
print(result.to_string(index=False))

print("\nMissing Mistral overall scores by model:")
print(
    df.groupby("model")["overall"]
    .apply(lambda x: x.isna().sum())
)

result.to_csv(OUTPUT_FILE, index=False)

print("\nCreated:")
print(OUTPUT_FILE)
