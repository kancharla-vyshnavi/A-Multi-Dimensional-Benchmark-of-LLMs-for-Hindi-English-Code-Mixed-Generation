import pandas as pd
from scipy.stats import friedmanchisquare

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered.csv"

MODELS = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

df = pd.read_csv(INPUT)

df["overall"] = pd.to_numeric(
    df["overall"],
    errors="coerce"
)

# Pivot: one row = sample_id
pivot = df.pivot(
    index="sample_id",
    columns="model",
    values="overall"
)

print("=" * 75)
print("FRIEDMAN TEST - COMPLETE CASES")
print("=" * 75)

# Keep ONLY samples having valid scores for all 4 models
complete = pivot.dropna(
    subset=MODELS
)

print()
print("Total samples:", len(pivot))
print("Complete paired samples:", len(complete))
print()

print("Samples used:")
print(list(complete.index))
print()

# Friedman test
statistic, p_value = friedmanchisquare(
    complete["HingGPT"],
    complete["Phi-3.5-mini"],
    complete["Qwen2.5-3B"],
    complete["Qwen2.5-7B"]
)

# Kendall's W effect size
k = len(MODELS)
n = len(complete)

kendall_w = statistic / (n * (k - 1))

print("=" * 75)
print("RESULT")
print("=" * 75)

print()
print(f"Friedman statistic: {statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Kendall's W: {kendall_w:.4f}")
print()

if p_value < 0.001:
    print("Interpretation: p < 0.001")
elif p_value < 0.01:
    print("Interpretation: p < 0.01")
elif p_value < 0.05:
    print("Interpretation: p < 0.05")
else:
    print("Interpretation: p >= 0.05")

print()
print("=" * 75)
print("MODEL MEANS ON COMPLETE CASES")
print("=" * 75)

print(
    complete[MODELS].mean()
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)