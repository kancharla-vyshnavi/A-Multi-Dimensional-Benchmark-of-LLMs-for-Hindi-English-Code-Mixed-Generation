import pandas as pd
import numpy as np
from scipy.stats import t

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered.csv"
OUTPUT = r".\outputs\llm_judge\descriptive_statistics.csv"

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

results = []

for model in MODELS:

    scores = df.loc[
        df["model"] == model,
        "overall"
    ].dropna().astype(float)

    n = len(scores)
    mean = scores.mean()
    median = scores.median()
    sd = scores.std(ddof=1)

    # 95% CI using t-distribution
    if n > 1:
        se = sd / np.sqrt(n)
        margin = t.ppf(
            0.975,
            df=n - 1
        ) * se

        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = np.nan
        ci_high = np.nan

    results.append({
        "Model": model,
        "N_valid": n,
        "Mean": mean,
        "Median": median,
        "SD": sd,
        "95%_CI_Lower": ci_low,
        "95%_CI_Upper": ci_high
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 75)
print("DESCRIPTIVE STATISTICS")
print("=" * 75)
print()

print(result_df.to_string(index=False))

print()
print("=" * 75)
print("OUTPUT")
print("=" * 75)
print(OUTPUT)
print()
print("DONE ✅")