import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

INPUT = r"outputs\independent_judge\independent_mistral_judge_v3_results.csv"
OUTDIR = r"outputs\statistical_analysis"

import os
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

# --------------------------------------------------
# 1. Pivot by prompt
# --------------------------------------------------
pivot = df.pivot_table(
    index="prompt",
    columns="model",
    values="overall",
    aggfunc="first"
)

# --------------------------------------------------
# 2. Friedman test
# Only complete prompts across ALL 4 models
# --------------------------------------------------
complete = pivot[models].dropna()

print("\n========================================")
print("V3 FRIEDMAN TEST")
print("========================================")
print("Complete prompts:", len(complete))
print("Expected complete prompts: 27")

stat, p = friedmanchisquare(
    complete["HingGPT"],
    complete["Phi-3.5-mini"],
    complete["Qwen2.5-3B"],
    complete["Qwen2.5-7B"]
)

print(f"Friedman statistic = {stat:.4f}")
print(f"p-value = {p:.8f}")

# Kendall's W
n = len(complete)
k = len(models)

kendall_w = stat / (n * (k - 1))

print(f"Kendall's W = {kendall_w:.4f}")

# --------------------------------------------------
# 3. Pairwise Wilcoxon signed-rank tests
# --------------------------------------------------
pairs = [
    ("HingGPT", "Phi-3.5-mini"),
    ("HingGPT", "Qwen2.5-3B"),
    ("HingGPT", "Qwen2.5-7B"),
    ("Phi-3.5-mini", "Qwen2.5-3B"),
    ("Phi-3.5-mini", "Qwen2.5-7B"),
    ("Qwen2.5-3B", "Qwen2.5-7B"),
]

results = []

for a, b in pairs:

    pair_data = pivot[[a, b]].dropna()

    x = pair_data[a]
    y = pair_data[b]

    try:
        w_stat, p_value = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided"
        )
    except ValueError:
        w_stat, p_value = np.nan, np.nan

    differences = x - y

    # Rank-biserial correlation
    nonzero = differences[differences != 0]

    if len(nonzero) > 0:
        abs_diff = nonzero.abs()
        ranks = abs_diff.rank(method="average")

        positive_rank_sum = ranks[nonzero > 0].sum()
        negative_rank_sum = ranks[nonzero < 0].sum()

        rbc = (
            positive_rank_sum - negative_rank_sum
        ) / (
            positive_rank_sum + negative_rank_sum
        )
    else:
        rbc = 0.0

    results.append({
        "model_a": a,
        "model_b": b,
        "n_pairs": len(pair_data),
        "wilcoxon_statistic": w_stat,
        "raw_p": p_value,
        "rank_biserial": rbc,
        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "median_a": x.median(),
        "median_b": y.median()
    })

pairwise = pd.DataFrame(results)

# --------------------------------------------------
# 4. Holm correction
# --------------------------------------------------
valid = pairwise["raw_p"].notna()

reject, p_holm, _, _ = multipletests(
    pairwise.loc[valid, "raw_p"],
    method="holm"
)

pairwise.loc[valid, "holm_p"] = p_holm
pairwise.loc[valid, "significant_after_holm"] = reject

# --------------------------------------------------
# 5. Save outputs
# --------------------------------------------------
pairwise.to_csv(
    os.path.join(
        OUTDIR,
        "v3_wilcoxon_holm_results.csv"
    ),
    index=False
)

friedman_summary = pd.DataFrame([{
    "test": "Friedman",
    "n_complete_prompts": n,
    "n_models": k,
    "statistic": stat,
    "p_value": p,
    "kendall_w": kendall_w
}])

friedman_summary.to_csv(
    os.path.join(
        OUTDIR,
        "v3_friedman_results.csv"
    ),
    index=False
)

# --------------------------------------------------
# 6. Console output
# --------------------------------------------------
print("\n========================================")
print("PAIRWISE WILCOXON RESULTS")
print("========================================")

print(
    pairwise[
        [
            "model_a",
            "model_b",
            "n_pairs",
            "raw_p",
            "holm_p",
            "rank_biserial",
            "significant_after_holm"
        ]
    ].round(4).to_string(index=False)
)

print("\n========================================")
print("FILES CREATED")
print("========================================")

print(
    os.path.join(
        OUTDIR,
        "v3_friedman_results.csv"
    )
)

print(
    os.path.join(
        OUTDIR,
        "v3_wilcoxon_holm_results.csv"
    )
)
