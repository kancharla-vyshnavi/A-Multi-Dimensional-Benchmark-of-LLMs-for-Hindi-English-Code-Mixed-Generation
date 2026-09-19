import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered.csv"
OUTPUT = r".\outputs\llm_judge\wilcoxon_pairwise_results.csv"

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

df = pd.read_csv(INPUT)

df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

print("=" * 75)
print("RECOVERED DATA CHECK")
print("=" * 75)
print()

print("Total rows:", len(df))
print()
print("Valid scores per model:")
print(
    df.groupby("model")["overall"]
      .apply(lambda x: x.notna().sum())
)
print()

results = []

# --------------------------------------------------
# Pairwise comparisons
# --------------------------------------------------

for model_a, model_b in combinations(models, 2):

    # Pair using sample_id
    a = df[df["model"] == model_a][
        ["sample_id", "overall"]
    ].rename(columns={"overall": "score_a"})

    b = df[df["model"] == model_b][
        ["sample_id", "overall"]
    ].rename(columns={"overall": "score_b"})

    pair = pd.merge(
        a,
        b,
        on="sample_id",
        how="inner"
    )

    # Only samples where BOTH models have valid scores
    pair = pair.dropna(
        subset=["score_a", "score_b"]
    )

    x = pair["score_a"].astype(float).values
    y = pair["score_b"].astype(float).values

    n = len(pair)

    if n == 0:
        statistic = np.nan
        p_value = np.nan
        effect = np.nan

    else:
        differences = x - y
        nonzero = differences[differences != 0]

        if len(nonzero) == 0:
            statistic = 0.0
            p_value = 1.0
            effect = 0.0

        elif len(nonzero) < 5:
            # Too few non-zero differences for a reliable
            # Wilcoxon significance calculation
            statistic = np.nan
            p_value = np.nan

            ranks = pd.Series(
                np.abs(nonzero)
            ).rank(method="average").values

            positive_rank_sum = ranks[nonzero > 0].sum()
            negative_rank_sum = ranks[nonzero < 0].sum()

            effect = (
                (positive_rank_sum - negative_rank_sum)
                /
                (positive_rank_sum + negative_rank_sum)
            )

        else:
            statistic, p_value = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto"
            )

            ranks = pd.Series(
                np.abs(nonzero)
            ).rank(method="average").values

            positive_rank_sum = ranks[nonzero > 0].sum()
            negative_rank_sum = ranks[nonzero < 0].sum()

            effect = (
                (positive_rank_sum - negative_rank_sum)
                /
                (positive_rank_sum + negative_rank_sum)
            )

    results.append({
        "Model_A": model_a,
        "Model_B": model_b,
        "N_Paired": n,
        "Wilcoxon_statistic": statistic,
        "Raw_p_value": p_value,
        "Rank_biserial_effect": effect
    })

results_df = pd.DataFrame(results)

# --------------------------------------------------
# Holm correction
# Only adjust valid p-values
# --------------------------------------------------

valid_p = results_df["Raw_p_value"].notna()

p_values = results_df.loc[
    valid_p,
    "Raw_p_value"
]

m = len(p_values)

if m > 0:

    sorted_indices = (
        p_values
        .sort_values()
        .index
    )

    adjusted = pd.Series(
        index=p_values.index,
        dtype=float
    )

    previous = 0.0

    for rank, idx in enumerate(sorted_indices):

        adjusted_value = (
            (m - rank)
            * p_values.loc[idx]
        )

        adjusted_value = max(
            adjusted_value,
            previous
        )

        adjusted_value = min(
            adjusted_value,
            1.0
        )

        adjusted.loc[idx] = adjusted_value
        previous = adjusted_value

    results_df["Holm_adjusted_p"] = np.nan

    for idx in adjusted.index:
        results_df.loc[
            idx,
            "Holm_adjusted_p"
        ] = adjusted.loc[idx]

else:
    results_df["Holm_adjusted_p"] = np.nan

results_df["Significant_0.05"] = (
    results_df["Holm_adjusted_p"] < 0.05
)

# --------------------------------------------------
# Save
# --------------------------------------------------

results_df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 75)
print("PAIRWISE WILCOXON RESULTS")
print("=" * 75)
print()

print(
    results_df.to_string(index=False)
)

print()
print("=" * 75)
print("OUTPUT")
print("=" * 75)
print(OUTPUT)
print()
print("DONE ✅")