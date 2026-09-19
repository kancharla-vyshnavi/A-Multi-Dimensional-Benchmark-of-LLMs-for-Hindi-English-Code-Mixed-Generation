import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

INPUT = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"
OUTDIR = r".\outputs\llm_judge"

df = pd.read_csv(INPUT)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

criteria = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

# Numeric conversion
for c in criteria:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("=" * 70)
print("FINAL LLM-JUDGE STATISTICS")
print("=" * 70)

# ---------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------

rows = []

for model in models:
    sub = df[df["model"] == model]

    for criterion in criteria:
        x = sub[criterion].dropna().astype(float)

        mean = x.mean()
        median = x.median()
        sd = x.std(ddof=1)
        n = len(x)

        ci_low = mean - 1.96 * sd / np.sqrt(n)
        ci_high = mean + 1.96 * sd / np.sqrt(n)

        rows.append({
            "model": model,
            "criterion": criterion,
            "N": n,
            "mean": mean,
            "median": median,
            "SD": sd,
            "CI95_low": ci_low,
            "CI95_high": ci_high
        })

desc = pd.DataFrame(rows)

desc.to_csv(
    OUTDIR + r"\final_llm_judge_descriptive_statistics.csv",
    index=False
)

print("\nDESCRIPTIVE STATISTICS")
print(desc.to_string(index=False))

# ---------------------------------------------------------
# 2. Overall means
# ---------------------------------------------------------

overall = (
    df.groupby("model")["overall"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

overall.columns = [
    "model", "N", "mean_overall",
    "median_overall", "SD_overall"
]

overall.to_csv(
    OUTDIR + r"\final_llm_judge_overall_summary.csv",
    index=False
)

print("\nOVERALL SUMMARY")
print(overall.to_string(index=False))

# ---------------------------------------------------------
# 3. Friedman test
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FRIEDMAN TEST")
print("=" * 70)

friedman_rows = []

for criterion in criteria:

    pivot = df.pivot_table(
        index="sample_id",
        columns="model",
        values=criterion,
        aggfunc="first"
    )

    pivot = pivot[models].dropna()

    if len(pivot) >= 3:

        stat, p = friedmanchisquare(
            pivot[models[0]],
            pivot[models[1]],
            pivot[models[2]],
            pivot[models[3]]
        )

        n = len(pivot)

        # Kendall's W
        W = stat / (n * (len(models) - 1))

        friedman_rows.append({
            "criterion": criterion,
            "N_complete": n,
            "Friedman_statistic": stat,
            "p_value": p,
            "Kendall_W": W
        })

        print(
            f"{criterion}: "
            f"N={n}, "
            f"χ²={stat:.4f}, "
            f"p={p:.6f}, "
            f"W={W:.4f}"
        )

friedman_df = pd.DataFrame(friedman_rows)

friedman_df.to_csv(
    OUTDIR + r"\final_friedman_results.csv",
    index=False
)

# ---------------------------------------------------------
# 4. Pairwise Wilcoxon + Holm
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("POST-HOC WILCOXON + HOLM")
print("=" * 70)

def holm_correction(pvalues):

    pvalues = np.array(pvalues, dtype=float)

    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))

    for rank, idx in enumerate(order):
        adjusted[idx] = min(
            1.0,
            (len(pvalues) - rank) * pvalues[idx]
        )

    # enforce monotonicity
    for i in range(1, len(order)):
        adjusted[order[i]] = max(
            adjusted[order[i]],
            adjusted[order[i - 1]]
        )

    return adjusted


for criterion in criteria:

    print(f"\n--- {criterion} ---")

    comparisons = []

    for m1, m2 in combinations(models, 2):

        pivot = df.pivot_table(
            index="sample_id",
            columns="model",
            values=criterion,
            aggfunc="first"
        )

        pivot = pivot[[m1, m2]].dropna()

        if len(pivot) == 0:
            continue

        try:
            stat, p = wilcoxon(
                pivot[m1],
                pivot[m2],
                zero_method="wilcox",
                alternative="two-sided"
            )
        except ValueError:
            stat, p = np.nan, 1.0

        comparisons.append({
            "criterion": criterion,
            "model1": m1,
            "model2": m2,
            "N": len(pivot),
            "W": stat,
            "raw_p": p
        })

    if comparisons:

        pvals = [x["raw_p"] for x in comparisons]
        adjusted = holm_correction(pvals)

        for x, hp in zip(comparisons, adjusted):
            x["Holm_p"] = hp

            print(
                f"{x['model1']} vs {x['model2']}: "
                f"N={x['N']}, "
                f"raw p={x['raw_p']:.6f}, "
                f"Holm p={hp:.6f}"
            )

        pd.DataFrame(comparisons).to_csv(
            OUTDIR + f"\\final_wilcoxon_{criterion}.csv",
            index=False
        )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nFiles created:")
print("final_llm_judge_descriptive_statistics.csv")
print("final_llm_judge_overall_summary.csv")
print("final_friedman_results.csv")
print("final_wilcoxon_<criterion>.csv")