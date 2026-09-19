import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"
)

OUTPUT_DIR = Path(r".\outputs\statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRIEDMAN_FILE = OUTPUT_DIR / "friedman_results.csv"
POSTHOC_FILE = OUTPUT_DIR / "wilcoxon_holm_results.csv"


# ============================================================
# Load FINAL LLM judge data
# ============================================================

df = pd.read_csv(INPUT_FILE)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]

criteria = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]


# ============================================================
# Validation
# ============================================================

required_columns = ["model", "sample_id"] + criteria

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


for model in models:
    n = len(df[df["model"] == model])

    if n != 34:
        raise ValueError(
            f"{model} has {n} samples instead of expected 34."
        )


print("\nVALIDATION")
print("=" * 70)
print(f"Total rows: {len(df)}")

for model in models:
    print(
        f"{model}: "
        f"{len(df[df['model'] == model])} samples"
    )


# ============================================================
# Friedman + Kendall's W
# ============================================================

friedman_rows = []

for criterion in criteria:

    pivot = df.pivot(
        index="sample_id",
        columns="model",
        values=criterion
    )

    # Ensure same model order
    pivot = pivot[models]

    # Remove incomplete rows if any
    pivot = pivot.dropna()

    if len(pivot) < 2:
        raise ValueError(
            f"Not enough complete samples for {criterion}"
        )

    arrays = [
        pivot[model].to_numpy()
        for model in models
    ]

    statistic, p_value = friedmanchisquare(*arrays)

    n = len(pivot)
    k = len(models)

    # Kendall's W derived from Friedman statistic
    kendall_w = statistic / (n * (k - 1))

    friedman_rows.append({
        "criterion": criterion,
        "N": n,
        "k": k,
        "friedman_chi_square": statistic,
        "friedman_p": p_value,
        "kendall_W": kendall_w,
        "significant_alpha_0.05": p_value < 0.05
    })


friedman_df = pd.DataFrame(friedman_rows)

friedman_df.to_csv(
    FRIEDMAN_FILE,
    index=False
)


# ============================================================
# Pairwise Wilcoxon + Holm
# ============================================================

pairwise_rows = []

pairs = [
    ("HingGPT", "Phi-3.5-mini"),
    ("HingGPT", "Qwen2.5-3B"),
    ("HingGPT", "Qwen2.5-7B"),
    ("Phi-3.5-mini", "Qwen2.5-3B"),
    ("Phi-3.5-mini", "Qwen2.5-7B"),
    ("Qwen2.5-3B", "Qwen2.5-7B"),
]


for criterion in criteria:

    pivot = df.pivot(
        index="sample_id",
        columns="model",
        values=criterion
    )

    pivot = pivot[models].dropna()

    raw_results = []

    for model_a, model_b in pairs:

        x = pivot[model_a].to_numpy()
        y = pivot[model_b].to_numpy()

        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        raw_results.append({
            "criterion": criterion,
            "model_a": model_a,
            "model_b": model_b,
            "N": len(x),
            "wilcoxon_statistic": result.statistic,
            "raw_p": result.pvalue
        })

    # --------------------------------------------------------
    # Holm correction across the six comparisons
    # for this criterion
    # --------------------------------------------------------

    p_values = [
        row["raw_p"]
        for row in raw_results
    ]

    _, adjusted_p, _, _ = multipletests(
        p_values,
        method="holm"
    )

    for row, adj_p in zip(raw_results, adjusted_p):

        row["holm_adjusted_p"] = adj_p
        row["significant_alpha_0.05"] = adj_p < 0.05

        pairwise_rows.append(row)


posthoc_df = pd.DataFrame(pairwise_rows)

posthoc_df.to_csv(
    POSTHOC_FILE,
    index=False
)


# ============================================================
# Print results
# ============================================================

print("\n\nFRIEDMAN RESULTS")
print("=" * 100)
print(
    friedman_df.to_string(index=False)
)

print("\n\nWILCOXON + HOLM RESULTS")
print("=" * 100)
print(
    posthoc_df.to_string(index=False)
)

print("\n\nFILES SAVED")
print("=" * 100)
print(FRIEDMAN_FILE)
print(POSTHOC_FILE)