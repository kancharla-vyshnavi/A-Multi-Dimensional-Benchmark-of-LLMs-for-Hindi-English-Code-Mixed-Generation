import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


# ============================================================
# CONFIG
# ============================================================

ORIGINAL = (
    "outputs/independent_judge_1000/"
    "independent_mistral_judge_1000_FINAL.csv"
)

PERTURBED = (
    "outputs/robustness_mistral_1000/"
    "mistral_robustness_1000_results.csv"
)

OUTDIR = "outputs/robustness_mistral_1000_analysis"

os.makedirs(OUTDIR, exist_ok=True)


DIMENSIONS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING ROBUSTNESS DATA")
print("=" * 70)

orig = pd.read_csv(ORIGINAL)
pert = pd.read_csv(PERTURBED)

print(f"Original rows   : {len(orig)}")
print(f"Perturbed rows  : {len(pert)}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required = ["row_index", "model"] + DIMENSIONS

for col in required:
    if col not in orig.columns:
        raise ValueError(f"Missing column in ORIGINAL: {col}")

    if col not in pert.columns:
        raise ValueError(f"Missing column in PERTURBED: {col}")


# ============================================================
# PREPARE DATA
# ============================================================

orig2 = orig[required].copy()
pert2 = pert[required].copy()

orig2 = orig2.rename(
    columns={d: f"{d}_original" for d in DIMENSIONS}
)

pert2 = pert2.rename(
    columns={d: f"{d}_perturbed" for d in DIMENSIONS}
)


# ============================================================
# MERGE ORIGINAL + PERTURBED
# ============================================================

df = pd.merge(
    orig2,
    pert2,
    on=["row_index", "model"],
    how="inner",
    validate="one_to_one",
)

print(f"Merged rows    : {len(df)}")

if len(df) != 4000:
    raise ValueError(
        f"Expected 4000 merged rows, got {len(df)}"
    )


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

summary_rows = []

for dimension in DIMENSIONS:

    original_col = f"{dimension}_original"
    perturbed_col = f"{dimension}_perturbed"

    pair = df[[original_col, perturbed_col]].dropna()

    n = len(pair)

    original = pair[original_col].astype(float)
    perturbed = pair[perturbed_col].astype(float)

    if n > 1 and original.nunique() > 1 and perturbed.nunique() > 1:

        spearman, spearman_p = spearmanr(
            original,
            perturbed,
        )

    else:
        spearman = np.nan
        spearman_p = np.nan

    if n > 0:

        qwk = cohen_kappa_score(
            original.astype(int),
            perturbed.astype(int),
            weights="quadratic",
        )

        difference = perturbed - original

        mean_absolute_change = np.mean(
            np.abs(difference)
        )

        signed_mean_change = np.mean(difference)

        original_mean = np.mean(original)
        perturbed_mean = np.mean(perturbed)

    else:

        qwk = np.nan
        mean_absolute_change = np.nan
        signed_mean_change = np.nan
        original_mean = np.nan
        perturbed_mean = np.nan

    summary_rows.append({
        "dimension": dimension,
        "n_paired": n,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "quadratic_weighted_kappa": qwk,
        "mean_absolute_change": mean_absolute_change,
        "signed_mean_change": signed_mean_change,
        "original_mean": original_mean,
        "perturbed_mean": perturbed_mean,
    })


robustness_summary = pd.DataFrame(summary_rows)


# ============================================================
# MODEL-LEVEL CHANGES
# ============================================================

model_rows = []

for model in sorted(df["model"].dropna().unique()):

    model_df = df[df["model"] == model]

    for dimension in DIMENSIONS:

        original_col = f"{dimension}_original"
        perturbed_col = f"{dimension}_perturbed"

        pair = model_df[
            [original_col, perturbed_col]
        ].dropna()

        if len(pair) == 0:
            continue

        original = pair[original_col].astype(float)
        perturbed = pair[perturbed_col].astype(float)

        difference = perturbed - original

        model_rows.append({
            "model": model,
            "dimension": dimension,
            "n": len(pair),
            "original_mean": original.mean(),
            "perturbed_mean": perturbed.mean(),
            "mean_absolute_change": np.abs(
                difference
            ).mean(),
            "signed_mean_change": difference.mean(),
        })


model_changes = pd.DataFrame(model_rows)


# ============================================================
# RANK / ORDER STABILITY
# ============================================================

rank_rows = []

for dimension in DIMENSIONS:

    original_col = f"{dimension}_original"
    perturbed_col = f"{dimension}_perturbed"

    model_means = (
        df.groupby("model")[
            [original_col, perturbed_col]
        ]
        .mean()
        .reset_index()
    )

    model_means["original_rank"] = (
        model_means[original_col]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    model_means["perturbed_rank"] = (
        model_means[perturbed_col]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    if (
        model_means["original_rank"].nunique() > 1
        and
        model_means["perturbed_rank"].nunique() > 1
    ):

        rank_rho, rank_p = spearmanr(
            model_means["original_rank"],
            model_means["perturbed_rank"],
        )

    else:
        rank_rho = np.nan
        rank_p = np.nan

    original_order = list(
        model_means
        .sort_values("original_rank")["model"]
    )

    perturbed_order = list(
        model_means
        .sort_values("perturbed_rank")["model"]
    )

    rank_rows.append({
        "dimension": dimension,
        "rank_spearman": rank_rho,
        "rank_spearman_p": rank_p,
        "exact_rank_order_stable": (
            original_order == perturbed_order
        ),
        "original_order": " > ".join(original_order),
        "perturbed_order": " > ".join(perturbed_order),
    })


rank_stability = pd.DataFrame(rank_rows)


# ============================================================
# BOOTSTRAP 95% CI
# ============================================================

def bootstrap_ci(values, n_boot=5000, seed=42):

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means[i] = np.mean(sample)

    return (
        np.mean(values),
        np.percentile(boot_means, 2.5),
        np.percentile(boot_means, 97.5),
    )


bootstrap_rows = []

for dimension in DIMENSIONS:

    original_col = f"{dimension}_original"
    perturbed_col = f"{dimension}_perturbed"

    pair = df[
        [original_col, perturbed_col]
    ].dropna()

    if len(pair) == 0:
        continue

    difference = (
        pair[perturbed_col].astype(float)
        -
        pair[original_col].astype(float)
    )

    absolute_difference = np.abs(difference)

    signed_mean, signed_low, signed_high = (
        bootstrap_ci(difference)
    )

    absolute_mean, absolute_low, absolute_high = (
        bootstrap_ci(absolute_difference)
    )

    bootstrap_rows.append({
        "dimension": dimension,
        "n": len(pair),

        "signed_mean_change": signed_mean,
        "signed_change_ci_lower": signed_low,
        "signed_change_ci_upper": signed_high,

        "mean_absolute_change": absolute_mean,
        "absolute_change_ci_lower": absolute_low,
        "absolute_change_ci_upper": absolute_high,
    })


bootstrap_results = pd.DataFrame(bootstrap_rows)


# ============================================================
# MISSING DATA AUDIT
# ============================================================

missing_rows = []

for dimension in DIMENSIONS:

    original_col = f"{dimension}_original"
    perturbed_col = f"{dimension}_perturbed"

    missing_rows.append({
        "dimension": dimension,
        "original_missing": int(
            df[original_col].isna().sum()
        ),
        "perturbed_missing": int(
            df[perturbed_col].isna().sum()
        ),
        "paired_complete": int(
            df[
                [original_col, perturbed_col]
            ].dropna().shape[0]
        ),
    })


missing_audit = pd.DataFrame(missing_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

robustness_summary.to_csv(
    f"{OUTDIR}/robustness_summary.csv",
    index=False,
)

model_changes.to_csv(
    f"{OUTDIR}/robustness_model_level_changes.csv",
    index=False,
)

rank_stability.to_csv(
    f"{OUTDIR}/robustness_rank_stability.csv",
    index=False,
)

bootstrap_results.to_csv(
    f"{OUTDIR}/robustness_bootstrap_95ci.csv",
    index=False,
)

missing_audit.to_csv(
    f"{OUTDIR}/robustness_missing_data_audit.csv",
    index=False,
)

df.to_csv(
    f"{OUTDIR}/robustness_merged_original_vs_perturbed.csv",
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)
print(robustness_summary.to_string(index=False))

print("\n" + "=" * 70)
print("MODEL-LEVEL CHANGES")
print("=" * 70)
print(model_changes.to_string(index=False))

print("\n" + "=" * 70)
print("RANK / ORDER STABILITY")
print("=" * 70)
print(rank_stability.to_string(index=False))

print("\n" + "=" * 70)
print("BOOTSTRAP 95% CI")
print("=" * 70)
print(bootstrap_results.to_string(index=False))

print("\n" + "=" * 70)
print("MISSING DATA AUDIT")
print("=" * 70)
print(missing_audit.to_string(index=False))

print("\n" + "=" * 70)
print("ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nOutput folder:")
print(OUTDIR)

print("\nFiles created:")
for filename in sorted(os.listdir(OUTDIR)):
    print(" -", filename)