import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HUMAN_FILE = "outputs/human_validation/human_validation_results.csv"
MISTRAL_FILE = "outputs/independent_judge/independent_mistral_judge_results.csv"

OUTPUT_DIR = "outputs/human_validation_v3"

AGREEMENT_FILE = f"{OUTPUT_DIR}/human_mistral_agreement_v3.csv"
MERGED_FILE = f"{OUTPUT_DIR}/human_mistral_merged_v3.csv"
SUMMARY_FILE = f"{OUTPUT_DIR}/human_validation_summary_v3.csv"

human = pd.read_csv(HUMAN_FILE)
mistral = pd.read_csv(MISTRAL_FILE)

human_columns = [
    "human_fluency",
    "human_code_mixing_naturalness",
    "human_hindi_grammar",
    "human_prompt_adherence",
    "human_spelling_consistency",
    "human_overall",
]

print("\n=== HUMAN VALIDATION SUMMARY ===")

human_summary = (
    human.groupby("model")[human_columns]
    .mean()
    .round(3)
)

print(human_summary)

merged = human.merge(
    mistral,
    on=["model", "prompt"],
    how="left",
    suffixes=("_h", "_m"),
)

print("\n=== MERGE CHECK ===")
print("Human rows:", len(human))
print("Merged rows:", len(merged))
print("Matched Mistral rows:", merged["judge_model"].notna().sum())

criteria = {
    "fluency": ("human_fluency", "fluency"),
    "code_mixing_naturalness": (
        "human_code_mixing_naturalness",
        "code_mixing_naturalness",
    ),
    "hindi_grammar": (
        "human_hindi_grammar",
        "hindi_grammar",
    ),
    "prompt_adherence": (
        "human_prompt_adherence",
        "prompt_adherence",
    ),
    "spelling_consistency": (
        "human_spelling_consistency",
        "spelling_consistency",
    ),
    "overall": (
        "human_overall",
        "overall",
    ),
}

agreement_rows = []

print("\n=== HUMAN vs MISTRAL AGREEMENT ===")

for criterion, (human_col, mistral_col) in criteria.items():

    # IMPORTANT:
    # Human columns remain unchanged after merge.
    # Mistral columns also remain unchanged because their
    # names are different.

    h = pd.to_numeric(
        merged[human_col],
        errors="coerce"
    )

    m = pd.to_numeric(
        merged[mistral_col],
        errors="coerce"
    )

    valid = (
        h.between(1, 5)
        & m.between(1, 5)
    )

    h_valid = h[valid].astype(int)
    m_valid = m[valid].astype(int)

    n = len(h_valid)

    if n >= 2:

        rho, p_value = spearmanr(
            h_valid,
            m_valid
        )

        kappa = cohen_kappa_score(
            h_valid,
            m_valid,
            weights="quadratic"
        )

        human_mean = h_valid.mean()
        mistral_mean = m_valid.mean()

        mean_difference = (
            human_mean - mistral_mean
        )

    else:

        rho = np.nan
        p_value = np.nan
        kappa = np.nan
        human_mean = np.nan
        mistral_mean = np.nan
        mean_difference = np.nan

    agreement_rows.append({
        "criterion": criterion,
        "n_paired": n,
        "human_mean": round(human_mean, 4)
        if not pd.isna(human_mean)
        else np.nan,
        "mistral_mean": round(mistral_mean, 4)
        if not pd.isna(mistral_mean)
        else np.nan,
        "mean_difference_human_minus_mistral":
            round(mean_difference, 4)
            if not pd.isna(mean_difference)
            else np.nan,
        "spearman_rho":
            round(rho, 4)
            if not pd.isna(rho)
            else np.nan,
        "spearman_p":
            round(p_value, 4)
            if not pd.isna(p_value)
            else np.nan,
        "weighted_cohen_kappa_quadratic":
            round(kappa, 4)
            if not pd.isna(kappa)
            else np.nan,
    })

agreement_df = pd.DataFrame(agreement_rows)

merged.to_csv(
    MERGED_FILE,
    index=False
)

agreement_df.to_csv(
    AGREEMENT_FILE,
    index=False
)

human_summary.to_csv(
    SUMMARY_FILE
)

print(
    agreement_df.to_string(index=False)
)

print("\nCreated:")
print(SUMMARY_FILE)
print(AGREEMENT_FILE)
print(MERGED_FILE)
