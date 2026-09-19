import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    r".\outputs\pairwise_judge\pairwise_all_judgments_parsed.csv"
)

OUTPUT_DIR = Path(r".\outputs\pairwise_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_FILE = OUTPUT_DIR / "pairwise_overall_results.csv"
CRITERION_FILE = OUTPUT_DIR / "pairwise_criterion_results.csv"
SIGNIFICANCE_FILE = OUTPUT_DIR / "pairwise_significance.csv"
MATRIX_FILE = OUTPUT_DIR / "pairwise_preference_matrix.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

criteria = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

required = [
    "sample_id",
    "pair_model_1",
    "pair_model_2",
    "displayed_A_model",
    "displayed_B_model",
    "position_swapped",
] + criteria

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


print("\nVALIDATION")
print("=" * 80)
print("Rows:", len(df))

if len(df) != 204:
    raise ValueError(
        f"Expected 204 pairwise judgments, found {len(df)}."
    )

print("All 204 pairwise judgments found.")


# ============================================================
# Validate A/B/T values
# ============================================================

valid_values = {"A", "B", "T"}

for criterion in criteria:

    values = set(
        df[criterion]
        .dropna()
        .astype(str)
        .str.strip()
    )

    invalid = values - valid_values

    if invalid:
        raise ValueError(
            f"{criterion} contains invalid values: {invalid}"
        )

    missing_count = df[criterion].isna().sum()

    if missing_count:
        raise ValueError(
            f"{criterion} has {missing_count} missing values."
        )


# ============================================================
# Model pairs
# ============================================================

pairs = sorted(
    set(
        tuple(sorted((a, b)))
        for a, b in zip(
            df["pair_model_1"],
            df["pair_model_2"]
        )
    )
)

print("\nMODEL PAIRS")
print("=" * 80)

for pair in pairs:
    count = (
        (df["pair_model_1"].eq(pair[0]) &
         df["pair_model_2"].eq(pair[1]))
        |
        (df["pair_model_1"].eq(pair[1]) &
         df["pair_model_2"].eq(pair[0]))
    ).sum()

    print(pair[0], "vs", pair[1], ":", count)


if len(pairs) != 6:
    raise ValueError(
        f"Expected 6 model pairs, found {len(pairs)}."
    )


# ============================================================
# Function: convert displayed A/B/T to pair_model_1/2
# ============================================================

def convert_result(row, value):

    value = str(value).strip()

    if value == "T":
        return "T"

    if value not in {"A", "B"}:
        raise ValueError(f"Invalid value: {value}")

    displayed_model = (
        row["displayed_A_model"]
        if value == "A"
        else row["displayed_B_model"]
    )

    if displayed_model == row["pair_model_1"]:
        return "model_1"

    if displayed_model == row["pair_model_2"]:
        return "model_2"

    raise ValueError(
        f"Displayed model {displayed_model} "
        f"does not belong to pair."
    )


# ============================================================
# Calculate criterion-level results
# ============================================================

criterion_rows = []

for criterion in criteria:

    for model_a, model_b in pairs:

        subset = df[
            (
                (df["pair_model_1"] == model_a) &
                (df["pair_model_2"] == model_b)
            )
            |
            (
                (df["pair_model_1"] == model_b) &
                (df["pair_model_2"] == model_a)
            )
        ].copy()

        model_a_wins = 0
        model_b_wins = 0
        ties = 0

        for _, row in subset.iterrows():

            result = convert_result(
                row,
                row[criterion]
            )

            if result == "model_1":
                winner = row["pair_model_1"]
            elif result == "model_2":
                winner = row["pair_model_2"]
            else:
                winner = "tie"

            if winner == model_a:
                model_a_wins += 1

            elif winner == model_b:
                model_b_wins += 1

            else:
                ties += 1

        n = len(subset)
        non_ties = model_a_wins + model_b_wins

        if non_ties > 0:

            win_rate_a = model_a_wins / non_ties

            p_value = binomtest(
                model_a_wins,
                non_ties,
                0.5,
                alternative="two-sided"
            ).pvalue

        else:

            win_rate_a = np.nan
            p_value = np.nan

        criterion_rows.append({
            "criterion": criterion,
            "model_a": model_a,
            "model_b": model_b,
            "N": n,
            "model_a_wins": model_a_wins,
            "model_b_wins": model_b_wins,
            "ties": ties,
            "non_ties": non_ties,
            "model_a_win_rate_non_ties": win_rate_a,
            "model_b_win_rate_non_ties": (
                model_b_wins / non_ties
                if non_ties > 0 else np.nan
            ),
            "tie_rate": ties / n,
            "raw_p": p_value,
        })


criterion_df = pd.DataFrame(criterion_rows)


# ============================================================
# Holm correction within each criterion
# ============================================================

criterion_df["holm_adjusted_p"] = np.nan
criterion_df["significant_alpha_0.05"] = False

for criterion in criteria:

    mask = criterion_df["criterion"] == criterion

    pvals = criterion_df.loc[mask, "raw_p"].to_numpy()

    adjusted = multipletests(
        pvals,
        method="holm"
    )[1]

    criterion_df.loc[
        mask,
        "holm_adjusted_p"
    ] = adjusted

    criterion_df.loc[
        mask,
        "significant_alpha_0.05"
    ] = adjusted < 0.05


# ============================================================
# Save criterion results
# ============================================================

criterion_df.to_csv(
    CRITERION_FILE,
    index=False
)


# ============================================================
# Overall pairwise summary
# ============================================================

overall = criterion_df[
    criterion_df["criterion"] == "overall"
].copy()

overall.to_csv(
    OVERALL_FILE,
    index=False
)


# ============================================================
# Preference matrix
# ============================================================

models = sorted(
    set(df["pair_model_1"]) |
    set(df["pair_model_2"])
)

matrix = pd.DataFrame(
    0,
    index=models,
    columns=models,
    dtype=int
)

for _, row in overall.iterrows():

    a = row["model_a"]
    b = row["model_b"]

    matrix.loc[a, b] = int(row["model_a_wins"])
    matrix.loc[b, a] = int(row["model_b_wins"])

matrix.to_csv(MATRIX_FILE)


# ============================================================
# Overall significance table
# ============================================================

significance = overall[
    [
        "model_a",
        "model_b",
        "N",
        "model_a_wins",
        "model_b_wins",
        "ties",
        "raw_p",
        "holm_adjusted_p",
        "significant_alpha_0.05",
    ]
].copy()

significance.to_csv(
    SIGNIFICANCE_FILE,
    index=False
)


# ============================================================
# Print results
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n\nOVERALL PAIRWISE RESULTS")
print("=" * 120)
print(
    overall.to_string(index=False)
)

print("\n\nCRITERION-LEVEL PAIRWISE RESULTS")
print("=" * 120)
print(
    criterion_df.to_string(index=False)
)

print("\n\nPREFERENCE MATRIX")
print("=" * 80)
print(matrix)


print("\n\nFILES SAVED")
print("=" * 80)
print(OVERALL_FILE)
print(CRITERION_FILE)
print(SIGNIFICANCE_FILE)
print(MATRIX_FILE)