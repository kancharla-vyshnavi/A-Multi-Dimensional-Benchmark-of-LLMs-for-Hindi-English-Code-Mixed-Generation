import pandas as pd
from scipy.stats import binomtest

INPUT = r".\outputs\pairwise_judge\pairwise_all_judgments_parsed.csv"
OUTPUT = r".\outputs\pairwise_judge\pairwise_significance.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 70)
print("PAIRWISE SIGNIFICANCE TEST")
print("=" * 70)

print()
print("Total judgments:", len(df))


# ============================================================
# CALCULATE WINS / LOSSES / TIES
# ============================================================

results = []

for (model_1, model_2), group in df.groupby(
    ["pair_model_1", "pair_model_2"]
):

    wins_1 = 0
    wins_2 = 0
    ties = 0

    for _, row in group.iterrows():

        choice = str(row["overall"]).strip().upper()

        model_a = row["displayed_A_model"]
        model_b = row["displayed_B_model"]

        if choice == "T":

            ties += 1

        elif choice == "A":

            if model_a == model_1:
                wins_1 += 1
            elif model_a == model_2:
                wins_2 += 1

        elif choice == "B":

            if model_b == model_1:
                wins_1 += 1
            elif model_b == model_2:
                wins_2 += 1


    non_ties = wins_1 + wins_2


    # --------------------------------------------------------
    # Two-sided exact binomial test
    # --------------------------------------------------------

    if non_ties > 0:

        test = binomtest(
            wins_1,
            n=non_ties,
            p=0.5,
            alternative="two-sided"
        )

        raw_p = test.pvalue

    else:

        raw_p = 1.0


    results.append({
        "model_1": model_1,
        "model_2": model_2,
        "total_judgments": len(group),
        "wins_model_1": wins_1,
        "wins_model_2": wins_2,
        "ties": ties,
        "non_ties": non_ties,
        "model_1_win_rate_excluding_ties":
            round(
                wins_1 / non_ties * 100,
                2
            ) if non_ties else 0,
        "model_2_win_rate_excluding_ties":
            round(
                wins_2 / non_ties * 100,
                2
            ) if non_ties else 0,
        "raw_p_value": raw_p
    })


result_df = pd.DataFrame(results)


# ============================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================

def holm_correction(p_values, alpha=0.05):

    p_values = list(p_values)

    m = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda x: x[1]
    )

    adjusted = [0.0] * m

    for rank, (original_index, p) in enumerate(
        indexed,
        start=1
    ):

        adjusted_value = (m - rank + 1) * p

        adjusted[original_index] = min(
            adjusted_value,
            1.0
        )


    # Ensure monotonicity
    for i in range(1, m):

        current_sorted_index = indexed[i][0]
        previous_sorted_index = indexed[i - 1][0]

        adjusted[current_sorted_index] = max(
            adjusted[current_sorted_index],
            adjusted[previous_sorted_index]
        )


    significant = [
        p < alpha
        for p in adjusted
    ]

    return adjusted, significant


adjusted_p, significant = holm_correction(
    result_df["raw_p_value"]
)


result_df["holm_adjusted_p"] = adjusted_p

result_df["significant_after_holm"] = significant


# ============================================================
# ROUND P VALUES
# ============================================================

result_df["raw_p_value"] = result_df[
    "raw_p_value"
].round(6)

result_df["holm_adjusted_p"] = result_df[
    "holm_adjusted_p"
].round(6)


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY
# ============================================================

print()

print(
    result_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("SIGNIFICANT AFTER HOLM CORRECTION")
print("=" * 70)

sig = result_df[
    result_df["significant_after_holm"] == True
]

if len(sig) == 0:

    print("No pairwise comparison is significant after Holm correction.")

else:

    print(
        sig.to_string(index=False)
    )


print()
print("=" * 70)
print("OUTPUT FILE")
print("=" * 70)

print(OUTPUT)

print()
print("DONE.")