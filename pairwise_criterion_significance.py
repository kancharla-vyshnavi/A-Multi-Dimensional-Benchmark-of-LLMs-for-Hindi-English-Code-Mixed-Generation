import pandas as pd
from scipy.stats import binomtest


INPUT = r".\outputs\pairwise_judge\pairwise_all_judgments_parsed.csv"

OUTPUT = r".\outputs\pairwise_judge\pairwise_criterion_significance.csv"


CRITERIA = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 75)
print("CRITERION-WISE PAIRWISE SIGNIFICANCE TEST")
print("=" * 75)

print()
print("Total judgments:", len(df))


# ============================================================
# COLLECT RESULTS
# ============================================================

results = []


for criterion in CRITERIA:

    for (model_1, model_2), group in df.groupby(
        ["pair_model_1", "pair_model_2"]
    ):

        wins_1 = 0
        wins_2 = 0
        ties = 0


        for _, row in group.iterrows():

            choice = str(
                row[criterion]
            ).strip().upper()

            model_a = row[
                "displayed_A_model"
            ]

            model_b = row[
                "displayed_B_model"
            ]


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


        # ----------------------------------------------------
        # Exact binomial test
        # ----------------------------------------------------

        if non_ties > 0:

            test = binomtest(
                wins_1,
                n=non_ties,
                p=0.5,
                alternative="two-sided"
            )

            p_value = test.pvalue

        else:

            p_value = 1.0


        results.append({

            "criterion": criterion,

            "model_1": model_1,

            "model_2": model_2,

            "total": len(group),

            "model_1_wins": wins_1,

            "model_2_wins": wins_2,

            "ties": ties,

            "non_ties": non_ties,

            "model_1_win_percent":
                round(
                    wins_1 /
                    non_ties * 100,
                    2
                )
                if non_ties else 0,

            "model_2_win_percent":
                round(
                    wins_2 /
                    non_ties * 100,
                    2
                )
                if non_ties else 0,

            "tie_percent":
                round(
                    ties /
                    len(group) * 100,
                    2
                ),

            "raw_p_value": p_value

        })


result_df = pd.DataFrame(results)


# ============================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================

def holm_correction(
    p_values,
    alpha=0.05
):

    p_values = list(p_values)

    m = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda x: x[1]
    )


    adjusted = [0.0] * m


    for rank, (
        original_index,
        p
    ) in enumerate(
        indexed,
        start=1
    ):

        adjusted_value = (
            m - rank + 1
        ) * p

        adjusted[
            original_index
        ] = min(
            adjusted_value,
            1.0
        )


    # --------------------------------------------------------
    # Enforce monotonicity
    # --------------------------------------------------------

    for i in range(
        1,
        m
    ):

        current_index = indexed[i][0]

        previous_index = indexed[i - 1][0]

        adjusted[
            current_index
        ] = max(
            adjusted[current_index],
            adjusted[previous_index]
        )


    significant = [
        p < alpha
        for p in adjusted
    ]


    return adjusted, significant


adjusted_p, significant = holm_correction(
    result_df["raw_p_value"]
)


result_df[
    "holm_adjusted_p"
] = adjusted_p


result_df[
    "significant_after_holm"
] = significant


# ============================================================
# ROUND
# ============================================================

result_df[
    "raw_p_value"
] = result_df[
    "raw_p_value"
].round(6)


result_df[
    "holm_adjusted_p"
] = result_df[
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
# DISPLAY ALL RESULTS
# ============================================================

print()
print("=" * 75)
print("ALL 36 COMPARISONS")
print("=" * 75)

print(
    result_df.to_string(
        index=False
    )
)


# ============================================================
# SIGNIFICANT RESULTS
# ============================================================

print()
print("=" * 75)
print("SIGNIFICANT AFTER HOLM CORRECTION")
print("=" * 75)


significant_df = result_df[
    result_df[
        "significant_after_holm"
    ] == True
]


if len(significant_df) == 0:

    print(
        "No criterion-level comparison "
        "is significant after Holm correction."
    )

else:

    print(
        significant_df.to_string(
            index=False
        )
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 75)
print("OUTPUT FILE")
print("=" * 75)

print(OUTPUT)

print()
print("DONE.")