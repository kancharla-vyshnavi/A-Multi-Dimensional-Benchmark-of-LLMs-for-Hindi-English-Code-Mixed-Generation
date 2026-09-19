import os
import re
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT = r".\outputs\pairwise_judge\pairwise_judge_results_final.csv"

OUTPUT_DIR = r".\outputs\pairwise_judge"

OVERALL_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "pairwise_overall_summary.csv"
)

CRITERIA_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "pairwise_criteria_summary.csv"
)

MODEL_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "pairwise_model_summary.csv"
)

PREFERENCE_MATRIX = os.path.join(
    OUTPUT_DIR,
    "pairwise_preference_matrix.csv"
)

DETAIL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "pairwise_all_judgments_parsed.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PAIRWISE PREFERENCE STATISTICS")
print("=" * 70)

df = pd.read_csv(INPUT)

print()
print("Total rows:", len(df))


# ============================================================
# CRITERIA
# ============================================================

criteria = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]


# ============================================================
# PARSE A/B/T
# ============================================================

def parse_choices(value):

    text = str(value).strip().upper()

    tokens = re.findall(
        r"\b(?:A|B|T|TIE)\b",
        text
    )

    if len(tokens) >= 6:

        tokens = tokens[-6:]

        result = []

        for token in tokens:

            if token == "TIE":
                token = "T"

            result.append(token)

        if len(result) == 6:
            return result

    compact = re.sub(
        r"[^ABT]",
        "",
        text
    )

    if len(compact) == 6:
        return list(compact)

    return None


# ============================================================
# PARSE ALL ROWS
# ============================================================

parsed_rows = []

invalid = 0

for _, row in df.iterrows():

    choices = parse_choices(
        row["overall"]
    )

    if choices is None:

        invalid += 1
        continue

    parsed_rows.append(
        {
            "sample_id": row.get(
                "sample_id"
            ),

            "pair_model_1": row.get(
                "pair_model_1"
            ),

            "pair_model_2": row.get(
                "pair_model_2"
            ),

            "displayed_A_model": row.get(
                "displayed_A_model"
            ),

            "displayed_B_model": row.get(
                "displayed_B_model"
            ),

            "position_swapped": row.get(
                "position_swapped"
            ),

            "fluency": choices[0],

            "code_mixing_naturalness": choices[1],

            "hindi_grammar": choices[2],

            "prompt_adherence": choices[3],

            "spelling_consistency": choices[4],

            "overall": choices[5]
        }
    )


parsed_df = pd.DataFrame(parsed_rows)


print(
    "Valid parsed rows:",
    len(parsed_df)
)

print(
    "Invalid rows:",
    invalid
)


# ============================================================
# SAVE PARSED DATA
# ============================================================

parsed_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FUNCTION:
# CONVERT A/B/T INTO ACTUAL MODEL RESULT
# ============================================================

def actual_result(choice, model_a, model_b):

    if choice == "T":
        return "TIE"

    if choice == "A":
        return model_a

    if choice == "B":
        return model_b

    return None


# ============================================================
# OVERALL PAIRWISE SUMMARY
# ============================================================

overall_rows = []


for _, row in parsed_df.iterrows():

    model_a = row["displayed_A_model"]
    model_b = row["displayed_B_model"]

    result = actual_result(
        row["overall"],
        model_a,
        model_b
    )

    overall_rows.append(
        {
            "pair_model_1": row["pair_model_1"],

            "pair_model_2": row["pair_model_2"],

            "model_A_displayed": model_a,

            "model_B_displayed": model_b,

            "sample_id": row["sample_id"],

            "result": result
        }
    )


overall_detail = pd.DataFrame(
    overall_rows
)


# ============================================================
# AGGREGATE EACH PAIR
# ============================================================

pair_summary_rows = []


for (
    model_1,
    model_2
), group in overall_detail.groupby(
    ["pair_model_1", "pair_model_2"]
):

    wins_1 = 0
    wins_2 = 0
    ties = 0

    for _, row in group.iterrows():

        result = row["result"]

        if result == model_1:
            wins_1 += 1

        elif result == model_2:
            wins_2 += 1

        elif result == "TIE":
            ties += 1


    total = len(group)

    pair_summary_rows.append(
        {
            "model_1": model_1,

            "model_2": model_2,

            "total_judgments": total,

            "model_1_wins": wins_1,

            "model_2_wins": wins_2,

            "ties": ties,

            "model_1_win_percent":
                round(
                    wins_1 / total * 100,
                    2
                ),

            "model_2_win_percent":
                round(
                    wins_2 / total * 100,
                    2
                ),

            "tie_percent":
                round(
                    ties / total * 100,
                    2
                )
        }
    )


pair_summary = pd.DataFrame(
    pair_summary_rows
)


# ============================================================
# SAVE OVERALL SUMMARY
# ============================================================

pair_summary.to_csv(
    OVERALL_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CRITERIA-WISE ANALYSIS
# ============================================================

criteria_rows = []


for criterion in criteria:

    for (
        model_1,
        model_2
    ), group in parsed_df.groupby(
        ["pair_model_1", "pair_model_2"]
    ):

        model_1_wins = 0
        model_2_wins = 0
        ties = 0

        for _, row in group.iterrows():

            choice = row[criterion]

            model_a = row[
                "displayed_A_model"
            ]

            model_b = row[
                "displayed_B_model"
            ]

            result = actual_result(
                choice,
                model_a,
                model_b
            )

            if result == model_1:
                model_1_wins += 1

            elif result == model_2:
                model_2_wins += 1

            elif result == "TIE":
                ties += 1


        total = len(group)

        criteria_rows.append(
            {
                "criterion": criterion,

                "model_1": model_1,

                "model_2": model_2,

                "total": total,

                "model_1_wins":
                    model_1_wins,

                "model_2_wins":
                    model_2_wins,

                "ties":
                    ties,

                "model_1_win_percent":
                    round(
                        model_1_wins /
                        total * 100,
                        2
                    ),

                "model_2_win_percent":
                    round(
                        model_2_wins /
                        total * 100,
                        2
                    ),

                "tie_percent":
                    round(
                        ties /
                        total * 100,
                        2
                    )
            }
        )


criteria_summary = pd.DataFrame(
    criteria_rows
)


criteria_summary.to_csv(
    CRITERIA_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MODEL-LEVEL SUMMARY
# ============================================================

models = sorted(
    set(parsed_df["pair_model_1"])
    | set(parsed_df["pair_model_2"])
)


model_rows = []


for model in models:

    wins = 0
    losses = 0
    ties = 0

    for _, row in parsed_df.iterrows():

        model_a = row[
            "displayed_A_model"
        ]

        model_b = row[
            "displayed_B_model"
        ]

        result = actual_result(
            row["overall"],
            model_a,
            model_b
        )

        if result == model:
            wins += 1

        elif (
            result != "TIE"
            and
            (
                result == model_a
                or
                result == model_b
            )
        ):

            losses += 1

        elif result == "TIE":

            if (
                model == model_a
                or
                model == model_b
            ):
                ties += 1


    total = wins + losses + ties

    model_rows.append(
        {
            "model": model,

            "pairwise_wins": wins,

            "pairwise_losses": losses,

            "pairwise_ties": ties,

            "total_pairwise_comparisons": total,

            "win_percent":
                round(
                    wins / total * 100,
                    2
                ),

            "loss_percent":
                round(
                    losses / total * 100,
                    2
                ),

            "tie_percent":
                round(
                    ties / total * 100,
                    2
                )
        }
    )


model_summary = pd.DataFrame(
    model_rows
)


model_summary.to_csv(
    MODEL_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PREFERENCE MATRIX
# ============================================================

matrix = pd.DataFrame(
    0.0,
    index=models,
    columns=models
)


for _, row in parsed_df.iterrows():

    model_a = row[
        "displayed_A_model"
    ]

    model_b = row[
        "displayed_B_model"
    ]

    result = actual_result(
        row["overall"],
        model_a,
        model_b
    )

    if result == model_a:

        matrix.loc[
            model_a,
            model_b
        ] += 1

    elif result == model_b:

        matrix.loc[
            model_b,
            model_a
        ] += 1


matrix.to_csv(
    PREFERENCE_MATRIX,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("OVERALL PAIRWISE RESULTS")
print("=" * 70)

print(
    pair_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("MODEL-LEVEL RESULTS")
print("=" * 70)

print(
    model_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("PREFERENCE MATRIX")
print("=" * 70)

print(matrix)


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nOverall:",
    OVERALL_SUMMARY
)

print(
    "\nCriteria:",
    CRITERIA_SUMMARY
)

print(
    "\nModel summary:",
    MODEL_SUMMARY
)

print(
    "\nPreference matrix:",
    PREFERENCE_MATRIX
)

print(
    "\nParsed judgments:",
    DETAIL_OUTPUT
)

print()
print("DONE.")