import os
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"outputs\pairwise_evaluation\pairwise_mistral_v3_results.csv"

OUTPUT_DIR = r"outputs\pairwise_evaluation"

MODEL_ORDER = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("PAIRWISE V3 ANALYSIS")
print("=" * 70)

print(f"Total judgments: {len(df)}")
print()


# ============================================================
# NORMALIZE WINNER TO ACTUAL MODEL
# ============================================================

def actual_winner(row):

    winner = str(row["winner"]).strip().upper()

    if winner == "T":
        return "T"

    if winner == "A":
        return row["model_A"]

    if winner == "B":
        return row["model_B"]

    return None


df["actual_winner"] = df.apply(
    actual_winner,
    axis=1
)


# ============================================================
# PAIR CANONICALIZATION
# ============================================================

def canonical_pair(row):

    models = sorted([
        str(row["model_A"]),
        str(row["model_B"])
    ])

    return f"{models[0]} vs {models[1]}"


df["pair"] = df.apply(
    canonical_pair,
    axis=1
)


# ============================================================
# CHECK INVALID
# ============================================================

invalid = df["actual_winner"].isna().sum()

print(
    f"Invalid judgments: {invalid}"
)

print()


# ============================================================
# MODEL-WISE PAIRWISE RESULTS
# ============================================================

rows = []

for model_a in MODEL_ORDER:

    for model_b in MODEL_ORDER:

        if model_a == model_b:
            continue

        # Avoid duplicate pair
        if MODEL_ORDER.index(model_a) >= MODEL_ORDER.index(model_b):
            continue

        pair_df = df[
            (
                (
                    df["model_A"] == model_a
                )
                &
                (
                    df["model_B"] == model_b
                )
            )
            |
            (
                (
                    df["model_A"] == model_b
                )
                &
                (
                    df["model_B"] == model_a
                )
            )
        ].copy()

        wins_a = (
            pair_df["actual_winner"] == model_a
        ).sum()

        wins_b = (
            pair_df["actual_winner"] == model_b
        ).sum()

        ties = (
            pair_df["actual_winner"] == "T"
        ).sum()

        total = len(pair_df)

        if total > 0:
            win_rate_a = wins_a / total
            win_rate_b = wins_b / total
        else:
            win_rate_a = 0
            win_rate_b = 0

        rows.append({
            "model_A": model_a,
            "model_B": model_b,
            "total": total,
            "wins_model_A": wins_a,
            "wins_model_B": wins_b,
            "ties": ties,
            "win_rate_model_A": round(
                win_rate_a,
                4
            ),
            "win_rate_model_B": round(
                win_rate_b,
                4
            ),
        })


pair_summary = pd.DataFrame(rows)


# ============================================================
# PRINT PAIRWISE SUMMARY
# ============================================================

print("=" * 70)
print("PAIRWISE MODEL COMPARISON")
print("=" * 70)

print()

print(
    pair_summary.to_string(
        index=False
    )
)

print()


# ============================================================
# MODEL AGGREGATE
# ============================================================

aggregate_rows = []

for model in MODEL_ORDER:

    wins = (
        df["actual_winner"] == model
    ).sum()

    losses = (
        (
            (
                df["model_A"] == model
            )
            |
            (
                df["model_B"] == model
            )
        )
        &
        (
            df["actual_winner"].notna()
        )
        &
        (
            df["actual_winner"] != model
        )
        &
        (
            df["actual_winner"] != "T"
        )
    ).sum()

    ties = (
        (
            (
                df["model_A"] == model
            )
            |
            (
                df["model_B"] == model
            )
        )
        &
        (
            df["actual_winner"] == "T"
        )
    ).sum()

    total = wins + losses + ties

    if total > 0:
        win_rate = wins / total
    else:
        win_rate = 0

    aggregate_rows.append({
        "model": model,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total,
        "win_rate": round(
            win_rate,
            4
        ),
    })


aggregate = pd.DataFrame(
    aggregate_rows
)


print("=" * 70)
print("OVERALL PAIRWISE MODEL RESULTS")
print("=" * 70)

print()

print(
    aggregate.to_string(
        index=False
    )
)

print()


# ============================================================
# A/B SWAP CONSISTENCY
# ============================================================

print("=" * 70)
print("A/B SWAP CONSISTENCY")
print("=" * 70)

swap_rows = []

for model_a in MODEL_ORDER:

    for model_b in MODEL_ORDER:

        if model_a == model_b:
            continue

        if MODEL_ORDER.index(model_a) >= MODEL_ORDER.index(model_b):
            continue

        # A_vs_B
        ab = df[
            (df["model_A"] == model_a)
            &
            (df["model_B"] == model_b)
            &
            (df["direction"] == "A_vs_B")
        ].copy()

        # B_vs_A
        ba = df[
            (df["model_A"] == model_b)
            &
            (df["model_B"] == model_a)
            &
            (df["direction"] == "B_vs_A")
        ].copy()

        # Match using row_index
        merged = pd.merge(
            ab[
                [
                    "row_index",
                    "actual_winner"
                ]
            ],
            ba[
                [
                    "row_index",
                    "actual_winner"
                ]
            ],
            on="row_index",
            suffixes=(
                "_ab",
                "_ba"
            )
        )

        consistent = 0
        inconsistent = 0

        for _, r in merged.iterrows():

            winner_ab = r[
                "actual_winner_ab"
            ]

            winner_ba = r[
                "actual_winner_ba"
            ]

            # Normalize perspective:
            # both should identify the same actual winner
            if winner_ab == winner_ba:
                consistent += 1
            else:
                inconsistent += 1

        total = len(merged)

        consistency_rate = (
            consistent / total
            if total > 0
            else 0
        )

        swap_rows.append({
            "model_A": model_a,
            "model_B": model_b,
            "matched_prompts": total,
            "consistent": consistent,
            "inconsistent": inconsistent,
            "consistency_rate": round(
                consistency_rate,
                4
            ),
        })


swap_summary = pd.DataFrame(
    swap_rows
)

print()

print(
    swap_summary.to_string(
        index=False
    )
)

print()


# ============================================================
# SAVE FILES
# ============================================================

pair_summary_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_model_comparison_v3.csv"
)

aggregate_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_model_aggregate_v3.csv"
)

swap_path = os.path.join(
    OUTPUT_DIR,
    "pairwise_swap_consistency_v3.csv"
)


pair_summary.to_csv(
    pair_summary_path,
    index=False
)

aggregate.to_csv(
    aggregate_path,
    index=False
)

swap_summary.to_csv(
    swap_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print()
print(pair_summary_path)
print(aggregate_path)
print(swap_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
