import os
import pandas as pd

BASE = r"C:\Users\vyshu\OneDrive\Desktop\HinglishLLM"

BENCH_PATH = os.path.join(
    BASE, "data", "benchmarks", "hinglish_bench_test.csv"
)

GEN_PATH = os.path.join(
    BASE, "outputs", "controlled_generation",
    "benchmark_generations_v3.csv"
)

JUDGE_PATH = os.path.join(
    BASE, "outputs", "independent_judge",
    "independent_mistral_judge_v3_results.csv"
)

PAIRWISE_PATH = os.path.join(
    BASE, "outputs", "pairwise_evaluation",
    "pairwise_mistral_v3_results.csv"
)

OUT_DIR = os.path.join(
    BASE, "outputs", "category_robustness"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

bench = pd.read_csv(BENCH_PATH)
gen = pd.read_csv(GEN_PATH)
judge = pd.read_csv(JUDGE_PATH)
pairwise = pd.read_csv(PAIRWISE_PATH)

print("Benchmark:", len(bench))
print("Generations:", len(gen))
print("Judge rows:", len(judge))
print("Pairwise rows:", len(pairwise))


# ============================================================
# CATEGORY MAPPING
# Use PROMPT because judge row_index repeats for every model
# ============================================================

category_map = bench[
    ["prompt", "category", "category_name"]
].drop_duplicates("prompt")


# ============================================================
# GENERATION + CATEGORY
# ============================================================

gen_cat = gen.merge(
    category_map,
    on="prompt",
    how="left",
    validate="many_to_one"
)

print(
    "\nGeneration category mapping missing:",
    gen_cat["category"].isna().sum()
)


# ============================================================
# JUDGE + CATEGORY
# IMPORTANT: merge by PROMPT, NOT row_index
# ============================================================

judge_cat = judge.merge(
    category_map,
    on="prompt",
    how="left",
    validate="many_to_one"
)

print(
    "Judge category mapping missing:",
    judge_cat["category"].isna().sum()
)


# ============================================================
# SAFETY CHECK
# ============================================================

if judge_cat["category"].isna().sum() > 0:
    print("\nERROR: Some judge rows could not be mapped.")
    print(
        judge_cat.loc[
            judge_cat["category"].isna(),
            ["row_index", "model", "prompt"]
        ].to_string(index=False)
    )
    raise ValueError("Judge category mapping failed.")


# ============================================================
# 1. CATEGORY Ã— MODEL OVERALL
# ============================================================

category_overall = (
    judge_cat
    .groupby(
        ["category", "category_name", "model"],
        as_index=False
    )
    .agg(
        overall_mean=("overall", "mean"),
        valid_judgments=("overall", "count")
    )
)

category_overall.to_csv(
    os.path.join(
        OUT_DIR,
        "category_overall_results_v3.csv"
    ),
    index=False
)


# ============================================================
# 2. CATEGORY Ã— MODEL â€” ALL JUDGE DIMENSIONS
# ============================================================

metrics = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

existing_metrics = [
    m for m in metrics
    if m in judge_cat.columns
]

category_judge_summary = (
    judge_cat
    .groupby(
        ["category", "category_name", "model"],
        as_index=False
    )
    .agg(
        **{
            f"{m}_mean": (m, "mean")
            for m in existing_metrics
        },
        **{
            f"{m}_valid": (m, "count")
            for m in existing_metrics
        }
    )
)

category_judge_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "category_independent_judge_summary_v3.csv"
    ),
    index=False
)


# ============================================================
# 3. CATEGORY ERROR FLAGS
# ============================================================

error_columns = [
    "E6_prompt_misunderstanding",
    "E10_irrelevant_response"
]

existing_errors = [
    e for e in error_columns
    if e in judge_cat.columns
]

if existing_errors:

    error_summary = (
        judge_cat
        .groupby(
            ["category", "category_name", "model"],
            as_index=False
        )
        .agg(
            **{
                f"{e}_rate": (e, "mean")
                for e in existing_errors
            },
            **{
                f"{e}_count": (e, "sum")
                for e in existing_errors
            }
        )
    )

    error_summary.to_csv(
        os.path.join(
            OUT_DIR,
            "category_error_flags_v3.csv"
        ),
        index=False
    )


# ============================================================
# 4. SAVE FULL CATEGORY-ANNOTATED JUDGE DATA
# ============================================================

judge_cat.to_csv(
    os.path.join(
        OUT_DIR,
        "category_annotated_judge_rows_v3.csv"
    ),
    index=False
)


# ============================================================
# 5. CATEGORY COUNTS
# ============================================================

category_counts = (
    bench
    .groupby(
        ["category", "category_name"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "prompt_count"})
)

category_counts.to_csv(
    os.path.join(
        OUT_DIR,
        "category_prompt_counts_v3.csv"
    ),
    index=False
)


# ============================================================
# 6. PAIRWISE + CATEGORY
# ============================================================

if "prompt" in pairwise.columns:

    pair_cat = pairwise.merge(
        category_map,
        on="prompt",
        how="left",
        validate="many_to_one"
    )

elif "row_index" in pairwise.columns:

    # Fallback only if prompt is unavailable
    bench_index = bench.copy()
    bench_index["row_index"] = bench_index.index

    pair_cat = pairwise.merge(
        bench_index[
            ["row_index", "category", "category_name"]
        ],
        on="row_index",
        how="left",
        validate="many_to_one"
    )

else:

    pair_cat = None
    print("\nWARNING: Pairwise file has no prompt or row_index.")


# ============================================================
# 7. CATEGORY-WISE PAIRWISE COMPARISON
# ============================================================

if pair_cat is not None:

    print(
        "Pairwise category mapping missing:",
        pair_cat["category"].isna().sum()
    )

    if pair_cat["category"].isna().sum() > 0:
        raise ValueError(
            "Pairwise category mapping failed."
        )

    print("\nPairwise columns:")
    print(pair_cat.columns.tolist())

    model_a_col = next(
        (
            c for c in
            ["model_A", "model_a", "A_model", "model1"]
            if c in pair_cat.columns
        ),
        None
    )

    model_b_col = next(
        (
            c for c in
            ["model_B", "model_b", "B_model", "model2"]
            if c in pair_cat.columns
        ),
        None
    )

    winner_col = next(
        (
            c for c in
            ["winner", "judgment", "result"]
            if c in pair_cat.columns
        ),
        None
    )

    if model_a_col is None:
        raise ValueError(
            "Could not find Model A column."
        )

    if model_b_col is None:
        raise ValueError(
            "Could not find Model B column."
        )

    if winner_col is None:
        raise ValueError(
            "Could not find winner column."
        )


    records = []

    for _, row in pair_cat.iterrows():

        model_a = str(row[model_a_col])
        model_b = str(row[model_b_col])
        winner_raw = str(row[winner_col]).strip()

        if winner_raw == "A":
            winner = model_a

        elif winner_raw == "B":
            winner = model_b

        else:
            winner = "TIE"

        records.append({
            "category": row["category"],
            "category_name": row["category_name"],
            "model_a": model_a,
            "model_b": model_b,
            "winner": winner
        })


    normalized_pairwise = pd.DataFrame(records)

    pairwise_results = []

    for keys, group in normalized_pairwise.groupby(
        [
            "category",
            "category_name",
            "model_a",
            "model_b"
        ]
    ):

        category, category_name, model_a, model_b = keys

        total = len(group)

        model_a_wins = int(
            (group["winner"] == model_a).sum()
        )

        model_b_wins = int(
            (group["winner"] == model_b).sum()
        )

        ties = int(
            (group["winner"] == "TIE").sum()
        )

        pairwise_results.append({
            "category": category,
            "category_name": category_name,
            "model_a": model_a,
            "model_b": model_b,
            "comparisons": total,
            "model_a_wins": model_a_wins,
            "model_b_wins": model_b_wins,
            "ties": ties,
            "model_a_win_rate": model_a_wins / total,
            "model_b_win_rate": model_b_wins / total
        })


    pairwise_summary = pd.DataFrame(
        pairwise_results
    )

    pairwise_summary.to_csv(
        os.path.join(
            OUT_DIR,
            "category_pairwise_comparison_v3.csv"
        ),
        index=False
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


print("\nIndependent judge valid counts by model:")

print(
    judge_cat
    .groupby("model")["overall"]
    .count()
    .to_string()
)


print("\nCategory Ã— model valid counts:")

print(
    judge_cat
    .groupby(
        ["category", "model"]
    )["overall"]
    .count()
    .to_string()
)


print("\nCategory counts:")

print(
    category_counts.to_string(index=False)
)


print("\nOutput folder:")

print(OUT_DIR)


print("\nGenerated files:")

for filename in sorted(os.listdir(OUT_DIR)):
    print(" -", filename)


print("\n" + "=" * 70)
print("CATEGORY-WISE ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 70)
