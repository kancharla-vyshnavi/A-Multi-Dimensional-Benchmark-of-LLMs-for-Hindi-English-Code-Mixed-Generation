import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT = r".\outputs\error_analysis\manual_error_annotation_final.csv"

OUTPUT_SUMMARY = r".\outputs\error_analysis\error_analysis_summary_final.csv"
OUTPUT_MODEL_COMPARISON = r".\outputs\error_analysis\error_analysis_model_comparison_final.csv"


# ============================================================
# LOAD FINAL MANUAL ANNOTATIONS
# ============================================================

df = pd.read_csv(INPUT)

print()
print("=" * 80)
print("FINAL ERROR-ANALYSIS SOURCE")
print("=" * 80)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# VALIDATION
# ============================================================

if len(df) != 136:
    raise ValueError(
        f"Expected 136 annotated responses, but found {len(df)}."
    )


required_columns = [
    "model",
    "sample",
    "E1_English_dominant",
    "E2_Hindi_dominant",
    "E3_Unnatural_code_switching",
    "E4_Grammatical_error",
    "E5_Repetition",
    "E6_Prompt_misunderstanding",
    "E7_Incomplete_response",
    "E8_Spelling_transliteration_error",
    "E9_Hallucination_factual_error",
    "E10_Irrelevant_response",
    "Review_Status"
]

for column in required_columns:
    if column not in df.columns:
        raise KeyError(
            f"Missing required column: {column}"
        )


# ============================================================
# ERROR DEFINITIONS
# ============================================================

error_columns = {
    "E1": (
        "E1_English_dominant",
        "English-dominant output"
    ),
    "E2": (
        "E2_Hindi_dominant",
        "Hindi-dominant output"
    ),
    "E3": (
        "E3_Unnatural_code_switching",
        "Unnatural code-switching"
    ),
    "E4": (
        "E4_Grammatical_error",
        "Grammatical error"
    ),
    "E5": (
        "E5_Repetition",
        "Repetition"
    ),
    "E6": (
        "E6_Prompt_misunderstanding",
        "Prompt misunderstanding"
    ),
    "E7": (
        "E7_Incomplete_response",
        "Incomplete response"
    ),
    "E8": (
        "E8_Spelling_transliteration_error",
        "Spelling/transliteration error"
    ),
    "E9": (
        "E9_Hallucination_factual_error",
        "Hallucination/factual error"
    ),
    "E10": (
        "E10_Irrelevant_response",
        "Irrelevant response"
    )
}


# ============================================================
# VALIDATE BINARY ANNOTATIONS
# ============================================================

for code, (column, label) in error_columns.items():

    values = set(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna().astype(int).unique()
    )

    invalid = values - {0, 1}

    if invalid:
        raise ValueError(
            f"{column} contains invalid values: {invalid}"
        )


# ============================================================
# VALIDATE REVIEW STATUS
# ============================================================

status_counts = df["Review_Status"].value_counts(dropna=False)

print()
print("Review status:")
print(status_counts.to_string())


# ============================================================
# OVERALL ERROR-ANALYSIS SUMMARY
# ============================================================

summary_rows = []

total_responses = len(df)

for code, (column, label) in error_columns.items():

    count = int(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0).sum()
    )

    percentage = (
        count / total_responses * 100
    )

    summary_rows.append({
        "error_code": code,
        "error_category": label,
        "number_of_errors": count,
        "total_responses": total_responses,
        "responses_affected_percent": round(
            percentage,
            2
        )
    })


summary = pd.DataFrame(summary_rows)


# ============================================================
# MODEL-WISE ERROR ANALYSIS
# ============================================================

model_rows = []

for model in sorted(df["model"].dropna().unique()):

    model_df = df[
        df["model"] == model
    ]

    model_total = len(model_df)

    for code, (column, label) in error_columns.items():

        count = int(
            pd.to_numeric(
                model_df[column],
                errors="coerce"
            ).fillna(0).sum()
        )

        percentage = (
            count / model_total * 100
        )

        model_rows.append({
            "model": model,
            "error_code": code,
            "error_category": label,
            "number_of_errors": count,
            "total_responses": model_total,
            "responses_affected_percent": round(
                percentage,
                2
            )
        })


model_comparison = pd.DataFrame(model_rows)


# ============================================================
# SAVE RESULTS
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

model_comparison.to_csv(
    OUTPUT_MODEL_COMPARISON,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 80)
print("OVERALL ERROR ANALYSIS")
print("=" * 80)

print(
    summary.to_string(index=False)
)


print()
print("=" * 80)
print("MODEL-WISE ERROR ANALYSIS")
print("=" * 80)

print(
    model_comparison.to_string(index=False)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

expected_summary_rows = 10
expected_model_rows = (
    df["model"].nunique() * 10
)

if len(summary) != expected_summary_rows:
    raise ValueError(
        f"Expected {expected_summary_rows} summary rows, "
        f"found {len(summary)}."
    )

if len(model_comparison) != expected_model_rows:
    raise ValueError(
        f"Expected {expected_model_rows} model-wise rows, "
        f"found {len(model_comparison)}."
    )


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_SUMMARY)
print(OUTPUT_MODEL_COMPARISON)

print()
print("ERROR ANALYSIS REGENERATED SUCCESSFULLY.")