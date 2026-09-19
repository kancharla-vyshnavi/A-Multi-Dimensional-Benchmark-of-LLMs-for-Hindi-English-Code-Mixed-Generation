import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = r".\outputs\error_analysis"

INPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "error_analysis_screening.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "manual_error_annotation.csv"
)

# ============================================================
# LOAD SCREENING DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"File not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("MANUAL ERROR ANALYSIS")
print("=" * 70)

print("\nInput shape:", df.shape)

# ============================================================
# CREATE MANUAL ANNOTATION COLUMNS
# ============================================================

manual_columns = [
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
    "Final_Error_Category",
    "Manual_Notes",
]

for column in manual_columns:
    df[column] = ""

# ============================================================
# REVIEW STATUS
# ============================================================

df["Review_Status"] = "NOT_REVIEWED"

# ============================================================
# ADD AUTOMATIC SCREENING AS REFERENCE
# ============================================================

# Keep automatic result, but DO NOT treat it as final.
df["Automatic_Screening_Reference"] = (
    df["automatic_errors"].fillna("")
)

# ============================================================
# REORDER COLUMNS
# ============================================================

column_order = [
    "model",
    "sample",
    "category",
    "prompt",
    "response",
    "Automatic_Screening_Reference",
] + manual_columns + [
    "Review_Status"
]

df = df[column_order]

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nCreated manual annotation file:")
print(OUTPUT_FILE)

print("\nTotal responses:", len(df))

print("\nResponses per model:")
print(df["model"].value_counts())

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nIMPORTANT:"
    "\nAutomatic labels are only references."
    "\nFinal E1-E10 labels must be manually verified."
)