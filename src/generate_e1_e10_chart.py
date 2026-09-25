import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# AUTOMATED E1-E10 FINAL CHART
# ============================================================

INPUT = r"outputs\final_research_tables\table_error_analysis_automated_4000.csv"
OUT_DIR = r"outputs\final_charts"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("GENERATING AUTOMATED E1-E10 CHART")
print("=" * 70)

# ------------------------------------------------------------
# LOAD ALREADY-PIVOTED TABLE
# ------------------------------------------------------------

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print("[OK] Loaded:", INPUT)
print("[OK] Shape:", df.shape)

# ------------------------------------------------------------
# REMOVE UNNAMED INDEX COLUMN IF PRESENT
# ------------------------------------------------------------

df = df.loc[
    :,
    ~df.columns.astype(str).str.startswith("Unnamed")
]

# ------------------------------------------------------------
# MODEL COLUMN
# ------------------------------------------------------------

if "model" in df.columns:

    df = df.set_index("model")

else:

    # If model became CSV index during previous save
    if df.index.name == "model":
        pass

    else:
        raise ValueError(
            "Could not find model column/index."
        )

# ------------------------------------------------------------
# E1-E10 COLUMNS
# ------------------------------------------------------------

error_columns = [
    "E1_English_dominant",
    "E2_Hindi_dominant",
    "E3_Unnatural_code_switching",
    "E4_Grammatical_error",
    "E5_Repetition",
    "E6_Prompt_misunderstanding",
    "E7_Incomplete_response",
    "E8_Spelling_transliteration_error",
    "E9_Hallucination_factual_error",
    "E10_Irrelevant_response"
]

missing = [
    c
    for c in error_columns
    if c not in df.columns
]

if missing:

    print()
    print("[ERROR] Missing E1-E10 columns:")
    print(missing)

    print()
    print("Available columns:")
    print(df.columns.tolist())

    raise ValueError(
        "E1-E10 columns missing."
    )

plot_df = df[
    error_columns
].copy()

# ------------------------------------------------------------
# DISPLAY NAMES
# ------------------------------------------------------------

plot_df.columns = [
    "E1 English",
    "E2 Hindi",
    "E3 Code-switching",
    "E4 Grammar",
    "E5 Repetition",
    "E6 Prompt",
    "E7 Incomplete",
    "E8 Spelling",
    "E9 Factual",
    "E10 Irrelevant"
]

# ------------------------------------------------------------
# ENSURE NUMERIC
# ------------------------------------------------------------

plot_df = plot_df.apply(
    pd.to_numeric,
    errors="coerce"
)

# ------------------------------------------------------------
# PRINT TABLE
# ------------------------------------------------------------

print()
print("-" * 70)
print("E1-E10 PERCENTAGES")
print("-" * 70)

print(
    plot_df.to_string()
)

# ------------------------------------------------------------
# CREATE CHART
# ------------------------------------------------------------

ax = plot_df.plot(
    kind="bar",
    figsize=(17, 9)
)

ax.set_title(
    "Automated E1-E10 Error Analysis Across Models",
    fontsize=16
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_ylabel(
    "Responses Flagged (%)",
    fontsize=12
)

ax.set_ylim(
    0,
    max(
        10,
        plot_df.max().max() * 1.15
    )
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Error Type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

# ------------------------------------------------------------
# SAVE PNG
# ------------------------------------------------------------

chart_path = os.path.join(
    OUT_DIR,
    "automated_e1_e10_error_analysis.png"
)

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("[OK] Chart saved:")
print(chart_path)

# ------------------------------------------------------------
# SAVE CLEAN FINAL CSV
# ------------------------------------------------------------

csv_path = os.path.join(
    OUT_DIR,
    "automated_e1_e10_final_results.csv"
)

plot_df.to_csv(
    csv_path
)

print("[OK] Final E1-E10 data saved:")
print(csv_path)

print()
print("=" * 70)
print("CHART GENERATION COMPLETE")
print("=" * 70)
