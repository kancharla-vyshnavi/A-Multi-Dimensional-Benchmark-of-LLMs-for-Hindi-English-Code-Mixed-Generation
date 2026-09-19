import os
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

OUTPUT_DIR = r".\outputs\final_figures"

MASTER_PATH = r".\outputs\MASTER_RESULTS.csv"
FRIEDMAN_PATH = r".\outputs\statistics\friedman_results.csv"
WILCOXON_PATH = r".\outputs\statistics\wilcoxon_holm_results.csv"
PAIRWISE_PATH = r".\outputs\pairwise_statistics\pairwise_overall_results.csv"
LINGUISTIC_PATH = (
    r".\outputs\linguistic_evaluation"
    r"\hinglish_bench_official_metrics_summary.csv"
)
ERROR_PATH = (
    r".\outputs\error_analysis"
    r"\error_analysis_summary_final.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# LOAD DATA
# ================================================================

master_df = pd.read_csv(MASTER_PATH)
friedman_df = pd.read_csv(FRIEDMAN_PATH)
wilcoxon_df = pd.read_csv(WILCOXON_PATH)
pairwise_df = pd.read_csv(PAIRWISE_PATH)
linguistic_df = pd.read_csv(LINGUISTIC_PATH)
error_df = pd.read_csv(ERROR_PATH)

print("=" * 70)
print("DATA LOADED")
print("=" * 70)

print(f"MASTER: {len(master_df)}")
print(f"FRIEDMAN: {len(friedman_df)}")
print(f"WILCOXON: {len(wilcoxon_df)}")
print(f"PAIRWISE: {len(pairwise_df)}")
print(f"LINGUISTIC: {len(linguistic_df)}")
print(f"ERROR ANALYSIS: {len(error_df)}")


# ================================================================
# VALIDATE ERROR ANALYSIS
# ================================================================

required_error_columns = [
    "error_code",
    "error_category",
    "number_of_errors",
    "total_responses",
    "responses_affected_percent",
]

missing_error_columns = [
    col
    for col in required_error_columns
    if col not in error_df.columns
]

if missing_error_columns:
    raise ValueError(
        f"Missing error-analysis columns: {missing_error_columns}"
    )

if len(error_df) != 10:
    raise ValueError(
        f"Expected 10 error categories, found {len(error_df)}"
    )


# ================================================================
# VALIDATE PAIRWISE DATA
# ================================================================

required_pairwise_columns = [
    "criterion",
    "model_a",
    "model_b",
    "N",
    "model_a_wins",
    "model_b_wins",
    "ties",
    "non_ties",
    "model_a_win_rate_non_ties",
    "model_b_win_rate_non_ties",
    "tie_rate",
    "raw_p",
    "holm_adjusted_p",
    "significant_alpha_0.05",
]

missing_pairwise_columns = [
    col
    for col in required_pairwise_columns
    if col not in pairwise_df.columns
]

if missing_pairwise_columns:
    raise ValueError(
        f"Missing pairwise columns: {missing_pairwise_columns}"
    )


# ================================================================
# FIGURE 1 — OVERALL QUALITY
# ================================================================

print("\nCreating Figure 1...")

plt.figure(figsize=(9, 6))

plt.bar(
    master_df["model"],
    master_df["overall_mean"]
)

plt.xlabel("Model")
plt.ylabel("Mean Overall Quality")
plt.title("Overall Quality Comparison Across Models")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_1_overall_quality.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FIGURE 2 — CRITERION COMPARISON
# ================================================================

print("Creating Figure 2...")

criteria = [
    "fluency_mean",
    "code_mixing_naturalness_mean",
    "hindi_grammar_mean",
    "prompt_adherence_mean",
    "spelling_consistency_mean",
    "overall_mean",
]

criterion_labels = [
    "Fluency",
    "Code-mixing",
    "Hindi grammar",
    "Prompt adherence",
    "Spelling",
    "Overall",
]

plt.figure(figsize=(12, 7))

for criterion, label in zip(criteria, criterion_labels):
    plt.plot(
        master_df["model"],
        master_df[criterion],
        marker="o",
        label=label
    )

plt.xlabel("Model")
plt.ylabel("Mean Score")
plt.title("Criterion-wise Model Comparison")
plt.legend()
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_2_criterion_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FIGURE 3 — LINGUISTIC METRICS
# ================================================================

print("Creating Figure 3...")

metric_columns = [
    "avg_CMI",
    "avg_M_index",
    "avg_SyMCoM_imbalance",
]

plt.figure(figsize=(10, 6))

for metric in metric_columns:
    plt.plot(
        linguistic_df["model"],
        linguistic_df[metric],
        marker="o",
        label=metric
    )

plt.xlabel("Model")
plt.ylabel("Metric Value")
plt.title("Linguistic Metrics Across Models")
plt.legend()
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_3_linguistic_metrics.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FIGURE 4 — KENDALL'S W
# ================================================================

print("Creating Figure 4...")

print("Friedman columns:")
print(friedman_df.columns.tolist())

kendall_column = None

for candidate in [
    "kendall_W",
    "kendall_w",
    "Kendall_W",
    "Kendall_w",
]:
    if candidate in friedman_df.columns:
        kendall_column = candidate
        break

if kendall_column is None:
    raise ValueError(
        "Kendall's W column not found in Friedman results."
    )

print(f"Using Kendall's W column: {kendall_column}")

plt.figure(figsize=(10, 6))

plt.bar(
    friedman_df["criterion"],
    friedman_df[kendall_column]
)

plt.xlabel("Criterion")
plt.ylabel("Kendall's W")
plt.title("Kendall's W for Friedman Tests")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_4_kendall_w.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FIGURE 5 — PAIRWISE WIN RATES
# ================================================================

print("Creating Figure 5...")

pair_labels = (
    pairwise_df["model_a"].astype(str)
    + " vs "
    + pairwise_df["model_b"].astype(str)
)

plt.figure(figsize=(11, 7))

plt.bar(
    pair_labels,
    pairwise_df["model_a_win_rate_non_ties"] * 100
)

plt.xlabel("Model Pair")
plt.ylabel("Model A Win Rate Among Non-Ties (%)")
plt.title("Pairwise Model Win Rates")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_5_pairwise_win_rates.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FIGURE 6 — ERROR ANALYSIS
# ================================================================

print("Creating Figure 6...")

error_plot_df = error_df.sort_values(
    "responses_affected_percent",
    ascending=True
).reset_index(drop=True)

error_labels = (
    error_plot_df["error_code"].astype(str)
    + " - "
    + error_plot_df["error_category"].astype(str)
)

plt.figure(figsize=(12, 8))

plt.barh(
    error_labels,
    error_plot_df["responses_affected_percent"]
)

plt.xlabel("Responses Affected (%)")
plt.ylabel("Error Category")
plt.title("Error Analysis Across All Model Responses")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_6_error_analysis.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# SAVE FINAL TABLES
# ================================================================

print("\nSaving final tables...")

master_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_master_results.csv"
    ),
    index=False
)

friedman_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_friedman_results.csv"
    ),
    index=False
)

wilcoxon_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_wilcoxon_holm_results.csv"
    ),
    index=False
)

pairwise_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_pairwise_results.csv"
    ),
    index=False
)

linguistic_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_linguistic_metrics.csv"
    ),
    index=False
)

error_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "table_error_analysis.csv"
    ),
    index=False
)


# ================================================================
# FINAL FILE CHECK
# ================================================================

expected_files = [
    "figure_1_overall_quality.png",
    "figure_2_criterion_comparison.png",
    "figure_3_linguistic_metrics.png",
    "figure_4_kendall_w.png",
    "figure_5_pairwise_win_rates.png",
    "figure_6_error_analysis.png",
    "table_master_results.csv",
    "table_friedman_results.csv",
    "table_wilcoxon_holm_results.csv",
    "table_pairwise_results.csv",
    "table_linguistic_metrics.csv",
    "table_error_analysis.csv",
]

print("\n" + "=" * 70)
print("FINAL FIGURES + TABLES CREATED SUCCESSFULLY")
print("=" * 70)

for filename in expected_files:
    print(filename)

actual_files = os.listdir(OUTPUT_DIR)

print(f"\nTotal files in output directory: {len(actual_files)}")
print(f"Expected files: {len(expected_files)}")
print(f"Output directory: {OUTPUT_DIR}")

missing_files = [
    filename
    for filename in expected_files
    if filename not in actual_files
]

if missing_files:
    raise RuntimeError(
        f"Missing expected files: {missing_files}"
    )

print("\nDONE.")
