from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_FILE = (
    BASE_DIR
    / "outputs"
    / "MASTER_RESULTS.csv"
)

FRIEDMAN_FILE = (
    BASE_DIR
    / "outputs"
    / "statistics"
    / "friedman_results.csv"
)

WILCOXON_FILE = (
    BASE_DIR
    / "outputs"
    / "statistics"
    / "wilcoxon_holm_results.csv"
)

PAIRWISE_FILE = (
    BASE_DIR
    / "outputs"
    / "pairwise_statistics"
    / "pairwise_overall_results.csv"
)

LINGUISTIC_FILE = (
    BASE_DIR
    / "outputs"
    / "linguistic_evaluation"
    / "hinglish_bench_official_metrics_summary.csv"
)

ERROR_FILE = (
    BASE_DIR
    / "outputs"
    / "error_analysis"
    / "error_analysis_summary_final.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "results_report"
)


# ============================================================
# HELPERS
# ============================================================

def check_file(path):
    """Check whether an input file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing required file:\n{path}\n"
        )

    print(f"[OK] {path}")


def format_p(p):
    """Format p-values for readable report output."""

    if pd.isna(p):
        return ""

    p = float(p)

    if p < 0.001:
        return "<0.001"

    return f"{p:.4f}"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 46)
    print("      HINGLISH BENCH RESULTS REPORT")
    print("=" * 46)
    print()

    # ========================================================
    # 1. CHECK INPUT FILES
    # ========================================================

    input_files = [
        MASTER_FILE,
        FRIEDMAN_FILE,
        WILCOXON_FILE,
        PAIRWISE_FILE,
        LINGUISTIC_FILE,
        ERROR_FILE,
    ]

    for file in input_files:
        check_file(file)

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    master = pd.read_csv(MASTER_FILE)

    friedman = pd.read_csv(
        FRIEDMAN_FILE
    )

    wilcoxon = pd.read_csv(
        WILCOXON_FILE
    )

    pairwise = pd.read_csv(
        PAIRWISE_FILE
    )

    linguistic = pd.read_csv(
        LINGUISTIC_FILE
    )

    errors = pd.read_csv(
        ERROR_FILE
    )

    print()
    print("Loaded files successfully.")

    print(
        f"MASTER_RESULTS rows      : {len(master)}"
    )

    print(
        f"Friedman rows            : {len(friedman)}"
    )

    print(
        f"Wilcoxon rows            : {len(wilcoxon)}"
    )

    print(
        f"Pairwise rows            : {len(pairwise)}"
    )

    print(
        f"Linguistic rows          : {len(linguistic)}"
    )

    print(
        f"Error-analysis rows      : {len(errors)}"
    )

    # ========================================================
    # 3. MODEL SUMMARY
    # ========================================================

    model_summary_columns = [
        "model",
        "samples",

        "fluency_mean",
        "fluency_std",

        "code_mixing_naturalness_mean",
        "code_mixing_naturalness_std",

        "hindi_grammar_mean",
        "hindi_grammar_std",

        "prompt_adherence_mean",
        "prompt_adherence_std",

        "spelling_consistency_mean",
        "spelling_consistency_std",

        "overall_mean",
        "overall_std",

        "avg_CMI",
        "avg_M_index",
        "avg_SyMCoM_imbalance",

        "total_hindi_tokens",
        "total_english_tokens",
        "total_other_tokens",
    ]

    available_columns = [
        column
        for column in model_summary_columns
        if column in master.columns
    ]

    model_summary = master[
        available_columns
    ].copy()

    model_summary.to_csv(
        OUTPUT_DIR
        / "results_model_summary.csv",
        index=False
    )

    # ========================================================
    # 4. FRIEDMAN RESULTS
    # ========================================================

    friedman_output = friedman.copy()

    # Use the ACTUAL columns from friedman_results.csv.
    #
    # criterion
    # N
    # k
    # friedman_chi_square
    # friedman_p
    # kendall_W
    # significant_alpha_0.05

    friedman_output[
        "p_value_formatted"
    ] = friedman_output[
        "friedman_p"
    ].apply(format_p)

    friedman_output.to_csv(
        OUTPUT_DIR
        / "results_friedman_significant.csv",
        index=False
    )

    # ========================================================
    # 5. WILCOXON + HOLM RESULTS
    # ========================================================

    wilcoxon_output = wilcoxon.copy()

    if "holm_adjusted_p" in wilcoxon_output.columns:

        wilcoxon_output[
            "significant_alpha_0.05"
        ] = (
            pd.to_numeric(
                wilcoxon_output[
                    "holm_adjusted_p"
                ],
                errors="coerce"
            ) < 0.05
        )

    wilcoxon_output.to_csv(
        OUTPUT_DIR
        / "results_wilcoxon_significant.csv",
        index=False
    )

    # ========================================================
    # 6. PAIRWISE OVERALL RESULTS
    # ========================================================

    pairwise_output = pairwise.copy()

    if "holm_adjusted_p" in pairwise_output.columns:

        pairwise_output[
            "significant_alpha_0.05"
        ] = (
            pd.to_numeric(
                pairwise_output[
                    "holm_adjusted_p"
                ],
                errors="coerce"
            ) < 0.05
        )

    pairwise_output.to_csv(
        OUTPUT_DIR
        / "results_pairwise_overall.csv",
        index=False
    )

    # ========================================================
    # 7. ERROR ANALYSIS
    # ========================================================

    # Use the ACTUAL columns from
    # error_analysis_summary_final.csv:
    #
    # error_code
    # error_category
    # number_of_errors
    # total_responses
    # responses_affected_percent

    errors_output = errors.copy()

    errors_output.to_csv(
        OUTPUT_DIR
        / "results_error_analysis.csv",
        index=False
    )

    # ========================================================
    # 8. TEXT REPORT
    # ========================================================

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "HINGLISH BENCH RESULTS REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    # ========================================================
    # DATASET / EVALUATION
    # ========================================================

    report.append(
        "DATASET / EVALUATION"
    )

    report.append(
        "-" * 70
    )

    if "model" in master.columns:

        models = master[
            "model"
        ].tolist()

        report.append(
            "Models evaluated: "
            + ", ".join(
                map(str, models)
            )
        )

    if "samples" in master.columns:

        total_samples = int(
            master["samples"].sum()
        )

        report.append(
            f"Total model-response evaluations: "
            f"{total_samples}"
        )

    report.append("")

    # ========================================================
    # MODEL-LEVEL RESULTS
    # ========================================================

    report.append(
        "MODEL-LEVEL RESULTS"
    )

    report.append(
        "-" * 70
    )

    required_model_columns = {
        "model",
        "overall_mean",
        "overall_std",
    }

    if required_model_columns.issubset(
        master.columns
    ):

        for _, row in master.iterrows():

            report.append(
                f"{row['model']}: "
                f"Overall Mean = "
                f"{float(row['overall_mean']):.3f}, "
                f"SD = "
                f"{float(row['overall_std']):.3f}"
            )

    report.append("")

    # ========================================================
    # LINGUISTIC METRICS
    # ========================================================

    report.append(
        "LINGUISTIC METRICS"
    )

    report.append(
        "-" * 70
    )

    linguistic_columns = {
        "model",
        "avg_CMI",
        "avg_M_index",
        "avg_SyMCoM_imbalance",
    }

    if linguistic_columns.issubset(
        linguistic.columns
    ):

        for _, row in linguistic.iterrows():

            report.append(
                f"{row['model']}: "
                f"CMI="
                f"{float(row['avg_CMI']):.4f}, "
                f"M-index="
                f"{float(row['avg_M_index']):.4f}, "
                f"SyMCoM imbalance="
                f"{float(row['avg_SyMCoM_imbalance']):.4f}"
            )

    report.append("")

    # ========================================================
    # FRIEDMAN TEST
    # ========================================================

    report.append(
        "FRIEDMAN TEST"
    )

    report.append(
        "-" * 70
    )

    for _, row in friedman.iterrows():

        criterion = row[
            "criterion"
        ]

        statistic = float(
            row[
                "friedman_chi_square"
            ]
        )

        p_value = float(
            row[
                "friedman_p"
            ]
        )

        kendall_w = float(
            row[
                "kendall_W"
            ]
        )

        significant = bool(
            row[
                "significant_alpha_0.05"
            ]
        )

        significance_text = (
            "Significant"
            if significant
            else "Not significant"
        )

        report.append(
            f"{criterion}: "
            f"Statistic={statistic:.4f}, "
            f"p={format_p(p_value)}, "
            f"Kendall's W={kendall_w:.4f}, "
            f"{significance_text}"
        )

    report.append("")

    # ========================================================
    # WILCOXON + HOLM
    # ========================================================

    report.append(
        "WILCOXON PAIRWISE TESTS WITH "
        "HOLM CORRECTION"
    )

    report.append(
        "-" * 70
    )

    if "holm_adjusted_p" in wilcoxon.columns:

        significant_rows = wilcoxon[
            pd.to_numeric(
                wilcoxon[
                    "holm_adjusted_p"
                ],
                errors="coerce"
            ) < 0.05
        ]

        if len(significant_rows) == 0:

            report.append(
                "No pairwise comparisons remained "
                "significant after Holm correction "
                "at alpha=0.05."
            )

        else:

            for _, row in significant_rows.iterrows():

                criterion = row.get(
                    "criterion",
                    ""
                )

                model_a = row.get(
                    "model_a",
                    ""
                )

                model_b = row.get(
                    "model_b",
                    ""
                )

                p_value = row[
                    "holm_adjusted_p"
                ]

                report.append(
                    f"{criterion}: "
                    f"{model_a} vs "
                    f"{model_b}, "
                    f"Holm-adjusted p="
                    f"{format_p(p_value)}"
                )

    report.append("")

    # ========================================================
    # PAIRWISE PREFERENCE RESULTS
    # ========================================================

    report.append(
        "PAIRWISE PREFERENCE RESULTS"
    )

    report.append(
        "-" * 70
    )

    for _, row in pairwise.iterrows():

        model_a = row.get(
            "model_a",
            ""
        )

        model_b = row.get(
            "model_b",
            ""
        )

        wins_a = int(
            row.get(
                "model_a_wins",
                0
            )
        )

        wins_b = int(
            row.get(
                "model_b_wins",
                0
            )
        )

        ties = int(
            row.get(
                "ties",
                0
            )
        )

        holm_p = row.get(
            "holm_adjusted_p",
            np.nan
        )

        report.append(
            f"{model_a} vs {model_b}: "
            f"{wins_a} vs {wins_b} wins, "
            f"{ties} ties, "
            f"Holm p={format_p(holm_p)}"
        )

    report.append("")

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    report.append(
        "ERROR ANALYSIS"
    )

    report.append(
        "-" * 70
    )

    required_error_columns = {
        "error_code",
        "error_category",
        "number_of_errors",
        "total_responses",
        "responses_affected_percent",
    }

    if required_error_columns.issubset(
        errors.columns
    ):

        for _, row in errors.iterrows():

            report.append(
                f"{row['error_code']} - "
                f"{row['error_category']}: "
                f"{int(row['number_of_errors'])}/"
                f"{int(row['total_responses'])} "
                f"("
                f"{float(row['responses_affected_percent']):.2f}"
                f"%)"
            )

    else:

        report.append(
            "ERROR: Expected error-analysis "
            "columns were not found."
        )

    report.append("")

    # ========================================================
    # GENERATED OUTPUT FILES
    # ========================================================

    report.append(
        "GENERATED OUTPUT FILES"
    )

    report.append(
        "-" * 70
    )

    # Only list files that already exist.
    # Avoid listing results_report.txt before it is created.

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if file.is_file():

            report.append(
                file.name
            )

    report.append(
        "results_report.txt"
    )

    report.append("")

    report.append(
        "=" * 70
    )

    report.append(
        "REPORT GENERATION COMPLETE"
    )

    report.append(
        "=" * 70
    )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    report_text = "\n".join(
        report
    )

    report_file = (
        OUTPUT_DIR
        / "results_report.txt"
    )

    report_file.write_text(
        report_text,
        encoding="utf-8"
    )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 46)
    print(
        "RESULTS REPORT GENERATED SUCCESSFULLY"
    )
    print("=" * 46)

    print()
    print("Output folder:")
    print(OUTPUT_DIR)

    print()
    print("Files created:")

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f"  - {file.name}"
            )

    print()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
