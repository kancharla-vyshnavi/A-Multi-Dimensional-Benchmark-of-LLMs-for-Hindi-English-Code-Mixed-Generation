from pathlib import Path
import pandas as pd
import subprocess
import sys
from datetime import datetime


# ============================================================
# HINGLISH LLM — FINAL REMAINING PIPELINE
#
# E1-E10
#    ↓
# Hinglish Metrics
#    ↓
# Final Sensitivity
#    ↓
# Final Tables
#    ↓
# Final Figures
#
# IMPORTANT:
# Existing completed 1000/4000 outputs are reused.
# No expensive model generation/judging is rerun.
# ============================================================


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

ERROR_ANALYSIS = OUTPUTS / "error_analysis"
HINGLISH = OUTPUTS / "hinglish_metrics"
SENSITIVITY = OUTPUTS / "final_robustness"
TABLES = OUTPUTS / "final_research_tables"
CHARTS = OUTPUTS / "final_charts"


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_script(script_path):
    script_path = Path(script_path)

    if not script_path.exists():
        print(f"[MISSING SCRIPT] {script_path}")
        return False

    print(f"\n[RUNNING] {script_path.name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT
    )

    if result.returncode == 0:
        print(f"[OK] {script_path.name}")
        return True

    print(f"[FAILED] {script_path.name}")
    return False


def csv_rows(path):
    try:
        return len(pd.read_csv(path))
    except Exception:
        return None


# ============================================================
# STAGE 1 — E1–E10
# ============================================================

def stage_e1_e10():

    header("STAGE 1 — E1–E10 ERROR ANALYSIS")

    # Only REAL annotation CSVs are accepted.
    # Python annotation apps are NOT accepted.

    candidates = [
        OUTPUTS / "error_analysis" / "manual_error_annotation_final.csv",
        OUTPUTS / "error_analysis" / "error_analysis_summary_final.csv",
        OUTPUTS / "error_analysis" / "error_analysis_model_comparison_final.csv",
    ]

    annotation_file = None

    for path in candidates:
        if path.exists():
            annotation_file = path
            break

    if annotation_file is None:

        print("[BLOCKED] Manual E1–E10 annotation CSV not found.")
        print()
        print("Expected:")
        print("  outputs/error_analysis/manual_error_annotation_final.csv")
        print()
        print("The existing human_annotation_app.py is only an annotation")
        print("interface and cannot be treated as completed annotations.")
        print()
        print("No E1–E10 labels will be fabricated.")

        return "BLOCKED"

    print(f"[FOUND] {annotation_file}")

    # If the final analysis script exists, run it.
    scripts = [
        ERROR_ANALYSIS / "analyze_manual_error_annotation.py",
        ERROR_ANALYSIS / "final_error_analysis.py",
        ERROR_ANALYSIS / "analyze_error_analysis.py",
    ]

    for script in scripts:
        if script.exists():
            print(f"[INFO] Found E1–E10 analysis script: {script.name}")
            run_script(script)
            break

    rows = csv_rows(annotation_file)

    if rows is not None:
        print(f"[OK] Annotation rows: {rows}")

    return "PASS"


# ============================================================
# STAGE 2 — HINGLISH METRICS
# ============================================================

def stage_hinglish_metrics():

    header("STAGE 2 — HINGLISH METRICS")

    required = [
        "code_mixing_index.csv",
        "code_mixing_summary.csv",
        "model_level_hinglish_metrics.csv",
        "response_level_hinglish_metrics.csv",
    ]

    missing = []

    for filename in required:

        path = HINGLISH / filename

        if path.exists():

            rows = csv_rows(path)

            if rows is not None:
                print(f"[OK] {filename} ({rows} rows)")
            else:
                print(f"[OK] {filename}")

        else:
            print(f"[MISSING] {filename}")
            missing.append(filename)

    # Existing outputs are reused.
    # Only calculate if required outputs are missing.

    if missing:

        script = HINGLISH / "calculate_hinglish_metrics.py"

        if script.exists():

            print()
            print("[INFO] Hinglish metric outputs missing.")
            print("[INFO] Running calculation script...")

            if not run_script(script):
                return "CHECK"

        else:
            print("[CHECK] Hinglish metrics script not found.")
            return "CHECK"

    # Re-check
    still_missing = [
        f for f in required
        if not (HINGLISH / f).exists()
    ]

    if still_missing:
        print("[CHECK] Missing Hinglish metric files:")
        for f in still_missing:
            print(" ", f)
        return "CHECK"

    print("[PASS] Hinglish metrics available.")
    return "PASS"


# ============================================================
# STAGE 3 — FINAL SENSITIVITY
# ============================================================

def stage_sensitivity():

    header("STAGE 3 — FINAL SENSITIVITY ANALYSIS")

    output = SENSITIVITY / "missing_data_sensitivity.csv"
    script = SENSITIVITY / "missing_data_sensitivity.py"

    if output.exists():

        rows = csv_rows(output)

        print(
            f"[OK] Existing sensitivity output "
            f"({rows if rows is not None else 'unknown'} rows)"
        )

        print("[INFO] Existing result will be reused.")
        return "PASS"

    if script.exists():

        print("[INFO] Sensitivity output missing.")
        print("[INFO] Running sensitivity analysis...")

        if run_script(script):
            if output.exists():
                print("[PASS] Sensitivity analysis completed.")
                return "PASS"

        return "CHECK"

    print("[MISSING] missing_data_sensitivity.py")
    return "CHECK"


# ============================================================
# STAGE 4 — FINAL TABLES
# ============================================================

def stage_tables():

    header("STAGE 4 — FINAL RESEARCH TABLES")

    required = [
        "table_category.csv",
        "table_error_analysis.csv",
        "table_overall.csv",
        "table_statistics.csv",
    ]

    missing = []

    for filename in required:

        path = TABLES / filename

        if path.exists():

            rows = csv_rows(path)

            if rows is not None:
                print(f"[OK] {filename} ({rows} rows)")
            else:
                print(f"[OK] {filename}")

        else:
            print(f"[MISSING] {filename}")
            missing.append(filename)

    if missing:

        print()
        print("[INFO] Some final tables are missing.")

        # Search for a table-building script.
        scripts = list(OUTPUTS.rglob("*.py"))

        table_scripts = [
            p for p in scripts
            if (
                "table" in p.name.lower()
                or "final" in p.name.lower()
            )
        ]

        if table_scripts:

            print("[INFO] Candidate table scripts:")

            for p in table_scripts[:10]:
                print(" ", p.relative_to(ROOT))

            # Do not blindly run arbitrary scripts.
            print()
            print(
                "[CHECK] Existing table files are incomplete, "
                "but no dedicated table-builder was safely identified."
            )

            return "CHECK"

        return "CHECK"

    print("[PASS] Final research tables available.")
    return "PASS"


# ============================================================
# STAGE 5 — FINAL FIGURES
# ============================================================

def stage_figures():

    header("STAGE 5 — FINAL FIGURES")

    required = [
        "category_performance.png",
        "code_mixing_index.png",
        "independent_judge_scores.png",
        "pairwise_win_rates.png",
    ]

    missing = []

    for filename in required:

        path = CHARTS / filename

        if path.exists():

            size_kb = path.stat().st_size / 1024

            print(
                f"[OK] {filename} "
                f"({size_kb:.1f} KB)"
            )

        else:

            print(f"[MISSING] {filename}")
            missing.append(filename)

    if not missing:

        print("[PASS] Final figures available.")
        return "PASS"

    # Run the known chart generator if available.
    script = CHARTS / "generate_final_charts.py"

    if script.exists():

        print()
        print("[INFO] Missing figures detected.")
        print("[INFO] Running final chart generator...")

        if run_script(script):

            still_missing = [
                f for f in required
                if not (CHARTS / f).exists()
            ]

            if not still_missing:
                print("[PASS] Final figures generated.")
                return "PASS"

    print("[CHECK] Final figures incomplete.")
    return "CHECK"


# ============================================================
# FINAL REPORT
# ============================================================

def create_report(results):

    header("FINAL COMPLETION REPORT")

    report = OUTPUTS / "FINAL_REMAINING_PIPELINE_REPORT.txt"

    with open(report, "w", encoding="utf-8") as f:

        f.write(
            "HINGLISH LLM — FINAL REMAINING PIPELINE REPORT\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            "Generated: "
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + "\n\n"
        )

        for stage, status in results.items():
            f.write(f"{stage}: {status}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("IMPORTANT\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "Existing 1,000-prompt benchmark and 4,000-response "
            "evaluation outputs were reused.\n\n"
        )

        f.write(
            "No CPT, controlled generation, Mistral judging, "
            "OLMo judging, reliability or main statistical analysis "
            "was rerun.\n\n"
        )

        if results["E1-E10"] == "BLOCKED":
            f.write(
                "E1-E10 is blocked because the completed manual "
                "annotation CSV is absent. The annotation interface "
                "was not treated as annotation data.\n\n"
            )

        f.write(
            "No unsupported E1-E10 labels were fabricated.\n"
        )

    print(f"[OK] Report:")
    print(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HINGLISH LLM — FINAL REMAINING PIPELINE")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # 1. E1-E10
    # --------------------------------------------------------

    results["E1-E10"] = stage_e1_e10()

    # --------------------------------------------------------
    # 2. Hinglish Metrics
    # --------------------------------------------------------

    results["Hinglish Metrics"] = stage_hinglish_metrics()

    # --------------------------------------------------------
    # 3. Final Sensitivity
    # --------------------------------------------------------

    results["Final Sensitivity"] = stage_sensitivity()

    # --------------------------------------------------------
    # 4. Final Tables
    # --------------------------------------------------------

    results["Final Tables"] = stage_tables()

    # --------------------------------------------------------
    # 5. Final Figures
    # --------------------------------------------------------

    results["Final Figures"] = stage_figures()

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    create_report(results)

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    header("FINAL STATUS")

    for stage, status in results.items():
        print(f"{stage:<30} {status}")

    print()
    print("=" * 70)
    print("REMAINING PIPELINE FINISHED")
    print("=" * 70)

    print()
    print("Sequence:")
    print("E1-E10")
    print("   ↓")
    print("Hinglish Metrics")
    print("   ↓")
    print("Final Sensitivity")
    print("   ↓")
    print("Final Tables")
    print("   ↓")
    print("Final Figures")
    print()


if __name__ == "__main__":
    main()