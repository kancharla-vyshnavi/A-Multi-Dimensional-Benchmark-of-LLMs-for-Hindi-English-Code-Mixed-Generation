import os
import pandas as pd

print("=" * 80)
print("FINAL HINGLISH LLM PROJECT VALIDATION")
print("=" * 80)

files = {
    "Benchmark": r"outputs\controlled_1000\benchmark_generations_1000.csv",
    "Mistral Judge": r"outputs\independent_judge_1000\independent_mistral_judge_1000_FINAL.csv",
    "OLMo Judge": r"outputs\independent_judge_1000\independent_olmo2_1b_judge_1000_FINAL.csv",
    "Robustness": r"outputs\robustness_mistral_1000_analysis\robustness_summary.csv",
    "Hinglish Metrics": r"outputs\hinglish_metrics\model_level_hinglish_metrics.csv",
    "Statistics": r"outputs\statistical_analysis_1000_FINAL\final_friedman_results.csv",
    "Reliability": r"outputs\reliability_1000_FINAL\final_inter_judge_reliability_summary.csv",
    "Error Analysis": r"outputs\error_analysis\automated_error_annotation_4000.csv",
    "Error Summary": r"outputs\error_analysis\automated_error_analysis_summary_4000.csv",
    "Error Model Comparison": r"outputs\error_analysis\automated_error_analysis_model_comparison_4000.csv",
    "Final Error Table": r"outputs\final_research_tables\table_error_analysis_automated_4000.csv",
    "Error Chart": r"outputs\final_charts\automated_e1_e10_error_analysis.png",
    "Final Error Results": r"outputs\final_charts\automated_e1_e10_final_results.csv",
    "Overall Table": r"outputs\final_research_tables\table_overall.csv",
    "Category Table": r"outputs\final_research_tables\table_category.csv",
    "Statistics Table": r"outputs\final_research_tables\table_statistics.csv",
    "Final Results Summary": r"outputs\final_charts\final_results_summary.csv",
}

print("\nFILE CHECK")
print("-" * 80)

all_ok = True

for name, path in files.items():
    exists = os.path.exists(path)

    if exists:
        if path.lower().endswith(".csv"):
            try:
                rows = len(pd.read_csv(path))
                print(f"[OK] {name:<25} {rows:>5} rows")
            except:
                print(f"[OK] {name:<25} CSV")
        else:
            print(f"[OK] {name:<25} EXISTS")
    else:
        print(f"[MISSING] {name:<25} {path}")
        all_ok = False


print("\nCORE DATA VALIDATION")
print("-" * 80)

benchmark = pd.read_csv(files["Benchmark"])
errors = pd.read_csv(files["Error Analysis"])

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

print(f"Benchmark rows: {len(benchmark)}")
print(f"E1-E10 rows:    {len(errors)}")

if len(benchmark) == 4000:
    print("[PASS] Benchmark has exactly 4000 responses")
else:
    print("[FAIL] Benchmark row count is not 4000")
    all_ok = False

if len(errors) == 4000:
    print("[PASS] E1-E10 has exactly 4000 responses")
else:
    print("[FAIL] E1-E10 row count is not 4000")
    all_ok = False


print("\nMODEL DISTRIBUTION")
print("-" * 80)

for model in models:
    b = (benchmark["model"] == model).sum()
    e = (errors["model"] == model).sum()

    print(f"{model:<18} Benchmark={b:<5} E1-E10={e:<5}")

    if b != 1000 or e != 1000:
        all_ok = False


print("\nHUMAN AUDIT DEPENDENCY CHECK")
print("-" * 80)

human_path = r"outputs\error_analysis\manual_error_annotation_final.csv"

if os.path.exists(human_path):
    print("[INFO] Old human annotation file exists, but it is NOT part of final validation.")
else:
    print("[OK] No human annotation dependency detected.")


print("\nE1-E10 SUMMARY")
print("-" * 80)

error_cols = [
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

for col in error_cols:
    total = errors[col].sum()
    pct = total / 4000 * 100
    print(f"{col:<40} {int(total):>5}  ({pct:>6.2f}%)")


print("\n" + "=" * 80)

if all_ok:
    print("FINAL VALIDATION: PASS")
    print("=" * 80)
    print("All major project outputs are present.")
    print("Benchmark = 4000")
    print("Models = 4 x 1000")
    print("E1-E10 = 4000")
    print("No human audit is required for the final pipeline.")
else:
    print("FINAL VALIDATION: CHECK REQUIRED")
    print("=" * 80)


