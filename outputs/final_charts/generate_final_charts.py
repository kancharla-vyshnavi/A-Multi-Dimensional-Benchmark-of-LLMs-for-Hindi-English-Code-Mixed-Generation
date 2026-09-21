import pandas as pd
import matplotlib.pyplot as plt
import os

OUT = "outputs/final_charts_v3"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. INDEPENDENT JUDGE SCORES
# ============================================================

overall = pd.read_csv(
    "outputs/final_research_tables/table_overall_v3.csv"
)

plt.figure(figsize=(9, 5))
plt.bar(overall["Model"], overall["Independent_Judge_Mean"])
plt.ylabel("Independent Judge Mean Score")
plt.xlabel("Model")
plt.title("Independent Judge Evaluation")
plt.ylim(0, 5)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    f"{OUT}/independent_judge_scores.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 2. PAIRWISE WIN RATES
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(overall["Model"], overall["Pairwise_Win_Rate"] * 100)
plt.ylabel("Pairwise Win Rate (%)")
plt.xlabel("Model")
plt.title("Pairwise Evaluation")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    f"{OUT}/pairwise_win_rates.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 3. CATEGORY PERFORMANCE
# ============================================================

category = pd.read_csv(
    "outputs/final_research_tables/table_category_v3.csv"
)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

plt.figure(figsize=(12, 6))

for model in models:
    plt.plot(
        category["category_name"],
        category[model],
        marker="o",
        label=model
    )

plt.ylabel("Category Mean Score")
plt.xlabel("Category")
plt.title("Category-Level Evaluation")
plt.ylim(0, 5)
plt.xticks(rotation=35, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{OUT}/category_performance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 4. CODE-MIXING INDEX
# ============================================================

cmi = pd.read_csv(
    "outputs/hinglish_metrics_v3/code_mixing_summary_v3.csv"
)

plt.figure(figsize=(9, 5))
plt.bar(cmi["model"], cmi["avg_cmi_percent"])
plt.ylabel("Approximate Code-Mixing Index (%)")
plt.xlabel("Model")
plt.title("Lexicon-Based Approximate Code-Mixing Index")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    f"{OUT}/code_mixing_index.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# FINAL SUMMARY CSV
# ============================================================

summary = overall[
    [
        "Model",
        "Independent_Judge_Mean",
        "Valid",
        "Pairwise_Wins",
        "Pairwise_Losses",
        "Pairwise_Ties",
        "Pairwise_Win_Rate",
        "Category_Mean",
        "Category_SD"
    ]
].copy()

cmi_small = cmi[
    [
        "model",
        "avg_cmi_percent",
        "avg_english_share",
        "avg_hindi_share",
        "both_languages_rate"
    ]
].rename(columns={"model": "Model"})

final = summary.merge(
    cmi_small,
    on="Model",
    how="left"
)

final.to_csv(
    f"{OUT}/final_results_summary_v3.csv",
    index=False
)

print("\n=== FINAL CHARTS CREATED ===")

for f in [
    "independent_judge_scores.png",
    "pairwise_win_rates.png",
    "category_performance.png",
    "code_mixing_index.png",
    "final_results_summary_v3.csv"
]:
    print(f"{OUT}/{f}")
