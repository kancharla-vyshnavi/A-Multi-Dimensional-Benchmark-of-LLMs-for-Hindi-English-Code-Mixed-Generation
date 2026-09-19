import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
JUDGE_FILE = Path(
    r".\outputs\llm_judge\hinglish_bench_llm_judge_results_final.csv"
)

LING_FILE = Path(
    r".\outputs\linguistic_evaluation\hinglish_bench_official_metrics_summary.csv"
)

OUT_FILE = Path(
    r".\outputs\MASTER_RESULTS.csv"
)

# ---------------------------------------------------------
# Read data
# ---------------------------------------------------------
judge = pd.read_csv(JUDGE_FILE)
ling = pd.read_csv(LING_FILE)

# ---------------------------------------------------------
# LLM Judge descriptive statistics
# ---------------------------------------------------------
criteria = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

stats = (
    judge.groupby("model")[criteria]
    .agg(["mean", "std"])
    .round(3)
)

# Flatten MultiIndex columns
stats.columns = [
    f"{criterion}_{stat}"
    for criterion, stat in stats.columns
]

stats = stats.reset_index()

# ---------------------------------------------------------
# Normalize model names for merging
# ---------------------------------------------------------
model_map = {
    "HingGPT": "hinggpt",
    "Phi-3.5-mini": "phi35_mini",
    "Qwen2.5-3B": "qwen25_3b",
    "Qwen2.5-7B": "qwen25_7b",
}

stats["model_key"] = stats["model"].map(model_map)

# ---------------------------------------------------------
# Select official linguistic metrics
# ---------------------------------------------------------
ling = ling.rename(columns={"model": "model_key"})

ling_cols = [
    "model_key",
    "samples",
    "avg_CMI",
    "avg_M_index",
    "avg_SyMCoM_imbalance",
    "total_hindi_tokens",
    "total_english_tokens",
    "total_other_tokens",
]

ling = ling[ling_cols]

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------
master = stats.merge(
    ling,
    on="model_key",
    how="left",
)

# Keep readable model name
master = master.drop(columns=["model_key"])

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
master.to_csv(OUT_FILE, index=False)

print("\nMASTER RESULTS")
print("=" * 100)
print(master.to_string(index=False))

print("\nSaved to:")
print(OUT_FILE)