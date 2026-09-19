import sys
import os
import glob
import pandas as pd

# Official Hinglish-Bench package
OFFICIAL_REPO = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "hinglish-bench-official",
    "hinglish-bench-main"
)

sys.path.insert(0, OFFICIAL_REPO)

from hinglish_bench.metrics import code_mix_stats


INPUT_DIR = r".\outputs\hinglish_bench"
OUTPUT_DIR = r".\outputs\linguistic_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_response_column(df):
    for col in ["response", "generated_response", "generated_text"]:
        if col in df.columns:
            return col

    raise ValueError(
        f"No response column found: {df.columns.tolist()}"
    )


files = glob.glob(
    os.path.join(INPUT_DIR, "*.csv")
)

all_results = []

for file in files:

    model_name = os.path.basename(file).replace(
        "_hinglish_bench_generations.csv", ""
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    df = pd.read_csv(file)

    response_col = get_response_column(df)

    print("Rows:", len(df))
    print("Response column:", response_col)

    model_results = []

    for i, row in df.iterrows():

        text = str(row[response_col])

        try:
            stats = code_mix_stats(text)

            counts = stats["token_counts"]

            result = {
                "model": model_name,
                "sample": i,
                "prompt": row.get("prompt", ""),
                "response": text,

                "hindi_tokens": counts.get("hi", 0),
                "english_tokens": counts.get("en", 0),
                "other_tokens": counts.get("other", 0),

                "CMI": stats["cmi"],
                "M_index": stats["m_index"],
                "SyMCoM_imbalance": stats["sycom_imbalance"],
            }

            model_results.append(result)
            all_results.append(result)

        except Exception as e:

            print(
                f"ERROR in {model_name}, sample {i}: {e}"
            )


    result_df = pd.DataFrame(model_results)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_official_metrics.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print("\nAverage CMI:",
          round(result_df["CMI"].mean(), 4))

    print("Average M-index:",
          round(result_df["M_index"].mean(), 4))

    print("Average SyMCoM imbalance:",
          round(result_df["SyMCoM_imbalance"].mean(), 4))

    print("Saved:", output_file)


# ---------------------------------------------------------
# Combined summary
# ---------------------------------------------------------

all_df = pd.DataFrame(all_results)

summary = (
    all_df
    .groupby("model")
    .agg(
        samples=("sample", "count"),
        avg_CMI=("CMI", "mean"),
        avg_M_index=("M_index", "mean"),
        avg_SyMCoM_imbalance=("SyMCoM_imbalance", "mean"),
        total_hindi_tokens=("hindi_tokens", "sum"),
        total_english_tokens=("english_tokens", "sum"),
        total_other_tokens=("other_tokens", "sum"),
    )
    .reset_index()
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "hinglish_bench_official_metrics_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)

print("\n" + "=" * 70)
print("OFFICIAL HINGLISH-BENCH METRICS COMPLETED")
print("=" * 70)

print("\n")
print(summary.to_string(index=False))

print("\nSaved:")
print(summary_csv)