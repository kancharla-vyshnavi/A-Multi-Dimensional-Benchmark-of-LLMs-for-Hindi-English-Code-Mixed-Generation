import pandas as pd
import re
import os

INPUT = "outputs/controlled_generation/benchmark_generations_v3.csv"
OUT_DIR = "outputs/hinglish_metrics_v3"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

def metrics(text):
    text = str(text)

    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    spaces = len(re.findall(r"\s", text))
    total_chars = len(text)

    alphabetic = devanagari + latin

    return pd.Series({
        "total_chars": total_chars,
        "devanagari_chars": devanagari,
        "latin_chars": latin,
        "digits": digits,
        "spaces": spaces,
        "devanagari_ratio": devanagari / total_chars if total_chars else 0,
        "latin_ratio": latin / total_chars if total_chars else 0,
        "devanagari_share_alphabetic": (
            devanagari / alphabetic if alphabetic else 0
        ),
        "latin_share_alphabetic": (
            latin / alphabetic if alphabetic else 0
        ),
        "mixed_script": int(devanagari > 0 and latin > 0),
        "roman_dominant": int(latin > devanagari),
        "devanagari_present": int(devanagari > 0),
    })

metric_df = df["response"].apply(metrics)

result = pd.concat(
    [df[["row_index", "model"]], metric_df],
    axis=1
)

response_file = f"{OUT_DIR}/response_level_hinglish_metrics_v3.csv"
result.to_csv(response_file, index=False)

summary = (
    result.groupby("model")
    .agg(
        responses=("row_index", "count"),
        avg_devanagari_ratio=("devanagari_ratio", "mean"),
        avg_latin_ratio=("latin_ratio", "mean"),
        avg_devanagari_share=("devanagari_share_alphabetic", "mean"),
        avg_latin_share=("latin_share_alphabetic", "mean"),
        mixed_script_rate=("mixed_script", "mean"),
        roman_dominant_rate=("roman_dominant", "mean"),
        devanagari_present_rate=("devanagari_present", "mean"),
    )
    .reset_index()
)

summary_file = f"{OUT_DIR}/model_level_hinglish_metrics_v3.csv"
summary.to_csv(summary_file, index=False)

print("\n=== HINGLISH-SPECIFIC METRICS ===")
print(summary.to_string(index=False))

print("\nCreated:")
print(response_file)
print(summary_file)
