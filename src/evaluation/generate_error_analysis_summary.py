import pandas as pd
import os

INPUT = r".\outputs\error_analysis\manual_error_annotation_final.csv"
OUT_DIR = r".\outputs\error_analysis"

df = pd.read_csv(INPUT)

models = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B"
]

errors = {
    "E1": "English_dominant",
    "E2": "Hindi_dominant",
    "E3": "Unnatural_code_switching",
    "E4": "Grammatical_error",
    "E5": "Repetition",
    "E6": "Prompt_misunderstanding",
    "E7": "Incomplete_response",
    "E8": "Spelling_transliteration_error",
    "E9": "Hallucination_factual_error",
    "E10": "Irrelevant_response"
}

rows = []

for model in models:

    subset = df[df["model"] == model]

    row = {
        "Model": model,
        "Total_Responses": len(subset)
    }

    for error_code, suffix in errors.items():

        column = f"{error_code}_{suffix}"

        count = int(subset[column].sum())
        percentage = round((count / len(subset)) * 100, 2)

        row[f"{error_code}_Count"] = count
        row[f"{error_code}_Percentage"] = percentage

    rows.append(row)

summary = pd.DataFrame(rows)

# Save complete table
output_file = os.path.join(
    OUT_DIR,
    "error_analysis_model_comparison.csv"
)

summary.to_csv(output_file, index=False)

print("=" * 70)
print("ERROR ANALYSIS SUMMARY GENERATED")
print("=" * 70)

print()
print(summary.to_string(index=False))

print()
print("Output:")
print(output_file)

print()
print("Responses per model:")
print(df["model"].value_counts())

print()
print("DONE ✅")