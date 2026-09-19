import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = r".\outputs\llm_judge"
OUTPUT_DIR = r".\outputs\error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FILE PATHS
# ============================================================

JUDGE_FILE = os.path.join(
    INPUT_DIR,
    "hinglish_bench_llm_judge_results.csv"
)

GENERATION_FILES = {
    "HingGPT":
        r".\outputs\hinglish_bench\hinggpt_hinglish_bench_generations.csv",

    "Phi-3.5-mini":
        r".\outputs\hinglish_bench\phi35_mini_hinglish_bench_generations.csv",

    "Qwen2.5-3B":
        r".\outputs\hinglish_bench\qwen25_3b_hinglish_bench_generations.csv",

    "Qwen2.5-7B":
        r".\outputs\hinglish_bench\qwen25_7b_hinglish_bench_generations.csv",
}


# ============================================================
# ERROR CATEGORIES
# ============================================================

ERROR_CATEGORIES = {
    "E1": "English-dominant output",
    "E2": "Hindi-dominant output",
    "E3": "Unnatural code-switching",
    "E4": "Grammatical error",
    "E5": "Repetition",
    "E6": "Prompt misunderstanding",
    "E7": "Incomplete response",
    "E8": "Spelling/transliteration error",
    "E9": "Hallucination/factual error",
    "E10": "Irrelevant response",
}


# ============================================================
# LOAD JUDGE RESULTS
# ============================================================

print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

if not os.path.exists(JUDGE_FILE):
    raise FileNotFoundError(
        f"\nLLM Judge results not found:\n{JUDGE_FILE}"
    )

judge_df = pd.read_csv(JUDGE_FILE)

print("\nLLM Judge results loaded")
print("Shape:", judge_df.shape)


# ============================================================
# LOAD GENERATION DATA
# ============================================================

generation_dfs = {}

for model, file_path in GENERATION_FILES.items():

    print("\n" + "-" * 70)
    print(model)

    if not os.path.exists(file_path):
        print("WARNING: File not found:")
        print(file_path)
        continue

    df = pd.read_csv(file_path)

    generation_dfs[model] = df

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


# ============================================================
# RESPONSE COLUMN MAPPING
# ============================================================

# Your actual files have different response-column names.

RESPONSE_COLUMNS = {
    "HingGPT": "generated_response",
    "Phi-3.5-mini": "response",
    "Qwen2.5-3B": "generated_text",
    "Qwen2.5-7B": "response",
}


# ============================================================
# AUTOMATIC SCREENING FUNCTIONS
# ============================================================

def detect_repetition(text):

    if not isinstance(text, str):
        return False

    words = text.lower().split()

    if len(words) < 8:
        return False

    unique_ratio = len(set(words)) / len(words)

    return unique_ratio <= 0.55


def detect_incomplete(text):

    if not isinstance(text, str):
        return True

    text = text.strip()

    if len(text) == 0:
        return True

    if len(text.split()) < 3:
        return True

    return False


def detect_language_dominance(text):

    """
    This is only an initial screening heuristic.
    Final E1/E2 labels should be manually verified.
    """

    if not isinstance(text, str):
        return None

    words = text.lower().split()

    if len(words) < 15:
        return None

    hindi_markers = {
        "hai", "hain", "tha", "thi", "the",
        "mujhe", "mujh", "aap", "tum",
        "kya", "kaise", "kyu", "kyon",
        "nahi", "nahin", "bahut", "acha",
        "achha", "karna", "karo", "hoga",
        "ho", "mein", "mai", "mera",
        "meri", "mere", "ke", "ki", "ka",
        "se", "ko", "par", "yeh", "yah",
        "woh", "wo", "hai", "chahiye"
    }

    english_markers = {
        "the", "is", "are", "was", "were",
        "this", "that", "these", "those",
        "with", "from", "because", "should",
        "would", "could", "can", "please",
        "what", "how", "why", "when",
        "where", "and", "but", "for",
        "you", "your", "they", "their",
        "have", "has", "been", "will"
    }

    hindi_count = sum(
        1 for word in words
        if word.strip(".,!?;:") in hindi_markers
    )

    english_count = sum(
        1 for word in words
        if word.strip(".,!?;:") in english_markers
    )

    total = len(words)

    hindi_ratio = hindi_count / total
    english_ratio = english_count / total

    if english_ratio >= 0.35 and hindi_ratio <= 0.10:
        return "E1"

    if hindi_ratio >= 0.35 and english_ratio <= 0.10:
        return "E2"

    return None


# ============================================================
# CREATE SCREENING DATA
# ============================================================

error_rows = []

for model, df in generation_dfs.items():

    response_col = RESPONSE_COLUMNS.get(model)

    if response_col not in df.columns:
        print(
            f"\nERROR: Response column '{response_col}' "
            f"not found for {model}"
        )
        continue

    print(
        f"\nProcessing {model} "
        f"using column: {response_col}"
    )

    for idx, row in df.iterrows():

        response = row[response_col]

        detected_errors = []

        # ----------------------------------------------------
        # E5 - Repetition
        # ----------------------------------------------------

        if detect_repetition(response):
            detected_errors.append("E5")

        # ----------------------------------------------------
        # E7 - Incomplete response
        # ----------------------------------------------------

        if detect_incomplete(response):
            detected_errors.append("E7")

        # ----------------------------------------------------
        # E1 / E2 - Language dominance
        # ----------------------------------------------------

        language_error = detect_language_dominance(response)

        if language_error:
            detected_errors.append(language_error)

        # ----------------------------------------------------
        # Remove duplicate labels
        # ----------------------------------------------------

        detected_errors = list(dict.fromkeys(detected_errors))

        error_rows.append({
            "model": model,
            "sample": idx + 1,
            "category": (
                row["category"]
                if "category" in df.columns
                else ""
            ),
            "prompt": (
                row["prompt"]
                if "prompt" in df.columns
                else ""
            ),
            "response": response,
            "automatic_errors": ", ".join(
                detected_errors
            )
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

error_df = pd.DataFrame(error_rows)

print("\n" + "=" * 70)
print("SCREENING RESULT")
print("=" * 70)

print("Total responses processed:", len(error_df))

print(
    "\nResponses per model:"
)

print(
    error_df["model"].value_counts()
)


# ============================================================
# SAVE SCREENING FILE
# ============================================================

screening_file = os.path.join(
    OUTPUT_DIR,
    "error_analysis_screening.csv"
)

error_df.to_csv(
    screening_file,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nAutomatic screening saved:"
)
print(screening_file)


# ============================================================
# ERROR SUMMARY
# ============================================================

summary_rows = []

for model in GENERATION_FILES.keys():

    model_df = error_df[
        error_df["model"].eq(model)
    ]

    total_responses = int(len(model_df))

    for code, description in ERROR_CATEGORIES.items():

        # Create a boolean Series for this error code
        mask = (
            model_df["automatic_errors"]
            .fillna("")
            .str.split(", ")
            .apply(
                lambda errors: code in errors
            )
        )

        error_count = int(mask.sum())

        if total_responses > 0:
            percentage = (
                error_count /
                total_responses
            ) * 100
        else:
            percentage = 0.0

        summary_rows.append({
            "model": model,
            "error_code": code,
            "error_category": description,
            "number_of_errors": error_count,
            "total_responses": total_responses,
            "responses_affected_percent": round(
                percentage,
                2
            )
        })


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "error_analysis_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)


# ============================================================
# SAVE MODEL-WISE SUMMARY
# ============================================================

model_summary = (
    summary_df
    .pivot(
        index="error_category",
        columns="model",
        values="responses_affected_percent"
    )
    .reset_index()
)

model_summary_file = os.path.join(
    OUTPUT_DIR,
    "error_analysis_model_comparison.csv"
)

model_summary.to_csv(
    model_summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nGenerated files:")

print(
    "1.",
    screening_file
)

print(
    "2.",
    summary_file
)

print(
    "3.",
    model_summary_file
)

print("\nTotal responses:", len(error_df))
print("Expected responses: 136")

if len(error_df) == 136:
    print("STATUS: ALL 136 RESPONSES PROCESSED SUCCESSFULLY")
else:
    print(
        "WARNING: Expected 136 responses "
        f"but processed {len(error_df)}"
    )