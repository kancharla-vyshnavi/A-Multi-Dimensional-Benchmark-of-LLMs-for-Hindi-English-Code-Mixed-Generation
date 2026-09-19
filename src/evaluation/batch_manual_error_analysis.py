import os
import pandas as pd
import re

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r".\outputs\error_analysis\manual_error_annotation.csv"

OUTPUT_DIR = r".\outputs\error_analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "manual_error_annotation_final.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# LOAD FILE
# ============================================================

print("=" * 70)
print("BATCH ERROR ANALYSIS")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nInput file loaded")
print("Shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "model",
    "sample",
    "category",
    "prompt",
    "response",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


def word_count(text):
    return len(clean_text(text).split())


def repeated_word_ratio(text):
    """
    Detects obvious repeated words.
    This is only a screening signal.
    """

    text = clean_text(text).lower()

    if not text:
        return 0

    words = re.findall(r"\b[\w']+\b", text)

    if len(words) < 8:
        return 0

    counts = pd.Series(words).value_counts()

    return counts.iloc[0] / len(words)


def repeated_line_or_phrase(text):
    """
    Detect obvious repeated phrases / lines.
    """

    text = clean_text(text)

    if not text:
        return False

    lines = [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) >= 4:
        unique_lines = len(set(lines))

        if unique_lines / len(lines) <= 0.50:
            return True

    return False


def detect_repetition(text):

    text = clean_text(text)

    if not text:
        return False

    ratio = repeated_word_ratio(text)

    if ratio >= 0.30:
        return True

    if repeated_line_or_phrase(text):
        return True

    # Very large repeated emoji sequences
    emoji_pattern = r"(?:😂|🤣|❤️|🇮🇳|🙏|😊|😍|🔥|👍|👏)"

    emojis = re.findall(emoji_pattern, text)

    if len(emojis) >= 15:
        return True

    return False


def detect_incomplete(text):

    text = clean_text(text)

    if not text:
        return True

    words = text.split()

    if len(words) < 3:
        return True

    # obvious unfinished endings
    if text.endswith(("...", "..", "—", "-", ":")):
        return True

    return False


def detect_language_dominance(text):

    text = clean_text(text)

    if len(text.split()) < 15:
        return None

    words = [
        w.lower().strip(".,!?;:'\"()[]{}")
        for w in text.split()
    ]

    hindi_markers = {
        "hai", "hain", "tha", "thi", "the",
        "mujhe", "mujh", "aap", "tum",
        "kya", "kaise", "kyu", "kyon",
        "nahi", "nahin", "bahut", "acha",
        "achha", "karna", "karo", "hoga",
        "ho", "mein", "mai", "mera",
        "meri", "mere", "ke", "ki", "ka",
        "se", "ko", "par", "yeh", "yah",
        "woh", "wo", "chahiye"
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
        word in hindi_markers
        for word in words
    )

    english_count = sum(
        word in english_markers
        for word in words
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
# INITIALIZE FINAL COLUMNS
# ============================================================

final_columns = [
    "E1_English_dominant",
    "E2_Hindi_dominant",
    "E3_Unnatural_code_switching",
    "E4_Grammatical_error",
    "E5_Repetition",
    "E6_Prompt_misunderstanding",
    "E7_Incomplete_response",
    "E8_Spelling_transliteration_error",
    "E9_Hallucination_factual_error",
    "E10_Irrelevant_response",
    "Final_Error_Category",
    "Manual_Notes",
    "Review_Status",
]

for col in final_columns:

    if col not in df.columns:
        df[col] = ""

# Make sure object/string columns can accept text
for col in [
    "Final_Error_Category",
    "Manual_Notes",
    "Review_Status",
]:
    df[col] = df[col].fillna("").astype(str)


# ============================================================
# BATCH SCREENING
# ============================================================

print("\nProcessing all responses...")
print("-" * 70)

for idx in range(len(df)):

    response = clean_text(
        df.loc[idx, "response"]
    )

    prompt = clean_text(
        df.loc[idx, "prompt"]
    )

    detected = []

    # --------------------------------------------------------
    # E5
    # --------------------------------------------------------

    if detect_repetition(response):
        detected.append("E5")

    # --------------------------------------------------------
    # E7
    # --------------------------------------------------------

    if detect_incomplete(response):
        detected.append("E7")

    # --------------------------------------------------------
    # E1 / E2
    # --------------------------------------------------------

    language_error = detect_language_dominance(
        response
    )

    if language_error:
        detected.append(language_error)

    detected = list(dict.fromkeys(detected))

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # We do NOT automatically assign:
    #
    # E3 unnatural code-switching
    # E4 grammatical error
    # E6 prompt misunderstanding
    # E8 spelling/transliteration
    # E9 hallucination/factual error
    # E10 irrelevant response
    #
    # because these require semantic/manual verification.
    # --------------------------------------------------------

    for code in ERROR_CATEGORIES:

        column_map = {
            "E1": "E1_English_dominant",
            "E2": "E2_Hindi_dominant",
            "E3": "E3_Unnatural_code_switching",
            "E4": "E4_Grammatical_error",
            "E5": "E5_Repetition",
            "E6": "E6_Prompt_misunderstanding",
            "E7": "E7_Incomplete_response",
            "E8": "E8_Spelling_transliteration_error",
            "E9": "E9_Hallucination_factual_error",
            "E10": "E10_Irrelevant_response",
        }

        col = column_map[code]

        # Preserve already manually reviewed labels
        existing = str(
            df.loc[idx, col]
        ).strip()

        if existing == "":
            df.loc[idx, col] = (
                1 if code in detected else 0
            )

    # --------------------------------------------------------
    # Store automatic screening reference
    # --------------------------------------------------------

    df.loc[idx, "Automatic_Screening_Reference"] = (
        ", ".join(detected)
        if detected
        else "NONE"
    )

    # --------------------------------------------------------
    # Do not overwrite manual review
    # --------------------------------------------------------

    status = str(
        df.loc[idx, "Review_Status"]
    ).strip()

    if status == "":
        df.loc[idx, "Review_Status"] = (
            "BATCH_SCREENED_NEEDS_SEMANTIC_REVIEW"
        )

    if (idx + 1) % 10 == 0 or idx == len(df) - 1:
        print(
            f"Processed {idx + 1}/{len(df)}"
        )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("BATCH SCREENING COMPLETED")
print("=" * 70)

print("\nTotal responses:", len(df))

print("\nModel counts:")
print(df["model"].value_counts())

print("\nOutput:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "E1/E2/E5/E7 are batch-screened where detectable."
)

print(
    "E3/E4/E6/E8/E9/E10 still require semantic verification."
)

print(
    "\nSTATUS: ALL RESPONSES PROCESSED"
)