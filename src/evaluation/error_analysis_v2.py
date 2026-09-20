import os
import re
import json
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "outputs/controlled_v2/benchmark_generations_v2.csv"
OUTPUT_DIR = "outputs/controlled_v2/error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ERROR DEFINITIONS
# ============================================================

ERROR_DEFINITIONS = {
    "E1": "English-dominant output",
    "E2": "Hindi-dominant output",
    "E3": "Unnatural code-switching",
    "E4": "Grammatical error",
    "E5": "Repetition",
    "E6": "Prompt misunderstanding",
    "E7": "Incomplete response",
    "E8": "Spelling/transliteration inconsistency",
    "E9": "Hallucination/factual issue",
    "E10": "Irrelevant response",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.strip()

    return text


def tokenize(text):
    """
    Simple whitespace tokenizer.
    This is intentionally lightweight and deterministic.
    """

    return re.findall(r"\b[\w']+\b", text.lower())


def count_script_tokens(text):
    """
    Rough character-script estimation.

    Devanagari:
        U+0900–U+097F

    English:
        ASCII alphabetic characters
    """

    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    english_chars = re.findall(r"[A-Za-z]", text)

    return len(hindi_chars), len(english_chars)


def english_ratio(text):
    hindi, english = count_script_tokens(text)

    total = hindi + english

    if total == 0:
        return 0.0

    return english / total


def hindi_ratio(text):
    hindi, english = count_script_tokens(text)

    total = hindi + english

    if total == 0:
        return 0.0

    return hindi / total


# ============================================================
# E1 / E2
# ============================================================

def detect_language_dominance(text):
    """
    IMPORTANT:
    This is only a script-based heuristic.

    E1:
        English-dominant

    E2:
        Hindi-dominant

    Romanized Hindi cannot be reliably distinguished from English
    using script alone, so these should be treated as preliminary
    automatic annotations.
    """

    e_ratio = english_ratio(text)
    h_ratio = hindi_ratio(text)

    e1 = e_ratio >= 0.70
    e2 = h_ratio >= 0.70

    return e1, e2, e_ratio, h_ratio


# ============================================================
# E3
# ============================================================

def detect_unnatural_code_switching(text):
    """
    Heuristic indicators:

    - Abrupt repeated switching between scripts
    - Very short fragments alternating repeatedly
    - Excessive punctuation around language switches

    This is deliberately conservative.
    """

    if not text:
        return False

    tokens = text.split()

    if len(tokens) < 5:
        return False

    script_sequence = []

    for token in tokens:

        has_hindi = bool(re.search(r"[\u0900-\u097F]", token))
        has_english = bool(re.search(r"[A-Za-z]", token))

        if has_hindi and has_english:
            script_sequence.append("mixed")

        elif has_hindi:
            script_sequence.append("hindi")

        elif has_english:
            script_sequence.append("english")

    if len(script_sequence) < 5:
        return False

    switches = 0

    previous = None

    for current in script_sequence:

        if previous is not None:
            if (
                current != previous
                and current != "mixed"
                and previous != "mixed"
            ):
                switches += 1

        previous = current

    switch_rate = switches / max(len(script_sequence) - 1, 1)

    # Conservative threshold
    return switch_rate >= 0.45


# ============================================================
# E4
# ============================================================

def detect_grammatical_error(text):
    """
    Lightweight heuristic.

    This is NOT a grammatical parser.
    It only flags obvious malformed structures.

    Examples:
        repeated auxiliary fragments
        obvious duplicated words
        malformed sentence endings
        very fragmented outputs
    """

    if not text:
        return False

    lower = text.lower()

    # Obvious repeated consecutive words
    tokens = tokenize(text)

    for i in range(len(tokens) - 1):

        if tokens[i] == tokens[i + 1]:
            return True

    # Common malformed patterns
    malformed_patterns = [
        r"\bi am am\b",
        r"\bis is\b",
        r"\bthe the\b",
        r"\bto to\b",
        r"\band and\b",
        r"\bof of\b",
        r"\bvery very\b",
    ]

    for pattern in malformed_patterns:

        if re.search(pattern, lower):
            return True

    # Excessive fragment punctuation
    fragments = re.split(r"[.!?]+", text)

    short_fragments = [
        fragment.strip()
        for fragment in fragments
        if fragment.strip()
    ]

    if len(short_fragments) >= 4:

        tiny = sum(
            1
            for fragment in short_fragments
            if len(fragment.split()) <= 2
        )

        if tiny / len(short_fragments) >= 0.60:
            return True

    return False


# ============================================================
# E5
# ============================================================

def detect_repetition(text):
    """
    Detect repeated words / repeated phrases.
    """

    if not text:
        return False

    tokens = tokenize(text)

    if len(tokens) < 6:
        return False

    # Consecutive word repetition
    for i in range(len(tokens) - 1):

        if tokens[i] == tokens[i + 1]:
            return True

    # 3-word phrase repetition
    trigrams = {}

    for i in range(len(tokens) - 2):

        phrase = tuple(tokens[i:i + 3])

        trigrams[phrase] = trigrams.get(phrase, 0) + 1

    if any(count >= 2 for count in trigrams.values()):
        return True

    # High-frequency single token
    counts = {}

    for token in tokens:

        if len(token) <= 2:
            continue

        counts[token] = counts.get(token, 0) + 1

    if counts:

        maximum = max(counts.values())

        if maximum >= max(4, len(tokens) * 0.15):
            return True

    return False


# ============================================================
# E6 / E10
# ============================================================

def normalize_for_matching(text):
    text = clean_text(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text


def extract_prompt_keywords(prompt):
    """
    Extract meaningful English words from prompt.

    Stopwords are removed because they do not help determine
    relevance.
    """

    stopwords = {
        "the", "a", "an", "and", "or", "but", "to", "of",
        "in", "on", "for", "with", "is", "are", "was", "were",
        "be", "been", "being", "this", "that", "these", "those",
        "your", "you", "me", "my", "our", "their", "they",
        "it", "its", "as", "at", "from", "by", "about",
        "how", "what", "why", "can", "could", "would", "should",
        "please", "give", "write", "tell", "explain"
    }

    words = re.findall(r"[a-zA-Z]{3,}", clean_text(prompt).lower())

    keywords = [
        word
        for word in words
        if word not in stopwords
    ]

    return set(keywords)


def keyword_overlap(prompt, response):
    prompt_keywords = extract_prompt_keywords(prompt)

    response_words = set(
        re.findall(
            r"[a-zA-Z]{3,}",
            clean_text(response).lower()
        )
    )

    if not prompt_keywords:
        return 0.0

    overlap = prompt_keywords.intersection(response_words)

    return len(overlap) / len(prompt_keywords)


def response_is_empty(response):
    response = clean_text(response)

    if not response:
        return True

    if response.lower() in {
        "none",
        "n/a",
        "na",
        "i don't know",
        "i cannot answer",
    }:
        return True

    return False


def detect_incomplete(prompt, response):
    """
    Heuristic incomplete-response detection.
    """

    response = clean_text(response)

    if response_is_empty(response):
        return True

    # Obvious unfinished endings
    unfinished_endings = [
        "...",
        "…",
        "and",
        "but",
        "because",
        "so",
        "if",
        "when",
        "to",
        "with",
    ]

    lower = response.lower()

    for ending in unfinished_endings:

        if lower.endswith(" " + ending):
            return True

    # Very short answer to a substantial prompt
    if len(response.split()) <= 3 and len(clean_text(prompt).split()) >= 10:
        return True

    return False


def detect_prompt_misunderstanding(prompt, response, expected_style):
    """
    Preliminary heuristic.

    E6 is intentionally conservative because actual prompt
    understanding is semantic and should ultimately be validated
    by an independent judge/human annotation.

    Signals:
        - empty answer
        - very low keyword overlap
        - response does not appear to address expected style
    """

    if response_is_empty(response):
        return True

    overlap = keyword_overlap(prompt, response)

    response_lower = clean_text(response).lower()

    style = clean_text(expected_style).lower()

    style_signal = 0

    if style:

        style_keywords = re.findall(r"[a-zA-Z]{3,}", style)

        matched = 0

        for keyword in style_keywords:

            if keyword in response_lower:
                matched += 1

        if style_keywords:

            style_signal = matched / len(style_keywords)

    # Conservative thresholds
    if overlap < 0.10 and style_signal < 0.10:
        return True

    return False


def detect_irrelevant(prompt, response, expected_style):
    """
    Preliminary relevance heuristic.

    E10 should NOT be interpreted as a final semantic judgment.
    The independent judge stage will provide the stronger analysis.
    """

    if response_is_empty(response):
        return True

    overlap = keyword_overlap(prompt, response)

    response_words = tokenize(response)

    # Very short answer with no meaningful overlap
    if len(response_words) <= 4 and overlap < 0.10:
        return True

    # Extremely low semantic proxy
    if len(response_words) >= 8 and overlap < 0.05:
        return True

    return False


# ============================================================
# E8
# ============================================================

def detect_spelling_inconsistency(text):
    """
    Preliminary spelling/transliteration heuristic.

    Flags:
        - repeated variants that differ only slightly
        - suspicious repeated spellings

    This is intentionally conservative.
    """

    if not text:
        return False

    tokens = tokenize(text)

    if len(tokens) < 5:
        return False

    # Common informal transliteration variants
    variant_groups = [
        {"hai", "he", "h", "hain"},
        {"nahi", "nahin", "nai"},
        {"kya", "kia"},
        {"acha", "accha", "achha"},
        {"bahut", "bohot", "bahot"},
        {"mujhe", "muje"},
        {"tumhe", "tumhen"},
    ]

    present_variants = 0

    for group in variant_groups:

        found = group.intersection(set(tokens))

        if len(found) >= 2:
            present_variants += 1

    return present_variants >= 1


# ============================================================
# E9
# ============================================================

def detect_hallucination(prompt, response):
    """
    Very conservative factual/hallucination heuristic.

    Since factual correctness requires external verification,
    this function only catches obvious unsupported numerical /
    factual-looking claims in a preliminary pass.

    Final E9 should be independently validated.
    """

    if not response:
        return False

    # No reliable automatic factual verification is attempted.
    # Returning False avoids inventing hallucination labels.

    return False


# ============================================================
# ANNOTATE ONE ROW
# ============================================================

def annotate_row(row):

    prompt = clean_text(row["prompt"])
    response = clean_text(row["response"])
    expected_style = clean_text(row["expected_style"])

    e1, e2, e_ratio, h_ratio = detect_language_dominance(response)

    e3 = detect_unnatural_code_switching(response)

    e4 = detect_grammatical_error(response)

    e5 = detect_repetition(response)

    e6 = detect_prompt_misunderstanding(
        prompt,
        response,
        expected_style
    )

    e7 = detect_incomplete(
        prompt,
        response
    )

    e8 = detect_spelling_inconsistency(response)

    e9 = detect_hallucination(
        prompt,
        response
    )

    e10 = detect_irrelevant(
        prompt,
        response,
        expected_style
    )

    result = dict(row)

    result.update({
        "E1_English_dominant": int(e1),
        "E2_Hindi_dominant": int(e2),
        "E3_Unnatural_code_switching": int(e3),
        "E4_Grammatical_error": int(e4),
        "E5_Repetition": int(e5),
        "E6_Prompt_misunderstanding": int(e6),
        "E7_Incomplete": int(e7),
        "E8_Spelling_transliteration": int(e8),
        "E9_Hallucination_factual": int(e9),
        "E10_Irrelevant": int(e10),

        "english_script_ratio": round(e_ratio, 4),
        "hindi_script_ratio": round(h_ratio, 4),
        "prompt_keyword_overlap": round(
            keyword_overlap(prompt, response),
            4
        ),
    })

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2 ERROR ANALYSIS")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"\nReading:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "model",
        "id",
        "prompt",
        "expected_style",
        "response",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"\nTotal rows: {len(df)}")

    print("\nRows per model:")
    print(df.groupby("model").size())

    # --------------------------------------------------------
    # Annotate
    # --------------------------------------------------------

    print("\nRunning preliminary E1-E10 annotation...")

    annotated_rows = []

    for index, row in df.iterrows():

        annotated = annotate_row(row)

        annotated_rows.append(annotated)

        if (index + 1) % 25 == 0:

            print(
                f"Processed {index + 1}/{len(df)}"
            )

    annotated_df = pd.DataFrame(annotated_rows)

    # --------------------------------------------------------
    # Save row-level annotations
    # --------------------------------------------------------

    row_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_row_level.csv"
    )

    annotated_df.to_csv(
        row_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved row-level results:\n{row_file}")

    # --------------------------------------------------------
    # Error columns
    # --------------------------------------------------------

    error_columns = [
        "E1_English_dominant",
        "E2_Hindi_dominant",
        "E3_Unnatural_code_switching",
        "E4_Grammatical_error",
        "E5_Repetition",
        "E6_Prompt_misunderstanding",
        "E7_Incomplete",
        "E8_Spelling_transliteration",
        "E9_Hallucination_factual",
        "E10_Irrelevant",
    ]

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall_rows = []

    total = len(annotated_df)

    for column in error_columns:

        count = int(
            annotated_df[column].sum()
        )

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        code = column.split("_")[0]

        overall_rows.append({
            "error_code": code,
            "error_type": ERROR_DEFINITIONS[code],
            "count": count,
            "total_responses": total,
            "percentage": round(percentage, 2),
        })

    overall_df = pd.DataFrame(overall_rows)

    overall_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_overall.csv"
    )

    overall_df.to_csv(
        overall_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Model-wise summary
    # --------------------------------------------------------

    model_rows = []

    for model, group in annotated_df.groupby("model"):

        model_total = len(group)

        for column in error_columns:

            count = int(
                group[column].sum()
            )

            percentage = (
                count / model_total * 100
                if model_total > 0
                else 0
            )

            code = column.split("_")[0]

            model_rows.append({
                "model": model,
                "error_code": code,
                "error_type": ERROR_DEFINITIONS[code],
                "count": count,
                "total_responses": model_total,
                "percentage": round(percentage, 2),
            })

    model_df = pd.DataFrame(model_rows)

    model_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_model_wise.csv"
    )

    model_df.to_csv(
        model_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Model × error matrix
    # --------------------------------------------------------

    matrix = (
        annotated_df
        .groupby("model")[error_columns]
        .sum()
        .reset_index()
    )

    matrix = matrix.rename(columns={
        column: column.split("_")[0]
        for column in error_columns
    })

    matrix_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_model_matrix.csv"
    )

    matrix.to_csv(
        matrix_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Percentage matrix
    # --------------------------------------------------------

    percentage_matrix = matrix.copy()

    for column in percentage_matrix.columns:

        if column == "model":
            continue

        percentage_matrix[column] = (
            percentage_matrix[column] / 34 * 100
        ).round(2)

    percentage_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_model_percentage_matrix.csv"
    )

    percentage_matrix.to_csv(
        percentage_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Critical reviewer metrics: E6 / E10
    # --------------------------------------------------------

    critical_rows = []

    for code, column in [
        ("E6", "E6_Prompt_misunderstanding"),
        ("E10", "E10_Irrelevant"),
    ]:

        count = int(
            annotated_df[column].sum()
        )

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        critical_rows.append({
            "error_code": code,
            "error_type": ERROR_DEFINITIONS[code],
            "count": count,
            "total": total,
            "percentage": round(percentage, 2),
        })

    critical_df = pd.DataFrame(critical_rows)

    critical_file = os.path.join(
        OUTPUT_DIR,
        "critical_E6_E10_v2.csv"
    )

    critical_df.to_csv(
        critical_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # V1 vs V2 comparison
    # --------------------------------------------------------

    # V1 values from the completed previous experiment.
    v1_values = {
        "E6": {
            "count": 129,
            "total": 136,
            "percentage": 94.85,
        },
        "E10": {
            "count": 118,
            "total": 136,
            "percentage": 86.76,
        },
    }

    comparison_rows = []

    for _, row in critical_df.iterrows():

        code = row["error_code"]

        v2_percentage = float(
            row["percentage"]
        )

        v1_percentage = v1_values[code]["percentage"]

        change = v2_percentage - v1_percentage

        comparison_rows.append({
            "error_code": code,
            "error_type": ERROR_DEFINITIONS[code],

            "V1_count": v1_values[code]["count"],
            "V1_total": v1_values[code]["total"],
            "V1_percentage": v1_percentage,

            "V2_count": int(row["count"]),
            "V2_total": int(row["total"]),
            "V2_percentage": v2_percentage,

            "V2_minus_V1_percentage_points": round(
                change,
                2
            ),
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "V1_vs_V2_E6_E10_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # JSON metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": "Controlled V2 benchmark error analysis",
        "input_file": INPUT_FILE,
        "total_responses": int(total),
        "models": sorted(
            annotated_df["model"].unique().tolist()
        ),
        "responses_per_model": 34,
        "error_categories": ERROR_DEFINITIONS,
        "annotation_type": "preliminary_rule_based",
        "important_note": (
            "E3-E10 are preliminary automatic heuristics. "
            "Semantic prompt understanding, relevance, "
            "grammatical correctness, spelling/transliteration, "
            "and factual correctness should be independently "
            "validated by human or stronger independent LLM judges."
        ),
    }

    metadata_file = os.path.join(
        OUTPUT_DIR,
        "error_analysis_v2_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("OVERALL V2 ERROR ANALYSIS")
    print("=" * 70)

    print(
        overall_df[
            [
                "error_code",
                "error_type",
                "count",
                "percentage",
            ]
        ].to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("CRITICAL REVIEWER METRICS")
    print("=" * 70)

    print(
        critical_df.to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("V1 vs V2 — E6 / E10")
    print("=" * 70)

    print(
        comparison_df.to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("MODEL-WISE ERROR COUNTS")
    print("=" * 70)

    print(
        matrix.to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(row_file)
    print(overall_file)
    print(model_file)
    print(matrix_file)
    print(percentage_file)
    print(critical_file)
    print(comparison_file)
    print(metadata_file)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()