import os
import re
import time
import pandas as pd


# ============================================================
# FAST AUTOMATED E1-E10 ERROR ANALYSIS
# ============================================================
# Input:
#   4000 controlled benchmark responses
#
# Output:
#   1. Row-level E1-E10 annotations
#   2. Overall E1-E10 summary
#   3. Model-level E1-E10 comparison
#
# IMPORTANT:
# This is a RULE-BASED automated analysis.
# It does NOT use an LLM judge and does NOT use human annotation.
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT = r"outputs\controlled_1000\benchmark_generations_1000.csv"

OUT_DIR = r"outputs\error_analysis"

ANNOTATION_OUT = os.path.join(
    OUT_DIR,
    "automated_error_annotation_4000.csv"
)

SUMMARY_OUT = os.path.join(
    OUT_DIR,
    "automated_error_analysis_summary_4000.csv"
)

MODEL_OUT = os.path.join(
    OUT_DIR,
    "automated_error_analysis_model_comparison_4000.csv"
)


os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# E1-E10 ERROR TAXONOMY
# ============================================================

ERROR_NAMES = [
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


# ============================================================
# ENGLISH WORD LIST
# ============================================================

ENGLISH_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "but",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "would",
    "will",
    "please",
    "your",
    "you",
    "we",
    "they",
    "it",
    "use",
    "using",
    "example",
    "answer",
    "write",
    "explain",
    "create",
    "give",
    "make",
    "important",
    "because",
    "about",
    "into",
    "than",
    "then",
    "also",
    "first",
    "second",
    "third",
    "code",
    "function",
    "data",
    "system",
    "model",
    "language",
    "information",
    "help",
    "provide",
    "describe",
    "list",
    "explain",
    "tell",
    "define",
    "difference",
    "advantages",
    "disadvantages",
    "reason",
    "solution",
    "problem",
    "method",
    "process",
    "result",
    "output",
    "input"
}


# ============================================================
# HINDI / ROMAN-HINDI WORD LIST
# ============================================================

HINDI_WORDS = {
    "hai",
    "hain",
    "tha",
    "thi",
    "the",
    "ho",
    "hoga",
    "hogi",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "me",
    "mein",
    "par",
    "aur",
    "ya",
    "lekin",
    "agar",
    "toh",
    "to",
    "yeh",
    "yah",
    "woh",
    "kya",
    "kyun",
    "kyu",
    "kaise",
    "kab",
    "kahan",
    "mujhe",
    "aap",
    "tum",
    "hum",
    "mera",
    "meri",
    "mere",
    "apna",
    "apni",
    "apne",
    "bahut",
    "bhi",
    "nahi",
    "nahin",
    "karna",
    "karo",
    "kare",
    "kar",
    "diya",
    "gaya",
    "raha",
    "rahi",
    "rahe",
    "achha",
    "accha",
    "acha",
    "liye",
    "wala",
    "wali",
    "wale",
    "chahiye",
    "sakta",
    "sakti",
    "sakte",
    "phir",
    "yahan",
    "wahan",
    "isme",
    "usme",
    "hamara",
    "hamari",
    "hamare",
    "tumhara",
    "tumhari",
    "tumhare",
    "thoda",
    "thodi",
    "kyunki",
    "kyonki"
}


ROMAN_HINDI = set(HINDI_WORDS)


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

def words(text):
    """
    Extract alphabetic words.
    """
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?",
        str(text).lower()
    )


def sentence_list(text):
    """
    Split text into approximate sentences.
    """
    return [
        s.strip()
        for s in re.split(
            r"[.!?]+",
            str(text)
        )
        if s.strip()
    ]


def word_count(text):
    return len(words(text))


# ============================================================
# LANGUAGE RATIOS
# ============================================================

def english_ratio(text):

    ws = words(text)

    if not ws:
        return 0.0

    count = sum(
        1
        for w in ws
        if w in ENGLISH_WORDS
    )

    return count / len(ws)


def hindi_ratio(text):

    ws = words(text)

    if not ws:
        return 0.0

    count = sum(
        1
        for w in ws
        if w in HINDI_WORDS
    )

    return count / len(ws)


def roman_hindi_ratio(text):

    ws = words(text)

    if not ws:
        return 0.0

    count = sum(
        1
        for w in ws
        if w in ROMAN_HINDI
    )

    return count / len(ws)


# ============================================================
# E1 - ENGLISH DOMINANT
# ============================================================

def detect_e1(text):

    er = english_ratio(text)
    hr = hindi_ratio(text)
    rr = roman_hindi_ratio(text)

    return int(
        er >= 0.45
        and hr < 0.10
        and rr < 0.10
    )


# ============================================================
# E2 - HINDI DOMINANT
# ============================================================

def detect_e2(text):

    er = english_ratio(text)
    hr = hindi_ratio(text)
    rr = roman_hindi_ratio(text)

    return int(
        hr + rr >= 0.45
        and er < 0.10
    )


# ============================================================
# E3 - UNNATURAL CODE SWITCHING
# ============================================================

def detect_e3(text):

    ws = words(text)

    if len(ws) < 10:
        return 0

    language_labels = []

    for w in ws:

        if w in ENGLISH_WORDS:
            language_labels.append("E")

        elif (
            w in HINDI_WORDS
            or w in ROMAN_HINDI
        ):
            language_labels.append("H")

    if len(language_labels) < 8:
        return 0

    switches = 0

    for i in range(
        1,
        len(language_labels)
    ):
        if (
            language_labels[i]
            != language_labels[i - 1]
        ):
            switches += 1

    switch_rate = (
        switches
        / len(language_labels)
    )

    # Excessive language switching
    if switch_rate >= 0.65:
        return 1

    # Fragmented switching around punctuation
    parts = re.split(
        r"[,;:()/\-]+",
        str(text)
    )

    valid_parts = [
        p
        for p in parts
        if p.strip()
    ]

    if len(valid_parts) >= 5:

        short_parts = sum(
            1
            for p in valid_parts
            if len(words(p)) <= 2
        )

        short_ratio = (
            short_parts
            / len(valid_parts)
        )

        if short_ratio >= 0.65:
            return 1

    return 0


# ============================================================
# E4 - GRAMMATICAL ERROR
# ============================================================

def detect_e4(text):

    s = str(text).strip().lower()

    if not s:
        return 0

    # Repeated punctuation
    if re.search(
        r"[,.!?]{3,}",
        s
    ):
        return 1

    # Obvious English agreement errors
    bad_patterns = [

        r"\bi is\b",
        r"\bhe are\b",
        r"\bshe are\b",
        r"\bthey is\b",
        r"\bwe is\b",
        r"\bthis are\b",
        r"\bthese is\b",
        r"\bthere is many\b",
        r"\bthere are a\b",
        r"\bdoes not knows\b",
        r"\bdid not went\b",
        r"\bhe don't\b",
        r"\bshe don't\b",
        r"\bit don't\b",
        r"\bthey doesn't\b",
        r"\bwe doesn't\b",
        r"\bi has\b"
    ]

    for pattern in bad_patterns:

        if re.search(
            pattern,
            s
        ):
            return 1

    return 0


# ============================================================
# E5 - REPETITION
# ============================================================

def repetition_score(text):

    ws = words(text)

    if len(ws) < 8:
        return 0.0

    # Adjacent repeated words
    adjacent = 0

    for i in range(
        len(ws) - 1
    ):

        if ws[i] == ws[i + 1]:
            adjacent += 1

    # Repeated 3-grams
    trigrams = []

    for i in range(
        len(ws) - 2
    ):

        trigrams.append(
            tuple(
                ws[i:i + 3]
            )
        )

    repeated_trigrams = (
        len(trigrams)
        - len(set(trigrams))
    )

    score = (
        adjacent * 2
        + repeated_trigrams
    ) / max(
        len(ws),
        1
    )

    return score


def detect_e5(text):

    return int(
        repetition_score(text)
        >= 0.18
    )


# ============================================================
# E6 - PROMPT MISUNDERSTANDING
# ============================================================

def detect_e6(prompt, response):

    p = str(prompt).lower()
    r = str(response).strip().lower()

    if not r:
        return 1

    # Explicit inability/refusal
    refusal_patterns = [
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bi am unable\b",
        r"\bnot possible\b"
    ]

    for pattern in refusal_patterns:

        if re.search(
            pattern,
            r
        ):
            return 1

    # Prompt asks for code/programming
    asks_code = bool(
        re.search(
            r"\b(code|python|program|function|sql|script)\b",
            p
        )
    )

    if asks_code:

        has_code = bool(
            re.search(
                r"```|def\s+\w+|import\s+\w+|print\s*\(",
                r
            )
        )

        # Very short response to code request
        if (
            not has_code
            and word_count(r) <= 8
        ):
            return 1

    # Prompt asks to list items
    asks_list = bool(
        re.search(
            r"\blist\b|\bexamples\b|\bpoints\b",
            p
        )
    )

    if asks_list:

        if (
            word_count(r) > 2
            and not re.search(
                r"(^|\n)([-*•]|\d+[.)])",
                r
            )
        ):
            # Only flag if response is extremely short
            if word_count(r) <= 8:
                return 1

    return 0


# ============================================================
# E7 - INCOMPLETE RESPONSE
# ============================================================

def detect_e7(prompt, response):

    r = str(response).strip()
    lower = r.lower()

    if not r:
        return 1

    if word_count(r) <= 3:
        return 1

    # Explicit cut-off indicators
    cutoff_patterns = [
        r"to be continued$",
        r"etc\.\.\.$",
        r"and so on\.\.\.$",
        r"continue\.\.\.$"
    ]

    for pattern in cutoff_patterns:

        if re.search(
            pattern,
            lower
        ):
            return 1

    # Ends with a conjunction / unfinished connector
    if re.search(
        r"\b(and|or|but|because|if|when|to|with|for)$",
        lower
    ):
        return 1

    # Very short answer to a substantial prompt
    p_words = word_count(prompt)
    r_words = word_count(response)

    if (
        p_words > 40
        and r_words <= 5
    ):
        return 1

    return 0


# ============================================================
# E8 - SPELLING / TRANSLITERATION ERROR
# ============================================================

def detect_e8(text):

    s = str(text).lower()
    ws = words(text)

    if not ws:
        return 0

    # Character spam
    if re.search(
        r"(.)\1{4,}",
        s
    ):
        return 1

    # Excessively long consonant-only words
    suspicious = 0

    for w in ws:

        if len(w) >= 8:

            vowels = sum(
                1
                for c in w
                if c in "aeiou"
            )

            if vowels == 0:
                suspicious += 1

    if suspicious >= 2:
        return 1

    return 0


# ============================================================
# E9 - HALLUCINATION / FACTUAL ERROR
# ============================================================

def detect_e9(prompt, response):

    """
    Conservative automated detector.

    It only catches obvious contradictions/patterns.
    It does NOT claim that every factual error can be
    detected automatically.
    """

    text = str(response).lower()

    obvious_errors = [

        r"\b2\s*\+\s*2\s*=\s*5\b",
        r"\b1\s*\+\s*1\s*=\s*3\b",
        r"\b10\s*\+\s*10\s*=\s*30\b",
        r"\b5\s*\*\s*5\s*=\s*50\b"
    ]

    for pattern in obvious_errors:

        if re.search(
            pattern,
            text
        ):
            return 1

    return 0


# ============================================================
# E10 - IRRELEVANT RESPONSE
# ============================================================

def detect_e10(prompt, response):

    r = str(response).strip()

    if not r:
        return 1

    if word_count(r) <= 3:
        return 1

    generic_responses = [

        "i don't know",
        "i do not know",
        "i cannot help",
        "i can't help",
        "as an ai",
        "sorry, i can't",
        "sorry, i cannot",
        "i am unable to answer"
    ]

    lower = r.lower()

    for phrase in generic_responses:

        if phrase in lower:
            return 1

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAST AUTOMATED E1-E10 ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    if not os.path.exists(INPUT):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT}"
        )

    df = pd.read_csv(
        INPUT
    )

    print(
        "[OK] Input rows:",
        len(df)
    )

    required_columns = [
        "row_index",
        "model",
        "prompt",
        "response"
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    if len(df) != 4000:

        raise ValueError(
            f"Expected exactly 4000 rows, "
            f"found {len(df)}"
        )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    results = []

    start_time = time.time()

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        prompt = str(
            row.prompt
        )

        response = str(
            row.response
        )

        er = english_ratio(
            response
        )

        hr = hindi_ratio(
            response
        )

        rr = roman_hindi_ratio(
            response
        )

        result = {

            "row_index":
                row.row_index,

            "model":
                row.model,

            "prompt":
                prompt,

            "response":
                response,

            # ----------------------------
            # E1-E10
            # ----------------------------

            "E1_English_dominant":
                detect_e1(
                    response
                ),

            "E2_Hindi_dominant":
                detect_e2(
                    response
                ),

            "E3_Unnatural_code_switching":
                detect_e3(
                    response
                ),

            "E4_Grammatical_error":
                detect_e4(
                    response
                ),

            "E5_Repetition":
                detect_e5(
                    response
                ),

            "E6_Prompt_misunderstanding":
                detect_e6(
                    prompt,
                    response
                ),

            "E7_Incomplete_response":
                detect_e7(
                    prompt,
                    response
                ),

            "E8_Spelling_transliteration_error":
                detect_e8(
                    response
                ),

            "E9_Hallucination_factual_error":
                detect_e9(
                    prompt,
                    response
                ),

            "E10_Irrelevant_response":
                detect_e10(
                    prompt,
                    response
                ),

            # ----------------------------
            # Supporting metrics
            # ----------------------------

            "english_ratio":
                round(
                    er,
                    6
                ),

            "hindi_ratio":
                round(
                    hr,
                    6
                ),

            "roman_hindi_ratio":
                round(
                    rr,
                    6
                ),

            "response_word_count":
                word_count(
                    response
                ),

            "judge_model":
                "RuleBased-Automated",

            "judge_valid":
                1
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            i % 250 == 0
            or i == len(df)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                i / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(df) - i
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0
            )

            eta_minutes = (
                eta_seconds / 60
            )

            print(
                f"[PROGRESS] "
                f"{i}/{len(df)} "
                f"| {rate:.1f} rows/sec "
                f"| ETA={eta_minutes:.2f} min"
            )

            # Intermediate save
            pd.DataFrame(
                results
            ).to_csv(
                ANNOTATION_OUT,
                index=False
            )

    # --------------------------------------------------------
    # FINAL DATAFRAME
    # --------------------------------------------------------

    final_df = pd.DataFrame(
        results
    )

    final_df.to_csv(
        ANNOTATION_OUT,
        index=False
    )

    print()
    print(
        "[DONE] Processed:",
        len(final_df),
        "rows"
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary_rows = []

    for error in ERROR_NAMES:

        count = int(
            final_df[error].sum()
        )

        percentage = (
            100
            * count
            / len(final_df)
        )

        summary_rows.append({

            "error":
                error,

            "count":
                count,

            "percentage":
                round(
                    percentage,
                    4
                )
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    model_rows = []

    for model_name, group in (
        final_df.groupby(
            "model"
        )
    ):

        for error in ERROR_NAMES:

            count = int(
                group[error].sum()
            )

            percentage = (
                100
                * count
                / len(group)
            )

            model_rows.append({

                "model":
                    model_name,

                "error":
                    error,

                "count":
                    count,

                "total_responses":
                    len(group),

                "percentage":
                    round(
                        percentage,
                        4
                    )
            })

    model_df = pd.DataFrame(
        model_rows
    )

    model_df.to_csv(
        MODEL_OUT,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("E1-E10 ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "[OK] Row-level:",
        ANNOTATION_OUT
    )

    print(
        "[OK] Overall summary:",
        SUMMARY_OUT
    )

    print(
        "[OK] Model comparison:",
        MODEL_OUT
    )

    print()
    print("-" * 70)
    print("OVERALL ERROR COUNTS")
    print("-" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("-" * 70)
    print("MODEL-LEVEL ERROR COUNTS")
    print("-" * 70)

    pivot = model_df.pivot(
        index="model",
        columns="error",
        values="count"
    )

    print(
        pivot.to_string()
    )

    elapsed_total = (
        time.time()
        - start_time
    )

    print()
    print(
        f"[TIME] Total: "
        f"{elapsed_total:.2f} seconds "
        f"({elapsed_total / 60:.2f} minutes)"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This E1-E10 analysis is "
        "rule-based automated analysis."
    )

    print(
        "It does not use human annotation "
        "or an LLM semantic judge."
    )

    print(
        "E9/factual error detection is "
        "conservative and cannot detect "
        "all factual hallucinations."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()