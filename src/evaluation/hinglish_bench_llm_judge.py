import os
import re
import json
import pandas as pd
from pathlib import Path

# ============================================================
# Hinglish-Bench LLM-as-Judge
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "outputs" / "hinglish_bench"
OUTPUT_DIR = BASE_DIR / "outputs" / "llm_judge"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "HingGPT": "hinggpt_hinglish_bench_generations.csv",
    "Qwen2.5-3B": "qwen25_3b_hinglish_bench_generations.csv",
    "Qwen2.5-7B": "qwen25_7b_hinglish_bench_generations.csv",
    "Phi-3.5-mini": "phi35_mini_hinglish_bench_generations.csv",
}

# ------------------------------------------------------------
# Hinglish-Bench evaluation dimensions
# ------------------------------------------------------------

RUBRIC = {
    "fluency": (
        "How fluent and natural is the generated response? "
        "Consider readability, grammatical flow, and natural expression."
    ),
    "code_mixing_naturalness": (
        "How natural and appropriate is the Hindi-English code-mixing? "
        "Penalize unnatural switching or excessive/unnecessary mixing."
    ),
    "hindi_grammar": (
        "How grammatically correct is the Hindi/Hinglish usage? "
        "Consider sentence structure, word order, and appropriate Hindi usage."
    ),
    "prompt_adherence": (
        "How well does the response follow and answer the given prompt?"
    ),
    "spelling_consistency": (
        "How consistent and understandable is the spelling, especially "
        "Romanized Hindi/Hinglish spelling?"
    ),
    "overall": (
        "Considering all dimensions, how good is the response overall?"
    ),
}

# ------------------------------------------------------------
# Score interpretation
# ------------------------------------------------------------

SCORE_GUIDE = """
Give an integer score from 1 to 5 for every dimension.

1 = Very poor
2 = Poor
3 = Acceptable / moderate
4 = Good
5 = Excellent

Be consistent across all models.
Do not compare models directly while scoring.
Score only the response against the prompt and rubric.
"""

# ------------------------------------------------------------
# Expected CSV columns
# ------------------------------------------------------------

def find_column(columns, candidates):
    """Find a column using exact or partial matching."""
    normalized = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for c in columns:
        c_low = str(c).strip().lower()
        for candidate in candidates:
            if candidate.lower() in c_low:
                return c

    return None


def load_model_data(model_name, filename):
    path = INPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    prompt_col = find_column(
        df.columns,
        ["prompt", "input", "question", "user_prompt"]
    )

    response_col = find_column(
        df.columns,
        ["generated_response", "response", "generated_text", "output", "answer"]
    )

    if prompt_col is None:
        raise ValueError(
            f"Could not find prompt column in {filename}. "
            f"Columns found: {list(df.columns)}"
        )

    if response_col is None:
        raise ValueError(
            f"Could not find response column in {filename}. "
            f"Columns found: {list(df.columns)}"
        )

    result = pd.DataFrame({
        "model": model_name,
        "prompt": df[prompt_col].fillna("").astype(str),
        "response": df[response_col].fillna("").astype(str),
    })

    return result


# ------------------------------------------------------------
# Build judge prompt
# ------------------------------------------------------------

def build_judge_prompt(prompt, response):
    return f"""
You are an expert evaluator for Hinglish generative language models.

Evaluate the following response using the exact rubric below.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

RUBRIC:

1. Fluency:
{RUBRIC["fluency"]}

2. Code-mixing naturalness:
{RUBRIC["code_mixing_naturalness"]}

3. Hindi grammar:
{RUBRIC["hindi_grammar"]}

4. Prompt adherence:
{RUBRIC["prompt_adherence"]}

5. Spelling consistency:
{RUBRIC["spelling_consistency"]}

6. Overall quality:
{RUBRIC["overall"]}

{SCORE_GUIDE}

Return ONLY valid JSON in this exact format:

{{
  "fluency": 1,
  "code_mixing_naturalness": 1,
  "hindi_grammar": 1,
  "prompt_adherence": 1,
  "spelling_consistency": 1,
  "overall": 1
}}
""".strip()


# ------------------------------------------------------------
# Load all 4 models
# ------------------------------------------------------------

def load_all_models():
    frames = []

    for model_name, filename in MODEL_FILES.items():
        print(f"Loading {model_name}...")
        df = load_model_data(model_name, filename)

        print(f"  Rows: {len(df)}")

        if len(df) != 34:
            print(
                f"  WARNING: expected 34 rows, found {len(df)}"
            )

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print()
    print("=" * 60)
    print("TOTAL RESPONSES:", len(combined))
    print("=" * 60)

    return combined


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("HINGLISH-BENCH LLM-AS-JUDGE")
    print("=" * 60)

    df = load_all_models()

    # Save the exact judge prompts so the evaluation is reproducible.
    judge_prompts = []

    for _, row in df.iterrows():
        judge_prompts.append(
            build_judge_prompt(
                row["prompt"],
                row["response"]
            )
        )

    df["judge_prompt"] = judge_prompts

    prompt_output = OUTPUT_DIR / "hinglish_bench_judge_prompts.jsonl"

    with open(prompt_output, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            record = {
                "index": int(idx),
                "model": row["model"],
                "prompt": row["prompt"],
                "response": row["response"],
                "judge_prompt": row["judge_prompt"],
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # Save metadata for reproducibility.
    metadata = {
        "task": "Hinglish-Bench LLM-as-Judge",
        "num_models": len(MODEL_FILES),
        "models": list(MODEL_FILES.keys()),
        "responses_per_model": 34,
        "total_responses": len(df),
        "score_range": "1-5",
        "dimensions": list(RUBRIC.keys()),
        "note": (
            "This file prepares the standardized judge prompts. "
            "No scores are generated yet."
        ),
    }

    metadata_path = OUTPUT_DIR / "judge_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Judge prompts created successfully.")
    print()
    print(f"Prompt file:   {prompt_output}")
    print(f"Metadata file: {metadata_path}")
    print()
    print("NO MODEL HAS BEEN JUDGED YET.")
    print("This step only prepares the reproducible evaluation input.")


if __name__ == "__main__":
    main()