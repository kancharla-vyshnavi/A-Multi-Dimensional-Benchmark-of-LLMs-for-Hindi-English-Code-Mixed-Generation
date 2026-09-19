import os
import json
import re
import random

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ============================================================
# CONFIG
# ============================================================

JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

INPUT_DIR = r".\outputs\hinglish_bench"

OUTPUT_DIR = r".\outputs\llm_judge"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 180


MODEL_FILES = {
    "HingGPT":
        "hinggpt_hinglish_bench_generations.csv",

    "Qwen2.5-3B":
        "qwen25_3b_hinglish_bench_generations.csv",

    "Qwen2.5-7B":
        "qwen25_7b_hinglish_bench_generations.csv",

    "Phi-3.5-mini":
        "phi35_mini_hinglish_bench_generations.csv"
}


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("PHI-3.5-MINI-INSTRUCT LLM-AS-JUDGE")
print("=" * 70)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        2
    ),
    "GB"
)


# ============================================================
# 4-BIT NF4
# ============================================================

print()
print("Creating 4-bit NF4 configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# MODEL
# ============================================================

print("Loading Phi judge in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()

# IMPORTANT:
# Prevent DynamicCache / seen_tokens problem
model.config.use_cache = False

print("Judge model loaded successfully.")


# ============================================================
# JUDGE PROMPT
# ============================================================

def create_judge_prompt(prompt, response):

    return f"""
You are an expert evaluator of Hinglish text generation.

Evaluate the MODEL RESPONSE for the USER PROMPT.

Give an integer score from 1 to 5 for each criterion.

CRITERIA:

1. fluency
Naturalness, readability, and linguistic fluency.

2. code_mixing_naturalness
How naturally Hindi and English are mixed.

3. hindi_grammar
Correctness and naturalness of Hindi grammar.

4. prompt_adherence
How well the response follows the requested task.

5. spelling_consistency
Consistency and readability of Roman Hinglish spelling.

6. overall
Overall quality of the response.

SCORING:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

IMPORTANT RULES:

- Evaluate only the response.
- Do not compare models.
- Do not explain your scores.
- Return ONLY valid JSON.
- Every score must be an integer from 1 to 5.

Return exactly:

{{
    "fluency": 1,
    "code_mixing_naturalness": 1,
    "hindi_grammar": 1,
    "prompt_adherence": 1,
    "spelling_consistency": 1,
    "overall": 1
}}

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}
""".strip()


# ============================================================
# PARSE JSON
# ============================================================

def parse_scores(text):

    if text is None:
        return None

    text = text.strip()

    # Remove markdown fences if model adds them
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    )

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    if match is None:

        return None


    try:

        data = json.loads(
            match.group(0)
        )

    except Exception:

        return None


    required_keys = [
        "fluency",
        "code_mixing_naturalness",
        "hindi_grammar",
        "prompt_adherence",
        "spelling_consistency",
        "overall"
    ]


    for key in required_keys:

        if key not in data:

            return None


    for key in required_keys:

        try:

            value = int(
                data[key]
            )

        except Exception:

            return None


        if value < 1 or value > 5:

            return None


        data[key] = value


    return data


# ============================================================
# JUDGE ONE RESPONSE
# ============================================================

def judge_one(prompt, response):

    judge_prompt = create_judge_prompt(
        prompt,
        response
    )


    messages = [
        {
            "role": "user",
            "content": judge_prompt
        }
    ]


    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )


    # ========================================================
    # IMPORTANT:
    # Do NOT use model.generate with cache.
    # This avoids DynamicCache.seen_tokens error.
    # ========================================================

    inputs = {
        key: value.to("cuda:0")
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=False,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )


    # Only decode newly generated tokens
    input_length = inputs[
        "input_ids"
    ].shape[1]


    generated_tokens = outputs[
        0,
        input_length:
    ]


    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    scores = parse_scores(
        generated_text
    )


    return scores, generated_text


# ============================================================
# FIND PROMPT COLUMN
# ============================================================

def find_prompt_column(df):

    candidates = [
        "prompt",
        "question",
        "user_prompt"
    ]

    for column in candidates:

        if column in df.columns:

            return column


    return None


# ============================================================
# FIND RESPONSE COLUMN
# ============================================================

def find_response_column(df):

    candidates = [
        "generated_response",
        "generated_text",
        "response",
        "generation",
        "output",
        "answer"
    ]

    for column in candidates:

        if column in df.columns:

            return column


    return None


# ============================================================
# LOAD BENCHMARK RESPONSES
# ============================================================

print()
print("=" * 70)
print("LOADING BENCHMARK RESPONSES")
print("=" * 70)


all_rows = []


for model_name, filename in MODEL_FILES.items():

    file_path = os.path.join(
        INPUT_DIR,
        filename
    )


    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nMissing benchmark file:\n{file_path}"
        )


    df = pd.read_csv(
        file_path
    )


    print()
    print(
        f"{model_name}: {len(df)} responses"
    )

    print(
        "Columns:",
        df.columns.tolist()
    )


    prompt_column = find_prompt_column(
        df
    )

    response_column = find_response_column(
        df
    )


    if prompt_column is None:

        raise KeyError(
            f"No prompt column found in {filename}.\n"
            f"Available columns: {df.columns.tolist()}"
        )


    if response_column is None:

        raise KeyError(
            f"No response column found in {filename}.\n"
            f"Available columns: {df.columns.tolist()}"
        )


    print(
        "Using prompt column:",
        prompt_column
    )

    print(
        "Using response column:",
        response_column
    )


    # Category
    if "category_name" in df.columns:

        category_column = "category_name"

    elif "category" in df.columns:

        category_column = "category"

    else:

        category_column = None


    for index, row in df.iterrows():

        category = ""

        if category_column is not None:

            category = str(
                row[category_column]
            )


        all_rows.append(
            {
                "model": model_name,

                "sample_id": index + 1,

                "category": category,

                "prompt": str(
                    row[prompt_column]
                ),

                "response": str(
                    row[response_column]
                )
            }
        )


print()
print("=" * 70)

print(
    "TOTAL RESPONSES:",
    len(all_rows)
)

print("=" * 70)


# ============================================================
# OUTPUT FILES
# ============================================================

results_jsonl = os.path.join(
    OUTPUT_DIR,
    "hinglish_bench_llm_judge_results.jsonl"
)

results_csv = os.path.join(
    OUTPUT_DIR,
    "hinglish_bench_llm_judge_results.csv"
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "hinglish_bench_llm_judge_summary.csv"
)


# ============================================================
# DELETE OLD INVALID RESULTS
# ============================================================

if os.path.exists(results_jsonl):

    os.remove(
        results_jsonl
    )


if os.path.exists(results_csv):

    os.remove(
        results_csv
    )


if os.path.exists(summary_csv):

    os.remove(
        summary_csv
    )


# ============================================================
# START JUDGING
# ============================================================

print()
print("=" * 70)
print("STARTING LLM-AS-JUDGE")
print("=" * 70)


results = []


for i, item in enumerate(
    all_rows,
    start=1
):

    print(
        f"[{i}/{len(all_rows)}] "
        f"{item['model']} - "
        f"sample {item['sample_id']}",
        flush=True
    )


    try:

        scores, raw_output = judge_one(
            item["prompt"],
            item["response"]
        )


        if scores is None:

            print(
                "   WARNING: Invalid JSON returned",
                flush=True
            )


            result = {
                **item,

                "fluency": np.nan,

                "code_mixing_naturalness": np.nan,

                "hindi_grammar": np.nan,

                "prompt_adherence": np.nan,

                "spelling_consistency": np.nan,

                "overall": np.nan,

                "judge_raw_output": raw_output
            }


        else:

            result = {
                **item,
                **scores,
                "judge_raw_output": raw_output
            }


    except Exception as error:

        print(
            "   ERROR:",
            str(error),
            flush=True
        )


        result = {
            **item,

            "fluency": np.nan,

            "code_mixing_naturalness": np.nan,

            "hindi_grammar": np.nan,

            "prompt_adherence": np.nan,

            "spelling_consistency": np.nan,

            "overall": np.nan,

            "judge_raw_output": "",

            "error": str(error)
        }


    results.append(
        result
    )


    # ========================================================
    # SAVE AFTER EVERY SAMPLE
    # ========================================================

    with open(
        results_jsonl,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# SAVE DETAILED CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    results_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SCORE COLUMNS
# ============================================================

score_columns = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]


# ============================================================
# VALIDITY
# ============================================================

print()
print("=" * 70)
print("JUDGMENT VALIDITY")
print("=" * 70)


for column in score_columns:

    valid_count = (
        results_df[column]
        .notna()
        .sum()
    )

    print(
        f"{column}: "
        f"{valid_count}/{len(results_df)}"
    )


# ============================================================
# MODEL-WISE MEANS
# ============================================================

summary = (
    results_df
    .groupby("model")[score_columns]
    .mean()
    .round(4)
)


# ============================================================
# MEAN ACROSS ALL CRITERIA
# ============================================================

summary["mean_all_criteria"] = (
    summary[score_columns]
    .mean(axis=1)
    .round(4)
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    summary_csv,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("MODEL-WISE LLM JUDGE RESULTS")
print("=" * 70)

print(
    summary
)


print()
print("=" * 70)
print("LLM-AS-JUDGE COMPLETED")
print("=" * 70)

print(
    "Detailed JSONL:",
    results_jsonl
)

print(
    "Detailed CSV:",
    results_csv
)

print(
    "Summary CSV:",
    summary_csv
)

print()
print("DONE.")
