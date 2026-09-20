import os
import json
import re
import time
import itertools

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"outputs\controlled_generation\benchmark_generations_v3.csv"

OUTPUT_DIR = r"outputs\pairwise_evaluation"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "pairwise_mistral_v3_results.csv"
)

OUTPUT_JSONL = os.path.join(
    OUTPUT_DIR,
    "pairwise_mistral_v3_results.jsonl"
)

CONFIG_JSON = os.path.join(
    OUTPUT_DIR,
    "pairwise_config_v3.json"
)

JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

MAX_ATTEMPTS = 3

# Six model pairs
MODELS = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD V3 GENERATIONS
# ============================================================

print("=" * 70)
print("PAIRWISE V3 EVALUATION")
print("=" * 70)

print()
print("Loading:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print()
print(f"Generation rows: {len(df)}")

print()
print("Models found:")
print(df["model"].value_counts().to_string())

print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "row_index",
    "model",
    "prompt",
    "response",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# BUILD LOOKUP
# ============================================================

generation_lookup = {}

for _, row in df.iterrows():

    key = (
        int(row["row_index"]),
        str(row["model"])
    )

    generation_lookup[key] = str(
        row["response"]
    )


# ============================================================
# VERIFY 34 PROMPTS Ã— 4 MODELS
# ============================================================

prompt_rows = {}

for _, row in df.iterrows():

    row_index = int(row["row_index"])

    if row_index not in prompt_rows:
        prompt_rows[row_index] = {
            "prompt": str(row["prompt"]),
            "models": {}
        }

    prompt_rows[row_index]["models"][
        str(row["model"])
    ] = str(row["response"])


print(
    f"Unique prompts: {len(prompt_rows)}"
)

for row_index, data in prompt_rows.items():

    missing_models = [
        model
        for model in MODELS
        if model not in data["models"]
    ]

    if missing_models:
        raise ValueError(
            f"Prompt {row_index} missing models: "
            f"{missing_models}"
        )

print(
    "All prompts have all 4 model generations."
)

print()


# ============================================================
# CREATE PAIR LIST
# ============================================================

unique_pairs = list(
    itertools.combinations(MODELS, 2)
)

print("Unique model pairs:")
for a, b in unique_pairs:
    print(f"  {a}  vs  {b}")

print()

print(
    f"Unique pairs       : {len(unique_pairs)}"
)

print(
    f"Directions per pair: 2"
)

print(
    f"Prompts            : {len(prompt_rows)}"
)

expected_total = (
    len(prompt_rows)
    * len(unique_pairs)
    * 2
)

print(
    f"Expected judgments : {expected_total}"
)

print()


# ============================================================
# LOAD PREVIOUS RESULTS IF THEY EXIST
# ============================================================

if os.path.exists(OUTPUT_CSV):

    print(
        "Existing pairwise result file found."
    )

    results_df = pd.read_csv(
        OUTPUT_CSV
    )

    print(
        f"Existing rows: {len(results_df)}"
    )

else:

    results_df = pd.DataFrame(
        columns=[
            "row_index",
            "prompt",
            "model_A",
            "model_B",
            "direction",
            "response_A",
            "response_B",
            "winner",
            "confidence",
            "reason",
            "raw_judge_output",
        ]
    )

    print(
        "Starting new pairwise result file."
    )

print()


# ============================================================
# CREATE COMPLETED-KEY SET
# ============================================================

completed_keys = set()

if len(results_df) > 0:

    for _, row in results_df.iterrows():

        try:

            key = (
                int(row["row_index"]),
                str(row["model_A"]),
                str(row["model_B"]),
                str(row["direction"]),
            )

            completed_keys.add(key)

        except Exception:
            pass


print(
    f"Already completed: {len(completed_keys)}"
)

print()


# ============================================================
# LOAD MISTRAL JUDGE
# ============================================================

print("=" * 70)
print("LOADING INDEPENDENT JUDGE")
print("=" * 70)

print(
    JUDGE_MODEL
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.eval()

print()
print("Judge loaded successfully.")
print()


# ============================================================
# PARSE PAIRWISE OUTPUT
# ============================================================

def parse_pairwise_output(text):

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Preferred format:
    #
    # winner=A
    # confidence=4
    # reason=...
    # --------------------------------------------------------

    winner_match = re.search(
        r"\bWINNER\s*[:=]\s*([ABT])\b",
        text,
        flags=re.IGNORECASE,
    )

    confidence_match = re.search(
        r"\bCONFIDENCE\s*[:=]\s*([1-5])\b",
        text,
        flags=re.IGNORECASE,
    )

    if winner_match and confidence_match:

        winner = winner_match.group(1).upper()

        confidence = int(
            confidence_match.group(1)
        )

        reason_match = re.search(
            r"\bREASON\s*[:=]\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = ""

        return {
            "winner": winner,
            "confidence": confidence,
            "reason": reason,
        }

    # --------------------------------------------------------
    # Fallback:
    #
    # A,4
    # B,3
    # T,2
    # --------------------------------------------------------

    fallback = re.search(
        r"(?<![A-Za-z])([ABT])\s*,\s*([1-5])(?!\d)",
        text,
        flags=re.IGNORECASE,
    )

    if fallback:

        winner = fallback.group(1).upper()

        confidence = int(
            fallback.group(2)
        )

        return {
            "winner": winner,
            "confidence": confidence,
            "reason": "",
        }

    # --------------------------------------------------------
    # Last fallback: find standalone A/B/T and score
    # --------------------------------------------------------

    winner_match = re.search(
        r"\b([ABT])\b",
        text,
        flags=re.IGNORECASE,
    )

    confidence_match = re.search(
        r"\b([1-5])\b",
        text,
    )

    if winner_match and confidence_match:

        winner = winner_match.group(1).upper()

        confidence = int(
            confidence_match.group(1)
        )

        # Avoid accidentally accepting arbitrary prose
        if winner in ["A", "B", "T"]:

            return {
                "winner": winner,
                "confidence": confidence,
                "reason": text,
            }

    return None


# ============================================================
# JUDGE ONE PAIR
# ============================================================

def judge_pair(
    prompt,
    response_a,
    response_b,
    model_a,
    model_b,
):

    system_instruction = """
You are an independent evaluator comparing two candidate
responses to the SAME Hinglish generation prompt.

You must compare the two responses fairly.

Evaluate:
- fluency
- natural Hindi-English code-mixing
- Hindi grammar
- spelling consistency
- adherence to the prompt
- overall quality

IMPORTANT:
Do NOT judge based on model name.
Do NOT assume either model is better.
Judge only the actual responses.

Winner options:
A = Response A is better overall
B = Response B is better overall
T = Tie / essentially equivalent

Confidence:
1 = very low confidence
2 = low confidence
3 = moderate confidence
4 = high confidence
5 = very high confidence

Return ONLY this format:

WINNER=A
CONFIDENCE=4
REASON=short factual reason

The reason must be concise.
Do not use JSON.
Do not provide scores for individual models.
"""

    user_instruction = f"""
PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

MODEL A:
{model_a}

MODEL B:
{model_b}

Compare the two responses and return the required format only.
"""

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": user_instruction,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return generated_text


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    global results_df

    new_row = pd.DataFrame(
        [result]
    )

    results_df = pd.concat(
        [
            results_df,
            new_row,
        ],
        ignore_index=True,
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSONL,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# MAIN PAIRWISE LOOP
# ============================================================

total = expected_total
completed_now = len(completed_keys)

print("=" * 70)
print("STARTING PAIRWISE EVALUATION")
print("=" * 70)

print()


for row_index, data in prompt_rows.items():

    prompt = data["prompt"]

    for model_a, model_b in unique_pairs:

        response_a = data["models"][model_a]
        response_b = data["models"][model_b]

        # ====================================================
        # DIRECTION 1: A vs B
        # ====================================================

        direction = "A_vs_B"

        key = (
            row_index,
            model_a,
            model_b,
            direction,
        )

        if key not in completed_keys:

            completed_now += 1

            print("=" * 70)
            print(
                f"[{completed_now}/{total}]"
            )
            print(
                f"Prompt : {row_index}"
            )
            print(
                f"A      : {model_a}"
            )
            print(
                f"B      : {model_b}"
            )
            print(
                "Direction: A_vs_B"
            )
            print("=" * 70)

            result_data = None

            for attempt in range(
                1,
                MAX_ATTEMPTS + 1
            ):

                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}"
                )

                try:

                    raw = judge_pair(
                        prompt,
                        response_a,
                        response_b,
                        model_a,
                        model_b,
                    )

                    print(
                        "Raw:",
                        repr(raw)
                    )

                    parsed = parse_pairwise_output(
                        raw
                    )

                    if parsed is not None:

                        result_data = {
                            "row_index": row_index,
                            "prompt": prompt,
                            "model_A": model_a,
                            "model_B": model_b,
                            "direction": direction,
                            "response_A": response_a,
                            "response_B": response_b,
                            "winner": parsed["winner"],
                            "confidence": parsed["confidence"],
                            "reason": parsed["reason"],
                            "raw_judge_output": raw,
                        }

                        break

                    else:

                        print(
                            "Invalid judge output."
                        )

                except Exception as e:

                    print(
                        "ERROR:",
                        repr(e)
                    )

                time.sleep(1)

            if result_data is not None:

                save_result(
                    result_data
                )

                completed_keys.add(
                    key
                )

                print(
                    "Saved:",
                    result_data["winner"],
                    "| confidence:",
                    result_data["confidence"],
                )

            else:

                print(
                    "FAILED - leaving this comparison for resume."
                )

            print()

        # ====================================================
        # DIRECTION 2: B vs A
        # ====================================================

        direction = "B_vs_A"

        key = (
            row_index,
            model_b,
            model_a,
            direction,
        )

        if key not in completed_keys:

            completed_now += 1

            print("=" * 70)
            print(
                f"[{completed_now}/{total}]"
            )
            print(
                f"Prompt : {row_index}"
            )
            print(
                f"A      : {model_b}"
            )
            print(
                f"B      : {model_a}"
            )
            print(
                "Direction: B_vs_A"
            )
            print("=" * 70)

            result_data = None

            for attempt in range(
                1,
                MAX_ATTEMPTS + 1
            ):

                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}"
                )

                try:

                    raw = judge_pair(
                        prompt,
                        response_b,
                        response_a,
                        model_b,
                        model_a,
                    )

                    print(
                        "Raw:",
                        repr(raw)
                    )

                    parsed = parse_pairwise_output(
                        raw
                    )

                    if parsed is not None:

                        result_data = {
                            "row_index": row_index,
                            "prompt": prompt,
                            "model_A": model_b,
                            "model_B": model_a,
                            "direction": direction,
                            "response_A": response_b,
                            "response_B": response_a,
                            "winner": parsed["winner"],
                            "confidence": parsed["confidence"],
                            "reason": parsed["reason"],
                            "raw_judge_output": raw,
                        }

                        break

                    else:

                        print(
                            "Invalid judge output."
                        )

                except Exception as e:

                    print(
                        "ERROR:",
                        repr(e)
                    )

                time.sleep(1)

            if result_data is not None:

                save_result(
                    result_data
                )

                completed_keys.add(
                    key
                )

                print(
                    "Saved:",
                    result_data["winner"],
                    "| confidence:",
                    result_data["confidence"],
                )

            else:

                print(
                    "FAILED - leaving this comparison for resume."
                )

            print()


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "input_csv": INPUT_CSV,
    "judge_model": JUDGE_MODEL,
    "models": MODELS,
    "num_prompts": len(prompt_rows),
    "num_unique_pairs": len(unique_pairs),
    "directions_per_pair": 2,
    "expected_total_judgments": expected_total,
    "max_attempts": MAX_ATTEMPTS,
    "pairwise_protocol": "A_vs_B_and_B_vs_A",
}

with open(
    CONFIG_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

if os.path.exists(OUTPUT_CSV):

    final_df = pd.read_csv(
        OUTPUT_CSV
    )

else:

    final_df = pd.DataFrame()


print()
print("=" * 70)
print("PAIRWISE EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Expected comparisons : {expected_total}"
)

print(
    f"Completed comparisons: {len(final_df)}"
)

print(
    f"Missing comparisons  : "
    f"{expected_total - len(final_df)}"
)

print()

if len(final_df) > 0:

    print("Winner counts:")
    print()

    print(
        final_df["winner"]
        .value_counts()
        .to_string()
    )

    print()

    print("Confidence distribution:")
    print()

    print(
        final_df["confidence"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("Files:")
    print(
        OUTPUT_CSV
    )
    print(
        OUTPUT_JSONL
    )
    print(
        CONFIG_JSON
    )

print()
print("=" * 70)
