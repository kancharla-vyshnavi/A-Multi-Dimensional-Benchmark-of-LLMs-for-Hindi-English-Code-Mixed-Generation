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

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

INPUT_FILE = r".\outputs\llm_judge\hinglish_bench_llm_judge_results.csv"

OUTPUT_FILE = r".\outputs\llm_judge\hinglish_bench_llm_judge_results_recovered.csv"

MAX_INPUT_LENGTH = 768
MAX_NEW_TOKENS = 80

SCORE_COLUMNS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall"
]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("FAST LLM-JUDGE RECOVERY V2")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

# Find rows where ANY score is missing
invalid_mask = df[SCORE_COLUMNS].isna().any(axis=1)
invalid_indices = df.index[invalid_mask].tolist()

print("Invalid rows:", len(invalid_indices))

if len(invalid_indices) == 0:
    print("ALL SCORES ALREADY VALID.")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    raise SystemExit

# ============================================================
# 4-BIT MODEL
# ============================================================

print("Loading Phi-3.5-mini judge...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()
model.config.use_cache = False

print("Model loaded.")

# ============================================================
# PROMPT
# ============================================================

def make_prompt(prompt, response):

    return f"""Rate the following Hinglish response from 1 to 5.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Return ONLY one JSON object.
No explanation.
No markdown.
No extra text.

Use exactly these keys:
fluency
code_mixing_naturalness
hindi_grammar
prompt_adherence
spelling_consistency
overall

Example:
{{"fluency":3,"code_mixing_naturalness":3,"hindi_grammar":3,"prompt_adherence":3,"spelling_consistency":3,"overall":3}}
"""

# ============================================================
# ROBUST PARSER
# ============================================================

def parse_scores(text):

    if not text:
        return None

    # Find every {...} candidate
    candidates = re.findall(
        r"\{[^{}]*\}",
        text,
        flags=re.DOTALL
    )

    for candidate in candidates:

        # Remove accidental standalone numeric lines
        cleaned = re.sub(
            r",\s*\d+\s*,",
            ",",
            candidate
        )

        try:
            data = json.loads(cleaned)
        except Exception:
            continue

        values = {}

        good = True

        for key in SCORE_COLUMNS:

            if key not in data:
                good = False
                break

            try:
                value = int(data[key])
            except Exception:
                good = False
                break

            if not 1 <= value <= 5:
                good = False
                break

            values[key] = value

        if good:
            return values

    return None

# ============================================================
# JUDGE
# ============================================================

def judge(prompt, response):

    text = make_prompt(
        prompt,
        response
    )

    messages = [
        {
            "role": "user",
            "content": text
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        k: v.to("cuda:0")
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]

    generated = output[0][input_len:]

    raw = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    scores = parse_scores(raw)

    return scores, raw

# ============================================================
# RECOVER
# ============================================================

success = 0
failed = 0

print()
print("=" * 70)
print("RECOVERING INVALID ROWS")
print("=" * 70)

for n, idx in enumerate(
    invalid_indices,
    start=1
):

    row = df.loc[idx]

    model_name = str(row["model"])
    sample_id = int(row["sample_id"])

    print(
        f"[{n}/{len(invalid_indices)}] "
        f"{model_name} - sample {sample_id}",
        flush=True
    )

    try:

        scores, raw = judge(
            str(row["prompt"]),
            str(row["response"])
        )

        # One retry if output is still invalid
        if scores is None:

            print("   Retry...", flush=True)

            scores, raw = judge(
                str(row["prompt"]),
                str(row["response"])
            )

        if scores is not None:

            for key in SCORE_COLUMNS:
                df.at[idx, key] = scores[key]

            if "judge_raw_output" in df.columns:
                df.at[idx, "judge_raw_output"] = raw

            if "error" in df.columns:
                df.at[idx, "error"] = ""

            success += 1

            print(
                "   OK:",
                scores,
                flush=True
            )

        else:

            failed += 1

            print(
                "   FAILED",
                flush=True
            )

    except Exception as e:

        failed += 1

        print(
            "   ERROR:",
            str(e),
            flush=True
        )

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FINAL CHECK
# ============================================================

final_invalid = int(
    df[SCORE_COLUMNS].isna().any(axis=1).sum()
)

valid = len(df) - final_invalid

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("Recovered:", success)
print("Failed:", failed)
print("Valid:", valid, "/", len(df))
print("Invalid:", final_invalid)

print()
print("Saved:")
print(OUTPUT_FILE)

if final_invalid == 0:
    print()
    print("🎉 SUCCESS — ALL 136 ROWS ARE VALID 🎉")
else:
    print()
    print("Remaining invalid rows:", final_invalid)